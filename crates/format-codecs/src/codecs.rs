// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Feature-gated container codecs, including exact Opus reference-clock trimming.

use gigaam_audio::{DecodedAudio, EncodedAudio, FileDecoder};

use gigaam_audio::{AudioMetadata, ChannelAudio, ChannelCount, SampleRate, resample_audio};
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use symphonia::core::{
    codecs::audio::{
        AudioDecoder, AudioDecoderOptions, CODEC_ID_NULL_AUDIO, well_known::CODEC_ID_OPUS,
    },
    errors::Error as SymphoniaError,
    formats::probe::Hint,
    formats::{FormatOptions, FormatReader},
    io::{MediaSource, MediaSourceStream},
    meta::MetadataOptions,
};

/// Opus decoded PCM and pre-skip use this fixed RFC reference clock.
const OPUS_REFERENCE_SAMPLE_RATE_HZ: u32 = 48_000;
/// GigaAM v3 Runtime's established decoded Opus delivery rate.
const OPUS_DELIVERY_SAMPLE_RATE_HZ: u32 = 16_000;
/// RFC 6716's maximum 120 ms packet duration expressed at the 48 kHz reference clock.
const OPUS_MAX_SAMPLES_PER_CHANNEL_PER_PACKET: usize = 5_760;

fn deinterleave(samples: &[f32], metadata: AudioMetadata) -> Result<DecodedAudio, String> {
    let channels = metadata.channel_count().get();
    let frames = samples.len() / channels;
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|_| "decoded channels cannot reserve memory".to_owned())?;
    for channel_index in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frames)
            .map_err(|_| "decoded channel samples cannot reserve memory".to_owned())?;
        channel.extend(
            samples
                .chunks_exact(channels)
                .map(|frame| frame[channel_index]),
        );
        output.push(ChannelAudio::new(channel)?);
    }
    DecodedAudio::new(metadata.sample_rate(), output)
}

struct SeekableSource<R> {
    input: R,
    length: u64,
}

impl<R: Read> Read for SeekableSource<R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.input.read(bytes)
    }
}

impl<R: Seek> Seek for SeekableSource<R> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.input.seek(position)
    }
}

impl<R: Read + Seek + Send + Sync> MediaSource for SeekableSource<R> {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.length)
    }
}

struct OpusPackets {
    decoder: opus_decoder::OpusDecoder,
    pending_skip: usize,
    scratch: Vec<f32>,
}

enum PacketDecoder {
    Standard(Box<dyn AudioDecoder>),
    Opus(Box<OpusPackets>),
}

/// The codec adapter retains one decoded packet at a time plus the linked demuxer's state.
struct CodecReader {
    format: Box<dyn FormatReader + 'static>,
    track_id: u32,
    decoder: PacketDecoder,
    metadata: Option<AudioMetadata>,
    samples: Vec<f32>,
    offset: usize,
}

impl CodecReader {
    fn open(
        mut input: impl Read + Seek + Send + Sync + 'static,
        format_hint: Option<&str>,
    ) -> Result<Self, String> {
        let length = input.seek(SeekFrom::End(0)).map_err(audio_io)?;
        let audio_offset = crate::id3v2::audio_offset(&mut input, length)?;
        input.rewind().map_err(audio_io)?;
        let mut source = MediaSourceStream::new(
            Box::new(SeekableSource { input, length }),
            Default::default(),
        );
        source
            .seek(SeekFrom::Start(audio_offset))
            .map_err(audio_io)?;
        let mut hint = Hint::new();
        if let Some(extension) = format_hint {
            hint.with_extension(extension);
        }
        let format = crate::batch_probe::BATCH_PROBE
            .probe(
                &hint,
                source,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|error| format!("audio format is not recognized: {error}"))?;
        let (track, parameters) = format
            .tracks()
            .iter()
            .filter_map(|track| Some((track, track.codec_params.as_ref()?.audio()?)))
            .find(|(_, parameters)| parameters.codec != CODEC_ID_NULL_AUDIO)
            .ok_or_else(|| "audio input has no decodable track".to_owned())?;
        let track_id = track.id;
        let (metadata, decoder) = if parameters.codec == CODEC_ID_OPUS {
            let channels = parameters
                .channels
                .as_ref()
                .map(|channels| ChannelCount::new(channels.count()))
                .transpose()?
                .ok_or_else(|| "Opus track channel layout is unknown".to_owned())?;
            let pre_skip_frames = track
                .delay
                .ok_or_else(|| "Opus track does not declare its required pre-skip".to_owned())
                .and_then(|delay| {
                    usize::try_from(delay)
                        .map_err(|_| "Opus declared pre-skip does not fit usize".to_owned())
                })?;
            let rate = SampleRate::new(OPUS_REFERENCE_SAMPLE_RATE_HZ)?;
            let decoder = opus_decoder::OpusDecoder::new(rate.hertz(), channels.get())
                .map_err(|error| format!("Opus decoder: {error:?}"))?;
            let buffer_length = OPUS_MAX_SAMPLES_PER_CHANNEL_PER_PACKET
                .checked_mul(channels.get())
                .ok_or_else(|| "Opus decoder buffer length overflows usize".to_owned())?;
            (
                Some(AudioMetadata::new(rate, channels)),
                PacketDecoder::Opus(Box::new(OpusPackets {
                    decoder,
                    pending_skip: pre_skip_frames,
                    scratch: vec![0.0_f32; buffer_length],
                })),
            )
        } else {
            let decoder = symphonia::default::get_codecs()
                .make_audio_decoder(parameters, &AudioDecoderOptions::default())
                .map_err(|error| format!("audio codec: {error}"))?;
            (None, PacketDecoder::Standard(decoder))
        };
        let mut reader = Self {
            format,
            track_id,
            decoder,
            metadata,
            samples: Vec::new(),
            offset: 0,
        };
        if !reader.read_packet()? {
            return Err("audio input produced no decoded samples".into());
        }
        Ok(reader)
    }

    fn metadata(&self) -> AudioMetadata {
        self.metadata
            .expect("opening a codec stream decodes its first nonempty packet")
    }

    fn uses_opus(&self) -> bool {
        matches!(self.decoder, PacketDecoder::Opus(_))
    }

    fn next_audio(&mut self, max_frames: usize) -> Result<Option<DecodedAudio>, String> {
        if self.offset == self.samples.len() && !self.read_packet()? {
            return Ok(None);
        }
        let metadata = self.metadata();
        let channels = metadata.channel_count().get();
        let frames = ((self.samples.len() - self.offset) / channels).min(max_frames);
        let end = self.offset + frames * channels;
        let result = deinterleave(&self.samples[self.offset..end], metadata)?;
        self.offset = end;
        Ok(Some(result))
    }

    fn read_packet(&mut self) -> Result<bool, String> {
        self.samples.clear();
        self.offset = 0;
        loop {
            let mut packet = match self.format.next_packet() {
                Ok(Some(packet)) if packet.track_id == self.track_id => packet,
                Ok(Some(_)) => continue,
                Ok(None) => {
                    if let PacketDecoder::Opus(opus) = &self.decoder
                        && opus.pending_skip != 0
                    {
                        return Err(
                            "Opus declared pre-skip exceeds the decoded audio duration".into()
                        );
                    }
                    return Ok(false);
                }
                Err(SymphoniaError::ResetRequired) => return Err(configuration_change()),
                Err(error) => return Err(format!("audio demux: {error}")),
            };
            match &mut self.decoder {
                PacketDecoder::Opus(opus) => {
                    let channels = self
                        .metadata
                        .expect("Opus carries validated source metadata before decoding")
                        .channel_count()
                        .get();
                    let frames = opus
                        .decoder
                        .decode_float(&packet.data, &mut opus.scratch, false)
                        .map_err(|error| format!("Opus decode: {error:?}"))?;
                    let skip = opus.pending_skip.min(frames);
                    opus.pending_skip -= skip;
                    let count = frames
                        .checked_mul(channels)
                        .ok_or_else(|| "Opus decoded frame count overflows usize".to_owned())?;
                    let decoded = opus.scratch.get(skip * channels..count).ok_or_else(|| {
                        "Opus decoder returned more frames than its fixed buffer permits".to_owned()
                    })?;
                    self.samples.extend_from_slice(decoded);
                }
                PacketDecoder::Standard(decoder) => {
                    // The previous decoder disabled gapless trimming. Retain
                    // its output sample clock for these container comparisons.
                    packet.dur = packet
                        .dur
                        .checked_add(packet.trim_start)
                        .and_then(|duration| duration.checked_add(packet.trim_end))
                        .ok_or_else(|| "decoded packet duration overflows".to_owned())?;
                    packet.trim_start = symphonia::core::units::Duration::ZERO;
                    packet.trim_end = symphonia::core::units::Duration::ZERO;
                    let audio = match decoder.decode(&packet) {
                        Ok(audio) => audio,
                        Err(SymphoniaError::DecodeError(_)) => continue,
                        Err(SymphoniaError::ResetRequired) => return Err(configuration_change()),
                        Err(error) => return Err(format!("audio decode: {error}")),
                    };
                    let spec = audio.spec();
                    let metadata = AudioMetadata::new(
                        SampleRate::new(spec.rate())?,
                        ChannelCount::new(spec.channels().count())?,
                    );
                    if self.metadata.is_some_and(|known| known != metadata) {
                        return Err(configuration_change());
                    }
                    self.metadata = Some(metadata);
                    audio.copy_to_vec_interleaved(&mut self.samples);
                }
            }
            if !self.samples.is_empty() {
                return Ok(true);
            }
        }
    }
}

/// Container file delivery on the decoded source sample clock.
pub(crate) struct CodecFileReader {
    inner: CodecReader,
}

impl CodecFileReader {
    pub(crate) fn open(
        input: impl Read + Seek + Send + Sync + 'static,
        format_hint: Option<&str>,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: CodecReader::open(input, format_hint)?,
        })
    }

    pub(crate) fn metadata(&self) -> AudioMetadata {
        self.inner.metadata()
    }

    pub(crate) fn next_audio(&mut self, max_frames: usize) -> Result<Option<DecodedAudio>, String> {
        self.inner.next_audio(max_frames)
    }
}

fn audio_io(error: io::Error) -> String {
    format!("audio source: {error}")
}

fn configuration_change() -> String {
    "unsupported midstream track/configuration change".into()
}

pub fn decode_bytes(encoded: EncodedAudio) -> Result<DecodedAudio, String> {
    let (bytes, format_hint) = encoded.into_parts();
    let reader = CodecReader::open(Cursor::new(bytes), format_hint.as_deref())?;
    let is_opus = reader.uses_opus();
    let mut decoder = FileDecoder::from_reader(Box::new(CodecFileReader { inner: reader }));
    let audio = decoder.collect_remaining()?;
    if is_opus {
        resample_audio(audio, SampleRate::new(OPUS_DELIVERY_SAMPLE_RATE_HZ)?)
    } else {
        Ok(audio)
    }
}

impl gigaam_audio::FileChunkReader for CodecFileReader {
    fn metadata(&self) -> AudioMetadata {
        CodecFileReader::metadata(self)
    }
    fn next_audio(
        &mut self,
        max_frames: std::num::NonZeroUsize,
    ) -> Result<Option<DecodedAudio>, String> {
        CodecFileReader::next_audio(self, max_frames.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gigaam_audio::FileDecoder;
    use std::num::NonZeroUsize;

    fn append_ogg_page(
        file: &mut Vec<u8>,
        sequence: u32,
        flags: u8,
        granule: u64,
        packet: &[u8],
    ) -> Result<(), String> {
        let packet_size = u8::try_from(packet.len()).map_err(|error| error.to_string())?;
        let start = file.len();
        file.extend_from_slice(b"OggS");
        file.extend_from_slice(&[0, flags]);
        file.extend_from_slice(&granule.to_le_bytes());
        file.extend_from_slice(&1_u32.to_le_bytes());
        file.extend_from_slice(&sequence.to_le_bytes());
        file.extend_from_slice(&0_u32.to_le_bytes());
        file.extend_from_slice(&[1, packet_size]);
        file.extend_from_slice(packet);
        let mut checksum = 0_u32;
        for byte in &file[start..] {
            checksum ^= u32::from(*byte) << 24;
            for _ in 0..8 {
                let carry = checksum & (1 << 31) != 0;
                checksum <<= 1;
                if carry {
                    checksum ^= 0x04c1_1db7;
                }
            }
        }
        file[start + 22..start + 26].copy_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    #[test]
    fn opus_trims_on_the_reference_clock_across_packets() -> Result<(), String> {
        let pre_skip = 961_u16;
        let mut head = b"OpusHead\x01\x01".to_vec();
        head.extend_from_slice(&pre_skip.to_le_bytes());
        head.extend_from_slice(&48_000_u32.to_le_bytes());
        head.extend_from_slice(&[0; 3]);
        let mut file = Vec::new();
        append_ogg_page(&mut file, 0, 2, 0, &head)?;
        append_ogg_page(&mut file, 1, 0, 0, b"OpusTags\0\0\0\0\0\0\0\0")?;
        let packet = [0xf8, 0x20, 0x56, 0xa7, 0x92, 0xe6, 0x18];
        append_ogg_page(&mut file, 2, 0, 960, &packet)?;
        append_ogg_page(&mut file, 3, 0, 1920, &packet)?;
        append_ogg_page(&mut file, 4, 4, 2880, &packet)?;

        let mut oracle =
            opus_decoder::OpusDecoder::new(48_000, 1).map_err(|error| format!("{error:?}"))?;
        let mut reference = vec![0_f32; 2880];
        for segment in reference.as_chunks_mut::<960>().0 {
            assert_eq!(
                oracle
                    .decode_float(&packet, segment, false)
                    .map_err(|error| format!("{error:?}"))?,
                960
            );
        }
        let trimmed = &reference[usize::from(pre_skip)..];
        assert!(trimmed.iter().any(|&sample| sample != 0.0));
        let aggregate = decode_bytes(EncodedAudio::new(file.clone(), None)?)?;
        let expected = resample_audio(
            DecodedAudio::new(
                SampleRate::new(48_000)?,
                vec![ChannelAudio::new(trimmed.to_vec())?],
            )?,
            SampleRate::new(16_000)?,
        )?;
        assert_eq!(aggregate, expected);

        for bound in [1, 137, 5700] {
            let limit = NonZeroUsize::new(bound).ok_or("nonzero test bound")?;
            let mut stream = FileDecoder::from_reader(Box::new(CodecFileReader::open(
                Cursor::new(file.clone()),
                None,
            )?));
            assert_eq!(stream.metadata().sample_rate(), SampleRate::new(48_000)?);
            let mut actual = Vec::new();
            while let Some(chunk) = stream.next_chunk(limit)? {
                assert!(chunk.audio().frames() <= bound);
                assert_eq!(
                    usize::try_from(chunk.first_frame()).map_err(|error| error.to_string())?,
                    actual.len()
                );
                actual.extend_from_slice(chunk.audio().channels()[0].samples());
            }
            assert!(stream.next_chunk(limit)?.is_none());
            assert_eq!(actual, trimmed);
            let at_model_rate = resample_audio(
                DecodedAudio::new(SampleRate::new(48_000)?, vec![ChannelAudio::new(actual)?])?,
                SampleRate::new(16_000)?,
            )?;
            assert_eq!(at_model_rate, expected);
        }

        Ok(())
    }
}
