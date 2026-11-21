// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Typed decoded audio shared by file and live inputs.

use std::num::NonZeroUsize;

use crate::wav::{self, WavReader};
use crate::{
    ChannelAudio, ChannelCount, DecodedAudio, InterleavedFrameDecoder, SampleFormat, SampleRate,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioMetadata {
    sample_rate: SampleRate,
    channels: ChannelCount,
}

impl AudioMetadata {
    pub const fn new(sample_rate: SampleRate, channels: ChannelCount) -> Self {
        Self {
            sample_rate,
            channels,
        }
    }

    pub const fn sample_rate(self) -> SampleRate {
        self.sample_rate
    }

    pub const fn channel_count(self) -> ChannelCount {
        self.channels
    }
}

/// A nonempty group of complete frames on the decoded source sample clock.
///
/// Frame zero is the first emitted sample after format-defined trimming. Positions
/// count emitted frames, without using container PTS or inserting gaps for
/// packets skipped by a decoder.
#[derive(Debug, PartialEq)]
pub struct DecodedChunk {
    first_frame: u64,
    end_frame: u64,
    audio: DecodedAudio,
}

impl DecodedChunk {
    pub fn new(first_frame: u64, audio: DecodedAudio) -> Result<Self, String> {
        let frames = u64::try_from(audio.frames())
            .map_err(|_| "decoded frame count exceeds u64".to_owned())?;
        let end_frame = first_frame
            .checked_add(frames)
            .ok_or_else(|| "decoded end frame overflows u64".to_owned())?;
        Ok(Self {
            first_frame,
            end_frame,
            audio,
        })
    }

    pub const fn first_frame(&self) -> u64 {
        self.first_frame
    }

    pub const fn end_frame(&self) -> u64 {
        self.end_frame
    }

    pub const fn audio(&self) -> &DecodedAudio {
        &self.audio
    }

    pub fn into_audio(self) -> DecodedAudio {
        self.audio
    }
}

/// Decoder for explicitly configured headerless live input.
pub struct RawDecoder {
    decoder: InterleavedFrameDecoder,
    metadata: AudioMetadata,
    next_frame: u64,
}

impl RawDecoder {
    pub const SUPPORTED_FORMATS: [SampleFormat; 4] = [
        SampleFormat::Pcm16,
        SampleFormat::F32,
        SampleFormat::Alaw,
        SampleFormat::Ulaw,
    ];

    pub fn new(
        format: SampleFormat,
        sample_rate: SampleRate,
        channels: ChannelCount,
    ) -> Result<Self, String> {
        Ok(Self {
            decoder: InterleavedFrameDecoder::new(format, channels)?,
            metadata: AudioMetadata::new(sample_rate, channels),
            next_frame: 0,
        })
    }

    pub const fn metadata(&self) -> AudioMetadata {
        self.metadata
    }

    /// Emits each available complete frame from this byte partition.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Option<DecodedChunk>, String> {
        let Some(channels) = self.decoder.push(bytes)? else {
            return Ok(None);
        };
        let chunk = DecodedChunk::new(
            self.next_frame,
            DecodedAudio::new(self.metadata.sample_rate(), channels)?,
        )?;
        self.next_frame = chunk.end_frame();
        Ok(Some(chunk))
    }

    pub fn finish(self) -> Result<(), String> {
        self.decoder.finish()
    }
}

/// Pulls source-rate, physical-channel chunks from an encoded file.
pub struct FileDecoder {
    source: FileSource,
    metadata: AudioMetadata,
    next_frame: u64,
    state: FileState,
}

enum FileState {
    Active,
    Eof,
    Failed(String),
}

enum FileSource {
    Wav(WavReader<std::io::Cursor<Vec<u8>>>),
    #[cfg(feature = "decoders")]
    Codec(crate::codecs::CodecFileReader),
}

impl FileDecoder {
    pub fn open(encoded: crate::EncodedAudio) -> Result<Self, String> {
        let (bytes, hint) = encoded.into_parts();
        let wav = wav::is_riff_wave(&bytes);
        let source = if wav {
            FileSource::Wav(WavReader::open(std::io::Cursor::new(bytes))?)
        } else {
            #[cfg(feature = "decoders")]
            {
                FileSource::Codec(crate::codecs::CodecFileReader::open(
                    std::io::Cursor::new(bytes),
                    hint.as_deref(),
                )?)
            }
            #[cfg(not(feature = "decoders"))]
            {
                return Err(crate::codecs::unsupported_format(hint.as_deref()));
            }
        };
        let metadata = match &source {
            FileSource::Wav(reader) => reader.metadata(),
            #[cfg(feature = "decoders")]
            FileSource::Codec(reader) => reader.metadata(),
        };
        Ok(Self {
            source,
            metadata,
            next_frame: 0,
            state: FileState::Active,
        })
    }
    pub const fn metadata(&self) -> AudioMetadata {
        self.metadata
    }
    pub fn next_chunk(&mut self, max_frames: NonZeroUsize) -> Result<Option<DecodedChunk>, String> {
        match &self.state {
            FileState::Active => {}
            FileState::Eof => return Ok(None),
            FileState::Failed(error) => return Err(error.clone()),
        }
        let result = match &mut self.source {
            FileSource::Wav(reader) => reader.next_audio(max_frames.get()),
            #[cfg(feature = "decoders")]
            FileSource::Codec(reader) => reader.next_audio(max_frames.get()),
        };
        match result {
            Ok(Some(audio)) => {
                let chunk = DecodedChunk::new(self.next_frame, audio)?;
                self.next_frame = chunk.end_frame();
                Ok(Some(chunk))
            }
            Ok(None) => {
                self.state = FileState::Eof;
                Ok(None)
            }
            Err(error) => {
                self.state = FileState::Failed(error.clone());
                Err(error)
            }
        }
    }
    /// Collects the decoded remainder at its source rate and in physical channel order.
    pub fn collect_remaining(&mut self) -> Result<DecodedAudio, String> {
        let chunk_size = NonZeroUsize::new(8192).expect("chunk size is nonzero");
        collect_audio(self.metadata(), || {
            self.next_chunk(chunk_size)
                .map(|chunk| chunk.map(DecodedChunk::into_audio))
        })
    }
}

pub(crate) fn collect_audio(
    metadata: AudioMetadata,
    mut next: impl FnMut() -> Result<Option<DecodedAudio>, String>,
) -> Result<DecodedAudio, String> {
    let mut channels = vec![Vec::new(); metadata.channel_count().get()];
    while let Some(audio) = next()? {
        for (destination, source) in channels.iter_mut().zip(audio.channels()) {
            destination
                .try_reserve(source.len())
                .map_err(|_| "decoded channel samples cannot reserve memory".to_owned())?;
            destination.extend_from_slice(source.samples());
        }
    }
    DecodedAudio::new(
        metadata.sample_rate(),
        channels
            .into_iter()
            .map(ChannelAudio::new)
            .collect::<Result<_, _>>()?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn stereo_float_wav(frames: u32, duplicate_data: bool) -> Vec<u8> {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        let bytes = frames * 8;
        let extra = if duplicate_data { 8 } else { 0 };
        wav.extend_from_slice(&(36 + bytes + extra).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&3_u16.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&48_000_u32.to_le_bytes());
        wav.extend_from_slice(&384_000_u32.to_le_bytes());
        wav.extend_from_slice(&8_u16.to_le_bytes());
        wav.extend_from_slice(&32_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&bytes.to_le_bytes());
        for _ in 0..frames {
            wav.extend_from_slice(&1.25_f32.to_le_bytes());
            wav.extend_from_slice(&(-0.25_f32).to_le_bytes());
        }
        if duplicate_data {
            wav.extend_from_slice(b"data\0\0\0\0");
        }
        wav
    }

    #[test]
    fn strict_wav_emits_bounded_chunks_and_rejects_duplicate_data() -> Result<(), String> {
        let mut decoder = FileDecoder::open(crate::EncodedAudio::new(
            stereo_float_wav(1003, false),
            Some("MP3".into()),
        )?)?;
        assert_eq!(decoder.metadata().sample_rate().hertz(), 48_000);
        assert_eq!(decoder.metadata().channel_count().get(), 2);
        let limit = NonZeroUsize::new(100).expect("100 is nonzero");
        let mut frames = 0;
        while let Some(chunk) = decoder.next_chunk(limit)? {
            assert_eq!(chunk.first_frame(), frames);
            let audio = chunk.audio();
            assert!(audio.frames() <= 100);
            assert!(
                audio.channels()[0]
                    .samples()
                    .iter()
                    .all(|sample| *sample == 1.25)
            );
            assert!(
                audio.channels()[1]
                    .samples()
                    .iter()
                    .all(|sample| *sample == -0.25)
            );
            frames = chunk.end_frame();
        }
        assert_eq!(frames, 1003);
        assert!(decoder.next_chunk(limit)?.is_none());
        assert!(
            FileDecoder::open(crate::EncodedAudio::new(stereo_float_wav(3, true), None)?).is_err()
        );
        Ok(())
    }

    #[test]
    fn wav_and_raw_share_source_positions_and_physical_samples() -> Result<(), String> {
        let frames = [
            (0_i16, 16384_i16),
            (-32768, -8192),
            (8192, 32767),
            (1234, -2345),
        ];
        let bytes: Vec<u8> = frames
            .iter()
            .flat_map(|(left, right)| left.to_le_bytes().into_iter().chain(right.to_le_bytes()))
            .collect();
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(
            &(36_u32 + u32::try_from(bytes.len()).map_err(|e| e.to_string())?).to_le_bytes(),
        );
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&8_000_u32.to_le_bytes());
        wav.extend_from_slice(&32_000_u32.to_le_bytes());
        wav.extend_from_slice(&4_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(
            &u32::try_from(bytes.len())
                .map_err(|e| e.to_string())?
                .to_le_bytes(),
        );
        wav.extend_from_slice(&bytes);
        let expected: [Vec<f32>; 2] = [
            frames
                .iter()
                .map(|(v, _)| f32::from(*v) / 32768.0)
                .collect(),
            frames
                .iter()
                .map(|(_, v)| f32::from(*v) / 32768.0)
                .collect(),
        ];
        let mut file = FileDecoder::open(crate::EncodedAudio::new(wav, None)?)?;
        let mut live = RawDecoder::new(
            SampleFormat::Pcm16,
            SampleRate::new(8000)?,
            ChannelCount::new(2)?,
        )?;
        assert_eq!(file.metadata(), live.metadata());
        let mut file_channels = [Vec::new(), Vec::new()];
        let mut file_position = 0;
        while let Some(chunk) = file.next_chunk(NonZeroUsize::new(3).ok_or("nonzero")?)? {
            assert_eq!(chunk.first_frame(), file_position);
            file_position = chunk.end_frame();
            for (out, channel) in file_channels.iter_mut().zip(chunk.audio().channels()) {
                out.extend_from_slice(channel.samples());
            }
        }
        let mut live_channels = [Vec::new(), Vec::new()];
        let mut live_position = 0;
        for part in bytes.chunks(3) {
            if let Some(chunk) = live.push(part)? {
                assert_eq!(chunk.first_frame(), live_position);
                live_position = chunk.end_frame();
                for (out, channel) in live_channels.iter_mut().zip(chunk.audio().channels()) {
                    out.extend_from_slice(channel.samples());
                }
            }
        }
        live.finish()?;
        assert_eq!(file_position, 4);
        assert_eq!(live_position, 4);
        assert_eq!(file_channels, expected);
        assert_eq!(live_channels, expected);
        Ok(())
    }

    #[test]
    fn live_raw_emits_before_eof_and_tracks_positions_across_partitions() -> Result<(), String> {
        let rate = SampleRate::new(8000)?;
        let channels = ChannelCount::new(2)?;
        let mut raw = RawDecoder::new(SampleFormat::Pcm16, rate, channels)?;
        assert_eq!(raw.metadata(), AudioMetadata::new(rate, channels));
        let input = [i16::MIN, 16384, 0, -16384, i16::MAX, 0];
        let bytes: Vec<u8> = input
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        assert!(raw.push(&bytes[..3])?.is_none());
        let first = raw
            .push(&bytes[3..5])?
            .ok_or("first full frame is available before EOF")?;
        assert_eq!((first.first_frame(), first.end_frame()), (0, 1));
        assert_eq!(first.audio().channels()[0].samples(), &[-1.0]);
        assert_eq!(first.audio().channels()[1].samples(), &[0.5]);
        let tail = raw
            .push(&bytes[5..])?
            .ok_or("two additional frames are available before EOF")?;
        assert_eq!((tail.first_frame(), tail.end_frame()), (1, 3));
        assert_eq!(
            tail.audio().channels()[0].samples(),
            &[0.0, f32::from(i16::MAX) / 32768.0]
        );
        assert_eq!(tail.audio().channels()[1].samples(), &[-0.5, 0.0]);
        raw.finish()?;

        let mut partial = RawDecoder::new(SampleFormat::Pcm16, rate, channels)?;
        assert!(partial.push(&bytes[..5])?.is_some());
        assert!(partial.finish().is_err());
        Ok(())
    }
}
