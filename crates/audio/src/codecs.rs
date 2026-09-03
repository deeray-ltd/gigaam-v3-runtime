// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Feature-gated container codecs, including exact Opus reference-clock trimming.

use crate::contracts::{DecodedAudio, EncodedAudio};

#[cfg(any(feature = "decoders", test))]
use crate::contracts::{ChannelAudio, ChannelCount, SampleRate};
#[cfg(any(feature = "decoders", test))]
use crate::resampling::resample_audio;

/// Opus decoded PCM and pre-skip use this fixed RFC reference clock.
#[cfg(any(feature = "decoders", test))]
const OPUS_REFERENCE_SAMPLE_RATE_HZ: u32 = 48_000;
/// GigaAM v3 Runtime's established decoded Opus delivery rate.
#[cfg(any(feature = "decoders", test))]
const OPUS_DELIVERY_SAMPLE_RATE_HZ: u32 = 16_000;
/// RFC 6716's maximum 120 ms packet duration expressed at the 48 kHz reference clock.
#[cfg(feature = "decoders")]
const OPUS_MAX_SAMPLES_PER_CHANNEL_PER_PACKET: usize = 5_760;

#[cfg(any(feature = "decoders", test))]
fn deinterleave(samples: &[f32], channels: ChannelCount) -> Result<Vec<ChannelAudio>, String> {
    if samples.is_empty() {
        return Err("decoded audio contains no samples".into());
    }
    if !samples.len().is_multiple_of(channels.get()) {
        return Err("decoded audio ends with an incomplete interleaved frame".into());
    }
    let frames = samples.len() / channels.get();
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels.get())
        .map_err(|_| "decoded channels cannot reserve memory".to_owned())?;
    for channel_index in 0..channels.get() {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frames)
            .map_err(|_| "decoded channel samples cannot reserve memory".to_owned())?;
        for frame in 0..frames {
            let index = frame
                .checked_mul(channels.get())
                .and_then(|value| value.checked_add(channel_index))
                .ok_or_else(|| "decoded interleaved index overflows usize".to_owned())?;
            channel.push(*samples.get(index).ok_or_else(|| {
                "decoded interleaved samples are shorter than their validated frame count"
                    .to_owned()
            })?);
        }
        output.push(ChannelAudio::new(channel)?);
    }
    Ok(output)
}

#[cfg(any(feature = "decoders", test))]
fn trim_opus_reference_samples(
    samples: &[f32],
    channels: ChannelCount,
    pre_skip_frames: usize,
) -> Result<&[f32], String> {
    let pre_skip_samples = pre_skip_frames
        .checked_mul(channels.get())
        .ok_or_else(|| "Opus interleaved pre-skip overflows usize".to_owned())?;
    samples
        .get(pre_skip_samples..)
        .ok_or_else(|| "Opus declared pre-skip exceeds the decoded audio duration".to_owned())
}

/// Applies Opus's exact 48 kHz pre-skip before the single established 16 kHz conversion.
#[cfg(any(feature = "decoders", test))]
fn opus_trim_then_resample(
    samples: &[f32],
    channels: ChannelCount,
    pre_skip_frames: usize,
) -> Result<DecodedAudio, String> {
    let trimmed = trim_opus_reference_samples(samples, channels, pre_skip_frames)?;
    let reference_rate = SampleRate::new(OPUS_REFERENCE_SAMPLE_RATE_HZ)?;
    let delivery_rate = SampleRate::new(OPUS_DELIVERY_SAMPLE_RATE_HZ)?;
    let decoded = DecodedAudio::new(reference_rate, deinterleave(trimmed, channels)?)?;
    resample_audio(decoded, delivery_rate)
}

#[cfg(feature = "decoders")]
pub fn decode_bytes(encoded: EncodedAudio) -> Result<DecodedAudio, String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{CODEC_TYPE_NULL, CODEC_TYPE_OPUS, DecoderOptions};
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let (bytes, hint_value) = encoded.into_parts();
    let mut hint = Hint::new();
    if let Some(extension) = hint_value.as_deref() {
        hint.with_extension(extension);
    }
    let source = MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| format!("audio format is not recognized: {error}"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "audio input has no decodable track".to_owned())?;
    let track_id = track.id;
    let parameters = track.codec_params.clone();

    if parameters.codec == CODEC_TYPE_OPUS {
        let declared_channels = parameters
            .channels
            .map(|channels| ChannelCount::new(channels.count()))
            .transpose()?
            .ok_or_else(|| "Opus track channel layout is unknown".to_owned())?;
        let pre_skip_frames = parameters
            .delay
            .ok_or_else(|| "Opus track does not declare its required pre-skip".to_owned())
            .and_then(|delay| {
                usize::try_from(delay)
                    .map_err(|_| "Opus declared pre-skip does not fit usize".to_owned())
            })?;
        let mut decoder =
            opus_decoder::OpusDecoder::new(OPUS_REFERENCE_SAMPLE_RATE_HZ, declared_channels.get())
                .map_err(|error| format!("Opus decoder: {error:?}"))?;
        let buffer_length = OPUS_MAX_SAMPLES_PER_CHANNEL_PER_PACKET
            .checked_mul(declared_channels.get())
            .ok_or_else(|| "Opus decoder buffer length overflows usize".to_owned())?;
        let mut buffer = vec![0.0_f32; buffer_length];
        let mut samples = Vec::new();
        loop {
            match format.next_packet() {
                Ok(packet) => {
                    if packet.track_id() != track_id {
                        continue;
                    }
                    let frames = decoder
                        .decode_float(&packet.data, &mut buffer, false)
                        .map_err(|error| format!("Opus decode: {error:?}"))?;
                    let count = frames
                        .checked_mul(declared_channels.get())
                        .ok_or_else(|| "Opus decoded frame count overflows usize".to_owned())?;
                    let decoded = buffer.get(0..count).ok_or_else(|| {
                        "Opus decoder returned more frames than its fixed buffer permits".to_owned()
                    })?;
                    samples.extend_from_slice(decoded);
                }
                Err(SymphoniaError::IoError(error))
                    if error.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(SymphoniaError::ResetRequired) => {
                    return Err("unsupported midstream track/configuration change".into());
                }
                Err(error) => return Err(format!("audio demux: {error}")),
            }
        }
        return opus_trim_then_resample(&samples, declared_channels, pre_skip_frames);
    }

    let mut decoder = symphonia::default::get_codecs()
        .make(&parameters, &DecoderOptions::default())
        .map_err(|error| format!("audio codec: {error}"))?;
    let mut specification = None;
    let mut samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err("unsupported midstream track/configuration change".into());
            }
            Err(error) => return Err(format!("audio demux: {error}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio) => {
                let spec = *audio.spec();
                let rate = SampleRate::new(spec.rate)?;
                let channels = ChannelCount::new(spec.channels.count())?;
                match specification {
                    Some((known_rate, known_channels))
                        if known_rate != rate || known_channels != channels =>
                    {
                        return Err("unsupported midstream track/configuration change".into());
                    }
                    Some(_) => {}
                    None => specification = Some((rate, channels)),
                }
                let capacity = u64::try_from(audio.capacity())
                    .map_err(|_| "decoded packet capacity does not fit u64".to_owned())?;
                // A fresh buffer per packet ensures a changed packet capacity/specification can
                // never be hidden behind a buffer built for an earlier packet.
                let mut packet_buffer = SampleBuffer::<f32>::new(capacity, spec);
                packet_buffer.copy_interleaved_ref(audio);
                samples.extend_from_slice(packet_buffer.samples());
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => {
                return Err("unsupported midstream track/configuration change".into());
            }
            Err(error) => return Err(format!("audio decode: {error}")),
        }
    }
    let (rate, channels) =
        specification.ok_or_else(|| "audio input produced no decoded packets".to_owned())?;
    DecodedAudio::new(rate, deinterleave(&samples, channels)?)
}

#[cfg(not(feature = "decoders"))]
pub fn decode_bytes(encoded: EncodedAudio) -> Result<DecodedAudio, String> {
    match encoded.format_hint() {
        Some(extension) => Err(format!(
            "audio extension {extension:?} is unsupported because this build has no `decoders` feature"
        )),
        None => Err(
            "audio format has no extension hint and is unsupported because this build has no `decoders` feature"
                .into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        OPUS_DELIVERY_SAMPLE_RATE_HZ, OPUS_REFERENCE_SAMPLE_RATE_HZ, deinterleave,
        opus_trim_then_resample, trim_opus_reference_samples,
    };
    use crate::{ChannelCount, DecodedAudio, SampleRate, resample_audio};

    #[test]
    fn opus_trim_is_exact_at_48khz_before_delivery_resampling() -> Result<(), String> {
        let channels = ChannelCount::new(2)?;
        let pre_skip_frames = 1_usize;
        assert!(!pre_skip_frames.is_multiple_of(3));
        let reference = [
            -0.9_f32, 0.9, -0.7, 0.7, -0.5, 0.5, -0.3, 0.3, -0.1, 0.1, 0.1, -0.1, 0.3, -0.3, 0.5,
            -0.5,
        ];
        let trimmed = trim_opus_reference_samples(&reference, channels, pre_skip_frames)?;
        assert_eq!(trimmed, &reference[2..]);

        let expected_reference = DecodedAudio::new(
            SampleRate::new(OPUS_REFERENCE_SAMPLE_RATE_HZ)?,
            deinterleave(trimmed, channels)?,
        )?;
        let expected = resample_audio(
            expected_reference,
            SampleRate::new(OPUS_DELIVERY_SAMPLE_RATE_HZ)?,
        )?;
        let actual = opus_trim_then_resample(&reference, channels, pre_skip_frames)?;
        assert_eq!(actual.sample_rate().hertz(), OPUS_DELIVERY_SAMPLE_RATE_HZ);
        assert_eq!(actual, expected);
        Ok(())
    }
}
