// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Strict RIFF/WAV container traversal and decoding.

use crate::contracts::{ChannelAudio, ChannelCount, DecodedAudio, SampleRate};
use crate::g711;
use gigaam_primitives::{i32_to_f32, u32_to_usize};

const RIFF_HEADER_LENGTH: usize = 12;
const CHUNK_HEADER_LENGTH: usize = 8;
const PCM_TAG: u16 = 1;
const IEEE_FLOAT_TAG: u16 = 3;
const ALAW_TAG: u16 = 6;
const ULAW_TAG: u16 = 7;
const EXTENSIBLE_TAG: u16 = 0xfffe;
const PCM_SUBFORMAT_GUID: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];
const IEEE_FLOAT_SUBFORMAT_GUID: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];

#[derive(Clone, Copy)]
struct WavFormat {
    tag: u16,
    channels: ChannelCount,
    sample_rate: SampleRate,
    bits_per_sample: u16,
    block_align: usize,
}

pub(crate) fn is_riff_wave(bytes: &[u8]) -> bool {
    bytes.len() >= RIFF_HEADER_LENGTH
        && bytes.get(0..4) == Some(b"RIFF")
        && bytes.get(8..12) == Some(b"WAVE")
}

fn range<'a>(
    bytes: &'a [u8],
    start: usize,
    length: usize,
    context: &str,
) -> Result<&'a [u8], String> {
    let end = start
        .checked_add(length)
        .ok_or_else(|| format!("{context}: offset overflows usize"))?;
    bytes
        .get(start..end)
        .ok_or_else(|| format!("{context}: truncated input"))
}

fn u16_at(bytes: &[u8], start: usize, context: &str) -> Result<u16, String> {
    let value: [u8; 2] = range(bytes, start, 2, context)?
        .try_into()
        .map_err(|_| format!("{context}: invalid 16-bit field"))?;
    Ok(u16::from_le_bytes(value))
}

fn u32_at(bytes: &[u8], start: usize, context: &str) -> Result<u32, String> {
    let value: [u8; 4] = range(bytes, start, 4, context)?
        .try_into()
        .map_err(|_| format!("{context}: invalid 32-bit field"))?;
    Ok(u32::from_le_bytes(value))
}

fn parse_format(body: &[u8]) -> Result<WavFormat, String> {
    if body.len() < 16 {
        return Err("WAV fmt chunk is shorter than 16 bytes".into());
    }
    let original_tag = u16_at(body, 0, "WAV fmt tag")?;
    let tag = if original_tag == EXTENSIBLE_TAG {
        if body.len() < 40 {
            return Err("WAV extensible fmt chunk is shorter than 40 bytes".into());
        }
        let extension_size = u16_at(body, 16, "WAV extensible fmt extension size")?;
        if extension_size != 22 {
            return Err("WAV extensible fmt chunk must have exactly 22 extension bytes".into());
        }
        let extension_end = 18_usize
            .checked_add(usize::from(extension_size))
            .ok_or_else(|| "WAV extensible fmt extension length overflows usize".to_owned())?;
        if body.len() < extension_end {
            return Err("WAV extensible fmt extension is truncated".into());
        }
        let bits_per_sample = u16_at(body, 14, "WAV bits per sample")?;
        let valid_bits = u16_at(body, 18, "WAV extensible valid bits")?;
        if valid_bits != bits_per_sample {
            return Err(
                "WAV extensible valid bits must exactly match the decoded sample width".into(),
            );
        }
        let subtype: [u8; 16] = range(body, 24, 16, "WAV extensible subtype GUID")?
            .try_into()
            .map_err(|_| "WAV extensible subtype GUID has invalid width".to_owned())?;
        if subtype == PCM_SUBFORMAT_GUID {
            PCM_TAG
        } else if subtype == IEEE_FLOAT_SUBFORMAT_GUID {
            IEEE_FLOAT_TAG
        } else {
            return Err("WAV extensible subtype GUID is unsupported".into());
        }
    } else {
        original_tag
    };
    let channels = ChannelCount::new(usize::from(u16_at(body, 2, "WAV channel count")?))?;
    let sample_rate = SampleRate::new(u32_at(body, 4, "WAV sample rate")?)?;
    let byte_rate = u32_to_usize(u32_at(body, 8, "WAV byte rate")?)
        .map_err(|error| format!("WAV byte rate: {error}"))?;
    let block_align = usize::from(u16_at(body, 12, "WAV block alignment")?);
    let bits_per_sample = u16_at(body, 14, "WAV bits per sample")?;
    if bits_per_sample == 0 {
        return Err("WAV bits per sample must be nonzero".into());
    }
    let bytes_per_sample = match (tag, bits_per_sample) {
        (PCM_TAG, 8 | 16 | 24 | 32) | (IEEE_FLOAT_TAG, 32) | (ALAW_TAG | ULAW_TAG, 8) => {
            usize::from(bits_per_sample / 8)
        }
        _ => {
            return Err(format!(
                "unsupported WAV format: tag={tag} bits={bits_per_sample}"
            ));
        }
    };
    let expected_block_align = bytes_per_sample
        .checked_mul(channels.get())
        .ok_or_else(|| "WAV block alignment overflows usize".to_owned())?;
    if block_align != expected_block_align {
        return Err(format!(
            "WAV block alignment {block_align} does not match format frame width {expected_block_align}"
        ));
    }
    let expected_byte_rate = sample_rate
        .as_usize()?
        .checked_mul(block_align)
        .ok_or_else(|| "WAV byte rate overflows usize".to_owned())?;
    if byte_rate != expected_byte_rate {
        return Err(format!(
            "WAV byte rate {byte_rate} does not match sample rate and frame width {expected_byte_rate}"
        ));
    }
    Ok(WavFormat {
        tag,
        channels,
        sample_rate,
        bits_per_sample,
        block_align,
    })
}

fn decode_wav_sample(format: WavFormat, sample: &[u8]) -> Result<f32, String> {
    match (format.tag, format.bits_per_sample) {
        (PCM_TAG, 8) => {
            let byte = *sample
                .first()
                .ok_or_else(|| "WAV PCM8 sample is truncated".to_owned())?;
            Ok((f32::from(byte) - 128.0) / 128.0)
        }
        (PCM_TAG, 16) => {
            let value: [u8; 2] = sample
                .try_into()
                .map_err(|_| "WAV PCM16 sample is truncated".to_owned())?;
            Ok(f32::from(i16::from_le_bytes(value)) / 32_768.0)
        }
        (PCM_TAG, 24) => {
            let value: [u8; 3] = sample
                .try_into()
                .map_err(|_| "WAV PCM24 sample is truncated".to_owned())?;
            let sign = if value[2] & 0x80 == 0 { 0 } else { 0xff };
            let signed = i32::from_le_bytes([value[0], value[1], value[2], sign]);
            Ok(i32_to_f32(signed) / 8_388_608.0)
        }
        (PCM_TAG, 32) => {
            let value: [u8; 4] = sample
                .try_into()
                .map_err(|_| "WAV PCM32 sample is truncated".to_owned())?;
            Ok(i32_to_f32(i32::from_le_bytes(value)) / 2_147_483_648.0)
        }
        (IEEE_FLOAT_TAG, 32) => {
            let value: [u8; 4] = sample
                .try_into()
                .map_err(|_| "WAV IEEE-float sample is truncated".to_owned())?;
            Ok(f32::from_le_bytes(value))
        }
        (ALAW_TAG, 8) => {
            let byte = *sample
                .first()
                .ok_or_else(|| "WAV A-law sample is truncated".to_owned())?;
            Ok(f32::from(g711::alaw_to_i16(byte)) / 32_768.0)
        }
        (ULAW_TAG, 8) => {
            let byte = *sample
                .first()
                .ok_or_else(|| "WAV mu-law sample is truncated".to_owned())?;
            Ok(f32::from(g711::ulaw_to_i16(byte)) / 32_768.0)
        }
        _ => Err("WAV format escaped its validated parser contract".into()),
    }
}

/// Parses exactly one unambiguous complete RIFF/WAV payload.
pub fn parse_wav(bytes: &[u8]) -> Result<DecodedAudio, String> {
    if !is_riff_wave(bytes) {
        return Err("input is not RIFF/WAVE".into());
    }
    let declared_size = u32_to_usize(u32_at(bytes, 4, "RIFF size")?)
        .map_err(|error| format!("RIFF size: {error}"))?;
    let actual_size = bytes
        .len()
        .checked_sub(8)
        .ok_or_else(|| "RIFF input is shorter than its header".to_owned())?;
    if declared_size != actual_size {
        return Err(format!(
            "RIFF size {declared_size} does not match payload size {actual_size}"
        ));
    }

    let (mut position, mut format, mut data) = (RIFF_HEADER_LENGTH, None, None);
    while position < bytes.len() {
        let header = range(bytes, position, CHUNK_HEADER_LENGTH, "RIFF chunk header")?;
        let identifier = header
            .get(0..4)
            .ok_or_else(|| "RIFF chunk identifier is truncated".to_owned())?;
        let body_size = u32_to_usize(u32::from_le_bytes(
            header
                .get(4..8)
                .ok_or_else(|| "RIFF chunk size is truncated".to_owned())?
                .try_into()
                .map_err(|_| "RIFF chunk size has invalid width".to_owned())?,
        ))
        .map_err(|error| format!("RIFF chunk size: {error}"))?;
        let body_start = position
            .checked_add(CHUNK_HEADER_LENGTH)
            .ok_or_else(|| "RIFF chunk body offset overflows usize".to_owned())?;
        let body = range(bytes, body_start, body_size, "RIFF chunk body")?;
        let padding = body_size % 2;
        let next_position = body_start
            .checked_add(body_size)
            .and_then(|value| value.checked_add(padding))
            .ok_or_else(|| "RIFF chunk end overflows usize".to_owned())?;
        if next_position > bytes.len() {
            return Err("RIFF chunk padding is truncated".into());
        }
        match identifier {
            b"fmt " => {
                if format.is_some() {
                    return Err("RIFF/WAV contains duplicate fmt chunks".into());
                }
                format = Some(parse_format(body)?);
            }
            b"data" => {
                if data.is_some() {
                    return Err("RIFF/WAV contains duplicate data chunks".into());
                }
                data = Some(body);
            }
            _ => {}
        }
        position = next_position;
    }
    let format = format.ok_or_else(|| "RIFF/WAV fmt chunk is missing".to_owned())?;
    let data = data.ok_or_else(|| "RIFF/WAV data chunk is missing".to_owned())?;
    if data.is_empty() {
        return Err("RIFF/WAV data chunk must not be empty".into());
    }
    if !data.len().is_multiple_of(format.block_align) {
        return Err("RIFF/WAV data ends with an incomplete terminal frame".into());
    }
    let frames = data.len() / format.block_align;
    if frames == 0 {
        return Err("RIFF/WAV data contains no complete frames".into());
    }
    let bytes_per_sample = format.block_align / format.channels.get();
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(format.channels.get())
        .map_err(|_| "WAV channels cannot reserve memory".to_owned())?;
    for _ in 0..format.channels.get() {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frames)
            .map_err(|_| "WAV channel samples cannot reserve memory".to_owned())?;
        samples.push(channel);
    }
    for frame in 0..frames {
        let frame_start = frame
            .checked_mul(format.block_align)
            .ok_or_else(|| "WAV frame offset overflows usize".to_owned())?;
        for (channel_index, channel) in samples.iter_mut().enumerate() {
            let sample_start = channel_index
                .checked_mul(bytes_per_sample)
                .and_then(|value| frame_start.checked_add(value))
                .ok_or_else(|| "WAV sample offset overflows usize".to_owned())?;
            let sample = range(data, sample_start, bytes_per_sample, "WAV sample")?;
            channel.push(decode_wav_sample(format, sample)?);
        }
    }
    let channels = samples
        .into_iter()
        .map(ChannelAudio::new)
        .collect::<Result<Vec<_>, _>>()?;
    DecodedAudio::new(format.sample_rate, channels)
}

#[cfg(test)]
mod tests {
    use super::parse_wav;

    fn pcm16_wav(samples: &[i16], sample_rate: u32) -> Result<Vec<u8>, String> {
        let data_length = samples
            .len()
            .checked_mul(2)
            .ok_or_else(|| "test WAV data length overflows".to_owned())?;
        let file_length = 44_usize
            .checked_add(data_length)
            .ok_or_else(|| "test WAV file length overflows".to_owned())?;
        let riff_size = u32::try_from(file_length - 8)
            .map_err(|_| "test RIFF size does not fit u32".to_owned())?;
        let data_size =
            u32::try_from(data_length).map_err(|_| "test data size does not fit u32".to_owned())?;
        let byte_rate = sample_rate
            .checked_mul(2)
            .ok_or_else(|| "test WAV byte rate overflows".to_owned())?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&riff_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        Ok(bytes)
    }

    #[test]
    fn generated_wav_matches_exact_reference_samples() -> Result<(), String> {
        let audio = parse_wav(&pcm16_wav(&[i16::MIN, 0, i16::MAX], 16_000)?)?;
        assert_eq!(audio.sample_rate().hertz(), 16_000);
        assert_eq!(audio.channels().len(), 1);
        assert_eq!(
            audio.channels()[0].samples(),
            &[-1.0_f32, 0.0_f32, f32::from(i16::MAX) / 32_768.0]
        );
        Ok(())
    }

    #[test]
    fn strict_wav_refuses_malformed_terminal_and_zero_rate_inputs() -> Result<(), String> {
        let valid = pcm16_wav(&[0, 1], 16_000)?;
        assert!(parse_wav(&valid[..valid.len() - 1]).is_err());
        let mut partial_terminal_frame = valid.clone();
        partial_terminal_frame.push(0);
        let partial_riff_size = u32::try_from(partial_terminal_frame.len() - 8)
            .map_err(|_| "test partial RIFF size does not fit u32".to_owned())?;
        partial_terminal_frame[4..8].copy_from_slice(&partial_riff_size.to_le_bytes());
        partial_terminal_frame[40..44].copy_from_slice(&5_u32.to_le_bytes());
        assert!(parse_wav(&partial_terminal_frame).is_err());
        let mut zero_rate = valid;
        zero_rate[24..28].copy_from_slice(&0_u32.to_le_bytes());
        assert!(parse_wav(&zero_rate).is_err());
        let empty = pcm16_wav(&[], 16_000)?;
        assert!(parse_wav(&empty).is_err());
        Ok(())
    }

    #[test]
    fn strict_wav_preserves_finite_float_overshoot_and_refuses_nonfinite_samples() {
        let mut nan = Vec::new();
        nan.extend_from_slice(b"RIFF");
        nan.extend_from_slice(&40_u32.to_le_bytes());
        nan.extend_from_slice(b"WAVEfmt ");
        nan.extend_from_slice(&16_u32.to_le_bytes());
        nan.extend_from_slice(&3_u16.to_le_bytes());
        nan.extend_from_slice(&1_u16.to_le_bytes());
        nan.extend_from_slice(&16_000_u32.to_le_bytes());
        nan.extend_from_slice(&64_000_u32.to_le_bytes());
        nan.extend_from_slice(&4_u16.to_le_bytes());
        nan.extend_from_slice(&32_u16.to_le_bytes());
        nan.extend_from_slice(b"data");
        nan.extend_from_slice(&4_u32.to_le_bytes());
        nan.extend_from_slice(&f32::NAN.to_le_bytes());
        assert!(parse_wav(&nan).is_err());
        let mut endpoint = nan.clone();
        endpoint[44..48].copy_from_slice(&1.0_f32.to_le_bytes());
        let accepted = parse_wav(&endpoint).expect("a finite IEEE-float endpoint is valid");
        assert_eq!(accepted.channels()[0].samples(), &[1.0_f32]);

        let above_one = f32::from_bits(
            1.0_f32
                .to_bits()
                .checked_add(1)
                .expect("positive unit IEEE-754 bits can advance once"),
        );
        let mut overshoot = endpoint;
        overshoot[44..48].copy_from_slice(&above_one.to_le_bytes());
        let accepted = parse_wav(&overshoot).expect("finite IEEE-float overshoot is valid");
        assert_eq!(accepted.channels()[0].samples(), &[above_one]);
    }
}
