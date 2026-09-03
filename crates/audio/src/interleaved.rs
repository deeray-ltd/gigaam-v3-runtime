// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Stateful raw interleaved sample decoding across arbitrary byte partitions.

use crate::contracts::{ChannelAudio, ChannelCount, SampleFormat};
use crate::g711;

/// Stateful raw interleaved decoder. It owns byte-partition handling; contracts own the sample
/// value types and format enum. External IEEE-754 float samples must be finite and in the closed
/// normalized interval `[-1, 1]`.
#[derive(Debug)]
pub struct InterleavedFrameDecoder {
    format: SampleFormat,
    channels: ChannelCount,
    frame_bytes: usize,
    remainder: Vec<u8>,
}

fn decode_sample(format: SampleFormat, bytes: &[u8]) -> Result<f32, String> {
    match format {
        SampleFormat::Pcm16 => {
            let sample: [u8; 2] = bytes
                .try_into()
                .map_err(|_| "interleaved PCM16 sample width is invalid".to_owned())?;
            Ok(f32::from(i16::from_le_bytes(sample)) / 32_768.0)
        }
        SampleFormat::F32 => {
            let sample: [u8; 4] = bytes
                .try_into()
                .map_err(|_| "interleaved f32 sample width is invalid".to_owned())?;
            let value = f32::from_le_bytes(sample);
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(
                    "raw interleaved IEEE-float samples must be finite and in normalized [-1, 1]"
                        .into(),
                );
            }
            Ok(value)
        }
        SampleFormat::Alaw => {
            let byte = *bytes
                .first()
                .ok_or_else(|| "interleaved A-law sample width is invalid".to_owned())?;
            Ok(f32::from(g711::alaw_to_i16(byte)) / 32_768.0)
        }
        SampleFormat::Ulaw => {
            let byte = *bytes
                .first()
                .ok_or_else(|| "interleaved mu-law sample width is invalid".to_owned())?;
            Ok(f32::from(g711::ulaw_to_i16(byte)) / 32_768.0)
        }
    }
}

impl InterleavedFrameDecoder {
    pub fn new(format: SampleFormat, channels: ChannelCount) -> Result<Self, String> {
        let frame_bytes = format
            .bytes_per_sample()
            .checked_mul(channels.get())
            .ok_or_else(|| "interleaved frame width overflows usize".to_owned())?;
        Ok(Self {
            format,
            channels,
            frame_bytes,
            remainder: Vec::new(),
        })
    }

    /// Returns complete nonempty channel chunks and retains a terminal partial interleaved frame.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Option<Vec<ChannelAudio>>, String> {
        let capacity = self
            .remainder
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| "interleaved input length overflows usize".to_owned())?;
        let mut combined = Vec::new();
        combined
            .try_reserve_exact(capacity)
            .map_err(|_| "interleaved input cannot reserve memory".to_owned())?;
        combined.extend_from_slice(&self.remainder);
        combined.extend_from_slice(bytes);
        let frames = combined.len() / self.frame_bytes;
        let complete_bytes = frames
            .checked_mul(self.frame_bytes)
            .ok_or_else(|| "interleaved complete frame length overflows usize".to_owned())?;
        self.remainder = combined[complete_bytes..].to_vec();
        if frames == 0 {
            return Ok(None);
        }
        let mut channels = Vec::new();
        channels
            .try_reserve_exact(self.channels.get())
            .map_err(|_| "interleaved output channels cannot reserve memory".to_owned())?;
        for _ in 0..self.channels.get() {
            let mut channel = Vec::new();
            channel
                .try_reserve_exact(frames)
                .map_err(|_| "interleaved output samples cannot reserve memory".to_owned())?;
            channels.push(channel);
        }
        let sample_width = self.format.bytes_per_sample();
        for frame in 0..frames {
            let frame_start = frame
                .checked_mul(self.frame_bytes)
                .ok_or_else(|| "interleaved frame offset overflows usize".to_owned())?;
            for (channel_index, channel) in channels.iter_mut().enumerate() {
                let sample_start = channel_index
                    .checked_mul(sample_width)
                    .and_then(|value| frame_start.checked_add(value))
                    .ok_or_else(|| "interleaved sample offset overflows usize".to_owned())?;
                let sample_end = sample_start
                    .checked_add(sample_width)
                    .ok_or_else(|| "interleaved sample end overflows usize".to_owned())?;
                let sample = combined
                    .get(sample_start..sample_end)
                    .ok_or_else(|| "interleaved sample is truncated".to_owned())?;
                channel.push(decode_sample(self.format, sample)?);
            }
        }
        channels
            .into_iter()
            .map(ChannelAudio::new)
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    /// A consuming terminal check: successful completion makes post-finish input unrepresentable.
    pub fn finish(self) -> Result<(), String> {
        if self.remainder.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "interleaved input ends with {} incomplete frame bytes",
                self.remainder.len()
            ))
        }
    }
}

/// Decodes one complete noninterleaved channel. Transport orchestration may retain its own
/// multichannel byte remainder, but sample-format conversion remains an Audio responsibility.
pub fn decode_samples(format: SampleFormat, bytes: &[u8]) -> Result<ChannelAudio, String> {
    if !bytes.len().is_multiple_of(format.bytes_per_sample()) {
        return Err(format!(
            "{format:?} payload length {} is not a whole number of {}-byte samples",
            bytes.len(),
            format.bytes_per_sample()
        ));
    }
    if bytes.is_empty() {
        return ChannelAudio::new(Vec::new());
    }
    let mut decoder = InterleavedFrameDecoder::new(format, ChannelCount::new(1)?)?;
    let mut decoded = decoder
        .push(bytes)?
        .ok_or_else(|| "complete nonempty sample input produced no channel output".to_owned())?;
    decoder.finish()?;
    decoded
        .pop()
        .ok_or_else(|| "one-channel decoder produced no channel output".to_owned())
}

#[cfg(test)]
mod tests {
    use super::{InterleavedFrameDecoder, decode_samples};
    use crate::{ChannelAudio, ChannelCount, SampleFormat};

    fn payload(format: SampleFormat) -> Vec<u8> {
        match format {
            SampleFormat::Pcm16 => [0_i16, i16::MAX, i16::MIN, 1]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect(),
            SampleFormat::F32 => [-1.0_f32, 1.0_f32]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            SampleFormat::Alaw => vec![0x00, 0xff, 0x80, 0x7f],
            SampleFormat::Ulaw => vec![0x00, 0xff, 0x80, 0x7f],
        }
    }

    fn one_shot(
        format: SampleFormat,
        channels: ChannelCount,
        bytes: &[u8],
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut decoder = InterleavedFrameDecoder::new(format, channels)?;
        let decoded = decoder
            .push(bytes)?
            .ok_or_else(|| "representative payload must contain a complete frame".to_owned())?;
        decoder.finish()?;
        Ok(decoded
            .into_iter()
            .map(ChannelAudio::into_samples)
            .collect())
    }

    fn append_chunk(target: &mut [Vec<f32>], chunk: Vec<ChannelAudio>) -> Result<(), String> {
        if target.len() != chunk.len() {
            return Err("interleaved decoder changed its validated channel count".into());
        }
        for (target_channel, decoded_channel) in target.iter_mut().zip(chunk) {
            target_channel.extend(decoded_channel.into_samples());
        }
        Ok(())
    }

    fn partitioned(
        format: SampleFormat,
        channels: ChannelCount,
        bytes: &[u8],
        cuts: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut decoder = InterleavedFrameDecoder::new(format, channels)?;
        let mut output: Vec<Vec<f32>> = (0..channels.get()).map(|_| Vec::new()).collect();
        let mut start = 0_usize;
        for boundary in 1..bytes.len() {
            let bit = u32::try_from(boundary - 1)
                .map_err(|_| "test partition bit does not fit u32".to_owned())?;
            let mask = 1_usize
                .checked_shl(bit)
                .ok_or_else(|| "test partition mask overflows usize".to_owned())?;
            if cuts & mask != 0 {
                if let Some(chunk) = decoder.push(&bytes[start..boundary])? {
                    append_chunk(&mut output, chunk)?;
                }
                start = boundary;
            }
        }
        if let Some(chunk) = decoder.push(&bytes[start..])? {
            append_chunk(&mut output, chunk)?;
        }
        decoder.finish()?;
        Ok(output)
    }

    #[test]
    fn every_nonempty_byte_partition_matches_one_shot_for_each_raw_format() -> Result<(), String> {
        let channels = ChannelCount::new(2)?;
        for format in [
            SampleFormat::Pcm16,
            SampleFormat::F32,
            SampleFormat::Alaw,
            SampleFormat::Ulaw,
        ] {
            let bytes = payload(format);
            let expected = one_shot(format, channels, &bytes)?;
            let cut_positions = bytes
                .len()
                .checked_sub(1)
                .ok_or_else(|| "representative payload must be nonempty".to_owned())?;
            let cut_count = u32::try_from(cut_positions)
                .map_err(|_| "test partition count does not fit u32".to_owned())?;
            let partitions = 1_usize
                .checked_shl(cut_count)
                .ok_or_else(|| "test partition count overflows usize".to_owned())?;
            for cuts in 0..partitions {
                assert_eq!(partitioned(format, channels, &bytes, cuts)?, expected);
            }

            let mut incomplete = InterleavedFrameDecoder::new(format, channels)?;
            incomplete.push(&bytes[..bytes.len() - 1])?;
            assert!(incomplete.finish().is_err(), "{format:?}");
        }
        Ok(())
    }

    #[test]
    fn f32_decoder_accepts_closed_endpoints_and_refuses_outside_or_nonfinite_values() {
        let endpoints = [-1.0_f32, 1.0_f32];
        let bytes = endpoints
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut decoder = InterleavedFrameDecoder::new(
            SampleFormat::F32,
            ChannelCount::new(1).expect("one channel is valid"),
        )
        .expect("one-channel f32 decoder has a valid frame width");
        let mut decoded = Vec::new();
        for byte in bytes.chunks(1) {
            if let Some(channels) = decoder
                .push(byte)
                .expect("closed-domain f32 input is valid")
            {
                assert_eq!(
                    channels.len(),
                    1,
                    "one input channel must produce one output channel"
                );
                let channel = channels
                    .into_iter()
                    .next()
                    .expect("one-channel decoder produces one channel");
                decoded.extend(channel.into_samples());
            }
        }
        decoder
            .finish()
            .expect("all endpoint bytes form complete f32 samples");
        assert_eq!(decoded, endpoints);

        let below_negative_one = f32::from_bits(
            (-1.0_f32)
                .to_bits()
                .checked_add(1)
                .expect("negative unit IEEE-754 bits can advance once"),
        );
        let above_one = f32::from_bits(
            1.0_f32
                .to_bits()
                .checked_add(1)
                .expect("positive unit IEEE-754 bits can advance once"),
        );
        for invalid in [
            below_negative_one,
            above_one,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let mut invalid_decoder = InterleavedFrameDecoder::new(
                SampleFormat::F32,
                ChannelCount::new(1).expect("one channel is valid"),
            )
            .expect("one-channel f32 decoder has a valid frame width");
            assert!(
                invalid_decoder.push(&invalid.to_le_bytes()).is_err(),
                "external f32 value {invalid:?} must be refused"
            );
        }
    }

    #[test]
    fn standalone_pcm16_samples_refuse_partial_input_and_preserve_complete_input() {
        assert!(decode_samples(SampleFormat::Pcm16, &[0]).is_err());
        assert_eq!(
            decode_samples(SampleFormat::Pcm16, &[0, 0]).map(|channel| channel.into_samples()),
            Ok(vec![0.0])
        );
    }
}
