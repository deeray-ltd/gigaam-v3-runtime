// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Seekable model-rate samples prepared from bounded decoder and resampler output.

use crate::channel_analysis::finish_correlation;
use crate::{FileDecoder, RatePair, Resampler, ResamplerConfig, SampleRate, StreamResampler};
use gigaam_primitives::usize_to_f64;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::num::NonZeroUsize;
use std::ops::Range;

const CHUNK_BYTES: usize = 64 * 1024;
const SAMPLE_BYTES: usize = size_of::<f32>();

/// Source and storage errors are distinguished so callers can preserve their input-error contract
/// while reporting exhausted working storage as an execution failure.
#[derive(Debug)]
pub enum ReplayError {
    Audio(String),
    Storage(io::Error),
    Cancelled(String),
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Audio(message) | Self::Cancelled(message) => output.write_str(message),
            Self::Storage(error) => error.fmt(output),
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<io::Error> for ReplayError {
    fn from(error: io::Error) -> Self {
        Self::Storage(error)
    }
}

/// Model-rate samples backed by caller-owned storage. The channel reader allocates only the
/// requested channel range and reuses a block buffer for the interleaved backing representation.
pub struct ReplayAudio<F> {
    storage: F,
    sample_rate: SampleRate,
    channels: usize,
    frames: usize,
    block: Vec<u8>,
}

impl<F: Read + Write + Seek> ReplayAudio<F> {
    /// Decodes one source without accumulating a full PCM buffer. Storage must be empty and
    /// positioned at its start. The checkpoint is invoked between successive decoder chunks.
    pub fn prepare(
        mut decoder: FileDecoder,
        storage: F,
        rate: SampleRate,
        downmix: bool,
        mut checkpoint: impl FnMut() -> Result<(), String>,
    ) -> Result<Self, ReplayError> {
        let metadata = decoder.metadata();
        let source_channels = metadata.channel_count().get();
        let chunk_frames = NonZeroUsize::new((CHUNK_BYTES / SAMPLE_BYTES / source_channels).max(1))
            .ok_or_else(|| ReplayError::Audio("audio chunk size must be positive".into()))?;
        let pair = RatePair::new(metadata.sample_rate(), rate).map_err(ReplayError::Audio)?;
        let mut resamplers = Vec::new();
        if !pair.is_identity() {
            let filter = Resampler::new(ResamplerConfig::new(pair)).map_err(ReplayError::Audio)?;
            let filter = std::sync::Arc::new(filter);
            resamplers.extend(
                (0..source_channels).map(|_| StreamResampler::with_shared_filter(filter.clone())),
            );
        }
        let mut output = Self {
            storage,
            sample_rate: rate,
            channels: if downmix { 1 } else { source_channels },
            frames: 0,
            block: Vec::new(),
        };
        loop {
            checkpoint().map_err(ReplayError::Cancelled)?;
            let Some(chunk) = decoder
                .next_chunk(chunk_frames)
                .map_err(ReplayError::Audio)?
            else {
                break;
            };
            let (_, channels) = chunk.into_audio().into_parts();
            let selected = if resamplers.is_empty() {
                channels
                    .into_iter()
                    .map(crate::ChannelAudio::into_samples)
                    .collect::<Vec<_>>()
            } else {
                resamplers
                    .iter_mut()
                    .zip(&channels)
                    .map(|(resampler, channel)| resampler.push(channel.samples()))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(ReplayError::Audio)?
            };
            output.append(&selected)?;
        }
        if !resamplers.is_empty() {
            checkpoint().map_err(ReplayError::Cancelled)?;
            let tail = resamplers
                .into_iter()
                .map(StreamResampler::finish)
                .collect::<Result<Vec<_>, _>>()
                .map_err(ReplayError::Audio)?;
            output.append(&tail)?;
        }
        if output.frames == 0 {
            return Err(ReplayError::Audio(
                "decoded audio contains no model-rate samples".into(),
            ));
        }
        output.storage.flush()?;
        Ok(output)
    }

    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    pub const fn frames(&self) -> usize {
        self.frames
    }

    fn append(&mut self, channels: &[Vec<f32>]) -> Result<(), ReplayError> {
        let frames = channels.first().map_or(0, Vec::len);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err(ReplayError::Audio(
                "resampled channel lengths diverged".into(),
            ));
        }
        if frames == 0 {
            return Ok(());
        }
        let total = self.frames.checked_add(frames).ok_or_else(|| {
            ReplayError::Audio("model-rate frame position overflows usize".into())
        })?;
        let frame_bytes = self.frame_bytes()?;
        let block_frames = (CHUNK_BYTES / frame_bytes).max(1);
        self.block.clear();
        let divisor = if self.channels == 1 && channels.len() > 1 {
            Some(f32::from(u16::try_from(channels.len()).map_err(|_| {
                ReplayError::Audio("too many source channels".into())
            })?))
        } else {
            None
        };
        for start in (0..frames).step_by(block_frames) {
            let end = start.saturating_add(block_frames).min(frames);
            for index in start..end {
                if let Some(divisor) = divisor {
                    let value = channels
                        .iter()
                        .fold(0.0, |mixed, channel| mixed + channel[index] / divisor);
                    if !value.is_finite() {
                        return Err(ReplayError::Audio("downmixed sample is not finite".into()));
                    }
                    self.block.extend_from_slice(&value.to_le_bytes());
                } else {
                    for channel in channels {
                        self.block.extend_from_slice(&channel[index].to_le_bytes());
                    }
                }
            }
            self.storage.write_all(&self.block)?;
            self.block.clear();
        }
        self.frames = total;
        Ok(())
    }

    fn frame_bytes(&self) -> Result<usize, ReplayError> {
        self.channels
            .checked_mul(SAMPLE_BYTES)
            .ok_or_else(|| ReplayError::Audio("model-rate frame width overflows usize".into()))
    }

    pub fn read_channel(
        &mut self,
        channel: usize,
        frames: Range<usize>,
    ) -> Result<Vec<f32>, ReplayError> {
        if channel >= self.channels {
            return Err(ReplayError::Audio(
                "requested channel is outside the source".into(),
            ));
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(frames.len())
            .map_err(|_| ReplayError::Audio("model-rate samples cannot reserve memory".into()))?;
        let offset = channel * SAMPLE_BYTES;
        self.read_blocks(frames, |block, frame_bytes| {
            for frame in block.chunks_exact(frame_bytes) {
                output.push(sample_at(frame, offset));
            }
            Ok(())
        })?;
        Ok(output)
    }

    /// Computes the same two-pass Pearson calculation as a complete stereo waveform.
    pub fn stereo_correlation(
        &mut self,
        mut checkpoint: impl FnMut() -> Result<(), String>,
    ) -> Result<f64, ReplayError> {
        if self.channels != 2 {
            return Err(ReplayError::Audio(
                "stereo correlation requires two channels".into(),
            ));
        }
        let (mut left_sum, mut right_sum) = (0.0, 0.0);
        self.read_blocks(0..self.frames, |block, frame_bytes| {
            checkpoint().map_err(ReplayError::Cancelled)?;
            for frame in block.chunks_exact(frame_bytes) {
                left_sum += f64::from(sample_at(frame, 0));
                right_sum += f64::from(sample_at(frame, SAMPLE_BYTES));
            }
            Ok(())
        })?;
        let frames = usize_to_f64(self.frames);
        let (left_mean, right_mean) = (left_sum / frames, right_sum / frames);
        let (mut numerator, mut left_energy, mut right_energy) = (0.0, 0.0, 0.0);
        self.read_blocks(0..self.frames, |block, frame_bytes| {
            checkpoint().map_err(ReplayError::Cancelled)?;
            for frame in block.chunks_exact(frame_bytes) {
                let left_delta = f64::from(sample_at(frame, 0)) - left_mean;
                let right_delta = f64::from(sample_at(frame, SAMPLE_BYTES)) - right_mean;
                numerator += left_delta * right_delta;
                left_energy += left_delta * left_delta;
                right_energy += right_delta * right_delta;
            }
            Ok(())
        })?;
        finish_correlation(numerator, left_energy, right_energy).map_err(ReplayError::Audio)
    }

    fn read_blocks(
        &mut self,
        frames: Range<usize>,
        mut consume: impl FnMut(&[u8], usize) -> Result<(), ReplayError>,
    ) -> Result<(), ReplayError> {
        if frames.start > frames.end || frames.end > self.frames {
            return Err(ReplayError::Audio(
                "requested frames are outside the source".into(),
            ));
        }
        let frame_bytes = self.frame_bytes()?;
        let start = frames
            .start
            .checked_mul(frame_bytes)
            .and_then(|offset| u64::try_from(offset).ok())
            .ok_or_else(|| ReplayError::Audio("model-rate frame offset overflows".into()))?;
        self.storage.seek(SeekFrom::Start(start))?;
        let mut remaining = frames.len();
        let block_frames = (CHUNK_BYTES / frame_bytes).max(1);
        while remaining > 0 {
            let count = remaining.min(block_frames);
            self.block.resize(count * frame_bytes, 0);
            self.storage.read_exact(&mut self.block)?;
            consume(&self.block, frame_bytes)?;
            remaining -= count;
        }
        Ok(())
    }
}

fn sample_at(frame: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        frame[offset],
        frame[offset + 1],
        frame[offset + 2],
        frame[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioMetadata, ChannelAudio, ChannelCount, DecodedAudio, FileChunkReader};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    struct Reader {
        dropped: Arc<AtomicBool>,
        remaining: usize,
        metadata: AudioMetadata,
    }
    impl Drop for Reader {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }
    impl FileChunkReader for Reader {
        fn metadata(&self) -> AudioMetadata {
            self.metadata
        }
        fn next_audio(&mut self, _: NonZeroUsize) -> Result<Option<DecodedAudio>, String> {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;
            Ok(Some(DecodedAudio::new(
                self.metadata.sample_rate(),
                vec![ChannelAudio::new(vec![0.25])?],
            )?))
        }
    }
    struct FailedWriter(std::io::Cursor<Vec<u8>>);
    impl Read for FailedWriter {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            self.0.read(bytes)
        }
    }
    impl Seek for FailedWriter {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.0.seek(pos)
        }
    }
    impl Write for FailedWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("injected write failure"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    fn reader(dropped: Arc<AtomicBool>) -> Result<FileDecoder, String> {
        Ok(FileDecoder::from_reader(Box::new(Reader {
            dropped,
            remaining: 2,
            metadata: AudioMetadata::new(SampleRate::new(16000)?, ChannelCount::new(1)?),
        })))
    }
    #[test]
    fn cancellation_and_storage_failure_release_decoder() -> Result<(), String> {
        let rate = SampleRate::new(16000)?;
        let dropped = Arc::new(AtomicBool::new(false));
        let mut calls = 0;
        let result = ReplayAudio::prepare(
            reader(Arc::clone(&dropped))?,
            std::io::Cursor::new(Vec::new()),
            rate,
            false,
            || {
                calls += 1;
                if calls == 2 {
                    Err("cancelled".into())
                } else {
                    Ok(())
                }
            },
        );
        assert!(
            matches!(result, Err(ReplayError::Cancelled(ref message)) if message == "cancelled")
        );
        assert!(dropped.load(Ordering::SeqCst));
        let dropped = Arc::new(AtomicBool::new(false));
        let result = ReplayAudio::prepare(
            reader(Arc::clone(&dropped))?,
            FailedWriter(std::io::Cursor::new(Vec::new())),
            rate,
            false,
            || Ok(()),
        );
        assert!(
            matches!(result, Err(ReplayError::Storage(ref error)) if error.to_string() == "injected write failure")
        );
        assert!(dropped.load(Ordering::SeqCst));
        Ok(())
    }
}
