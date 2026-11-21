// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Bounded observations and recognition for one replayable physical channel.

use crate::batch::{BatchTranscriber, SpanOrigin, WindowObservations};
use crate::channel_selection::OriginalChannel;
use crate::contracts::{TranscriptWord, checkpoint};
use crate::vad_policy::{SpeechSegment, VadPolicyConfig, VadSegmenter};
use gigaam_audio::ReplayAudio;
use gigaam_recognition::{ExecutionControl, SpeechProbabilityDetector, WindowDecoder};
use std::borrow::Cow;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::time::Instant;

/// Acoustic observations required by the selected batch operation.
pub struct ReplayObservations<'a> {
    pub detector: &'a mut dyn SpeechProbabilityDetector,
    pub policy: Option<VadPolicyConfig>,
    pub scored: bool,
}

pub(crate) fn transcribe_channel<D: WindowDecoder, F: Read + Write + Seek>(
    batch: &mut BatchTranscriber<D>,
    audio: &mut ReplayAudio<F>,
    channel: OriginalChannel,
    control: &ExecutionControl,
    observations: Option<&mut ReplayObservations<'_>>,
    create_file: &mut impl FnMut() -> io::Result<F>,
    emit: &mut impl FnMut(TranscriptWord) -> Result<(), String>,
) -> Result<(), String> {
    let length = audio.frames();
    let Some(observations) = observations else {
        batch.transcribe_span_to(
            length,
            SpanOrigin {
                channel: Some(channel),
                ..SpanOrigin::default()
            },
            None,
            |range| read_channel(audio, channel, range),
            emit,
        )?;
        return Ok(());
    };
    batch.validate_vad_sample_rate()?;
    let started = Instant::now();
    let mut probabilities = observations
        .scored
        .then(&mut *create_file)
        .transpose()
        .map_err(io_message)?;
    let mut groups = observations
        .policy
        .map(|policy| Ok::<_, io::Error>((VadSegmenter::new(length, policy), create_file()?)))
        .transpose()
        .map_err(io_message)?;
    let mut pending = None;
    let mut group_count = 0_usize;
    let mut next_sample = 0_usize;
    let mut probability_count = 0_usize;
    let expected_probabilities = length.div_ceil(512);
    let window_samples = batch.full_window_samples();
    observations.detector.probabilities_bounded(
        length,
        &mut |output| {
            checkpoint(control)?;
            let end = next_sample
                .checked_add(output.len())
                .ok_or_else(|| "VAD read position overflows".to_owned())?;
            let samples = read_channel(audio, channel, next_sample..end)?;
            output.copy_from_slice(&samples);
            next_sample = end;
            Ok(())
        },
        &mut |block| {
            checkpoint(control)?;
            accept_probabilities(&mut probability_count, expected_probabilities, block)?;
            if let Some(file) = &mut probabilities {
                let bytes = block
                    .iter()
                    .flat_map(|probability| probability.to_le_bytes())
                    .collect::<Vec<_>>();
                file.write_all(&bytes).map_err(io_message)?;
            }
            if let Some((segmenter, file)) = &mut groups {
                for &probability in block {
                    if let Some(segment) = segmenter.push(probability)? {
                        accept_segment(
                            file,
                            &mut pending,
                            &mut group_count,
                            segment,
                            window_samples,
                        )?;
                    }
                }
            }
            Ok(())
        },
    )?;
    if next_sample != length {
        return Err("VAD did not consume the declared waveform".into());
    }
    if probability_count != expected_probabilities {
        return Err(format!(
            "VAD emitted {probability_count} probabilities, expected {expected_probabilities}"
        ));
    }
    let groups = if let Some((segmenter, mut file)) = groups {
        for segment in segmenter.finish()?.into_iter().flatten() {
            accept_segment(
                &mut file,
                &mut pending,
                &mut group_count,
                segment,
                window_samples,
            )?;
        }
        if let Some(group) = pending {
            write_group(&mut file, group, &mut group_count)?;
        }
        file.rewind().map_err(io_message)?;
        Some(file)
    } else {
        None
    };
    batch.record_vad_time(started.elapsed().as_secs_f64());

    let mut read_probabilities = probabilities
        .as_mut()
        .map(|file| move |range| maximum_probability(file, range));
    let mut decode = |range: Range<usize>, window_index| {
        let sample = range.start;
        batch.transcribe_span_to(
            range.len(),
            SpanOrigin {
                channel: Some(channel),
                sample,
                window_index,
            },
            read_probabilities
                .as_mut()
                .map(|read| read as &mut WindowObservations<'_>),
            |local| read_channel(audio, channel, sample + local.start..sample + local.end),
            &mut *emit,
        )
    };
    if let Some(mut file) = groups {
        let mut window_index = 0_usize;
        for _ in 0..group_count {
            checkpoint(control)?;
            let group = read_group(&mut file)?;
            if group.start >= group.end || group.end > length {
                return Err("stored VAD group is outside the source waveform".into());
            }
            window_index = window_index
                .checked_add(decode(group, window_index)?)
                .ok_or_else(|| "VAD batch planned window count overflows".to_owned())?;
        }
    } else {
        decode(0..length, 0)?;
    }
    Ok(())
}

fn read_channel<'a, F: Read + Write + Seek>(
    audio: &mut ReplayAudio<F>,
    channel: OriginalChannel,
    range: Range<usize>,
) -> Result<Cow<'a, [f32]>, String> {
    audio
        .read_channel(channel.index(), range)
        .map(Cow::Owned)
        .map_err(|error| error.to_string())
}

fn accept_segment(
    file: &mut impl Write,
    pending: &mut Option<Range<usize>>,
    count: &mut usize,
    segment: SpeechSegment,
    window_samples: usize,
) -> Result<(), String> {
    let end = segment.end_sample();
    if let Some(group) = pending.as_mut()
        && end
            .checked_sub(group.start)
            .is_some_and(|length| length <= window_samples)
    {
        group.end = end;
    } else if let Some(closed) = pending.replace(segment.start_sample()..end) {
        write_group(file, closed, count)?;
    }
    Ok(())
}

fn write_group(
    file: &mut impl Write,
    group: Range<usize>,
    count: &mut usize,
) -> Result<(), String> {
    let mut bytes = [0; 16];
    let parts = [group.start, group.end];
    for (part, buffer) in parts.into_iter().zip(bytes.as_chunks_mut::<8>().0) {
        let part = u64::try_from(part).map_err(|_| "VAD group position exceeds u64".to_owned())?;
        buffer.copy_from_slice(&part.to_le_bytes());
    }
    file.write_all(&bytes).map_err(io_message)?;
    *count = count
        .checked_add(1)
        .ok_or_else(|| "VAD group count overflows".to_owned())?;
    Ok(())
}

fn read_group(file: &mut impl Read) -> Result<Range<usize>, String> {
    fn position(file: &mut impl Read) -> Result<usize, String> {
        let mut bytes = [0; 8];
        file.read_exact(&mut bytes).map_err(io_message)?;
        usize::try_from(u64::from_le_bytes(bytes))
            .map_err(|_| "VAD group position exceeds usize".into())
    }
    Ok(position(file)?..position(file)?)
}

fn maximum_probability(file: &mut (impl Read + Seek), frames: Range<usize>) -> Result<f32, String> {
    let offset = u64::try_from(frames.start)
        .ok()
        .and_then(|start| start.checked_mul(4))
        .ok_or_else(|| "VAD probability offset overflows".to_owned())?;
    file.seek(SeekFrom::Start(offset)).map_err(io_message)?;
    let mut remaining = frames.len();
    let mut maximum = 0.0_f32;
    let mut block = [0_u8; 512];
    while remaining > 0 {
        let count = remaining.min(block.len() / 4);
        let bytes = &mut block[..count * 4];
        file.read_exact(bytes).map_err(io_message)?;
        let (values, _) = bytes.as_chunks::<4>();
        for &value in values {
            maximum = maximum.max(f32::from_le_bytes(value));
        }
        remaining -= count;
    }
    Ok(maximum)
}

fn io_message(error: io::Error) -> String {
    format!("batch VAD working storage: {error}")
}

fn accept_probabilities(count: &mut usize, expected: usize, block: &[f32]) -> Result<(), String> {
    if block.iter().any(|value| !(0.0..=1.0).contains(value)) {
        return Err("speech detector returned a probability outside [0, 1]".into());
    }
    *count = count
        .checked_add(block.len())
        .filter(|&total| total <= expected)
        .ok_or_else(|| "VAD emitted more observations than the waveform accommodates".to_owned())?;
    Ok(())
}
