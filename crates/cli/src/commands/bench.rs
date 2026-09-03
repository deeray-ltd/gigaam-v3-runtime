// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Direct encoder-window benchmark command adapter.

use crate::composition;
use crate::configuration;
use crate::grammar::BenchInvocation;
use crate::numeric::BenchmarkSamplePlan;
use crate::projection::{self, BenchmarkSummary};
use gigaam_audio::{ChannelAudio, DecodedAudio, SampleRate, read_wav};
use gigaam_recognition::{Encoder, RequiredEncoderRoles};
use std::path::Path;
use std::time::Instant;

/// Benchmarks one warmed direct CTC encoder shape and an optional alternate shape.
pub(crate) fn run(invocation: BenchInvocation) -> Result<(), String> {
    let runtime = configuration::runtime(&invocation.runtime, RequiredEncoderRoles::ctc())
        .map_err(|error| format!("runtime configuration: {error}"))?;
    let frontend_mode =
        configuration::frontend_mode().map_err(|error| format!("frontend: {error}"))?;
    let package = composition::open_package(&invocation.model)?;
    let sample_rate = composition::package_sample_rate(&package)?;
    let sample_plan = BenchmarkSamplePlan::new(
        sample_rate,
        invocation.window_seconds,
        invocation.alternate_window_seconds,
    )?;
    let loaded = read_wav(&invocation.audio)
        .map_err(|error| format!("bench audio {}: {error}", invocation.audio.display()))?;
    let buffers = prepare_buffers(loaded, sample_rate, sample_plan, &invocation.audio)?;
    let frontend = composition::frontend_for_package(&package, frontend_mode)
        .map_err(|error| format!("frontend: {error}"))?;
    composition::initialize_runtime(&runtime)?;
    let mut encoder = composition::ctc_encoder(&package, &runtime, invocation.precision)
        .map_err(|error| format!("encoder: {error}"))?;
    projection::emit_unverified_cuda_assignment(
        runtime.plan(),
        gigaam_recognition::EncoderRole::Ctc,
        encoder.assignment_evidence(),
    );

    for _ in 0..3 {
        let mel = frontend
            .log_mel(&buffers.primary)
            .map_err(|error| format!("frontend warmup: {error}"))?;
        encoder
            .forward(mel.view())
            .map_err(|error| format!("encoder warmup: {error}"))?;
        if let Some(alternate) = &buffers.alternate {
            let mel = frontend
                .log_mel(alternate)
                .map_err(|error| format!("frontend alternate warmup: {error}"))?;
            encoder
                .forward(mel.view())
                .map_err(|error| format!("encoder alternate warmup: {error}"))?;
        }
    }

    let mut frontend_milliseconds = Vec::with_capacity(invocation.iterations);
    let mut encoder_milliseconds = Vec::with_capacity(invocation.iterations);
    for _ in 0..invocation.iterations {
        if invocation.gap_milliseconds > 0 {
            std::thread::sleep(std::time::Duration::from_millis(
                invocation.gap_milliseconds,
            ));
        }
        if let Some(alternate) = &buffers.alternate {
            let mel = frontend
                .log_mel(alternate)
                .map_err(|error| format!("frontend alternate run: {error}"))?;
            encoder
                .forward(mel.view())
                .map_err(|error| format!("encoder alternate run: {error}"))?;
        }
        let frontend_started = Instant::now();
        let mel = frontend
            .log_mel(&buffers.primary)
            .map_err(|error| format!("frontend run: {error}"))?;
        let encoder_started = Instant::now();
        encoder
            .forward(mel.view())
            .map_err(|error| format!("encoder run: {error}"))?;
        let finished = Instant::now();
        frontend_milliseconds.push((encoder_started - frontend_started).as_secs_f64() * 1_000.0);
        encoder_milliseconds.push((finished - encoder_started).as_secs_f64() * 1_000.0);
    }
    frontend_milliseconds.sort_by(|left, right| left.total_cmp(right));
    encoder_milliseconds.sort_by(|left, right| left.total_cmp(right));
    let median_index = invocation.iterations / 2;
    let frontend_median = measurement(&frontend_milliseconds, median_index, "frontend median")?;
    let encoder_median = measurement(&encoder_milliseconds, median_index, "encoder median")?;
    let encoder_minimum = measurement(&encoder_milliseconds, 0, "encoder minimum")?;
    let encoder_maximum = measurement(
        &encoder_milliseconds,
        invocation
            .iterations
            .checked_sub(1)
            .ok_or_else(|| "benchmark iterations must be positive".to_owned())?,
        "encoder maximum",
    )?;
    projection::benchmark(BenchmarkSummary {
        window_seconds: invocation.window_seconds,
        alternate_window_seconds: invocation.alternate_window_seconds,
        precision: invocation.precision,
        device: runtime.plan().device(),
        gap_milliseconds: invocation.gap_milliseconds,
        frontend_median_milliseconds: frontend_median,
        encoder_median_milliseconds: encoder_median,
        encoder_minimum_milliseconds: encoder_minimum,
        encoder_maximum_milliseconds: encoder_maximum,
        iterations: invocation.iterations,
    });
    Ok(())
}

/// Fully prepared, model-rate buffers used by one benchmark's warmup and measurement loops.
struct BenchmarkBuffers {
    primary: Vec<f32>,
    alternate: Option<Vec<f32>>,
}

/// Resamples validated WAV input before deriving the benchmark's cyclic model-rate buffers.
fn prepare_buffers(
    audio: DecodedAudio,
    sample_rate: SampleRate,
    sample_plan: BenchmarkSamplePlan,
    audio_path: &Path,
) -> Result<BenchmarkBuffers, String> {
    let audio = composition::resample_to(audio, sample_rate)
        .map_err(|error| format!("bench resample {}: {error}", audio_path.display()))?;
    let channel = first_nonempty_channel(&audio, audio_path)?;
    let primary = cyclic_buffer(channel, sample_plan.primary(), audio_path)?;
    let alternate = match sample_plan.alternate() {
        Some(samples) => Some(cyclic_buffer(channel, samples, audio_path)?),
        None => None,
    };
    Ok(BenchmarkBuffers { primary, alternate })
}

fn first_nonempty_channel<'a>(
    audio: &'a DecodedAudio,
    audio_path: &Path,
) -> Result<&'a ChannelAudio, String> {
    match audio
        .channels()
        .first()
        .filter(|channel| !channel.is_empty())
    {
        Some(channel) => Ok(channel),
        None => Err(format!(
            "bench audio {}: no audio samples",
            audio_path.display()
        )),
    }
}

fn cyclic_buffer(
    channel: &ChannelAudio,
    requested_samples: usize,
    audio_path: &Path,
) -> Result<Vec<f32>, String> {
    let source_length = channel.len();
    if source_length == 0 {
        return Err(format!(
            "bench audio {}: no audio samples",
            audio_path.display()
        ));
    }
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(requested_samples).map_err(|_| {
        format!(
            "bench audio {}: cannot reserve {requested_samples} benchmark samples",
            audio_path.display()
        )
    })?;
    for index in 0..requested_samples {
        let sample = channel
            .samples()
            .get(index % source_length)
            .copied()
            .ok_or_else(|| "validated benchmark source channel index must exist".to_owned())?;
        buffer.push(sample);
    }
    Ok(buffer)
}

fn measurement(values: &[f64], index: usize, context: &str) -> Result<f64, String> {
    values
        .get(index)
        .copied()
        .ok_or_else(|| format!("benchmark {context} index is unavailable"))
}

#[cfg(test)]
mod tests {
    use super::prepare_buffers;
    use crate::numeric::BenchmarkSamplePlan;
    use gigaam_audio::{ChannelAudio, DecodedAudio, SampleRate, resample_audio};
    use std::path::Path;

    fn cyclic(samples: &[f32], length: usize) -> Vec<f32> {
        (0..length)
            .map(|index| samples[index % samples.len()])
            .collect()
    }

    #[test]
    fn benchmark_buffers_resample_to_the_package_rate_before_cyclic_projection()
    -> Result<(), String> {
        let input_rate = SampleRate::new(8_000)?;
        let package_rate = SampleRate::new(24_000)?;
        let audio =
            DecodedAudio::new(input_rate, vec![ChannelAudio::new(vec![0.25, -0.5, 0.75])?])?;
        let plan = BenchmarkSamplePlan::new(package_rate, 0.001, Some(0.0005))?;
        let expected = resample_audio(audio.clone(), package_rate)?;
        let expected_samples = expected
            .channels()
            .first()
            .ok_or_else(|| "validated expected audio must retain one channel".to_owned())?
            .samples();

        let buffers = prepare_buffers(audio, package_rate, plan, Path::new("benchmark.wav"))?;

        assert_eq!(expected.sample_rate(), package_rate);
        assert_eq!(buffers.primary.len(), plan.primary());
        assert_eq!(buffers.primary, cyclic(expected_samples, plan.primary()));
        let alternate = buffers
            .alternate
            .ok_or_else(|| "an alternate benchmark duration must retain its buffer".to_owned())?;
        let alternate_length = plan
            .alternate()
            .ok_or_else(|| "an alternate benchmark duration must retain its plan".to_owned())?;
        assert_eq!(alternate.len(), alternate_length);
        assert_eq!(alternate, cyclic(expected_samples, alternate_length));
        Ok(())
    }
}
