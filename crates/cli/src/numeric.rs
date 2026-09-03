// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Checked CLI-specific numeric projections and benchmark sample planning.

use gigaam_audio::SampleRate;
use gigaam_primitives::{f64_to_f32, trunc_f32_to_usize, usize_to_f32, usize_to_f64};

/// Complete package-rate benchmark buffer shapes selected before native runtime initialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BenchmarkSamplePlan {
    primary: usize,
    alternate: Option<usize>,
}

impl BenchmarkSamplePlan {
    /// Computes every requested package-rate benchmark shape from validated CLI durations.
    pub(crate) fn new(
        sample_rate: SampleRate,
        window_seconds: f32,
        alternate_window_seconds: Option<f32>,
    ) -> Result<Self, String> {
        let primary = benchmark_samples(sample_rate, window_seconds, "--window sample count")?;
        let alternate = match alternate_window_seconds {
            Some(seconds) => Some(benchmark_samples(
                sample_rate,
                seconds,
                "--alt-window sample count",
            )?),
            None => None,
        };
        Ok(Self { primary, alternate })
    }

    pub(crate) const fn primary(self) -> usize {
        self.primary
    }

    pub(crate) const fn alternate(self) -> Option<usize> {
        self.alternate
    }
}

/// Converts the CLI simulator chunk from milliseconds at the package-bound model rate.
pub(crate) fn stream_chunk(
    chunk_milliseconds: usize,
    sample_rate: SampleRate,
) -> Result<usize, String> {
    let samples_per_second = sample_rate
        .as_usize()
        .map_err(|error| format!("stream sample rate: {error}"))?;
    chunk_milliseconds
        .checked_mul(samples_per_second)
        .and_then(|samples| samples.checked_div(1_000))
        .filter(|samples| *samples > 0)
        .ok_or_else(|| {
            "stream chunk: --chunk-ms must produce at least one model-rate sample".into()
        })
}

/// Projects a finite nonnegative rounded `f32` to a platform index without saturation.
pub(crate) fn rounded_f32_to_usize(value: f32, context: &str) -> Result<usize, String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{context}: value must be finite and non-negative"));
    }
    trunc_f32_to_usize(value.round()).map_err(|_| format!("{context}: value exceeds usize"))
}

/// Converts one CLI benchmark duration to a package-rate sample count.
fn benchmark_samples(
    sample_rate: SampleRate,
    seconds: f32,
    context: &str,
) -> Result<usize, String> {
    let samples_per_second = sample_rate
        .as_usize()
        .map_err(|error| format!("{context}: {error}"))?;
    rounded_f32_to_usize(seconds * usize_to_f32(samples_per_second), context)
}

/// Converts a platform count for stable human-facing output without an unchecked cast.
pub(crate) fn count_to_f32(value: usize) -> f32 {
    usize_to_f32(value)
}

/// Converts a platform count for stable human-facing output without an unchecked cast.
pub(crate) fn count_to_f64(value: usize) -> f64 {
    usize_to_f64(value)
}

/// Narrows one finite elapsed duration only when its output projection remains finite.
pub(crate) fn duration_to_f32(value: f64, context: &str) -> Result<f32, String> {
    let narrowed = f64_to_f32(value);
    if !value.is_finite() || !narrowed.is_finite() {
        return Err(format!(
            "{context}: value cannot be represented as finite f32"
        ));
    }
    Ok(narrowed)
}

/// Converts one platform rate to Audio's checked 32-bit rate domain.
pub(crate) fn usize_to_u32(value: usize, context: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{context}: value {value} exceeds u32"))
}

/// Returns a checked percentile projection over the caller's mutable numeric collection.
pub(crate) fn percentile(values: &mut [f32], percentile: f32) -> Result<f32, String> {
    if values.is_empty() {
        return Ok(0.0);
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let last_index = values
        .len()
        .checked_sub(1)
        .ok_or_else(|| "percentile length underflows".to_owned())?;
    let index = rounded_f32_to_usize(count_to_f32(last_index) * percentile, "percentile index")?;
    values
        .get(index)
        .copied()
        .ok_or_else(|| "percentile index exceeds the sorted value range".to_owned())
}

#[cfg(test)]
mod tests {
    use super::BenchmarkSamplePlan;
    use gigaam_audio::SampleRate;

    #[test]
    fn benchmark_plan_rounds_each_duration_at_the_validated_package_rate() -> Result<(), String> {
        let sixteen_kilohertz = SampleRate::new(16_000)?;
        assert_eq!(
            BenchmarkSamplePlan::new(sixteen_kilohertz, 30.0, Some(10.0)),
            Ok(BenchmarkSamplePlan {
                primary: 480_000,
                alternate: Some(160_000),
            })
        );

        let twenty_four_kilohertz = SampleRate::new(24_000)?;
        assert_eq!(
            BenchmarkSamplePlan::new(twenty_four_kilohertz, 30.0, Some(10.0)),
            Ok(BenchmarkSamplePlan {
                primary: 720_000,
                alternate: Some(240_000),
            })
        );
        assert_eq!(
            BenchmarkSamplePlan::new(twenty_four_kilohertz, 30.0, None),
            Ok(BenchmarkSamplePlan {
                primary: 720_000,
                alternate: None,
            })
        );

        let binary_exact_rate = SampleRate::new(16_384)?;
        assert_eq!(
            BenchmarkSamplePlan::new(binary_exact_rate, 2.5 / 16_384.0, Some(3.5 / 16_384.0),),
            Ok(BenchmarkSamplePlan {
                primary: 3,
                alternate: Some(4),
            })
        );
        Ok(())
    }

    #[test]
    fn benchmark_plan_refuses_primary_and_alternate_overflow_before_runtime_work()
    -> Result<(), String> {
        let sample_rate = SampleRate::new(16_000)?;
        assert_eq!(
            BenchmarkSamplePlan::new(sample_rate, f32::MAX, None),
            Err("--window sample count: value must be finite and non-negative".into())
        );
        assert_eq!(
            BenchmarkSamplePlan::new(sample_rate, 30.0, Some(f32::MAX)),
            Err("--alt-window sample count: value must be finite and non-negative".into())
        );
        Ok(())
    }
}
