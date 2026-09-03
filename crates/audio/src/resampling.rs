// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Validated rational polyphase FIR resampling.

use crate::contracts::{ChannelAudio, DecodedAudio, SampleRate};
use gigaam_primitives::{f64_to_f32, usize_to_f64};

/// Maximum factor in either direction after reducing a rate ratio.
const MAX_REDUCED_RATE_FACTOR: usize = 1000;
const DEFAULT_TAPS_PER_PHASE: usize = 48;
const DEFAULT_KAISER_BETA: f64 = 9.0;
const DEFAULT_ROLLOFF: f64 = 0.945;
const MINIMUM_TAPS_PER_PHASE: usize = 2;
const BESSEL_MAX_ITERATIONS: usize = 60;
const BESSEL_RELATIVE_TOLERANCE: f64 = 1e-14;
const SINC_NEAR_ZERO: f64 = 1e-12;

/// A reduced, bounded input/output rate ratio. This is the sole owner of the rate-ratio
/// predicate used by batch and streaming audio.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RatePair {
    input: SampleRate,
    output: SampleRate,
    numerator: usize,
    denominator: usize,
}

impl RatePair {
    pub fn new(input: SampleRate, output: SampleRate) -> Result<Self, String> {
        let input_hz = input.as_usize()?;
        let output_hz = output.as_usize()?;
        let divisor = gcd(input_hz, output_hz);
        let numerator = output_hz
            .checked_div(divisor)
            .ok_or_else(|| "rate-ratio divisor must be nonzero".to_owned())?;
        let denominator = input_hz
            .checked_div(divisor)
            .ok_or_else(|| "rate-ratio divisor must be nonzero".to_owned())?;
        if numerator > MAX_REDUCED_RATE_FACTOR || denominator > MAX_REDUCED_RATE_FACTOR {
            return Err(format!(
                "reduced sample-rate ratio {numerator}/{denominator} exceeds the supported {MAX_REDUCED_RATE_FACTOR} bound"
            ));
        }
        Ok(Self {
            input,
            output,
            numerator,
            denominator,
        })
    }

    pub const fn input(self) -> SampleRate {
        self.input
    }

    pub const fn output(self) -> SampleRate {
        self.output
    }

    pub const fn numerator(self) -> usize {
        self.numerator
    }

    pub const fn denominator(self) -> usize {
        self.denominator
    }

    pub const fn is_identity(self) -> bool {
        self.numerator == 1 && self.denominator == 1
    }
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Filter parameters paired with a validated rational rate conversion.
#[derive(Clone, Copy, Debug)]
pub struct ResamplerConfig {
    rate_pair: RatePair,
    taps_per_phase: usize,
    beta: f64,
    rolloff: f64,
}

impl ResamplerConfig {
    /// The supported default: 48 taps per phase, Kaiser beta 9.0, and rolloff 0.945.
    pub fn new(rate_pair: RatePair) -> Self {
        Self {
            rate_pair,
            taps_per_phase: DEFAULT_TAPS_PER_PHASE,
            beta: DEFAULT_KAISER_BETA,
            rolloff: DEFAULT_ROLLOFF,
        }
    }

    pub fn with_filter(
        rate_pair: RatePair,
        taps_per_phase: usize,
        beta: f64,
        rolloff: f64,
    ) -> Result<Self, String> {
        if taps_per_phase < MINIMUM_TAPS_PER_PHASE {
            return Err(format!(
                "resampler taps per phase must be at least {MINIMUM_TAPS_PER_PHASE}"
            ));
        }
        if !beta.is_finite() || beta < 0.0 {
            return Err("resampler Kaiser beta must be finite and nonnegative".into());
        }
        if !rolloff.is_finite() || !(0.0..=1.0).contains(&rolloff) || rolloff == 0.0 {
            return Err("resampler rolloff must be finite and in (0, 1]".into());
        }
        Ok(Self {
            rate_pair,
            taps_per_phase,
            beta,
            rolloff,
        })
    }

    pub const fn rate_pair(self) -> RatePair {
        self.rate_pair
    }

    pub const fn taps_per_phase(self) -> usize {
        self.taps_per_phase
    }
}

/// Polyphase FIR state built once per validated configuration.
pub struct Resampler {
    pair: RatePair,
    taps_per_phase: usize,
    phases: Vec<Vec<f64>>,
    center: usize,
}

fn bessel_i0(value: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let squared_quarter = value * value / 4.0;
    for index in 1..BESSEL_MAX_ITERATIONS {
        let index_f64 = usize_to_f64(index);
        term *= squared_quarter / (index_f64 * index_f64);
        sum += term;
        if term.abs() < BESSEL_RELATIVE_TOLERANCE * sum.abs() {
            break;
        }
    }
    sum
}

fn checked_ceil_div(value: usize, divisor: usize, context: &str) -> Result<usize, String> {
    if divisor == 0 {
        return Err(format!("{context}: zero divisor"));
    }
    let quotient = value / divisor;
    if value.is_multiple_of(divisor) {
        Ok(quotient)
    } else {
        quotient
            .checked_add(1)
            .ok_or_else(|| format!("{context}: arithmetic overflows usize"))
    }
}

impl Resampler {
    pub fn new(config: ResamplerConfig) -> Result<Self, String> {
        let pair = config.rate_pair();
        let interpolation = pair.numerator();
        let decimation = pair.denominator();
        let unrounded_length = config
            .taps_per_phase
            .checked_mul(interpolation.max(decimation))
            .ok_or_else(|| "resampler filter length overflows usize".to_owned())?;
        let length = unrounded_length
            .checked_add(1 - unrounded_length % 2)
            .ok_or_else(|| "resampler odd filter length overflows usize".to_owned())?;
        if length < 3 {
            return Err("resampler filter length must support a centered FIR".into());
        }
        let input_hz = pair.input().as_usize()?;
        let output_hz = pair.output().as_usize()?;
        let upsampled_hz = input_hz
            .checked_mul(interpolation)
            .ok_or_else(|| "resampler upsampled rate overflows usize".to_owned())?;
        let cutoff = usize_to_f64(input_hz.min(output_hz)) * 0.5 * config.rolloff;
        let normalized_cutoff = 2.0 * std::f64::consts::PI * cutoff / usize_to_f64(upsampled_hz);
        if !normalized_cutoff.is_finite() || normalized_cutoff <= 0.0 {
            return Err("resampler cutoff is not finite and positive".into());
        }
        let center = usize_to_f64(length - 1) / 2.0;
        let beta_denominator = bessel_i0(config.beta);
        if !beta_denominator.is_finite() || beta_denominator == 0.0 {
            return Err("resampler Kaiser normalization is invalid".into());
        }

        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(length)
            .map_err(|_| "resampler coefficients cannot reserve memory".to_owned())?;
        let length_minus_one = usize_to_f64(length - 1);
        for index in 0..length {
            let distance = usize_to_f64(index) - center;
            let sinc = if distance.abs() < SINC_NEAR_ZERO {
                normalized_cutoff / std::f64::consts::PI
            } else {
                (normalized_cutoff * distance).sin() / (std::f64::consts::PI * distance)
            };
            let radius = 2.0 * usize_to_f64(index) / length_minus_one - 1.0;
            let window =
                bessel_i0(config.beta * (1.0 - radius * radius).max(0.0).sqrt()) / beta_denominator;
            let coefficient = sinc * window * usize_to_f64(interpolation);
            if !coefficient.is_finite() {
                return Err("resampler coefficients must be finite".into());
            }
            coefficients.push(coefficient);
        }

        let per_phase = checked_ceil_div(length, interpolation, "resampler phase length")?;
        let mut phases = Vec::new();
        phases
            .try_reserve_exact(interpolation)
            .map_err(|_| "resampler phases cannot reserve memory".to_owned())?;
        for phase in 0..interpolation {
            let mut branch = Vec::new();
            branch
                .try_reserve_exact(per_phase)
                .map_err(|_| "resampler phase cannot reserve memory".to_owned())?;
            for tap in 0..per_phase {
                let offset = tap
                    .checked_mul(interpolation)
                    .and_then(|value| value.checked_add(phase))
                    .ok_or_else(|| "resampler phase offset overflows usize".to_owned())?;
                if offset < coefficients.len() {
                    branch.push(coefficients[offset]);
                } else {
                    // Every branch has the same length; a zero only pads the FIR tail beyond
                    // the designed odd filter, not a recovery/fallback coefficient.
                    branch.push(0.0);
                }
            }
            phases.push(branch);
        }

        Ok(Self {
            pair,
            taps_per_phase: config.taps_per_phase,
            phases,
            center: (length - 1) / 2,
        })
    }

    pub const fn rate_pair(&self) -> RatePair {
        self.pair
    }

    pub const fn taps_per_phase(&self) -> usize {
        self.taps_per_phase
    }

    fn output_count(&self, input_frames: usize) -> Result<usize, String> {
        checked_ceil_div(
            input_frames
                .checked_mul(self.pair.numerator())
                .ok_or_else(|| "resampler output length overflows usize".to_owned())?,
            self.pair.denominator(),
            "resampler output length",
        )
    }

    fn output_at(&self, input: &[f32], output_index: usize) -> Result<f32, String> {
        let upsampled_index = output_index
            .checked_mul(self.pair.denominator())
            .and_then(|value| value.checked_add(self.center))
            .ok_or_else(|| "resampler output position overflows usize".to_owned())?;
        let base = upsampled_index / self.pair.numerator();
        let phase = upsampled_index % self.pair.numerator();
        let branch = self
            .phases
            .get(phase)
            .ok_or_else(|| "resampler phase is outside the validated plan".to_owned())?;
        let mut accumulator = 0.0_f64;
        for (tap, coefficient) in branch.iter().enumerate() {
            if base < tap {
                break;
            }
            let input_index = base - tap;
            if let Some(sample) = input.get(input_index) {
                accumulator += coefficient * f64::from(*sample);
            }
        }
        let output = f64_to_f32(accumulator);
        if !output.is_finite() {
            return Err("resampler produced a non-finite sample".into());
        }
        Ok(output)
    }

    /// Resamples a complete finite input. Identity conversion returns exact input samples.
    pub fn process(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.iter().any(|sample| !sample.is_finite()) {
            return Err("resampler input samples must be finite".into());
        }
        if self.pair.is_identity() {
            return Ok(input.to_vec());
        }
        let output_count = self.output_count(input.len())?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_count)
            .map_err(|_| "resampler output cannot reserve memory".to_owned())?;
        for output_index in 0..output_count {
            output.push(self.output_at(input, output_index)?);
        }
        Ok(output)
    }
}

/// Resamples all validated channels to an explicit validated rate.
///
/// Whole-audio orchestration belongs with the rate-pair and FIR owner so every caller, including
/// the Opus reference-clock path, receives the same checked conversion semantics.
pub fn resample_audio(
    audio: DecodedAudio,
    output_rate: SampleRate,
) -> Result<DecodedAudio, String> {
    let (input_rate, channels) = audio.into_parts();
    let pair = RatePair::new(input_rate, output_rate)?;
    if pair.is_identity() {
        return DecodedAudio::new(input_rate, channels);
    }
    let resampler = Resampler::new(ResamplerConfig::new(pair))?;
    let channels = channels
        .into_iter()
        .map(|channel| {
            resampler
                .process(channel.samples())
                .and_then(ChannelAudio::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    DecodedAudio::new(output_rate, channels)
}

/// A streaming version of [`Resampler`]. `finish(self)` consumes the state, so no input can be
/// accepted after its terminal tail has been emitted.
pub struct StreamResampler {
    resampler: Resampler,
    buffer: Vec<f32>,
    input_start: usize,
    outputs_emitted: usize,
}

impl StreamResampler {
    pub fn new(resampler: Resampler) -> Self {
        Self {
            resampler,
            buffer: Vec::new(),
            input_start: 0,
            outputs_emitted: 0,
        }
    }

    fn per_phase(&self) -> Result<usize, String> {
        self.resampler
            .phases
            .first()
            .map(Vec::len)
            .ok_or_else(|| "resampler plan has no validated phases".to_owned())
    }

    fn total_input(&self) -> Result<usize, String> {
        self.input_start
            .checked_add(self.buffer.len())
            .ok_or_else(|| "stream resampler input position overflows usize".to_owned())
    }

    fn output_at(&self, output_index: usize) -> Result<f32, String> {
        let pair = self.resampler.rate_pair();
        let upsampled_index = output_index
            .checked_mul(pair.denominator())
            .and_then(|value| value.checked_add(self.resampler.center))
            .ok_or_else(|| "stream resampler output position overflows usize".to_owned())?;
        let base = upsampled_index / pair.numerator();
        let phase = upsampled_index % pair.numerator();
        let branch = self
            .resampler
            .phases
            .get(phase)
            .ok_or_else(|| "stream resampler phase is outside the validated plan".to_owned())?;
        let mut accumulator = 0.0_f64;
        let input_end = self.total_input()?;
        for (tap, coefficient) in branch.iter().enumerate() {
            if base < tap {
                break;
            }
            let absolute_index = base - tap;
            if absolute_index >= self.input_start && absolute_index < input_end {
                let local_index = absolute_index
                    .checked_sub(self.input_start)
                    .ok_or_else(|| "stream resampler buffer index underflows".to_owned())?;
                let sample = self.buffer.get(local_index).ok_or_else(|| {
                    "stream resampler buffer is shorter than its validated index".to_owned()
                })?;
                accumulator += coefficient * f64::from(*sample);
            }
        }
        let output = f64_to_f32(accumulator);
        if !output.is_finite() {
            return Err("stream resampler produced a non-finite sample".into());
        }
        Ok(output)
    }

    pub fn push(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.iter().any(|sample| !sample.is_finite()) {
            return Err("stream resampler input samples must be finite".into());
        }
        if self.resampler.rate_pair().is_identity() {
            return Ok(input.to_vec());
        }
        self.buffer
            .try_reserve(input.len())
            .map_err(|_| "stream resampler input cannot reserve memory".to_owned())?;
        self.buffer.extend_from_slice(input);
        let total_input = self.total_input()?;
        let mut output = Vec::new();
        if total_input != 0 {
            loop {
                let pair = self.resampler.rate_pair();
                let base = self
                    .outputs_emitted
                    .checked_mul(pair.denominator())
                    .and_then(|value| value.checked_add(self.resampler.center))
                    .ok_or_else(|| "stream resampler output position overflows usize".to_owned())?
                    / pair.numerator();
                if base >= total_input {
                    break;
                }
                output.push(self.output_at(self.outputs_emitted)?);
                self.outputs_emitted = self
                    .outputs_emitted
                    .checked_add(1)
                    .ok_or_else(|| "stream resampler output count overflows usize".to_owned())?;
            }
        }

        let pair = self.resampler.rate_pair();
        let next_base = self
            .outputs_emitted
            .checked_mul(pair.denominator())
            .and_then(|value| value.checked_add(self.resampler.center))
            .ok_or_else(|| "stream resampler output position overflows usize".to_owned())?
            / pair.numerator();
        let history = self.per_phase()?.checked_sub(1).ok_or_else(|| {
            "stream resampler phase length must be nonzero after plan validation".to_owned()
        })?;
        let keep_from = next_base.saturating_sub(history);
        if keep_from > self.input_start {
            let drop_count = keep_from
                .checked_sub(self.input_start)
                .ok_or_else(|| "stream resampler drop count underflows".to_owned())?
                .min(self.buffer.len());
            self.buffer.drain(..drop_count);
            self.input_start = self
                .input_start
                .checked_add(drop_count)
                .ok_or_else(|| "stream resampler input position overflows usize".to_owned())?;
        }
        Ok(output)
    }

    pub fn finish(mut self) -> Result<Vec<f32>, String> {
        if self.resampler.rate_pair().is_identity() {
            return Ok(Vec::new());
        }
        let total_input = self.total_input()?;
        let total_output = self.resampler.output_count(total_input)?;
        let mut output = Vec::new();
        let remaining = total_output
            .checked_sub(self.outputs_emitted)
            .ok_or_else(|| {
                "stream resampler emitted more outputs than its input permits".to_owned()
            })?;
        output
            .try_reserve_exact(remaining)
            .map_err(|_| "stream resampler tail cannot reserve memory".to_owned())?;
        while self.outputs_emitted < total_output {
            output.push(self.output_at(self.outputs_emitted)?);
            self.outputs_emitted = self
                .outputs_emitted
                .checked_add(1)
                .ok_or_else(|| "stream resampler output count overflows usize".to_owned())?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::{RatePair, Resampler, ResamplerConfig, StreamResampler};
    use crate::SampleRate;
    use gigaam_primitives::{f64_to_f32, usize_to_f64};

    fn rate(value: u32) -> Result<SampleRate, String> {
        SampleRate::new(value)
    }

    fn resampler(input: u32, output: u32) -> Result<Resampler, String> {
        let pair = RatePair::new(rate(input)?, rate(output)?)?;
        Resampler::new(ResamplerConfig::new(pair))
    }

    fn tones(sample_rate: usize, frames: usize, frequencies: &[f64]) -> Vec<f32> {
        (0..frames)
            .map(|index| {
                let mixed = frequencies
                    .iter()
                    .map(|frequency| {
                        (2.0 * std::f64::consts::PI * frequency * usize_to_f64(index)
                            / usize_to_f64(sample_rate))
                        .sin()
                            / usize_to_f64(frequencies.len())
                    })
                    .sum::<f64>();
                f64_to_f32(mixed)
            })
            .collect()
    }

    fn snr_db(reference: &[f32], actual: &[f32], skip: usize) -> Result<f64, String> {
        let frames = reference.len().min(actual.len());
        let end = frames
            .checked_sub(skip)
            .ok_or_else(|| "SNR skip exceeds available frames".to_owned())?;
        let (mut signal, mut error) = (0.0_f64, 0.0_f64);
        for index in skip..end {
            signal += f64::from(reference[index]).powi(2);
            error += (f64::from(reference[index]) - f64::from(actual[index])).powi(2);
        }
        Ok(10.0 * (signal / error.max(1e-30)).log10())
    }

    fn check_snr(input: u32, output: u32, frequencies: &[f64]) -> Result<(), String> {
        let input_frames = usize::try_from(input)
            .map_err(|_| "test sample rate does not fit usize".to_owned())?
            .checked_mul(2)
            .ok_or_else(|| "test input frame count overflows".to_owned())?;
        let output_frames = usize::try_from(output)
            .map_err(|_| "test sample rate does not fit usize".to_owned())?
            .checked_mul(2)
            .ok_or_else(|| "test output frame count overflows".to_owned())?;
        let source = tones(
            usize::try_from(input).map_err(|_| "test rate".to_owned())?,
            input_frames,
            frequencies,
        );
        let actual = resampler(input, output)?.process(&source)?;
        let reference = tones(
            usize::try_from(output).map_err(|_| "test rate".to_owned())?,
            output_frames,
            frequencies,
        );
        let skip = usize::try_from(output).map_err(|_| "test rate".to_owned())? / 10;
        let snr = snr_db(&reference, &actual, skip)?;
        assert!(snr > 60.0, "{input}->{output}: SNR {snr:.1} dB");
        Ok(())
    }

    #[test]
    fn rational_resampling_preserves_reference_snr() -> Result<(), String> {
        check_snr(8000, 16000, &[200.0, 1000.0, 3000.0])?;
        check_snr(48000, 16000, &[200.0, 1000.0, 3000.0, 6000.0])?;
        check_snr(44100, 16000, &[300.0, 1500.0, 5000.0])
    }

    #[test]
    fn streaming_matches_offline_for_arbitrary_partitions() -> Result<(), String> {
        let source = tones(44100, 44_100, &[300.0, 1200.0, 4000.0]);
        let offline = resampler(44100, 16000)?.process(&source)?;
        let mut stream = StreamResampler::new(resampler(44100, 16000)?);
        let partitions = [1_usize, 100, 333, 7];
        let (mut offset, mut partition_index, mut actual) = (0_usize, 0_usize, Vec::new());
        while offset < source.len() {
            let count = partitions[partition_index % partitions.len()].min(source.len() - offset);
            actual.extend(stream.push(&source[offset..offset + count])?);
            offset += count;
            partition_index += 1;
        }
        actual.extend(stream.finish()?);
        assert_eq!(actual, offline);
        Ok(())
    }

    #[test]
    fn invalid_rate_pairs_and_filter_configurations_refuse() -> Result<(), String> {
        assert!(RatePair::new(rate(16_000)?, rate(1)?).is_err());
        let pair = RatePair::new(rate(16_000)?, rate(16_000)?)?;
        assert!(ResamplerConfig::with_filter(pair, 1, 9.0, 0.945).is_err());
        assert!(ResamplerConfig::with_filter(pair, 48, f64::NAN, 0.945).is_err());
        assert!(ResamplerConfig::with_filter(pair, 48, 9.0, 0.0).is_err());
        Ok(())
    }

    #[test]
    fn identity_is_exact() -> Result<(), String> {
        let source = tones(16000, 1600, &[440.0]);
        assert_eq!(resampler(16000, 16000)?.process(&source)?, source);
        Ok(())
    }
}
