// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Model-ready log-mel features from typed model-package definitions and weights.

use crate::contracts::{FeatureMatrix, FrontendMode, SampleRate};
use crate::fft::{Complex, RealFft};
use gigaam_model_package::{FrontendDefinition, FrontendWeights};
use gigaam_primitives::f64_to_f32;

struct MelBand {
    first_bin: usize,
    weights: Vec<f32>,
}

/// Immutable frontend plan. Construction consumes typed package projections; it never reads
/// environment variables or resolves package paths.
pub struct FrontendProcessor {
    mode: FrontendMode,
    sample_rate: SampleRate,
    fft_length: usize,
    hop_length: usize,
    mel_bins: usize,
    centered: bool,
    clamp_min: f32,
    clamp_max: f32,
    window: Vec<f32>,
    bands: Vec<MelBand>,
    fft: RealFft,
}

/// Reusable scalar-frame workspace allocated outside the frontend hot path.
pub struct FrontendScratch {
    frame: Vec<f32>,
    power: Vec<f32>,
    buffer: Vec<Complex>,
    scratch: Vec<Complex>,
}

fn checked_product(left: usize, right: usize, context: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{context} overflows usize"))
}

impl FrontendProcessor {
    pub fn new(
        definition: &FrontendDefinition,
        weights: FrontendWeights,
        mode: FrontendMode,
    ) -> Result<Self, String> {
        let fft_length = definition.n_fft();
        let sample_rate = SampleRate::from_usize(definition.sample_rate(), "frontend")?;
        let mel_bins = definition.n_mels();
        let expected_window = [fft_length];
        if weights.window_dimensions() != expected_window {
            return Err(format!(
                "frontend window dimensions must be {expected_window:?}, got {:?}",
                weights.window_dimensions()
            ));
        }
        let frequency_bins = fft_length
            .checked_div(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "frontend FFT frequency dimensions overflow usize".to_owned())?;
        let expected_filterbank = [frequency_bins, mel_bins];
        if weights.filterbank_dimensions() != expected_filterbank {
            return Err(format!(
                "frontend filterbank dimensions must be {expected_filterbank:?}, got {:?}",
                weights.filterbank_dimensions()
            ));
        }
        if weights
            .window_values()
            .iter()
            .any(|value| !value.is_finite())
            || weights
                .filterbank_values()
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err("frontend weight values must be finite".into());
        }
        let expected_filter_values =
            checked_product(frequency_bins, mel_bins, "frontend filterbank dimensions")?;
        if weights.window_values().len() != fft_length
            || weights.filterbank_values().len() != expected_filter_values
        {
            return Err("frontend weight values do not match their declared dimensions".into());
        }
        let fft = RealFft::new(fft_length)?;
        let mut bands = Vec::new();
        bands
            .try_reserve_exact(mel_bins)
            .map_err(|_| "frontend mel bands cannot reserve memory".to_owned())?;
        for mel_bin in 0..mel_bins {
            let mut first_nonzero = None;
            let mut last_nonzero = None;
            for frequency_bin in 0..frequency_bins {
                let index = checked_product(frequency_bin, mel_bins, "frontend filterbank index")?
                    .checked_add(mel_bin)
                    .ok_or_else(|| "frontend filterbank index overflows usize".to_owned())?;
                let value = *weights.filterbank_values().get(index).ok_or_else(|| {
                    "frontend filterbank values are shorter than their validated dimensions"
                        .to_owned()
                })?;
                if value != 0.0 {
                    if first_nonzero.is_none() {
                        first_nonzero = Some(frequency_bin);
                    }
                    last_nonzero = Some(frequency_bin);
                }
            }
            let first_bin = first_nonzero.ok_or_else(|| {
                format!("frontend mel band {mel_bin} must contain at least one nonzero weight")
            })?;
            let last_bin = last_nonzero.ok_or_else(|| {
                format!("frontend mel band {mel_bin} lost its final nonzero weight")
            })?;
            let weight_count = last_bin
                .checked_sub(first_bin)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| "frontend mel band dimensions overflow usize".to_owned())?;
            let mut band_weights = Vec::new();
            band_weights
                .try_reserve_exact(weight_count)
                .map_err(|_| "frontend mel band cannot reserve memory".to_owned())?;
            for frequency_bin in first_bin..=last_bin {
                let index = checked_product(frequency_bin, mel_bins, "frontend filterbank index")?
                    .checked_add(mel_bin)
                    .ok_or_else(|| "frontend filterbank index overflows usize".to_owned())?;
                let value = *weights.filterbank_values().get(index).ok_or_else(|| {
                    "frontend filterbank values are shorter than their validated dimensions"
                        .to_owned()
                })?;
                band_weights.push(value);
            }
            bands.push(MelBand {
                first_bin,
                weights: band_weights,
            });
        }
        let clamp_min = f64_to_f32(definition.log_clamp_min());
        let clamp_max = f64_to_f32(definition.log_clamp_max());
        if !clamp_min.is_finite()
            || !clamp_max.is_finite()
            || clamp_min <= 0.0
            || clamp_max < clamp_min
        {
            return Err("frontend f32 clamp values must be finite, positive, and ordered".into());
        }
        Ok(Self {
            mode,
            sample_rate,
            fft_length,
            hop_length: definition.hop_length(),
            mel_bins,
            centered: definition.center(),
            clamp_min,
            clamp_max,
            window: weights.window_values().to_vec(),
            bands,
            fft,
        })
    }

    pub const fn mode(&self) -> FrontendMode {
        self.mode
    }

    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    pub const fn fft_length(&self) -> usize {
        self.fft_length
    }

    pub const fn hop_length(&self) -> usize {
        self.hop_length
    }

    pub const fn mel_bins(&self) -> usize {
        self.mel_bins
    }

    pub const fn is_centered(&self) -> bool {
        self.centered
    }

    pub fn scratch(&self) -> Result<FrontendScratch, String> {
        let frequency_bins = self
            .fft_length
            .checked_div(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "frontend scratch frequency dimensions overflow usize".to_owned())?;
        let half_length = self
            .fft_length
            .checked_div(2)
            .ok_or_else(|| "frontend scratch FFT dimensions are invalid".to_owned())?;
        let mut frame = Vec::new();
        let mut power = Vec::new();
        let mut buffer = Vec::new();
        let mut scratch = Vec::new();
        frame
            .try_reserve_exact(self.fft_length)
            .map_err(|_| "frontend frame workspace cannot reserve memory".to_owned())?;
        power
            .try_reserve_exact(frequency_bins)
            .map_err(|_| "frontend power workspace cannot reserve memory".to_owned())?;
        buffer
            .try_reserve_exact(half_length)
            .map_err(|_| "frontend FFT workspace cannot reserve memory".to_owned())?;
        scratch
            .try_reserve_exact(half_length)
            .map_err(|_| "frontend FFT scratch cannot reserve memory".to_owned())?;
        frame.resize(self.fft_length, 0.0);
        power.resize(frequency_bins, 0.0);
        buffer.resize(half_length, Complex::default());
        scratch.resize(half_length, Complex::default());
        Ok(FrontendScratch {
            frame,
            power,
            buffer,
            scratch,
        })
    }

    pub fn frame_log_mel(
        &self,
        samples: &[f32],
        output: &mut [f32],
        scratch: &mut FrontendScratch,
    ) -> Result<(), String> {
        if samples.len() != self.fft_length || output.len() != self.mel_bins {
            return Err("frontend frame buffers do not match the validated dimensions".into());
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err("frontend frame samples must be finite".into());
        }
        if scratch.frame.len() != self.fft_length
            || scratch.power.len() != self.fft_length / 2 + 1
            || scratch.buffer.len() != self.fft_length / 2
            || scratch.scratch.len() != self.fft_length / 2
        {
            return Err("frontend scratch does not match this processor".into());
        }
        for ((destination, sample), window) in
            scratch.frame.iter_mut().zip(samples).zip(&self.window)
        {
            *destination = sample * window;
        }
        self.fft.power_spectrum(
            &scratch.frame,
            &mut scratch.power,
            &mut scratch.buffer,
            &mut scratch.scratch,
        )?;
        for (mel_bin, band) in self.bands.iter().enumerate() {
            let mut accumulated = 0.0_f32;
            for (offset, weight) in band.weights.iter().enumerate() {
                let frequency_bin = band
                    .first_bin
                    .checked_add(offset)
                    .ok_or_else(|| "frontend mel frequency index overflows usize".to_owned())?;
                let power = *scratch.power.get(frequency_bin).ok_or_else(|| {
                    "frontend mel band exceeds the validated FFT power dimension".to_owned()
                })?;
                accumulated += power * weight;
            }
            output[mel_bin] = accumulated.clamp(self.clamp_min, self.clamp_max).ln();
            if !output[mel_bin].is_finite() {
                return Err("frontend produced a non-finite log-mel value".into());
            }
        }
        Ok(())
    }

    pub fn log_mel(&self, samples: &[f32]) -> Result<FeatureMatrix, String> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err("frontend input samples must be finite".into());
        }
        match self.mode {
            FrontendMode::Scalar => self.log_mel_scalar(samples),
            FrontendMode::Batched => self.log_mel_batched(samples),
        }
    }

    fn frame_count(&self, samples: usize) -> Result<usize, String> {
        let effective_length = if self.centered {
            let padding = self
                .fft_length
                .checked_div(2)
                .ok_or_else(|| "frontend padding dimensions are invalid".to_owned())?;
            samples
                .checked_add(checked_product(padding, 2, "frontend centered input")?)
                .ok_or_else(|| "frontend centered input length overflows usize".to_owned())?
        } else {
            samples
        };
        if effective_length < self.fft_length {
            Ok(0)
        } else {
            effective_length
                .checked_sub(self.fft_length)
                .and_then(|value| value.checked_div(self.hop_length))
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| "frontend frame count overflows usize".to_owned())
        }
    }

    fn centered<'a>(
        &self,
        samples: &'a [f32],
        padded: &'a mut Vec<f32>,
    ) -> Result<&'a [f32], String> {
        if !self.centered {
            return Ok(samples);
        }
        let padding = self.fft_length / 2;
        if samples.len() <= padding {
            return Err(
                "centered frontend input must be longer than its reflect-padding width".into(),
            );
        }
        let capacity = samples
            .len()
            .checked_add(checked_product(padding, 2, "frontend centered input")?)
            .ok_or_else(|| "frontend centered input length overflows usize".to_owned())?;
        padded.clear();
        padded
            .try_reserve(capacity)
            .map_err(|_| "frontend centered input cannot reserve memory".to_owned())?;
        for offset in (1..=padding).rev() {
            padded.push(samples[offset]);
        }
        padded.extend_from_slice(samples);
        for offset in 1..=padding {
            let source = samples
                .len()
                .checked_sub(1 + offset)
                .ok_or_else(|| "frontend reflect-padding index underflows".to_owned())?;
            padded.push(samples[source]);
        }
        Ok(padded)
    }

    fn log_mel_scalar(&self, samples: &[f32]) -> Result<FeatureMatrix, String> {
        let frames = self.frame_count(samples.len())?;
        if frames == 0 {
            return FeatureMatrix::new(self.mel_bins, 0, Vec::new());
        }
        let mut padded = Vec::new();
        let centered = self.centered(samples, &mut padded)?;
        let output_len = checked_product(self.mel_bins, frames, "frontend output dimensions")?;
        let mut output = Vec::new();
        let mut column = Vec::new();
        output
            .try_reserve_exact(output_len)
            .map_err(|_| "frontend output cannot reserve memory".to_owned())?;
        column
            .try_reserve_exact(self.mel_bins)
            .map_err(|_| "frontend mel column cannot reserve memory".to_owned())?;
        output.resize(output_len, 0.0);
        column.resize(self.mel_bins, 0.0);
        let mut scratch = self.scratch()?;
        for frame in 0..frames {
            let offset = checked_product(frame, self.hop_length, "frontend frame offset")?;
            let end = offset
                .checked_add(self.fft_length)
                .ok_or_else(|| "frontend frame end overflows usize".to_owned())?;
            self.frame_log_mel(&centered[offset..end], &mut column, &mut scratch)?;
            for (mel_bin, value) in column.iter().copied().enumerate() {
                let destination = checked_product(mel_bin, frames, "frontend output index")?
                    .checked_add(frame)
                    .ok_or_else(|| "frontend output index overflows usize".to_owned())?;
                output[destination] = value;
            }
        }
        FeatureMatrix::new(self.mel_bins, frames, output)
    }

    fn log_mel_batched(&self, samples: &[f32]) -> Result<FeatureMatrix, String> {
        let frames = self.frame_count(samples.len())?;
        if frames == 0 {
            return FeatureMatrix::new(self.mel_bins, 0, Vec::new());
        }
        let mut padded = Vec::new();
        let centered = self.centered(samples, &mut padded)?;
        let half_length = self.fft_length / 2;
        let packed_len = checked_product(half_length, frames, "batched frontend dimensions")?;
        let output_len =
            checked_product(self.mel_bins, frames, "batched frontend output dimensions")?;
        let power_len = checked_product(
            half_length
                .checked_add(1)
                .ok_or_else(|| "batched frontend power dimension overflows usize".to_owned())?,
            frames,
            "batched frontend power dimensions",
        )?;
        let mut real = vec![0.0_f32; packed_len];
        let mut imaginary = vec![0.0_f32; packed_len];
        for frame in 0..frames {
            let offset = checked_product(frame, self.hop_length, "batched frontend frame offset")?;
            for index in 0..half_length {
                let left = index
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(offset))
                    .ok_or_else(|| "batched frontend frame index overflows usize".to_owned())?;
                let right = left
                    .checked_add(1)
                    .ok_or_else(|| "batched frontend frame index overflows usize".to_owned())?;
                let target = checked_product(index, frames, "batched frontend packed index")?
                    .checked_add(frame)
                    .ok_or_else(|| "batched frontend packed index overflows usize".to_owned())?;
                real[target] = centered[left] * self.window[index * 2];
                imaginary[target] = centered[right] * self.window[index * 2 + 1];
            }
        }
        let mut scratch_real = vec![0.0_f32; packed_len];
        let mut scratch_imaginary = vec![0.0_f32; packed_len];
        let mut power = vec![0.0_f32; power_len];
        self.fft.power_spectrum_batched(
            &mut real,
            &mut imaginary,
            &mut scratch_real,
            &mut scratch_imaginary,
            &mut power,
            frames,
        )?;
        let mut output = vec![0.0_f32; output_len];
        for (mel_bin, band) in self.bands.iter().enumerate() {
            let output_offset = checked_product(mel_bin, frames, "batched frontend output index")?;
            for (offset, weight) in band.weights.iter().enumerate() {
                let bin = band.first_bin.checked_add(offset).ok_or_else(|| {
                    "batched frontend mel frequency index overflows usize".to_owned()
                })?;
                let power_offset = checked_product(bin, frames, "batched frontend power index")?;
                for frame in 0..frames {
                    let output_index = output_offset.checked_add(frame).ok_or_else(|| {
                        "batched frontend output index overflows usize".to_owned()
                    })?;
                    let power_index = power_offset
                        .checked_add(frame)
                        .ok_or_else(|| "batched frontend power index overflows usize".to_owned())?;
                    output[output_index] += power[power_index] * weight;
                }
            }
            for frame in 0..frames {
                let output_index = output_offset
                    .checked_add(frame)
                    .ok_or_else(|| "batched frontend output index overflows usize".to_owned())?;
                output[output_index] = output[output_index]
                    .clamp(self.clamp_min, self.clamp_max)
                    .ln();
                if !output[output_index].is_finite() {
                    return Err("frontend produced a non-finite log-mel value".into());
                }
            }
        }
        FeatureMatrix::new(self.mel_bins, frames, output)
    }
}
