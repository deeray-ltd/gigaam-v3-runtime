// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Mixed-radix Stockham FFT plans. Execution uses caller-provided work buffers and does not
//! allocate on the frontend hot path.

use gigaam_primitives::{f64_to_f32, usize_to_f64};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Complex {
    pub(crate) re: f32,
    pub(crate) im: f32,
}

impl Complex {
    #[inline]
    fn multiply(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    #[inline]
    fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

struct Stage {
    radix: usize,
    groups: usize,
    stride: usize,
    twiddles: Vec<Complex>,
    roots: Vec<Complex>,
}

/// A validated FFT plan using radices no wider than the allocation-free butterfly kernel.
pub(crate) struct Fft {
    length: usize,
    stages: Vec<Stage>,
}

fn checked_product(left: usize, right: usize, context: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{context} overflows usize"))
}

fn supported_factor(length: usize) -> Result<usize, String> {
    for factor in [2_usize, 3, 5, 7, 11, 13] {
        if length.is_multiple_of(factor) {
            return Ok(factor);
        }
    }
    let mut factor = 17_usize;
    while factor <= length / factor {
        if length.is_multiple_of(factor) {
            return Err(format!(
                "FFT length {length} requires unsupported radix {factor}; the allocation-free kernel supports radices through 16"
            ));
        }
        factor = factor
            .checked_add(2)
            .ok_or_else(|| "FFT factor search overflows usize".to_owned())?;
    }
    if length > 16 {
        return Err(format!(
            "FFT length {length} requires unsupported radix {length}; the allocation-free kernel supports radices through 16"
        ));
    }
    Ok(length)
}

fn twiddle(index: usize, length: usize) -> Complex {
    let angle = -2.0 * std::f64::consts::PI * usize_to_f64(index % length) / usize_to_f64(length);
    Complex {
        re: f64_to_f32(angle.cos()),
        im: f64_to_f32(angle.sin()),
    }
}

impl Fft {
    pub(crate) fn new(length: usize) -> Result<Self, String> {
        if length == 0 {
            return Err("FFT length must be nonzero".into());
        }
        let mut stages = Vec::new();
        let (mut current, mut stride) = (length, 1_usize);
        while current > 1 {
            let radix = supported_factor(current)?;
            let groups = current / radix;
            let twiddle_count = checked_product(groups, radix, "FFT twiddle count")?;
            let root_count = checked_product(radix, radix, "FFT root count")?;
            let mut twiddles = Vec::new();
            twiddles
                .try_reserve_exact(twiddle_count)
                .map_err(|_| "FFT twiddles cannot reserve memory".to_owned())?;
            for group in 0..groups {
                for lane in 0..radix {
                    let index = checked_product(group, lane, "FFT twiddle index")?;
                    twiddles.push(twiddle(index, current));
                }
            }
            let mut roots = Vec::new();
            roots
                .try_reserve_exact(root_count)
                .map_err(|_| "FFT roots cannot reserve memory".to_owned())?;
            for index in 0..root_count {
                let row = index / radix;
                let column = index % radix;
                roots.push(twiddle(
                    checked_product(row, column, "FFT root index")?,
                    radix,
                ));
            }
            stages.push(Stage {
                radix,
                groups,
                stride,
                twiddles,
                roots,
            });
            current = groups;
            stride = checked_product(stride, radix, "FFT stage stride")?;
        }
        Ok(Self { length, stages })
    }

    fn validate_buffers(&self, input: &[Complex], scratch: &[Complex]) -> Result<(), String> {
        if input.len() != self.length || scratch.len() != self.length {
            return Err(format!(
                "FFT buffers must both have length {}, got {} and {}",
                self.length,
                input.len(),
                scratch.len()
            ));
        }
        Ok(())
    }

    pub(crate) fn forward_inplace(
        &self,
        input: &mut [Complex],
        scratch: &mut [Complex],
    ) -> Result<(), String> {
        self.validate_buffers(input, scratch)?;
        let (mut source, mut destination): (&mut [Complex], &mut [Complex]) = (input, scratch);
        let mut values = [Complex::default(); 16];
        for stage in &self.stages {
            // Plan construction proves every stage index is below `self.length`; buffer
            // validation above proves the two slices have exactly that length. Keep the
            // butterfly free of fallible arithmetic on the frontend hot path.
            for group in 0..stage.groups {
                for offset in 0..stage.stride {
                    for (radix_index, value) in values.iter_mut().take(stage.radix).enumerate() {
                        let position = offset + stage.stride * (group + radix_index * stage.groups);
                        *value = source[position];
                    }
                    for output_index in 0..stage.radix {
                        let mut value = values[0];
                        for (radix_index, input) in
                            values.iter().take(stage.radix).enumerate().skip(1)
                        {
                            let root_index = radix_index * stage.radix + output_index;
                            value = value.add(input.multiply(stage.roots[root_index]));
                        }
                        let destination_index =
                            offset + stage.stride * (stage.radix * group + output_index);
                        let twiddle_index = group * stage.radix + output_index;
                        destination[destination_index] =
                            value.multiply(stage.twiddles[twiddle_index]);
                    }
                }
            }
            std::mem::swap(&mut source, &mut destination);
        }
        if !self.stages.len().is_multiple_of(2) {
            destination.copy_from_slice(source);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn forward(&self, input: &[Complex]) -> Result<Vec<Complex>, String> {
        if input.len() != self.length {
            return Err(format!(
                "FFT input must have length {}, got {}",
                self.length,
                input.len()
            ));
        }
        let mut output = input.to_vec();
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(self.length)
            .map_err(|_| "FFT scratch cannot reserve memory".to_owned())?;
        scratch.resize(self.length, Complex::default());
        self.forward_inplace(&mut output, &mut scratch)?;
        Ok(output)
    }

    pub(crate) fn forward_batched(
        &self,
        real: &mut [f32],
        imaginary: &mut [f32],
        scratch_real: &mut [f32],
        scratch_imaginary: &mut [f32],
        frames: usize,
    ) -> Result<(), String> {
        let expected = checked_product(self.length, frames, "batched FFT dimensions")?;
        if real.len() != expected
            || imaginary.len() != expected
            || scratch_real.len() != expected
            || scratch_imaginary.len() != expected
        {
            return Err("batched FFT buffers do not match the validated dimensions".into());
        }
        let (mut source_real, mut source_imaginary): (&mut [f32], &mut [f32]) = (real, imaginary);
        let (mut destination_real, mut destination_imaginary): (&mut [f32], &mut [f32]) =
            (scratch_real, scratch_imaginary);
        for stage in &self.stages {
            // `expected` and the validated plan prove all scalar ranges below are within
            // the four caller-provided buffers. Do not put checked-arithmetic errors in
            // the batched butterfly's per-frame execution path.
            for group in 0..stage.groups {
                for offset in 0..stage.stride {
                    for output_index in 0..stage.radix {
                        let output_complex =
                            offset + stage.stride * (stage.radix * group + output_index);
                        let output_start = output_complex * frames;
                        let output_end = output_start + frames;
                        let input_complex = offset + stage.stride * group;
                        let input_start = input_complex * frames;
                        let input_end = input_start + frames;
                        destination_real[output_start..output_end]
                            .copy_from_slice(&source_real[input_start..input_end]);
                        destination_imaginary[output_start..output_end]
                            .copy_from_slice(&source_imaginary[input_start..input_end]);
                        for radix_index in 1..stage.radix {
                            let root_index = radix_index * stage.radix + output_index;
                            let root = stage.roots[root_index];
                            let input_group = group + radix_index * stage.groups;
                            let input_complex = offset + stage.stride * input_group;
                            let input_start = input_complex * frames;
                            for frame in 0..frames {
                                let input_index = input_start + frame;
                                let output_index = output_start + frame;
                                let source_re = source_real[input_index];
                                let source_im = source_imaginary[input_index];
                                destination_real[output_index] +=
                                    source_re * root.re - source_im * root.im;
                                destination_imaginary[output_index] +=
                                    source_re * root.im + source_im * root.re;
                            }
                        }
                        let twiddle_index = group * stage.radix + output_index;
                        let twiddle = stage.twiddles[twiddle_index];
                        for frame in 0..frames {
                            let index = output_start + frame;
                            let value_real = destination_real[index];
                            let value_imaginary = destination_imaginary[index];
                            destination_real[index] =
                                value_real * twiddle.re - value_imaginary * twiddle.im;
                            destination_imaginary[index] =
                                value_real * twiddle.im + value_imaginary * twiddle.re;
                        }
                    }
                }
            }
            std::mem::swap(&mut source_real, &mut destination_real);
            std::mem::swap(&mut source_imaginary, &mut destination_imaginary);
        }
        if !self.stages.len().is_multiple_of(2) {
            destination_real.copy_from_slice(source_real);
            destination_imaginary.copy_from_slice(source_imaginary);
        }
        Ok(())
    }
}

/// Real-input FFT through a packed complex FFT of length N/2.
pub(crate) struct RealFft {
    length: usize,
    half: Fft,
    recombination: Vec<Complex>,
}

impl RealFft {
    pub(crate) fn new(length: usize) -> Result<Self, String> {
        if length == 0 || !length.is_multiple_of(2) {
            return Err("real FFT length must be a nonzero even number".into());
        }
        let half_length = length / 2;
        let half = Fft::new(half_length)?;
        let mut recombination = Vec::new();
        recombination
            .try_reserve_exact(
                half_length
                    .checked_add(1)
                    .ok_or_else(|| "real FFT recombination length overflows usize".to_owned())?,
            )
            .map_err(|_| "real FFT recombination cannot reserve memory".to_owned())?;
        for index in 0..=half_length {
            recombination.push(twiddle(index, length));
        }
        Ok(Self {
            length,
            half,
            recombination,
        })
    }

    pub(crate) fn power_spectrum(
        &self,
        frame: &[f32],
        output: &mut [f32],
        buffer: &mut [Complex],
        scratch: &mut [Complex],
    ) -> Result<(), String> {
        let half_length = self.length / 2;
        if frame.len() != self.length
            || output.len() != half_length + 1
            || buffer.len() != half_length
            || scratch.len() != half_length
        {
            return Err("real FFT buffers do not match the validated dimensions".into());
        }
        if frame.iter().any(|sample| !sample.is_finite()) {
            return Err("real FFT input samples must be finite".into());
        }
        for (index, packed) in buffer.iter_mut().enumerate() {
            let left = index * 2;
            let right = left + 1;
            *packed = Complex {
                re: frame[left],
                im: frame[right],
            };
        }
        self.half.forward_inplace(buffer, scratch)?;
        for index in 0..=half_length {
            let first = buffer[index % half_length];
            let second = buffer[(half_length - index) % half_length].conjugate();
            let even = Complex {
                re: (first.re + second.re) * 0.5,
                im: (first.im + second.im) * 0.5,
            };
            let difference = Complex {
                re: (first.re - second.re) * 0.5,
                im: (first.im - second.im) * 0.5,
            };
            let odd = Complex {
                re: difference.im,
                im: -difference.re,
            };
            let value = even.add(self.recombination[index].multiply(odd));
            output[index] = value.re * value.re + value.im * value.im;
        }
        Ok(())
    }

    pub(crate) fn power_spectrum_batched(
        &self,
        real: &mut [f32],
        imaginary: &mut [f32],
        scratch_real: &mut [f32],
        scratch_imaginary: &mut [f32],
        output: &mut [f32],
        frames: usize,
    ) -> Result<(), String> {
        let half_length = self.length / 2;
        let half_values = checked_product(half_length, frames, "batched real FFT dimensions")?;
        let output_values = checked_product(
            half_length
                .checked_add(1)
                .ok_or_else(|| "batched real FFT output dimension overflows usize".to_owned())?,
            frames,
            "batched real FFT output dimensions",
        )?;
        if real.len() != half_values
            || imaginary.len() != half_values
            || scratch_real.len() != half_values
            || scratch_imaginary.len() != half_values
            || output.len() != output_values
        {
            return Err("batched real FFT buffers do not match the validated dimensions".into());
        }
        self.half
            .forward_batched(real, imaginary, scratch_real, scratch_imaginary, frames)?;
        for index in 0..=half_length {
            let first_offset = (index % half_length) * frames;
            let second_offset = ((half_length - index) % half_length) * frames;
            let output_offset = index * frames;
            let recombination = self.recombination[index];
            for frame_index in 0..frames {
                let first_index = first_offset + frame_index;
                let second_index = second_offset + frame_index;
                let output_index = output_offset + frame_index;
                let first_real = real[first_index];
                let first_imaginary = imaginary[first_index];
                let second_real = real[second_index];
                let second_imaginary = -imaginary[second_index];
                let even_real = (first_real + second_real) * 0.5;
                let even_imaginary = (first_imaginary + second_imaginary) * 0.5;
                let difference_real = (first_real - second_real) * 0.5;
                let difference_imaginary = (first_imaginary - second_imaginary) * 0.5;
                let odd_real = difference_imaginary;
                let odd_imaginary = -difference_real;
                let value_real =
                    even_real + (recombination.re * odd_real - recombination.im * odd_imaginary);
                let value_imaginary = even_imaginary
                    + (recombination.re * odd_imaginary + recombination.im * odd_real);
                output[output_index] = value_real * value_real + value_imaginary * value_imaginary;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Complex, Fft, RealFft};
    use gigaam_primitives::{f64_to_f32, usize_to_f32, usize_to_f64};

    fn naive_power(input: &[f32]) -> Vec<f32> {
        let length = input.len();
        (0..=length / 2)
            .map(|frequency| {
                let (mut real, mut imaginary) = (0.0_f64, 0.0_f64);
                for (sample_index, sample) in input.iter().enumerate() {
                    let angle =
                        -2.0 * std::f64::consts::PI * usize_to_f64(sample_index * frequency)
                            / usize_to_f64(length);
                    real += f64::from(*sample) * angle.cos();
                    imaginary += f64::from(*sample) * angle.sin();
                }
                f64_to_f32(real * real + imaginary * imaginary)
            })
            .collect()
    }

    #[test]
    fn fft_matches_naive_dft() -> Result<(), String> {
        for length in [2_usize, 3, 5, 8, 12, 30, 160, 320, 512] {
            let input: Vec<Complex> = (0..length)
                .map(|index| Complex {
                    re: usize_to_f32((index * 7919) % 13) - 6.0,
                    im: usize_to_f32((index * 31) % 5),
                })
                .collect();
            let output = Fft::new(length)?.forward(&input)?;
            for (frequency, actual) in output.iter().enumerate() {
                let (mut real, mut imaginary) = (0.0_f64, 0.0_f64);
                for (sample_index, sample) in input.iter().enumerate() {
                    let angle =
                        -2.0 * std::f64::consts::PI * usize_to_f64(sample_index * frequency)
                            / usize_to_f64(length);
                    real += f64::from(sample.re) * angle.cos() - f64::from(sample.im) * angle.sin();
                    imaginary +=
                        f64::from(sample.re) * angle.sin() + f64::from(sample.im) * angle.cos();
                }
                assert!(
                    (f64::from(actual.re) - real).abs() < 1e-3
                        && (f64::from(actual.im) - imaginary).abs() < 1e-3,
                    "length={length} frequency={frequency}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn real_fft_matches_naive_power() -> Result<(), String> {
        for length in [8_usize, 12, 160, 320] {
            let frame: Vec<f32> = (0..length)
                .map(|index| {
                    (usize_to_f32(index) * 0.3).sin() + usize_to_f32((index * 7) % 5) * 0.1
                })
                .collect();
            let fft = RealFft::new(length)?;
            let mut output = vec![0.0_f32; length / 2 + 1];
            let mut buffer = vec![Complex::default(); length / 2];
            let mut scratch = vec![Complex::default(); length / 2];
            fft.power_spectrum(&frame, &mut output, &mut buffer, &mut scratch)?;
            let expected = naive_power(&frame);
            for (index, (actual, expected)) in output.iter().zip(expected).enumerate() {
                assert!(
                    (*actual - expected).abs() < 1e-2 * (1.0 + expected.abs()),
                    "length={length} frequency={index}: {actual} vs {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn unsupported_or_invalid_plans_refuse() {
        assert!(Fft::new(0).is_err());
        assert!(Fft::new(17).is_err());
        assert!(RealFft::new(0).is_err());
        assert!(RealFft::new(15).is_err());
    }
}
