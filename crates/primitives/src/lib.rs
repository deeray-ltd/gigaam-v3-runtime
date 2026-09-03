// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Checked and bit-preserving numeric conversions used at audio and public-input
//! boundaries. Keeping them here makes every lossy conversion explicit without
//! duplicating unchecked casts through DSP and service code.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversionError {
    target: &'static str,
    value: String,
}

impl ConversionError {
    fn new(target: &'static str, value: impl ToString) -> Self {
        Self {
            target,
            value: value.to_string(),
        }
    }
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} cannot be represented as {}", self.value, self.target)
    }
}

impl std::error::Error for ConversionError {}

fn low_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn usize_to_u64(value: usize) -> u64 {
    match u64::try_from(value) {
        Ok(value) => value,
        Err(_) => panic!("supported usize values must fit the runtime's u64 conversion boundary"),
    }
}

fn round_shift_right(value: u64, shift: u32) -> u64 {
    if shift == 0 {
        return value;
    }
    if shift >= 64 {
        return 0;
    }
    let truncated = value >> shift;
    let remainder = value & ((1_u64 << shift) - 1);
    let halfway = 1_u64 << (shift - 1);
    if remainder > halfway || (remainder == halfway && truncated & 1 == 1) {
        truncated + 1
    } else {
        truncated
    }
}

/// Converts an unsigned counter to `f64` with the same IEEE-754 rounding as a
/// direct conversion, without relying on a language-level numeric cast.
pub fn u64_to_f64(value: u64) -> f64 {
    if value == 0 {
        return 0.0;
    }

    let mut exponent = 63_u32 - value.leading_zeros();
    let mut significand = if exponent > 52 {
        round_shift_right(value, exponent - 52)
    } else {
        value << (52 - exponent)
    };
    if significand == (1_u64 << 53) {
        significand >>= 1;
        exponent += 1;
    }
    let exponent_bits = u64::from(exponent + 1023);
    f64::from_bits((exponent_bits << 52) | (significand & 0x000f_ffff_ffff_ffff))
}

/// Converts an unsigned counter to `f32` with direct IEEE-754 rounding. This
/// deliberately avoids an intermediate `f64`, whose prior rounding can differ
/// for large platform indices.
fn u64_to_f32(value: u64) -> f32 {
    if value == 0 {
        return 0.0;
    }

    let mut exponent = 63_u32 - value.leading_zeros();
    let mut significand = if exponent > 23 {
        round_shift_right(value, exponent - 23)
    } else {
        value << (23 - exponent)
    };
    if significand == (1_u64 << 24) {
        significand >>= 1;
        exponent += 1;
    }
    f32::from_bits(((exponent + 127) << 23) | low_u32(significand & 0x007f_ffff))
}

/// Converts a platform index to `f64` for duration and DSP calculations.
pub fn usize_to_f64(value: usize) -> f64 {
    u64_to_f64(usize_to_u64(value))
}

/// Converts a platform index to `f32` through the IEEE-754 narrowing routine.
pub fn usize_to_f32(value: usize) -> f32 {
    u64_to_f32(usize_to_u64(value))
}

/// Converts a `u32` to `f32` through the same explicit narrowing boundary.
pub fn u32_to_f32(value: u32) -> f32 {
    u64_to_f32(u64::from(value))
}

/// Converts an `i32` PCM sample to `f32` while preserving its sign and nearest
/// IEEE-754 representable magnitude.
pub fn i32_to_f32(value: i32) -> f32 {
    let magnitude = u64::from(value.unsigned_abs());
    let converted = u64_to_f32(magnitude);
    if value.is_negative() {
        -converted
    } else {
        converted
    }
}

/// Narrows `f64` to `f32` by reconstructing the IEEE-754 result bits with
/// round-to-nearest, ties-to-even. This keeps resampling and FFT output on a
/// numeric path rather than allocating and parsing decimal text per sample.
pub fn f64_to_f32(value: f64) -> f32 {
    let bits = value.to_bits();
    let sign = low_u32((bits >> 32) & 0x8000_0000);
    let exponent = (bits >> 52) & 0x07ff;
    let fraction = bits & 0x000f_ffff_ffff_ffff;

    if exponent == 0x07ff {
        if fraction == 0 {
            return f32::from_bits(sign | 0x7f80_0000);
        }
        let payload = low_u32(fraction >> 29) & 0x007f_ffff;
        return f32::from_bits(sign | 0x7f80_0000 | 0x0040_0000 | payload.max(1));
    }
    if exponent == 0 {
        return f32::from_bits(sign);
    }

    let exponent_i32 = match i32::try_from(exponent) {
        Ok(exponent) => exponent,
        Err(_) => panic!("f64 exponent field always fits i32"),
    };
    let unbiased = exponent_i32 - 1023;
    if unbiased > 127 {
        return f32::from_bits(sign | 0x7f80_0000);
    }

    let significand = fraction | (1_u64 << 52);
    if unbiased >= -126 {
        let mut rounded = round_shift_right(significand, 29);
        let mut exponent32 = match u32::try_from(unbiased + 127) {
            Ok(exponent) => exponent,
            Err(_) => panic!("normal f32 exponent is non-negative"),
        };
        if rounded == (1_u64 << 24) {
            rounded >>= 1;
            exponent32 += 1;
            if exponent32 >= 0xff {
                return f32::from_bits(sign | 0x7f80_0000);
            }
        }
        return f32::from_bits(sign | (exponent32 << 23) | low_u32(rounded & 0x007f_ffff));
    }

    let shift = match u32::try_from(-unbiased - 97) {
        Ok(shift) => shift,
        Err(_) => panic!("subnormal f32 shift is non-negative"),
    };
    let rounded = round_shift_right(significand, shift);
    if rounded >= (1_u64 << 23) {
        f32::from_bits(sign | (1_u32 << 23))
    } else {
        f32::from_bits(sign | low_u32(rounded))
    }
}

fn trunc_nonnegative_f64_to_u64(value: f64) -> Option<u64> {
    if !value.is_finite() || (value.is_sign_negative() && value != 0.0) {
        return None;
    }

    let bits = value.to_bits();
    let exponent = (bits >> 52) & 0x07ff;
    if exponent == 0 {
        return Some(0);
    }
    let exponent_i32 = i32::try_from(exponent).ok()? - 1023;
    if exponent_i32 < 0 {
        return Some(0);
    }

    let significand = (bits & 0x000f_ffff_ffff_ffff) | (1_u64 << 52);
    if exponent_i32 >= 52 {
        let shift = u32::try_from(exponent_i32 - 52).ok()?;
        if shift >= 64 || significand > (u64::MAX >> shift) {
            return None;
        }
        Some(significand << shift)
    } else {
        let shift = u32::try_from(52 - exponent_i32).ok()?;
        Some(significand >> shift)
    }
}

/// Strictly truncates a finite non-negative `f32` to a platform index. Invalid
/// public configuration is returned to the caller instead of saturating silently.
pub fn trunc_f32_to_usize(value: f32) -> Result<usize, ConversionError> {
    if !value.is_finite() || value < 0.0 {
        return Err(ConversionError::new("usize", value));
    }
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 0xff;
    if exponent == 0 {
        return Ok(0);
    }
    let exponent_i32 = match i32::try_from(exponent) {
        Ok(exponent) => exponent,
        Err(_) => return Err(ConversionError::new("usize", value)),
    } - 127;
    if exponent_i32 < 0 {
        return Ok(0);
    }
    let significand = u64::from((bits & 0x007f_ffff) | (1_u32 << 23));
    let integer = if exponent_i32 >= 23 {
        let shift = match u32::try_from(exponent_i32 - 23) {
            Ok(shift) => shift,
            Err(_) => return Err(ConversionError::new("usize", value)),
        };
        if shift >= 64 || significand > (u64::MAX >> shift) {
            return Err(ConversionError::new("usize", value));
        }
        significand << shift
    } else {
        let shift = match u32::try_from(23 - exponent_i32) {
            Ok(shift) => shift,
            Err(_) => return Err(ConversionError::new("usize", value)),
        };
        significand >> shift
    };
    usize::try_from(integer).map_err(|_| ConversionError::new("usize", value))
}

/// Strictly truncates a finite `f32` to `isize`, rejecting values outside the
/// platform range rather than wrapping or saturating.
pub fn trunc_f32_to_isize(value: f32) -> Result<isize, ConversionError> {
    if !value.is_finite() {
        return Err(ConversionError::new("isize", value));
    }
    if value < 0.0 {
        let magnitude = trunc_f32_to_usize(-value)?;
        if magnitude == isize::MIN.unsigned_abs() {
            return Ok(isize::MIN);
        }
        let magnitude =
            isize::try_from(magnitude).map_err(|_| ConversionError::new("isize", value))?;
        Ok(-magnitude)
    } else {
        let magnitude = trunc_f32_to_usize(value)?;
        isize::try_from(magnitude).map_err(|_| ConversionError::new("isize", value))
    }
}

/// Converts a finite integral `f64` to `i64` for JSON's integer presentation.
pub fn integral_f64_to_i64(value: f64) -> Result<i64, ConversionError> {
    if !value.is_finite() || value != value.trunc() {
        return Err(ConversionError::new("i64", value));
    }
    if value.is_sign_negative() {
        let magnitude = trunc_nonnegative_f64_to_u64(-value)
            .ok_or_else(|| ConversionError::new("i64", value))?;
        if magnitude == (1_u64 << 63) {
            return Ok(i64::MIN);
        }
        let magnitude = i64::try_from(magnitude).map_err(|_| ConversionError::new("i64", value))?;
        Ok(-magnitude)
    } else {
        let magnitude = trunc_nonnegative_f64_to_u64(value)
            .ok_or_else(|| ConversionError::new("i64", value))?;
        i64::try_from(magnitude).map_err(|_| ConversionError::new("i64", value))
    }
}

pub fn usize_to_isize(value: usize) -> Result<isize, ConversionError> {
    isize::try_from(value).map_err(|_| ConversionError::new("isize", value))
}

pub fn isize_to_usize(value: isize) -> Result<usize, ConversionError> {
    usize::try_from(value).map_err(|_| ConversionError::new("usize", value))
}

pub fn u32_to_usize(value: u32) -> Result<usize, ConversionError> {
    usize::try_from(value).map_err(|_| ConversionError::new("usize", value))
}

pub fn usize_to_u64_checked(value: usize) -> Result<u64, ConversionError> {
    u64::try_from(value).map_err(|_| ConversionError::new("u64", value))
}

pub fn usize_to_i64(value: usize) -> Result<i64, ConversionError> {
    i64::try_from(value).map_err(|_| ConversionError::new("i64", value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrowing_keeps_representable_values_and_signs() {
        assert_eq!(f64_to_f32(1.5).to_bits(), 1.5_f32.to_bits());
        assert_eq!(f64_to_f32(-0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(f64_to_f32(f64::INFINITY), f32::INFINITY);
        assert_eq!(f64_to_f32(f64::NEG_INFINITY), f32::NEG_INFINITY);
        assert_eq!(f64_to_f32(16_777_217.0), 16_777_216.0_f32);

        let smallest_subnormal = f64::from(f32::from_bits(1));
        assert_eq!(f64_to_f32(smallest_subnormal / 2.0).to_bits(), 0);
        assert_eq!(
            f64_to_f32(smallest_subnormal * 1.5).to_bits(),
            f32::from_bits(2).to_bits()
        );
    }

    #[test]
    fn index_truncation_refuses_invalid_values() {
        assert_eq!(trunc_f32_to_usize(123.9), Ok(123));
        assert_eq!(trunc_f32_to_isize(-123.9), Ok(-123));
        assert_eq!(trunc_f32_to_usize(-0.0), Ok(0));
        assert_eq!(integral_f64_to_i64(-0.0), Ok(0));
        assert!(trunc_f32_to_usize(-0.5).is_err());
        assert!(trunc_f32_to_usize(f32::INFINITY).is_err());
    }

    #[test]
    fn integer_conversions_are_exact_at_small_boundaries() {
        assert_eq!(usize_to_f32(16_000), 16_000.0);
        assert_eq!(usize_to_f64(4_294_967_296), 4_294_967_296.0);
        assert_eq!(i32_to_f32(-32_124), -32_124.0);
    }

    #[test]
    fn direct_unsigned_to_f32_rounding_avoids_an_intermediate_f64_round() {
        let lower = (1_u64 << 63) + (1_u64 << 40);
        let immediately_below_halfway = lower + (1_u64 << 39) - 1;
        let expected = f32::from_bits((190_u32 << 23) | 1);

        assert_eq!(
            u64_to_f32(immediately_below_halfway).to_bits(),
            expected.to_bits()
        );
        match usize::try_from(immediately_below_halfway) {
            Ok(platform_index) => {
                assert_eq!(usize_to_f32(platform_index).to_bits(), expected.to_bits());
            }
            Err(_) => assert_eq!(usize::BITS, 32),
        }
    }

    #[test]
    fn bit_decoded_integer_limits_preserve_signed_zero_and_minimum() {
        let isize_minimum = if isize::BITS == 64 {
            f32::from_bits(0xdf00_0000)
        } else {
            f32::from_bits(0xcf00_0000)
        };
        assert_eq!(trunc_f32_to_isize(isize_minimum), Ok(isize::MIN));
        assert_eq!(
            integral_f64_to_i64(-9_223_372_036_854_775_808.0),
            Ok(i64::MIN)
        );

        let largest_u64_f64 = f64::from_bits(0x43ef_ffff_ffff_ffff);
        assert_eq!(
            trunc_nonnegative_f64_to_u64(largest_u64_f64),
            Some(u64::MAX - 2_047)
        );
        assert_eq!(
            trunc_nonnegative_f64_to_u64(f64::from_bits(0x43f0_0000_0000_0000)),
            None
        );
        assert_eq!(u64_to_f64(u64::MAX).to_bits(), 0x43f0_0000_0000_0000);
    }
}
