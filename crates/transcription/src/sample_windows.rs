// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Lazy batch windows on the source sample clock.

/// The exact rate-scaled, rounded f32 duration, clamped to [1, width].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SampleStride {
    whole: u64,
    remainder: u64,
    denominator: u64,
}

impl SampleStride {
    pub(crate) fn new(seconds: f32, rate: u32, width: u64) -> Self {
        let bits = seconds.to_bits();
        let exponent = (bits >> 23) & 0xff;
        let fraction = bits & 0x7f_ffff;
        let significand = if exponent == 0 {
            fraction
        } else {
            fraction | (1 << 23)
        };
        let product = u64::from(significand) * u64::from(rate);
        let shift = if exponent == 0 {
            -149
        } else {
            i32::try_from(exponent).expect("f32 exponent fits i32") - 150
        };
        let (numerator, denominator) = if shift >= 0 {
            let shift = u32::try_from(shift).expect("nonnegative shift fits u32");
            if shift >= 64 || product > (width >> shift) {
                return Self {
                    whole: width,
                    remainder: 0,
                    denominator: 1,
                };
            }
            (product << shift, 1)
        } else {
            let shift = shift.unsigned_abs();
            if shift >= 64 || product < (1_u64 << shift) {
                return Self {
                    whole: 1,
                    remainder: 0,
                    denominator: 1,
                };
            }
            (product, 1_u64 << shift)
        };
        if u128::from(numerator) >= u128::from(width) * u128::from(denominator) {
            return Self {
                whole: width,
                remainder: 0,
                denominator: 1,
            };
        }
        Self {
            whole: numerator / denominator,
            remainder: numerator % denominator,
            denominator,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SampleWindow {
    pub start: u64,
    pub end: u64,
    pub tail: bool,
}

pub(crate) struct SampleWindows {
    width: u64,
    length: u64,
    tail: u64,
    next: Option<u64>,
    carry: u64,
    stride: SampleStride,
    count: u64,
}

impl SampleWindows {
    pub(crate) fn new(length: u64, width: u64, stride: SampleStride) -> Result<Self, String> {
        let tail = length.saturating_sub(width);
        let numerator = u128::from(stride.whole) * u128::from(stride.denominator)
            + u128::from(stride.remainder);
        let scaled_tail = u128::from(tail) * u128::from(stride.denominator);
        let count = 1_u128 + scaled_tail.div_ceil(numerator);
        Ok(Self {
            width,
            length,
            tail,
            next: Some(0),
            carry: 0,
            stride,
            count: u64::try_from(count)
                .map_err(|_| "batch window count overflows u64".to_owned())?,
        })
    }

    pub(crate) const fn planned_count(&self) -> u64 {
        self.count
    }
}

impl Iterator for SampleWindows {
    type Item = SampleWindow;
    fn next(&mut self) -> Option<Self::Item> {
        let start = self.next?;
        if start == self.tail {
            self.next = None;
            return Some(SampleWindow {
                start,
                end: self.length,
                tail: true,
            });
        }
        let sum = self.carry + self.stride.remainder;
        let increment = self.stride.whole + sum / self.stride.denominator;
        self.carry = sum % self.stride.denominator;
        self.next = Some(if increment >= self.tail - start {
            self.tail
        } else {
            start + increment
        });
        Some(SampleWindow {
            start,
            end: start + self.width,
            tail: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{SampleStride, SampleWindows};

    fn starts(length: u64, width: u64, seconds: f32, rate: u32) -> Vec<(u64, u64, bool)> {
        SampleWindows::new(length, width, SampleStride::new(seconds, rate, width))
            .expect("the test schedule count fits")
            .map(|window| (window.start, window.end, window.tail))
            .collect()
    }

    #[test]
    fn fractional_stride_and_unique_tail_are_exact() {
        assert_eq!(
            starts(204, 64, 46.5 / 64.0, 64),
            vec![
                (0, 64, false),
                (46, 110, false),
                (93, 157, false),
                (139, 203, false),
                (140, 204, true)
            ]
        );
        assert_eq!(
            starts(157, 64, 46.5 / 64.0, 64),
            vec![(0, 64, false), (46, 110, false), (93, 157, true)]
        );
        assert_eq!(starts(64, 64, 0.5, 64), vec![(0, 64, true)]);
        assert_eq!(starts(0, 64, 0.5, 64), vec![(0, 0, true)]);
    }

    #[test]
    fn stride_clamping_and_exact_integer_rate_preserve_coverage() {
        assert_eq!(
            starts(5, 2, 0.001, 64),
            vec![(0, 2, false), (1, 3, false), (2, 4, false), (3, 5, true)]
        );
        assert_eq!(
            starts(6, 2, 1.0, 64),
            vec![(0, 2, false), (2, 4, false), (4, 6, true)]
        );
        assert_eq!(
            starts(40_000_000, 20_000_000, 1.0, 16_777_217),
            vec![
                (0, 20_000_000, false),
                (16_777_217, 36_777_217, false),
                (20_000_000, 40_000_000, true)
            ]
        );
        let stride = SampleStride::new(1.0, 16_777_217, 20_000_000);
        assert_eq!(stride.whole, 16_777_217);
    }

    #[test]
    fn eof_tail_uses_exact_sample_before_frame_projection() {
        let result = starts(480_481, 480_000, 24.0, 16_000);
        assert_eq!(result, vec![(0, 480_000, false), (481, 480_481, true)]);
        assert_eq!(result[1].0.div_ceil(160), 4);
    }

    #[test]
    fn count_matches_independent_fractional_oracle() {
        for length in 0_u64..200 {
            let stride = SampleStride::new(46.5 / 64.0, 64, 64);
            let windows = SampleWindows::new(length, 64, stride).expect("small schedule fits");
            let count = windows.planned_count();
            let actual: Vec<_> = windows.collect();
            let tail = length.saturating_sub(64);
            let mut expected = Vec::new();
            let mut ordinal = 0_u64;
            while ordinal * 93 / 2 < tail {
                expected.push(ordinal * 93 / 2);
                ordinal += 1;
            }
            expected.push(tail);
            assert_eq!(
                actual.iter().map(|window| window.start).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                usize::try_from(count).expect("small count fits"),
                expected.len()
            );
        }
    }

    #[test]
    fn eof_transition_near_u64_max_emits_unique_tail() {
        assert_eq!(
            starts(
                u64::MAX,
                9_223_372_036_854_775_808,
                4_611_686_018_427_387_904.0,
                1,
            ),
            vec![
                (0, 9_223_372_036_854_775_808, false),
                (4_611_686_018_427_387_904, 13_835_058_055_282_163_712, false),
                (9_223_372_036_854_775_807, 18_446_744_073_709_551_615, true),
            ]
        );
    }

    #[test]
    fn logical_u64_boundary_count_and_prefix_do_not_overflow() {
        let mut windows = SampleWindows::new(u64::MAX, 1, SampleStride::new(1.0, 1, 1))
            .expect("maximum schedule fits u64");
        assert_eq!(windows.planned_count(), u64::MAX);
        assert_eq!(windows.next().expect("first window").start, 0);
        assert_eq!(windows.next().expect("second window").start, 1);
        let stride = SampleStride::new(1.5, 1, 2);
        let windows =
            SampleWindows::new(u64::MAX, 2, stride).expect("fractional maximum schedule fits");
        assert_eq!(
            windows.planned_count(),
            1 + u64::try_from((u128::from(u64::MAX - 2) * 2).div_ceil(3)).expect("count fits")
        );
    }
}
