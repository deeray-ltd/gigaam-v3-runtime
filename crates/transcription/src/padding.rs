// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Deterministic low-amplitude waveform padding.

use gigaam_primitives::{f64_to_f32, u64_to_f64};

const MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const INCREMENT: u64 = 1_442_695_040_888_963_407;
const UNIT_DENOMINATOR: u64 = 1_u64 << 53;
const AMPLITUDE: f64 = 1e-3;

/// Produces deterministic low-amplitude noise for a fixed requested length and seed.
pub(crate) fn pad_noise(length: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
    (0..length)
        .map(|_| {
            state = state.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
            let unit = u64_to_f64(state >> 11) / u64_to_f64(UNIT_DENOMINATOR);
            f64_to_f32((unit * 2.0 - 1.0) * AMPLITUDE)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{AMPLITUDE, pad_noise};

    #[test]
    fn padding_noise_is_deterministic_and_bounded() {
        let first = pad_noise(128, 0x5eed);
        let repeated = pad_noise(128, 0x5eed);
        let distinct_seed = pad_noise(128, 0x5eee);

        assert_eq!(first, repeated);
        assert_ne!(first, distinct_seed);
        assert_eq!(first.len(), 128);
        assert!(
            first
                .iter()
                .all(|sample| sample.is_finite() && f64::from(sample.abs()) <= AMPLITUDE)
        );
    }

    #[test]
    fn padding_noise_respects_requested_empty_length() {
        assert!(pad_noise(0, 0x5eed).is_empty());
    }
}
