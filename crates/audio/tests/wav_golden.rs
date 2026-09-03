// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Golden WAV decoding properties at the Audio ownership boundary.

mod common;

use gigaam_audio::read_wav;

#[test]
fn wav_decoding_matches_the_golden_waveform() {
    for clip in ["example", "long_example"] {
        let decoded = read_wav(&common::root().join(format!("fixtures/{clip}.wav")))
            .expect("golden WAV fixture must decode");
        assert_eq!(decoded.channels().len(), 1, "{clip}: golden WAV is mono");
        let (dimensions, expected) =
            common::read_f32(&common::root().join(format!("fixtures/{clip}.wav.f32")))
                .expect("golden waveform fixture must be readable");
        let count = dimensions
            .iter()
            .try_fold(1_usize, |total, dimension| {
                total
                    .checked_mul(*dimension)
                    .ok_or("golden fixture dimensions must fit usize")
            })
            .expect("golden fixture dimensions must fit usize");
        assert_eq!(
            count,
            expected.len(),
            "{clip}: fixture dimensions describe its samples"
        );
        assert_eq!(
            decoded.channels()[0].samples().len(),
            expected.len(),
            "{clip}: sample count"
        );
        let maximum_delta = decoded.channels()[0]
            .samples()
            .iter()
            .zip(&expected)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            maximum_delta < 1e-6,
            "{clip}: WAV decoding diverged from golden samples, max|Δ|={maximum_delta}"
        );
    }
}
