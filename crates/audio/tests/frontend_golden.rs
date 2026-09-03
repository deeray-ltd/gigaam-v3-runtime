// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Full model-package frontend acceptance at the Audio ownership boundary.

mod common;

use gigaam_audio::{FrontendMode, FrontendProcessor};
use gigaam_model_package::ModelPackage;
use gigaam_primitives::usize_to_f64;

#[test]
fn log_mel_matches_golden_for_the_model_package() {
    let pack = ModelPackage::open(&common::root().join("model"))
        .expect("golden frontend test requires a valid model pack");
    let frontend = FrontendProcessor::new(
        pack.frontend(),
        pack.frontend_weights()
            .expect("golden model pack must expose frontend weights"),
        FrontendMode::Scalar,
    )
    .expect("golden model pack must construct the frontend");
    for clip in ["example", "long_example"] {
        let (_, waveform) =
            common::read_f32(&common::root().join(format!("fixtures/{clip}.wav.f32")))
                .expect("golden waveform fixture must be readable");
        let (dimensions, expected) =
            common::read_f32(&common::root().join(format!("fixtures/{clip}.mel.f32")))
                .expect("golden log-mel fixture must be readable");
        let actual = frontend
            .log_mel(&waveform)
            .expect("golden waveform must produce log-mel features");
        assert_eq!(
            dimensions,
            vec![frontend.mel_bins(), actual.frames()],
            "{clip}: frame count"
        );
        let (mut maximum_delta, mut sum_delta) = (0.0_f32, 0.0_f64);
        for (left, right) in actual.values().iter().zip(&expected) {
            let delta = (left - right).abs();
            maximum_delta = maximum_delta.max(delta);
            sum_delta += f64::from(delta);
        }
        let mean_delta = sum_delta / usize_to_f64(expected.len());
        eprintln!("{clip}: mel max|Δ|={maximum_delta:.2e} mean|Δ|={mean_delta:.2e}");
        assert!(
            maximum_delta < 5e-3 && mean_delta < 1e-4,
            "{clip}: frontend diverged from golden output"
        );
    }
}
