// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Shared model-backed helpers for opt-in transcription acceptance tests.

use gigaam_audio::{FrontendMode, FrontendProcessor};
use gigaam_model_package::ModelPackage;
use gigaam_recognition::{Device, OrtConfig, ProviderPlan, init_runtime};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

#[path = "../../../../tests/support/mod.rs"]
mod external_artifacts;

pub use external_artifacts::external_artifact_root as root;

pub fn init_ort() {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    let result = INIT.get_or_init(|| {
        let value = std::env::var("ORT_DYLIB_PATH").map_err(|error| {
            format!("ORT_DYLIB_PATH must name the ONNX Runtime library: {error}")
        })?;
        if value.is_empty() {
            return Err("ORT_DYLIB_PATH must not be empty".into());
        }
        init_runtime(&PathBuf::from(value))
    });
    if let Err(error) = result {
        panic!("initialize ORT: {error}");
    }
}

pub fn frontend(pack: &ModelPackage) -> Arc<FrontendProcessor> {
    Arc::new(
        FrontendProcessor::new(
            pack.frontend(),
            pack.frontend_weights()
                .expect("test model pack must expose frontend weights"),
            FrontendMode::Scalar,
        )
        .expect("test model pack frontend must be supported"),
    )
}

pub fn cpu_plan() -> ProviderPlan {
    ProviderPlan::new(Device::Cpu, OrtConfig::default())
        .expect("the CPU provider plan must be valid")
}
