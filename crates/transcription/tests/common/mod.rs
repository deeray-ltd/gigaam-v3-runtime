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
use gigaam_recognition::{Device, NativeRuntime, OrtConfig, ProviderPlan, init_runtime};
use std::path::PathBuf;
use std::sync::Arc;

#[path = "../../../../tests/support/mod.rs"]
mod external_artifacts;

pub use external_artifacts::external_artifact_root as root;

/// Initializes the one scoped environment owned by a model-backed test suite.
pub fn init_ort() -> NativeRuntime {
    let value =
        std::env::var("ORT_DYLIB_PATH").expect("ORT_DYLIB_PATH must name the ONNX Runtime library");
    assert!(!value.is_empty(), "ORT_DYLIB_PATH must not be empty");
    init_runtime(&PathBuf::from(value)).expect("the ORT environment must initialize")
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
