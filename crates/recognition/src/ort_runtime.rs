// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Explicit ONNX Runtime dynamic-library initialization.

use std::path::Path;

/// Initializes ONNX Runtime from one explicit regular dynamic-library file before any session is
/// created.
pub fn init_runtime(dylib: &Path) -> Result<(), String> {
    let metadata = std::fs::metadata(dylib)
        .map_err(|error| format!("ORT dylib {}: {error}", dylib.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "ORT dylib {} is not a regular file",
            dylib.display()
        ));
    }
    let builder =
        ort::init_from(dylib).map_err(|error| format!("ORT dylib {}: {error}", dylib.display()))?;
    if !builder.with_name("asr").commit() {
        return Err(
            "ORT runtime is already initialized; startup must initialize it before creating sessions"
                .into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::init_runtime;
    use std::path::Path;

    #[test]
    fn runtime_rejects_a_missing_library_before_ort_initialization() {
        let result = init_runtime(Path::new("/nonexistent/deeray-asr/libonnxruntime.so"));
        assert!(result.is_err());
        let error = result.expect_err("a missing explicit ORT library must refuse");
        assert!(error.contains("ORT dylib"));
    }
}
