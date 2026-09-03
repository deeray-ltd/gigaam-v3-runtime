// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Test-only resolution of externally supplied golden artifacts.

use std::path::PathBuf;

/// The sole environment key that selects the model-and-fixture artifact directory.
pub const EXTERNAL_ARTIFACT_ROOT_ENV: &str = "ASR_GOLDEN_ARTIFACT_ROOT";

/// Returns the validated external artifact directory for a golden suite.
///
/// The directory is intentionally resolved only when an opt-in golden test invokes this function.
pub fn external_artifact_root() -> PathBuf {
    let root = match std::env::var_os(EXTERNAL_ARTIFACT_ROOT_ENV) {
        None => {
            panic!("{EXTERNAL_ARTIFACT_ROOT_ENV} must name an absolute existing artifact directory")
        }
        Some(value) if value.is_empty() => panic!(
            "{EXTERNAL_ARTIFACT_ROOT_ENV} must not be empty; it must name an absolute existing artifact directory"
        ),
        Some(value) => PathBuf::from(value),
    };
    if !root.is_absolute() {
        panic!(
            "{EXTERNAL_ARTIFACT_ROOT_ENV} must be an absolute path, got {}",
            root.display()
        );
    }
    let metadata = match std::fs::metadata(&root) {
        Ok(metadata) => metadata,
        Err(error) => panic!(
            "{EXTERNAL_ARTIFACT_ROOT_ENV} {} must name an existing directory: {error}",
            root.display()
        ),
    };
    if !metadata.is_dir() {
        panic!(
            "{EXTERNAL_ARTIFACT_ROOT_ENV} {} must name a directory",
            root.display()
        );
    }
    root
}
