// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

pub(crate) use gigaam_transcription::{ScratchFile, ScratchSpace};

pub(crate) fn configured() -> std::io::Result<ScratchSpace> {
    let directory = std::env::var_os("ASR_BATCH_SCRATCH_DIR").map(std::path::PathBuf::from);
    let limit = match std::env::var("ASR_BATCH_SCRATCH_MAX_BYTES") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, error)),
    };
    ScratchSpace::from_settings(directory, limit.as_deref())
}
