// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! File/content dispatch into the strict WAV parser or feature-gated codec adapter.

use crate::contracts::{DecodedAudio, EncodedAudio};
use crate::{codecs, wav};
use std::fs;
use std::path::Path;

fn format_hint_from_path(path: &Path) -> Result<Option<String>, String> {
    match path.extension() {
        Some(extension) => {
            let extension = extension
                .to_str()
                .ok_or_else(|| format!("{}: audio extension is not Unicode", path.display()))?;
            Ok(Some(extension.to_owned()))
        }
        None => Ok(None),
    }
}

/// Loads an encoded file through the typed encoded-audio boundary.
pub fn load(path: &Path) -> Result<DecodedAudio, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let encoded = EncodedAudio::new(bytes, format_hint_from_path(path)?)?;
    load_bytes(encoded)
}

/// Dispatches a pre-validated encoded input. RIFF/WAV is decoded by the strict built-in parser;
/// other inputs require the optional decoder feature.
pub fn load_bytes(encoded: EncodedAudio) -> Result<DecodedAudio, String> {
    if wav::is_riff_wave(encoded.bytes()) {
        wav::parse_wav(encoded.bytes())
    } else {
        codecs::decode_bytes(encoded)
    }
}

/// Reads one strict RIFF/WAV file from disk.
pub fn read_wav(path: &Path) -> Result<DecodedAudio, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    wav::parse_wav(&bytes)
}
