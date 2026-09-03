// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

use std::fmt;
use std::io;
use std::path::PathBuf;

/// A structured refusal while parsing, validating, or selecting a model package.
#[derive(Debug)]
pub enum PackageError {
    Io {
        context: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    MalformedLine {
        line: usize,
        reason: &'static str,
    },
    DuplicateKey {
        line: usize,
        key: String,
    },
    UnknownKey {
        line: usize,
        key: String,
    },
    MissingKey {
        key: &'static str,
    },
    UnsupportedFormatVersion {
        value: String,
    },
    InvalidValue {
        key: &'static str,
        value: String,
        reason: &'static str,
    },
    UnsafeAssetPath {
        key: &'static str,
        value: String,
        reason: &'static str,
    },
    AssetNotRegularFile {
        key: &'static str,
        path: PathBuf,
    },
    InvalidAssetData {
        key: &'static str,
        path: PathBuf,
        reason: String,
    },
    Compatibility {
        field: &'static str,
        reason: &'static str,
    },
}

impl PackageError {
    pub(crate) fn io(context: &'static str, path: PathBuf, source: io::Error) -> Self {
        Self::Io {
            context,
            path,
            source,
        }
    }
}

impl fmt::Display for PackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                context,
                path,
                source,
            } => write!(f, "{context} {}: {source}", path.display()),
            Self::MalformedLine { line, reason } => {
                write!(f, "config.kv line {line}: {reason}")
            }
            Self::DuplicateKey { line, key } => {
                write!(f, "config.kv line {line}: duplicate key {key}")
            }
            Self::UnknownKey { line, key } => {
                write!(f, "config.kv line {line}: unknown key {key}")
            }
            Self::MissingKey { key } => write!(f, "config.kv: required key {key} is missing"),
            Self::UnsupportedFormatVersion { value } => {
                write!(f, "config.kv format_version: expected 1, got {value}")
            }
            Self::InvalidValue { key, value, reason } => {
                write!(f, "config.kv {key}={value:?}: {reason}")
            }
            Self::UnsafeAssetPath { key, value, reason } => {
                write!(f, "config.kv {key}={value:?}: unsafe asset path ({reason})")
            }
            Self::AssetNotRegularFile { key, path } => {
                write!(
                    f,
                    "config.kv {key}: {} is not a regular file",
                    path.display()
                )
            }
            Self::InvalidAssetData { key, path, reason } => {
                write!(f, "config.kv {key}: invalid {} ({reason})", path.display())
            }
            Self::Compatibility { field, reason } => {
                write!(f, "config.kv {field}: {reason}")
            }
        }
    }
}

impl std::error::Error for PackageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
