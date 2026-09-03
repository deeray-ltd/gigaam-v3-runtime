// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

use crate::error::PackageError;
use gigaam_primitives::u32_to_usize;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

/// A regular-file path that was validated immediately before this value was returned.
#[derive(Clone, Debug)]
pub struct ValidatedArtifact {
    key: &'static str,
    path: PathBuf,
}

impl ValidatedArtifact {
    pub(crate) fn new(key: &'static str, path: PathBuf) -> Self {
        Self { key, path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn key(&self) -> &'static str {
        self.key
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RelativeAsset {
    key: &'static str,
    path: PathBuf,
}

impl RelativeAsset {
    pub(crate) fn parse(key: &'static str, value: String) -> Result<Self, PackageError> {
        if value.contains('\\') {
            return Err(PackageError::UnsafeAssetPath {
                key,
                value,
                reason: "backslash separators are not accepted",
            });
        }
        let path = Path::new(&value);
        if path.as_os_str().is_empty() {
            return Err(PackageError::UnsafeAssetPath {
                key,
                value,
                reason: "path is empty",
            });
        }
        let mut normal_components = 0usize;
        for component in path.components() {
            match component {
                Component::Normal(_) => normal_components += 1,
                Component::CurDir => {
                    return Err(PackageError::UnsafeAssetPath {
                        key,
                        value,
                        reason: "current-directory components are not accepted",
                    });
                }
                Component::ParentDir => {
                    return Err(PackageError::UnsafeAssetPath {
                        key,
                        value,
                        reason: "parent traversal is not accepted",
                    });
                }
                Component::RootDir => {
                    return Err(PackageError::UnsafeAssetPath {
                        key,
                        value,
                        reason: "absolute paths are not accepted",
                    });
                }
                Component::Prefix(_) => {
                    return Err(PackageError::UnsafeAssetPath {
                        key,
                        value,
                        reason: "platform path prefixes are not accepted",
                    });
                }
            }
        }
        if normal_components == 0 {
            return Err(PackageError::UnsafeAssetPath {
                key,
                value,
                reason: "path has no file component",
            });
        }
        Ok(Self {
            key,
            path: path.to_path_buf(),
        })
    }

    pub(crate) fn fixed(key: &'static str, value: &'static str) -> Self {
        Self {
            key,
            path: PathBuf::from(value),
        }
    }
}

pub(crate) fn validate_regular_asset(
    root: &Path,
    asset: &RelativeAsset,
) -> Result<ValidatedArtifact, PackageError> {
    let selected_path = root.join(&asset.path);
    let canonical_path = fs::canonicalize(&selected_path).map_err(|source| {
        PackageError::io("canonicalize selected asset", selected_path.clone(), source)
    })?;
    if !canonical_path.starts_with(root) {
        return Err(PackageError::UnsafeAssetPath {
            key: asset.key,
            value: asset.path.display().to_string(),
            reason: "canonical target escapes the model package root",
        });
    }
    let metadata = fs::metadata(&canonical_path).map_err(|source| {
        PackageError::io(
            "inspect canonical selected asset",
            canonical_path.clone(),
            source,
        )
    })?;
    if !metadata.is_file() {
        return Err(PackageError::AssetNotRegularFile {
            key: asset.key,
            path: canonical_path,
        });
    }
    Ok(ValidatedArtifact::new(asset.key, canonical_path))
}

fn invalid_asset(artifact: &ValidatedArtifact, reason: impl Into<String>) -> PackageError {
    PackageError::InvalidAssetData {
        key: artifact.key(),
        path: artifact.path().to_path_buf(),
        reason: reason.into(),
    }
}

/// `.f32` format: `u32 ndim`, `u32 dims[ndim]`, then `f32` data, all little-endian.
/// This reader is intentionally crate-private: callers load only a typed selected artifact.
pub(crate) fn read_f32(
    artifact: &ValidatedArtifact,
) -> Result<(Vec<usize>, Vec<f32>), PackageError> {
    let mut file = fs::File::open(artifact.path()).map_err(|source| {
        PackageError::io(
            "open selected f32 asset",
            artifact.path().to_path_buf(),
            source,
        )
    })?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|source| {
        PackageError::io(
            "read selected f32 asset",
            artifact.path().to_path_buf(),
            source,
        )
    })?;
    let u32_at = |offset: usize| -> Result<u32, PackageError> {
        let end = offset
            .checked_add(4)
            .ok_or_else(|| invalid_asset(artifact, "f32 header offset overflows"))?;
        let word = buffer
            .get(offset..end)
            .ok_or_else(|| invalid_asset(artifact, "f32 header is truncated"))?;
        let bytes: [u8; 4] = word
            .try_into()
            .map_err(|_| invalid_asset(artifact, "f32 header word has invalid length"))?;
        Ok(u32::from_le_bytes(bytes))
    };
    let ndim = u32_to_usize(u32_at(0)?)
        .map_err(|error| invalid_asset(artifact, format!("f32 ndim: {error}")))?;
    let dimensions: Vec<usize> = (0..ndim)
        .map(|index| {
            let offset = index
                .checked_mul(4)
                .and_then(|value| value.checked_add(4))
                .ok_or_else(|| invalid_asset(artifact, "f32 dimension offset overflows"))?;
            u32_to_usize(u32_at(offset)?)
                .map_err(|error| invalid_asset(artifact, format!("f32 dimension: {error}")))
        })
        .collect::<Result<_, _>>()?;
    let offset = ndim
        .checked_add(1)
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| invalid_asset(artifact, "f32 header size overflows"))?;
    let values = dimensions.iter().try_fold(1usize, |total, &dimension| {
        total
            .checked_mul(dimension)
            .ok_or_else(|| invalid_asset(artifact, "f32 dimensions overflow"))
    })?;
    let expected_len = values
        .checked_mul(4)
        .and_then(|data_bytes| offset.checked_add(data_bytes))
        .ok_or_else(|| invalid_asset(artifact, "f32 data size overflows"))?;
    if buffer.len() != expected_len {
        return Err(invalid_asset(
            artifact,
            "f32 data size does not match dimensions",
        ));
    }
    let (chunks, remainder) = buffer[offset..].as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(invalid_asset(artifact, "f32 data has a partial word"));
    }
    let data = chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    Ok((dimensions, data))
}

pub(crate) fn read_vocabulary(artifact: &ValidatedArtifact) -> Result<Vec<String>, PackageError> {
    let text = fs::read_to_string(artifact.path()).map_err(|source| {
        PackageError::io(
            "read selected vocabulary",
            artifact.path().to_path_buf(),
            source,
        )
    })?;
    Ok(text.lines().map(str::to_string).collect())
}
