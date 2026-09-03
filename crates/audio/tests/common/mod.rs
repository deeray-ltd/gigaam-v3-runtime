// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Checked reader for opaque binary golden fixture values.

use gigaam_primitives::u32_to_usize;
use std::io;
use std::path::Path;

#[path = "../../../../tests/support/mod.rs"]
mod external_artifacts;

pub use external_artifacts::external_artifact_root as root;

pub fn read_f32(path: &Path) -> io::Result<(Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(path)?;
    let u32_at = |offset: usize| -> io::Result<u32> {
        let end = offset.checked_add(4).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "fixture f32 header offset overflows",
            )
        })?;
        let word = bytes.get(offset..end).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "fixture f32 header is truncated",
            )
        })?;
        let array: [u8; 4] = word.try_into().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "fixture f32 header word is invalid",
            )
        })?;
        Ok(u32::from_le_bytes(array))
    };
    let dimensions = (0..u32_to_usize(u32_at(0)?).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("fixture f32 ndim: {error}"),
        )
    })?)
        .map(|index| {
            let offset = index
                .checked_add(1)
                .and_then(|value| value.checked_mul(4))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "fixture f32 dimension offset overflows",
                    )
                })?;
            u32_to_usize(u32_at(offset)?).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("fixture f32 dimension: {error}"),
                )
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    let offset = dimensions
        .len()
        .checked_add(1)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "fixture f32 header overflows")
        })?;
    let count = dimensions.iter().try_fold(1_usize, |total, dimension| {
        total.checked_mul(*dimension).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "fixture f32 dimensions overflow",
            )
        })
    })?;
    let expected = count
        .checked_mul(4)
        .and_then(|value| offset.checked_add(value))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "fixture f32 size overflows"))?;
    if bytes.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "fixture f32 size does not match dimensions",
        ));
    }
    let (words, remainder) = bytes[offset..].as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "fixture f32 payload has a partial word",
        ));
    }
    Ok((
        dimensions,
        words.iter().map(|word| f32::from_le_bytes(*word)).collect(),
    ))
}
