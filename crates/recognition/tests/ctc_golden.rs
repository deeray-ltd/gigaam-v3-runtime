// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Pure CTC decoding acceptance against fixed logits and transcript fixtures.

#[path = "../../../tests/support/mod.rs"]
mod external_artifacts;

use gigaam_model_package::ModelPackage;
use gigaam_primitives::u32_to_usize;
use gigaam_recognition::{FrameRate, ctc};
use std::io;
use std::path::Path;

fn read_f32(path: &Path) -> io::Result<(Vec<usize>, Vec<f32>)> {
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

#[test]
fn ctc_greedy_reproduces_golden_text_and_timestamps() {
    let root = external_artifacts::external_artifact_root();
    let pack = ModelPackage::open(&root.join("model"))
        .expect("golden CTC test requires a valid model pack");
    let vocabulary = pack
        .ctc_vocabulary()
        .expect("golden CTC model pack must include vocabulary");
    let blank = pack.ctc().blank_id();
    let frame_rate = FrameRate::new(25.0).expect("golden CTC frame rate is valid");
    for clip in ["example", "long_example"] {
        let (dimensions, log_probabilities) =
            read_f32(&root.join(format!("fixtures/{clip}.ctc.logprobs.f32")))
                .expect("golden CTC logits fixture must be readable");
        let frames = *dimensions
            .first()
            .expect("golden CTC logits must declare a frame dimension");
        let vocabulary_size = *dimensions
            .get(1)
            .expect("golden CTC logits must declare a vocabulary dimension");
        let tokens = ctc::greedy(&log_probabilities, frames, vocabulary_size, blank)
            .expect("golden CTC logits are finite and match their declared matrix shape");
        let text = ctc::tokens_to_text(&tokens, &vocabulary);
        let expected = std::fs::read_to_string(root.join(format!("fixtures/{clip}.ctc.gold.txt")))
            .expect("golden CTC transcript fixture must be readable");
        assert_eq!(text, expected, "{clip}");
        let words = ctc::tokens_to_words(&tokens, &vocabulary, frame_rate)
            .expect("golden CTC token timestamps are valid");
        assert!(
            words.len() > 10 && words[0].start() < 0.5,
            "{clip}: words/timestamps"
        );
        assert!(
            words.iter().all(|word| word.end() > word.start()),
            "{clip}: word end follows start"
        );
        assert!(
            words
                .windows(2)
                .all(|pair| pair[0].end() <= pair[1].start()),
            "{clip}: words do not overlap"
        );
    }
}
