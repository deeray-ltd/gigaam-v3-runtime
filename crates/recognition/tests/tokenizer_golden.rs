// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! External SentencePiece CPU-oracle witnesses for both selected V3 tokenizers.

#[path = "../../../tests/support/mod.rs"]
mod external_artifacts;

use gigaam_model_package::VariantPackage;
use gigaam_recognition::{Token, TokenTextProjector};
use std::fmt::Write;

#[test]
fn real_sentencepiece_models_match_text_and_occurrence_ranges() {
    let root = external_artifacts::external_artifact_root().join("tokenizer-fixtures");
    for (name, expected_cases) in [("ctc", 2390), ("rnnt", 6998)] {
        let fixture = root.join(name);
        let package = VariantPackage::open(&fixture).expect("valid external V3 package metadata");
        let metadata = package
            .load_tokenizer()
            .expect("model and vocabulary agree");
        let projector = TokenTextProjector::new(metadata).expect("supported model capabilities");
        let cases = std::fs::read_to_string(fixture.join("cases.tsv"))
            .expect("external oracle cases must be readable");
        let mut count = 0;
        for line in cases.lines() {
            let mut fields = line.split('\t');
            let ids = fields.next().expect("IDs field");
            let expected_text = fields.next().expect("text hex field");
            let expected_ranges = fields.next().expect("ranges field");
            assert!(fields.next().is_none(), "unexpected oracle fields");
            let tokens: Vec<_> = if ids.is_empty() {
                Vec::new()
            } else {
                ids.split(',')
                    .enumerate()
                    .map(|(frame, id)| Token::new(id.parse().expect("token ID"), frame))
                    .collect()
            };
            let result = projector.project(&tokens).expect("oracle IDs are valid");
            let mut text_hex = String::new();
            for byte in result.text.as_bytes() {
                write!(&mut text_hex, "{byte:02x}").expect("writing to String succeeds");
            }
            assert_eq!(text_hex, expected_text, "{name} case {count}");
            let ranges = result
                .token_byte_ranges
                .iter()
                .map(|range| format!("{}:{}", range.start, range.end))
                .collect::<Vec<_>>()
                .join(",");
            assert_eq!(ranges, expected_ranges, "{name} case {count}");
            count += 1;
        }
        assert_eq!(count, expected_cases, "{name} oracle case count");
    }
}
