// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Exact tokenizer text and occurrence-to-word projection.

use crate::Token;
use gigaam_model_package::{PieceKind, TokenizerMetadata};
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectedWord {
    pub text_range: Range<usize>,
    pub first_token: usize,
    pub last_token: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenTextProjection {
    pub text: String,
    pub token_byte_ranges: Vec<Range<usize>>,
    pub words: Vec<ProjectedWord>,
}
#[derive(Clone, Debug)]
pub struct TokenTextProjector {
    metadata: TokenizerMetadata,
}
impl TokenTextProjector {
    pub fn new(metadata: TokenizerMetadata) -> Result<Self, String> {
        if let TokenizerMetadata::SentencePiece {
            pieces,
            unknown_surface,
            active_denormalizer,
            ..
        } = &metadata
        {
            if *active_denormalizer || pieces.iter().any(|p| p.kind() == PieceKind::Byte) {
                return Err(
                    "tokenizer requires unsupported byte reconstruction or denormalization".into(),
                );
            }
            for p in pieces {
                let surface = if p.kind() == PieceKind::Unknown {
                    unknown_surface.as_str()
                } else {
                    p.text()
                };
                let rendered = if p.kind() == PieceKind::Unknown {
                    surface.to_string()
                } else {
                    surface.replace('▁', " ")
                };
                if p.kind() != PieceKind::Control && rendered.split_whitespace().count() > 1 {
                    return Err("tokenizer piece can contribute to multiple words".into());
                }
            }
        } else if let TokenizerMetadata::Characters(pieces) = &metadata
            && pieces.iter().any(|p| p.split_whitespace().count() > 1)
        {
            return Err("character entry can contribute to multiple words".into());
        }
        Ok(Self { metadata })
    }
    pub fn project(&self, tokens: &[Token]) -> Result<TokenTextProjection, String> {
        let mut text = String::new();
        let mut ranges = Vec::with_capacity(tokens.len());
        let mut beginning = true;
        for token in tokens {
            let start = text.len();
            match &self.metadata {
                TokenizerMetadata::Characters(pieces) => {
                    let piece = pieces
                        .get(token.id())
                        .ok_or("token id outside vocabulary")?;
                    text.push_str(piece);
                }
                TokenizerMetadata::SentencePiece {
                    pieces,
                    unknown_surface,
                    add_dummy_prefix,
                    remove_extra_whitespaces,
                    ..
                } => {
                    let piece = pieces
                        .get(token.id())
                        .ok_or("token id outside vocabulary")?;
                    match piece.kind() {
                        PieceKind::Control => {}
                        PieceKind::Unknown => {
                            text.push_str(unknown_surface);
                            if !unknown_surface.is_empty() {
                                beginning = false;
                            }
                        }
                        PieceKind::Byte => return Err("byte piece is unsupported".into()),
                        PieceKind::Normal | PieceKind::UserDefined | PieceKind::Unused => {
                            let mut surface = piece.text();
                            if beginning
                                && (*add_dummy_prefix || *remove_extra_whitespaces)
                                && let Some(rest) = surface.strip_prefix('▁')
                            {
                                surface = rest;
                                if !*remove_extra_whitespaces {
                                    beginning = false;
                                }
                            }
                            for c in surface.chars() {
                                text.push(if c == '▁' { ' ' } else { c });
                            }
                            if !surface.is_empty() {
                                beginning = false;
                            }
                        }
                    }
                }
            }
            ranges.push(start..text.len());
        }
        let mut words = Vec::new();
        let mut current: Option<ProjectedWord> = None;
        for (index, range) in ranges.iter().enumerate() {
            for (offset, ch) in text[range.clone()].char_indices() {
                let start = range.start + offset;
                if ch.is_whitespace() {
                    if let Some(word) = current.take() {
                        words.push(word);
                    }
                } else if let Some(word) = &mut current {
                    word.text_range.end = start + ch.len_utf8();
                    word.last_token = index;
                } else {
                    current = Some(ProjectedWord {
                        text_range: start..start + ch.len_utf8(),
                        first_token: index,
                        last_token: index,
                    });
                }
            }
        }
        if let Some(word) = current {
            words.push(word);
        }
        Ok(TokenTextProjection {
            text,
            token_byte_ranges: ranges,
            words,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_projection_preserves_text_ranges_and_token_provenance() {
        let projector = TokenTextProjector::new(TokenizerMetadata::Characters(vec![
            "а".into(),
            " ".into(),
            "б".into(),
        ]))
        .expect("supported characters");
        let tokens = [
            Token::new(0, 2),
            Token::new(1, 3),
            Token::new(1, 4),
            Token::new(2, 5),
        ];
        let result = projector.project(&tokens).expect("valid tokens");
        assert_eq!(result.text, "а  б");
        assert_eq!(result.token_byte_ranges, vec![0..2, 2..3, 3..4, 4..6]);
        assert_eq!(
            result.words,
            vec![
                ProjectedWord {
                    text_range: 0..2,
                    first_token: 0,
                    last_token: 0
                },
                ProjectedWord {
                    text_range: 4..6,
                    first_token: 3,
                    last_token: 3
                }
            ]
        );
        assert_eq!(tokens[3].frame(), 5);
        assert!(projector.project(&[Token::new(3, 0)]).is_err());
    }
    #[test]
    fn one_entry_cannot_own_two_word_spans() {
        assert!(
            TokenTextProjector::new(TokenizerMetadata::Characters(vec!["a b".into()])).is_err()
        );
    }
}

#[cfg(test)]
mod sentencepiece_tests {
    use super::*;
    use prost::Message;
    use sentencepiece_model::{ModelProto, SentencePiece};

    fn projector() -> TokenTextProjector {
        let entries = [("<unk>", 2), ("▁", 1), ("а", 1), ("б", 1), ("<s>", 3)];
        let pieces = entries
            .iter()
            .map(|(text, kind)| SentencePiece {
                piece: Some((*text).into()),
                r#type: Some(*kind),
                ..Default::default()
            })
            .collect();
        let model = ModelProto {
            pieces,
            ..Default::default()
        };
        let vocabulary = entries
            .iter()
            .map(|(text, _)| (*text).to_string())
            .collect();
        let metadata = TokenizerMetadata::from_sentencepiece(&model.encode_to_vec(), vocabulary)
            .expect("valid model");
        TokenTextProjector::new(metadata).expect("supported model")
    }
    #[test]
    fn exact_unknown_and_boundary_surfaces_keep_occurrence_ranges() {
        let projector = projector();
        let tokens = [
            Token::new(1, 0),
            Token::new(1, 1),
            Token::new(2, 2),
            Token::new(0, 3),
            Token::new(4, 4),
            Token::new(3, 5),
            Token::new(1, 6),
        ];
        let result = projector.project(&tokens).expect("valid tokens");
        assert_eq!(result.text, "а ⁇ б ");
        assert_eq!(
            result.token_byte_ranges,
            vec![0..0, 0..0, 0..2, 2..7, 7..7, 7..9, 9..10]
        );
        assert_eq!(
            result
                .words
                .iter()
                .map(|w| &result.text[w.text_range.clone()])
                .collect::<Vec<_>>(),
            vec!["а", "⁇", "б"]
        );
        assert_eq!(
            result
                .words
                .iter()
                .map(|w| (w.first_token, w.last_token))
                .collect::<Vec<_>>(),
            vec![(2, 2), (3, 3), (5, 5)]
        );
    }
    #[test]
    fn rejects_byte_and_active_denormalizer_and_multiword_surface() {
        use sentencepiece_model::NormalizerSpec;
        let entries = [("<unk>", 2), ("a", 6)];
        let make = |pieces: Vec<SentencePiece>, denormalizer_spec| ModelProto {
            pieces,
            denormalizer_spec,
            ..Default::default()
        };
        let pieces: Vec<_> = entries
            .iter()
            .map(|(text, kind)| SentencePiece {
                piece: Some((*text).into()),
                r#type: Some(*kind),
                ..Default::default()
            })
            .collect();
        let vocab = vec!["<unk>".to_string(), "a".to_string()];
        let metadata = TokenizerMetadata::from_sentencepiece(
            &make(pieces.clone(), None).encode_to_vec(),
            vocab.clone(),
        )
        .expect("model parses");
        assert!(TokenTextProjector::new(metadata).is_err());
        let mut normal = pieces;
        normal[1].r#type = Some(1);
        let denormalizer = NormalizerSpec {
            precompiled_charsmap: Some(vec![1]),
            ..Default::default()
        };
        let metadata = TokenizerMetadata::from_sentencepiece(
            &make(normal.clone(), Some(denormalizer)).encode_to_vec(),
            vocab,
        )
        .expect("model parses");
        assert!(TokenTextProjector::new(metadata).is_err());
        normal[1].piece = Some("a▁b".into());
        let metadata = TokenizerMetadata::from_sentencepiece(
            &make(normal, None).encode_to_vec(),
            vec!["<unk>".into(), "a▁b".into()],
        )
        .expect("model parses");
        assert!(TokenTextProjector::new(metadata).is_err());
    }
    #[test]
    fn add_dummy_only_consumes_exactly_one_initial_boundary() {
        use sentencepiece_model::NormalizerSpec;
        let entries = [("<unk>", 2), ("▁", 1), ("а", 4), ("б", 5)];
        let pieces = entries
            .iter()
            .map(|(text, kind)| SentencePiece {
                piece: Some((*text).into()),
                r#type: Some(*kind),
                ..Default::default()
            })
            .collect();
        let model = ModelProto {
            pieces,
            normalizer_spec: Some(NormalizerSpec {
                add_dummy_prefix: Some(true),
                remove_extra_whitespaces: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        };
        let vocabulary = entries
            .iter()
            .map(|(text, _)| (*text).to_string())
            .collect();
        let metadata = TokenizerMetadata::from_sentencepiece(&model.encode_to_vec(), vocabulary)
            .expect("model parses");
        let projector = TokenTextProjector::new(metadata).expect("supported model");
        let result = projector
            .project(&[
                Token::new(1, 0),
                Token::new(1, 1),
                Token::new(2, 2),
                Token::new(3, 3),
            ])
            .expect("valid tokens");
        assert_eq!(result.text, " аб");
        assert_eq!(result.token_byte_ranges, vec![0..0, 0..1, 1..3, 3..5]);
        assert_eq!(
            (result.words[0].first_token, result.words[0].last_token),
            (2, 3)
        );
    }
    // SentencePiece 0.2.3 CPU oracle; source model is the immutable official RNNT
    // tokenizer at https://huggingface.co/ai-sage/GigaAM-v3/resolve/7655ad717f8122257385bb4b2f373db3697e8680/tokenizer.model
    // SHA-256 828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a.
    // Equivalent selected pieces and changed normalizer/unknown metadata are used here.
    fn controlled_model(add: bool, remove: bool, unknown_surface: &str) -> TokenTextProjector {
        use sentencepiece_model::{NormalizerSpec, TrainerSpec};
        let entries = [("а", 1), ("▁", 1), ("б", 1), ("<unk>", 2)];
        let pieces = entries
            .iter()
            .map(|(text, kind)| SentencePiece {
                piece: Some((*text).into()),
                r#type: Some(*kind),
                ..Default::default()
            })
            .collect();
        let model = ModelProto {
            pieces,
            trainer_spec: Some(TrainerSpec {
                unk_id: Some(3),
                unk_surface: Some(unknown_surface.into()),
                ..Default::default()
            }),
            normalizer_spec: Some(NormalizerSpec {
                add_dummy_prefix: Some(add),
                remove_extra_whitespaces: Some(remove),
                ..Default::default()
            }),
            ..Default::default()
        };
        let vocabulary = entries
            .iter()
            .map(|(text, _)| (*text).to_string())
            .collect();
        TokenTextProjector::new(
            TokenizerMetadata::from_sentencepiece(&model.encode_to_vec(), vocabulary)
                .expect("valid model"),
        )
        .expect("supported model")
    }
    #[test]
    fn source_oracle_witnesses_normalizer_flags() {
        let tokens = [
            Token::new(1, 0),
            Token::new(1, 1),
            Token::new(0, 2),
            Token::new(3, 3),
            Token::new(2, 4),
            Token::new(1, 5),
        ];
        for (add, remove, text, ranges) in [
            (
                false,
                false,
                "  а ⁇ б ",
                vec![0..1, 1..2, 2..4, 4..9, 9..11, 11..12],
            ),
            (
                false,
                true,
                "а ⁇ б ",
                vec![0..0, 0..0, 0..2, 2..7, 7..9, 9..10],
            ),
        ] {
            let result = controlled_model(add, remove, " ⁇ ")
                .project(&tokens)
                .expect("valid tokens");
            assert_eq!(result.text, text);
            assert_eq!(result.token_byte_ranges, ranges);
        }
    }
    #[test]
    fn source_oracle_witnesses_custom_and_empty_unknown_at_nonzero_id() {
        let tokens = [
            Token::new(1, 0),
            Token::new(3, 1),
            Token::new(1, 2),
            Token::new(2, 3),
        ];
        for (surface, text, ranges) in [
            ("?", "? б", vec![0..0, 0..1, 1..2, 2..4]),
            ("", "б", vec![0..0, 0..0, 0..0, 0..2]),
        ] {
            let result = controlled_model(true, true, surface)
                .project(&tokens)
                .expect("valid tokens");
            assert_eq!(result.text, text);
            assert_eq!(result.token_byte_ranges, ranges);
        }
    }
}
