// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

use sentencepiece_model::SentencePieceModel;

/// Decoder-relevant SentencePiece piece kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizerPiece {
    text: String,
    kind: PieceKind,
}
impl TokenizerPiece {
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn kind(&self) -> PieceKind {
        self.kind
    }
}

/// Owned tokenizer data, independent of the model reader implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenizerMetadata {
    Characters(Vec<String>),
    SentencePiece {
        pieces: Vec<TokenizerPiece>,
        unknown_surface: String,
        add_dummy_prefix: bool,
        remove_extra_whitespaces: bool,
        active_denormalizer: bool,
    },
}
impl TokenizerMetadata {
    pub fn from_sentencepiece(bytes: &[u8], vocabulary: Vec<String>) -> Result<Self, String> {
        read_sentencepiece(bytes, vocabulary)
    }
    pub(crate) fn characters(vocabulary: Vec<String>) -> Self {
        Self::Characters(vocabulary)
    }
}

pub(crate) fn read_sentencepiece(
    bytes: &[u8],
    vocabulary: Vec<String>,
) -> Result<TokenizerMetadata, String> {
    let model = SentencePieceModel::from_slice(bytes).map_err(|e| format!("invalid model: {e}"))?;
    if model.pieces().len() != vocabulary.len() {
        return Err("piece count differs from vocabulary".into());
    }
    let mut pieces = Vec::with_capacity(vocabulary.len());
    let mut unknown = None;
    for (index, (piece, word)) in model.pieces().iter().zip(vocabulary).enumerate() {
        if piece.piece() != word || word.is_empty() {
            return Err(format!("piece {index} differs from vocabulary"));
        }
        let kind = match piece.r#type {
            None | Some(1) => PieceKind::Normal,
            Some(2) => PieceKind::Unknown,
            Some(3) => PieceKind::Control,
            Some(4) => PieceKind::UserDefined,
            Some(5) => PieceKind::Unused,
            Some(6) => PieceKind::Byte,
            Some(value) => return Err(format!("piece {index} has unknown type {value}")),
        };
        if kind == PieceKind::Unknown && unknown.replace(index).is_some() {
            return Err("multiple unknown pieces".into());
        }
        pieces.push(TokenizerPiece { text: word, kind });
    }
    let trainer = model.trainer();
    let unknown_id = trainer.map_or(0, |t| t.unk_id());
    let unknown_id = usize::try_from(unknown_id).map_err(|_| "invalid unknown id".to_string())?;
    if unknown != Some(unknown_id) {
        return Err("unknown id does not identify the unique unknown piece".into());
    }
    if let Some(trainer) = trainer
        && let Some(declared) = &trainer.unk_piece
        && pieces
            .get(unknown_id)
            .is_none_or(|piece| piece.text() != declared)
    {
        return Err("unknown piece declaration differs from indexed piece".into());
    }
    let unknown_surface = trainer.map_or(" ⁇ ", |t| t.unk_surface()).to_string();
    let normalizer = model.normalizer();
    let add_dummy_prefix = normalizer.is_none_or(|n| n.add_dummy_prefix());
    let remove_extra_whitespaces = normalizer.is_none_or(|n| n.remove_extra_whitespaces());
    let active_denormalizer = model
        .denormalizer()
        .is_some_and(|n| !n.precompiled_charsmap().is_empty());
    Ok(TokenizerMetadata::SentencePiece {
        pieces,
        unknown_surface,
        add_dummy_prefix,
        remove_extra_whitespaces,
        active_denormalizer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use sentencepiece_model::{ModelProto, NormalizerSpec, SentencePiece, TrainerSpec};

    fn fixture() -> ModelProto {
        ModelProto {
            pieces: vec![
                SentencePiece {
                    piece: Some("<unk>".into()),
                    r#type: Some(2),
                    ..Default::default()
                },
                SentencePiece {
                    piece: Some("▁а".into()),
                    r#type: Some(1),
                    ..Default::default()
                },
            ],
            trainer_spec: Some(TrainerSpec::default()),
            normalizer_spec: Some(NormalizerSpec::default()),
            ..Default::default()
        }
    }
    #[test]
    fn reads_model_and_refuses_inconsistent_vocabulary_and_unknown_types() {
        let model = fixture();
        let vocab = vec!["<unk>".into(), "▁а".into()];
        let result =
            read_sentencepiece(&model.encode_to_vec(), vocab.clone()).expect("valid model");
        match result {
            TokenizerMetadata::SentencePiece {
                pieces,
                unknown_surface,
                add_dummy_prefix,
                remove_extra_whitespaces,
                ..
            } => {
                assert_eq!(pieces[0].kind(), PieceKind::Unknown);
                assert_eq!(unknown_surface, " ⁇ ");
                assert!(add_dummy_prefix && remove_extra_whitespaces);
            }
            TokenizerMetadata::Characters(_) => panic!("expected SentencePiece metadata"),
        }
        assert!(
            read_sentencepiece(&model.encode_to_vec(), vec!["wrong".into(), "▁а".into()]).is_err()
        );
        assert!(read_sentencepiece(&[255], vocab.clone()).is_err());
        let mut bad = model.clone();
        bad.pieces[1].r#type = Some(99);
        assert!(read_sentencepiece(&bad.encode_to_vec(), vocab.clone()).is_err());
        bad = model;
        bad.pieces[1].r#type = Some(2);
        assert!(read_sentencepiece(&bad.encode_to_vec(), vocab).is_err());
    }
}
