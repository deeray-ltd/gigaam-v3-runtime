// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Greedy CTC decoding: per-frame argmax, repeated-token collapse, blank removal, and words.

use crate::contracts::{Decoded, Encoder, FrameRate, WindowDecoder};
use gigaam_audio::FeatureMatrixView;
use gigaam_model_package::{ModelPackage, VariantPackage};
use gigaam_primitives::{f64_to_f32, usize_to_f32};
use std::time::Instant;

pub use crate::contracts::{Token, Word};

/// CTC silence interpretation derived from the vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CtcSilencePolicy {
    /// Only the blank token denotes silence because the vocabulary has no standalone `▁` token.
    BlankOnly,
    /// Both blank and the standalone SentencePiece boundary token denote silence.
    BlankAndSpace { space_token: usize },
}

impl CtcSilencePolicy {
    fn space_token(self) -> Option<usize> {
        match self {
            Self::BlankOnly => None,
            Self::BlankAndSpace { space_token } => Some(space_token),
        }
    }

    pub const fn construction_notice(self) -> Option<CtcConstructionNotice> {
        match self {
            Self::BlankOnly => Some(CtcConstructionNotice::BlankOnlySilenceMask),
            Self::BlankAndSpace { .. } => None,
        }
    }
}

/// A process-boundary notice resulting from one valid CTC construction decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CtcConstructionNotice {
    BlankOnlySilenceMask,
}

impl CtcConstructionNotice {
    pub const fn message(self) -> &'static str {
        match self {
            Self::BlankOnlySilenceMask => {
                "# warning: vocabulary has no standalone ▁ — silence mask uses blank only, so endpointing is less precise"
            }
        }
    }
}

/// `log_probabilities` is a contiguous `[frames][vocabulary]` matrix.
pub fn greedy(
    log_probabilities: &[f32],
    frames: usize,
    vocabulary: usize,
    blank: usize,
) -> Result<Vec<Token>, String> {
    let (tokens, _, _) = greedy_masked(log_probabilities, frames, vocabulary, blank, None)?;
    Ok(tokens)
}

/// Greedy CTC output: the collapsed non-blank tokens, the per-frame silence mask, and the raw
/// per-frame argmax label (one entry per frame, blank included).
pub type GreedyMaskedOutput = (Vec<Token>, Vec<bool>, Vec<usize>);

/// The greedy output plus a per-frame silence mask and the raw per-frame argmax label. A frame is
/// silent when its argmax is blank or the bare SentencePiece space token emitted during pauses
/// between words. `labels` has one entry per output frame, blank included, so a caller can
/// reproduce the collapse behind `tokens` without re-running the encoder.
pub fn greedy_masked(
    log_probabilities: &[f32],
    frames: usize,
    vocabulary: usize,
    blank: usize,
    space: Option<usize>,
) -> Result<GreedyMaskedOutput, String> {
    greedy_masked_with_detail(log_probabilities, frames, vocabulary, blank, space, false)
}

fn greedy_masked_with_detail(
    log_probabilities: &[f32],
    frames: usize,
    vocabulary: usize,
    blank: usize,
    space: Option<usize>,
    scored: bool,
) -> Result<GreedyMaskedOutput, String> {
    greedy_masked_in_context(
        log_probabilities,
        frames,
        vocabulary,
        blank,
        space,
        scored,
        &crate::ExecutionControl::without_deadline(),
    )
}

fn greedy_masked_in_context(
    log_probabilities: &[f32],
    frames: usize,
    vocabulary: usize,
    blank: usize,
    space: Option<usize>,
    scored: bool,
    control: &crate::ExecutionControl,
) -> Result<GreedyMaskedOutput, String> {
    control.checkpoint()?;
    validate_greedy_input(log_probabilities, frames, vocabulary, blank, space)?;
    let mut out = Vec::new();
    let mut silence = Vec::with_capacity(frames);
    let mut labels = Vec::with_capacity(frames);
    let mut previous = usize::MAX;
    for frame in 0..frames {
        control.checkpoint()?;
        let row = &log_probabilities[frame * vocabulary..(frame + 1) * vocabulary];
        let mut best = 0_usize;
        for (index, &value) in row.iter().enumerate() {
            if value > row[best] {
                best = index;
            }
        }
        silence.push(best == blank || Some(best) == space);
        if best != blank && best != previous {
            out.push(if scored {
                Token::scored(
                    best,
                    frame,
                    crate::contracts::selected_log_probability(row, best)?,
                )?
            } else {
                Token::new(best, frame)
            });
        }
        labels.push(best);
        previous = best;
    }
    Ok((out, silence, labels))
}

fn validate_greedy_input(
    log_probabilities: &[f32],
    frames: usize,
    vocabulary: usize,
    blank: usize,
    space: Option<usize>,
) -> Result<(), String> {
    if vocabulary == 0 {
        return Err("CTC vocabulary size must be nonzero".into());
    }
    if blank >= vocabulary {
        return Err(format!(
            "CTC blank identifier {blank} does not fit vocabulary size {vocabulary}"
        ));
    }
    if let Some(space_token) = space
        && space_token >= vocabulary
    {
        return Err(format!(
            "CTC space identifier {space_token} does not fit vocabulary size {vocabulary}"
        ));
    }
    let expected_values = frames
        .checked_mul(vocabulary)
        .ok_or_else(|| "CTC matrix dimensions overflow usize".to_owned())?;
    if log_probabilities.len() != expected_values {
        return Err(format!(
            "CTC log-probability storage has {} values, expected {expected_values}",
            log_probabilities.len()
        ));
    }
    for (index, value) in log_probabilities.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "CTC log probability at flattened index {index} must be finite"
            ));
        }
    }
    Ok(())
}

/// Decode SentencePiece output without a tokenizer library by concatenating pieces and replacing
/// the word-boundary marker with spaces.
pub fn tokens_to_text(tokens: &[Token], vocabulary: &[String]) -> String {
    let mut text = String::new();
    for token in tokens {
        text.push_str(&vocabulary[token.id()]);
    }
    text.replace('\u{2581}', " ").trim_start().to_owned()
}

/// Returns the token-index span `(first, last)` behind each word `tokens_to_words` would emit,
/// in word order and inclusive on both ends, using this token slice's own 0-based indices.
/// `tokens_to_words` is built on this function, so the two can never disagree about which tokens
/// form a word: an external reader can reproduce the decoder's own word grouping from the same
/// tokens without re-implementing the merge rule. A bare SentencePiece word-boundary token
/// (piece text `▁` with no trailing text) contributes no text and is excluded from every span; a
/// run of only such tokens yields no span.
pub fn word_token_spans(tokens: &[Token], vocabulary: &[String]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut current = String::new();
    let (mut first, mut last) = (0_usize, 0_usize);
    for (index, token) in tokens.iter().enumerate() {
        let piece = &vocabulary[token.id()];
        if piece.starts_with('\u{2581}') && !current.is_empty() {
            let text = std::mem::take(&mut current);
            if !text.trim().is_empty() {
                spans.push((first, last));
            }
        }
        if current.is_empty() {
            first = index;
        }
        last = index;
        current.push_str(piece.trim_start_matches('\u{2581}'));
    }
    if !current.is_empty() && !current.trim().is_empty() {
        spans.push((first, last));
    }
    spans
}

/// Build words using their first and last token times. A piece beginning with `▁` starts a word.
pub fn tokens_to_words(
    tokens: &[Token],
    vocabulary: &[String],
    frame_rate: FrameRate,
) -> Result<Vec<Word>, String> {
    let mut words = Vec::new();
    for (first_index, last_index) in word_token_spans(tokens, vocabulary) {
        let mut text = String::new();
        for token in &tokens[first_index..=last_index] {
            text.push_str(vocabulary[token.id()].trim_start_matches('\u{2581}'));
        }
        let first_frame = tokens[first_index].frame();
        let last_frame = tokens[last_index].frame();
        words.push(word_from_token_range(
            text,
            first_frame,
            last_frame,
            frame_rate,
        )?);
    }
    Ok(words)
}

fn word_from_token_range(
    text: String,
    first: usize,
    last: usize,
    frame_rate: FrameRate,
) -> Result<Word, String> {
    let end_frame = last
        .checked_add(1)
        .ok_or_else(|| "CTC token frame end overflows usize".to_owned())?;
    Word::new(
        text,
        usize_to_f32(first) / frame_rate.get(),
        usize_to_f32(end_frame) / frame_rate.get(),
    )
}

fn validate_beam_alignment(
    labels: &[usize],
    tokens: &[Token],
    blank: usize,
    vocabulary: usize,
) -> Result<(), String> {
    let mut prior = None;
    let mut emitted = 0;
    for (frame, &label) in labels.iter().enumerate() {
        if label >= vocabulary {
            return Err("CTC beam label outside vocabulary".into());
        }
        if label != blank && prior != Some(label) {
            let token = tokens
                .get(emitted)
                .ok_or("CTC beam alignment lacks token")?;
            if token.id() != label || token.frame() != frame {
                return Err("CTC beam alignment does not match tokens".into());
            }
            emitted += 1;
        }
        prior = Some(label);
    }
    if emitted != tokens.len() {
        return Err("CTC beam tokens exceed alignment".into());
    }
    Ok(())
}

pub(crate) struct ProjectedWords {
    pub(crate) words: Vec<Word>,
    pub(crate) spans: Vec<(usize, usize)>,
}

pub(crate) fn project_words(
    tokens: &[Token],
    vocabulary: &[String],
    projector: Option<&crate::TokenTextProjector>,
    frame_rate: FrameRate,
) -> Result<ProjectedWords, String> {
    match projector {
        None => Ok(ProjectedWords {
            words: tokens_to_words(tokens, vocabulary, frame_rate)?,
            spans: word_token_spans(tokens, vocabulary),
        }),
        Some(projector) => {
            let projection = projector.project(tokens)?;
            let mut words = Vec::with_capacity(projection.words.len());
            let mut spans = Vec::with_capacity(projection.words.len());
            for word in projection.words {
                let text = projection.text[word.text_range].to_string();
                words.push(word_from_token_range(
                    text,
                    tokens[word.first_token].frame(),
                    tokens[word.last_token].frame(),
                    frame_rate,
                )?);
                spans.push((word.first_token, word.last_token));
            }
            Ok(ProjectedWords { words, spans })
        }
    }
}

/// CTC decoder over an injected encoder port.
pub struct CtcDecoder<E: Encoder> {
    encoder: E,
    vocabulary: Vec<String>,
    projector: Option<crate::TokenTextProjector>,
    mode: crate::DecodingMode,
    blank: usize,
    silence_policy: CtcSilencePolicy,
    frame_rate: FrameRate,
}

impl<E: Encoder> CtcDecoder<E> {
    /// Constructs a decoder from exact vocabulary and encoder properties.
    pub fn new(
        encoder: E,
        vocabulary: Vec<String>,
        blank: usize,
        frame_rate: FrameRate,
    ) -> Result<Self, String> {
        let expected_output_dimension = blank
            .checked_add(1)
            .ok_or_else(|| "CTC blank identifier overflows output dimension".to_owned())?;
        if encoder.out_dim() != expected_output_dimension || vocabulary.len() < blank {
            return Err(format!(
                "CTC: encoder dimension {}, blank {}, vocabulary words {}",
                encoder.out_dim(),
                blank,
                vocabulary.len()
            ));
        }
        let silence_policy = match vocabulary.iter().position(|piece| piece == "\u{2581}") {
            Some(space_token) => CtcSilencePolicy::BlankAndSpace { space_token },
            None => CtcSilencePolicy::BlankOnly,
        };
        Ok(Self {
            encoder,
            vocabulary,
            projector: None,
            mode: crate::DecodingMode::Greedy,
            blank,
            silence_policy,
            frame_rate,
        })
    }

    /// Builds the pure decoder from one typed model-package projection.
    pub fn from_pack(pack: &ModelPackage, encoder: E) -> Result<Self, String> {
        let vocabulary = pack.ctc_vocabulary().map_err(|error| error.to_string())?;
        let frame_rate = FrameRate::new(f64_to_f32(pack.frontend().frames_per_second()))?;
        Self::new(encoder, vocabulary, pack.ctc().blank_id(), frame_rate)
    }

    pub(crate) fn from_variant(
        pack: &VariantPackage,
        encoder: E,
        projector: crate::TokenTextProjector,
    ) -> Result<Self, String> {
        if pack.variant().family() != gigaam_model_package::ModelFamily::Ctc {
            return Err("selected variant is not CTC".into());
        }
        let vocabulary = pack.vocabulary().map_err(|error| error.to_string())?;
        let frame_rate = FrameRate::new(f64_to_f32(pack.frontend().frames_per_second()))?;
        let mut decoder = Self::new(encoder, vocabulary, pack.tokenizer().blank_id(), frame_rate)?;
        if pack.tokenizer().kind() == gigaam_model_package::TokenizerKind::Characters {
            decoder.silence_policy = match decoder.vocabulary.iter().position(|piece| piece == " ")
            {
                Some(space_token) => CtcSilencePolicy::BlankAndSpace { space_token },
                None => CtcSilencePolicy::BlankOnly,
            };
        }
        decoder.projector = Some(projector);
        Ok(decoder)
    }

    pub(crate) fn with_mode(mut self, mode: crate::DecodingMode) -> Self {
        self.mode = mode;
        self
    }

    pub const fn silence_policy(&self) -> CtcSilencePolicy {
        self.silence_policy
    }

    pub const fn construction_notice(&self) -> Option<CtcConstructionNotice> {
        self.silence_policy.construction_notice()
    }
}

impl<E: Encoder> WindowDecoder for CtcDecoder<E> {
    fn silence_capability(&self) -> crate::SilenceCapability {
        match self.mode {
            crate::DecodingMode::Greedy => crate::SilenceCapability::LegacyBlank,
            crate::DecodingMode::Beam { .. } => crate::SilenceCapability::RepresentativeAlignment,
        }
    }

    fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
        self.decode_with_detail(features, crate::contracts::ScoreDetail::None)
    }

    fn decode_with_detail(
        &mut self,
        features: FeatureMatrixView<'_>,
        detail: crate::contracts::ScoreDetail,
    ) -> Result<Decoded, String> {
        let control = crate::ExecutionControl::without_deadline();
        self.decode_in_context(features, detail, &control)
    }

    fn decode_in_context(
        &mut self,
        features: FeatureMatrixView<'_>,
        detail: crate::contracts::ScoreDetail,
        control: &crate::ExecutionControl,
    ) -> Result<Decoded, String> {
        control.checkpoint()?;
        let started = Instant::now();
        let (log_probabilities, output_frames) = self.encoder.forward(features)?;
        control.checkpoint()?;
        let encoder_seconds = started.elapsed().as_secs_f64();
        let (tokens, silence, labels) = match self.mode {
            crate::DecodingMode::Greedy => {
                let (tokens, silence, labels) = greedy_masked_in_context(
                    &log_probabilities,
                    output_frames,
                    self.encoder.out_dim(),
                    self.blank,
                    self.silence_policy.space_token(),
                    detail == crate::ScoreDetail::TokenLogProbabilities,
                    control,
                )?;
                (
                    tokens,
                    silence,
                    crate::FrameLabels::new(crate::FrameLabelKind::GreedyArgmax, labels),
                )
            }
            crate::DecodingMode::Beam { width } => {
                let input = crate::ctc_beam::CtcBeamInput::new(
                    &log_probabilities,
                    output_frames,
                    self.encoder.out_dim(),
                    crate::ctc_beam::CtcBeamOptions {
                        blank: self.blank,
                        width: width.get(),
                        detail,
                    },
                )?;
                let (labels, tokens) =
                    crate::ctc_beam::decode(&input, || control.checkpoint())?.into_parts();
                validate_beam_alignment(&labels, &tokens, self.blank, self.encoder.out_dim())?;
                let silence = labels
                    .iter()
                    .map(|label| {
                        *label == self.blank || Some(*label) == self.silence_policy.space_token()
                    })
                    .collect();
                (
                    tokens,
                    silence,
                    crate::FrameLabels::new(crate::FrameLabelKind::CtcRepresentativePath, labels),
                )
            }
        };
        control.checkpoint()?;
        let ProjectedWords {
            words,
            spans: word_spans,
        } = project_words(
            &tokens,
            &self.vocabulary,
            self.projector.as_ref(),
            self.frame_rate,
        )?;
        control.checkpoint()?;
        Decoded::new_with_frame_labels(crate::DecodedFrameLabelsInput {
            words,
            silence,
            output_frames,
            encoder_seconds,
            frame_labels: labels,
            tokens,
            word_spans,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gigaam_audio::FeatureMatrix;

    #[test]
    fn selected_projector_maps_original_occurrences_to_words() {
        let metadata = gigaam_model_package::TokenizerMetadata::Characters(vec![
            "а".into(),
            " ".into(),
            "б".into(),
        ]);
        let projector = crate::TokenTextProjector::new(metadata).expect("valid tokenizer");
        let tokens = vec![
            Token::new(0, 2),
            Token::new(1, 3),
            Token::new(1, 4),
            Token::new(2, 5),
        ];
        let ProjectedWords { words, spans } =
            project_words(&tokens, &[], Some(&projector), frame_rate()).expect("valid projection");
        assert_eq!(
            words.iter().map(Word::text).collect::<Vec<_>>(),
            vec!["а", "б"]
        );
        assert_eq!(spans, vec![(0, 0), (3, 3)]);
        assert_eq!(tokens[3].frame(), 5);
    }

    struct FakeEncoder {
        output: Vec<f32>,
        frames: usize,
        dimension: usize,
    }

    impl Encoder for FakeEncoder {
        fn out_dim(&self) -> usize {
            self.dimension
        }

        fn forward(
            &mut self,
            features: FeatureMatrixView<'_>,
        ) -> Result<(Vec<f32>, usize), String> {
            if features.mel_bins() != 2 {
                return Err("fake encoder received the wrong feature shape".into());
            }
            Ok((self.output.clone(), self.frames))
        }
    }

    fn features() -> FeatureMatrix {
        FeatureMatrix::from_values(2, 1, vec![0.0, 0.0])
            .expect("test feature dimensions and values are valid")
    }

    fn frame_rate() -> FrameRate {
        FrameRate::new(25.0).expect("test frame rate is finite and positive")
    }

    #[test]
    fn beam_preserves_representative_labels_and_token_provenance() {
        let encoder = FakeEncoder {
            output: vec![0.0, 4.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0],
            frames: 3,
            dimension: 3,
        };
        let mut decoder = CtcDecoder::new(encoder, vec!["а".into(), "б".into()], 2, frame_rate())
            .expect("valid decoder")
            .with_mode(crate::DecodingMode::Beam {
                width: std::num::NonZeroUsize::new(3).expect("positive width"),
            });
        assert_eq!(
            decoder.silence_capability(),
            crate::SilenceCapability::RepresentativeAlignment
        );
        let decoded = decoder
            .decode_with_detail(features().view(), crate::ScoreDetail::TokenLogProbabilities)
            .expect("beam decoding succeeds");
        assert_eq!(decoded.labels(), None);
        let labels = decoded.frame_labels().expect("typed labels");
        assert_eq!(labels.kind(), crate::FrameLabelKind::CtcRepresentativePath);
        validate_beam_alignment(labels.values(), decoded.tokens().expect("tokens"), 2, 3)
            .expect("alignment matches");
        assert!(
            decoded
                .tokens()
                .expect("tokens")
                .iter()
                .all(|token| token.log_probability().is_some())
        );
    }
    #[test]
    fn alignment_refuses_wrong_id_frame_and_length() {
        assert!(
            validate_beam_alignment(&[1, 2, 1], &[Token::new(1, 0), Token::new(1, 2)], 2, 3)
                .is_ok()
        );
        assert!(
            validate_beam_alignment(&[1, 2, 1], &[Token::new(1, 0), Token::new(1, 1)], 2, 3)
                .is_err()
        );
        assert!(validate_beam_alignment(&[3], &[], 2, 3).is_err());
        assert!(validate_beam_alignment(&[1], &[], 2, 3).is_err());
    }

    #[test]
    fn words_have_start_and_end() {
        let vocabulary: Vec<String> = ["<blk>", "▁hel", "lo", "▁", "▁world"]
            .iter()
            .map(ToString::to_string)
            .collect();
        let tokens = vec![
            Token::new(1, 2),
            Token::new(2, 4),
            Token::new(3, 7),
            Token::new(4, 9),
        ];
        let words = tokens_to_words(&tokens, &vocabulary, frame_rate())
            .expect("test CTC token ranges are valid");
        assert_eq!(words.len(), 2);
        assert_eq!(
            (words[0].text(), words[0].start(), words[0].end()),
            ("hello", 0.08, 0.2)
        );
        assert_eq!(
            (words[1].text(), words[1].start(), words[1].end()),
            ("world", 0.36, 0.4)
        );
    }

    #[test]
    fn word_token_spans_reproduce_the_words_tokens_to_words_builds_and_exclude_the_bare_marker() {
        let vocabulary: Vec<String> = ["<blk>", "▁hel", "lo", "▁", "▁world"]
            .iter()
            .map(ToString::to_string)
            .collect();
        let tokens = vec![
            Token::new(1, 2),
            Token::new(2, 4),
            Token::new(3, 7), // a bare word-marker token between the two words
            Token::new(4, 9),
        ];
        let spans = word_token_spans(&tokens, &vocabulary);
        assert_eq!(spans, vec![(0, 1), (3, 3)]);
        assert!(
            spans
                .iter()
                .all(|&(first, last)| !(first..=last).contains(&2)),
            "the bare word-marker token at index 2 must belong to no span"
        );

        let words = tokens_to_words(&tokens, &vocabulary, frame_rate())
            .expect("test CTC token ranges are valid");
        assert_eq!(words.len(), spans.len());
        for (&(first, last), word) in spans.iter().zip(&words) {
            let mut text = String::new();
            for token in &tokens[first..=last] {
                text.push_str(vocabulary[token.id()].trim_start_matches('\u{2581}'));
            }
            assert_eq!(
                word.text(),
                text,
                "the span's own tokens must reproduce the decoded word text"
            );
        }
    }

    #[test]
    fn word_token_spans_of_only_marker_tokens_yields_no_span_or_word() {
        let vocabulary: Vec<String> = ["<blk>", "\u{2581}"]
            .iter()
            .map(ToString::to_string)
            .collect();
        let tokens = vec![Token::new(1, 0), Token::new(1, 1), Token::new(1, 2)];
        assert_eq!(word_token_spans(&tokens, &vocabulary), Vec::new());
        let words = tokens_to_words(&tokens, &vocabulary, frame_rate())
            .expect("bare-marker-only tokens are still a valid decode");
        assert!(words.is_empty());
    }

    #[test]
    fn finite_logits_preserve_current_silence_mask_and_tokens() {
        let log_probabilities = [
            0.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0,
        ];
        let (tokens, silence, labels) = greedy_masked(&log_probabilities, 4, 3, 0, Some(2))
            .expect("finite CTC logits satisfy the declared matrix contract");
        assert_eq!(silence, vec![true, false, true, false]);
        assert_eq!(labels, vec![0, 1, 2, 1]);
        assert_eq!(
            tokens,
            vec![Token::new(1, 1), Token::new(2, 2), Token::new(1, 3)]
        );
    }

    #[test]
    fn greedy_masked_labels_have_one_entry_per_frame_and_agree_with_emitted_tokens() {
        // Frames 0-1 repeat label 1 (one token), frame 2 is blank, frames 3-4 repeat label 2
        // (one token), frame 5 returns to label 1 (a new token because the intervening blank
        // reset the previous-label state).
        let log_probabilities = [
            0.0, 5.0, 0.0, // frame 0: label 1
            0.0, 5.0, 0.0, // frame 1: label 1 (repeat)
            5.0, 0.0, 0.0, // frame 2: blank
            0.0, 0.0, 5.0, // frame 3: label 2
            0.0, 0.0, 5.0, // frame 4: label 2 (repeat)
            0.0, 5.0, 0.0, // frame 5: label 1
        ];
        let (tokens, _silence, labels) = greedy_masked(&log_probabilities, 6, 3, 0, None)
            .expect("finite CTC logits satisfy the declared matrix contract");
        assert_eq!(labels.len(), 6);
        assert_eq!(labels, vec![1, 1, 0, 2, 2, 1]);
        assert_eq!(
            tokens,
            vec![Token::new(1, 0), Token::new(2, 3), Token::new(1, 5)]
        );
        for token in &tokens {
            let frame = token.frame();
            assert_eq!(
                labels[frame],
                token.id(),
                "an emitted token must name the label actually decoded at its frame"
            );
            assert!(
                frame == 0 || labels[frame - 1] != token.id(),
                "an emitted token must start a new run, not continue the previous frame's label"
            );
        }
    }

    #[test]
    fn greedy_refuses_invalid_matrix_contracts() {
        assert!(greedy(&[], 0, 0, 0).is_err());
        assert!(greedy(&[], 0, 3, 3).is_err());
        assert!(greedy_masked(&[], 0, 3, 0, Some(3)).is_err());
        assert!(greedy(&[], usize::MAX, 2, 0).is_err());
        assert!(greedy(&[0.0, -1.0], 1, 3, 0).is_err());
    }

    #[test]
    fn greedy_refuses_nonfinite_logits_at_every_position() {
        for (name, invalid) in [
            ("NaN", f32::NAN),
            ("positive infinity", f32::INFINITY),
            ("negative infinity", f32::NEG_INFINITY),
        ] {
            for position in [0, 1, 2] {
                let mut log_probabilities = vec![0.0, -1.0, -2.0];
                log_probabilities[position] = invalid;
                assert!(
                    greedy_masked(&log_probabilities, 1, 3, 0, Some(2)).is_err(),
                    "{name} at CTC logit position {position} must refuse"
                );
            }
        }
    }

    #[test]
    fn blank_only_policy_preserves_the_exact_adapter_notice() {
        let decoder = CtcDecoder::new(
            FakeEncoder {
                output: Vec::new(),
                frames: 0,
                dimension: 1,
            },
            vec!["<blk>".into()],
            0,
            frame_rate(),
        )
        .expect("blank-only fake CTC contract is internally consistent");
        assert_eq!(decoder.silence_policy(), CtcSilencePolicy::BlankOnly);
        let notice = decoder
            .construction_notice()
            .expect("blank-only CTC policy must expose its construction notice");
        assert_eq!(
            notice.message(),
            "# warning: vocabulary has no standalone ▁ — silence mask uses blank only, so endpointing is less precise"
        );
    }

    #[test]
    fn frame_rate_and_recognition_values_refuse_invalid_state() {
        assert!(FrameRate::new(0.0).is_err());
        assert!(FrameRate::new(f32::NAN).is_err());
        assert!(Word::new("word".into(), 1.0, 0.5).is_err());
        assert!(Word::new("word".into(), f32::INFINITY, 1.0).is_err());
        assert!(Word::new(" \t\n".into(), 0.0, 1.0).is_err());
        assert!(Decoded::new(Vec::new(), vec![true], 0, 0.0).is_err());
        assert!(Decoded::new(Vec::new(), Vec::new(), 0, f64::NAN).is_err());
    }

    #[test]
    fn word_transforms_consume_valid_words_and_refuse_a_preceding_cap() {
        let shifted = Word::new("word".into(), 0.5, 1.0)
            .expect("test word timestamps are valid")
            .shifted(2.0)
            .expect("finite word offset preserves a valid word");
        assert_eq!(
            (shifted.text(), shifted.start(), shifted.end()),
            ("word", 2.5, 3.0)
        );

        let unchanged = Word::new("word".into(), 0.5, 1.0)
            .expect("test word timestamps are valid")
            .capped_end(2.0)
            .expect("a later finite cap preserves a valid word");
        assert_eq!(
            (unchanged.text(), unchanged.start(), unchanged.end()),
            ("word", 0.5, 1.0)
        );

        let capped = Word::new("word".into(), 0.5, 1.0)
            .expect("test word timestamps are valid")
            .capped_end(0.75)
            .expect("an in-range finite cap preserves a valid word");
        assert_eq!(
            (capped.text(), capped.start(), capped.end()),
            ("word", 0.5, 0.75)
        );

        assert!(
            Word::new("word".into(), 1.0, 2.0)
                .expect("test word timestamps are valid")
                .capped_end(0.5)
                .is_err()
        );
    }

    #[test]
    fn injected_encoder_produces_current_ctc_words() {
        let mut decoder = CtcDecoder::new(
            FakeEncoder {
                output: vec![0.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, 0.0],
                frames: 3,
                dimension: 3,
            },
            vec!["▁hi".into(), "there".into(), "<blk>".into()],
            2,
            frame_rate(),
        )
        .expect("fake CTC contract is internally consistent");
        let features = features();
        let decoded = decoder
            .decode(features.view())
            .expect("fake encoder returns a valid CTC matrix");
        assert_eq!(decoded.words().len(), 1);
        assert_eq!(decoded.words()[0].text(), "hithere");
        assert_eq!(decoded.silence(), &[false, false, true]);
    }
    #[test]
    fn scored_occurrences_use_first_emission_and_ignore_row_offsets() -> Result<(), String> {
        let rows = [3.0, 0.0, 4.0, 0.0, 0.0, 5.0, 2.0, 0.0];
        let shifted: Vec<f32> = rows.iter().map(|value| value + 17.0).collect();
        let (tokens, _, _) = super::greedy_masked_with_detail(&rows, 4, 2, 1, None, true)?;
        let (other, _, _) = super::greedy_masked_with_detail(&shifted, 4, 2, 1, None, true)?;
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].frame(), 0);
        assert_eq!(tokens[1].frame(), 3);
        assert_eq!(tokens[0].log_probability(), other[0].log_probability());
        assert_eq!(tokens[1].log_probability(), other[1].log_probability());
        Ok(())
    }
}
