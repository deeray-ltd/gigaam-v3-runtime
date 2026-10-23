// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Process-independent per-window observation contracts.

use gigaam_recognition::Token;
use std::sync::Arc;

/// Immutable observation emitted after a successful recognition window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowTiming {
    offset_sec: f32,
    frames: usize,
    encoder_seconds: f64,
}

impl WindowTiming {
    pub(crate) fn new(
        offset_sec: f32,
        frames: usize,
        encoder_seconds: f64,
    ) -> Result<Self, String> {
        if !offset_sec.is_finite() || offset_sec < 0.0 {
            return Err("window observation offset must be finite and nonnegative".into());
        }
        if !encoder_seconds.is_finite() || encoder_seconds < 0.0 {
            return Err(
                "window observation encoder duration must be finite and nonnegative".into(),
            );
        }
        Ok(Self {
            offset_sec,
            frames,
            encoder_seconds,
        })
    }

    pub const fn offset_sec(&self) -> f32 {
        self.offset_sec
    }

    pub const fn frames(&self) -> usize {
        self.frames
    }

    pub const fn encoder_seconds(&self) -> f64 {
        self.encoder_seconds
    }
}

/// A process adapter for successful-window observations.
pub trait WindowTimingObserver: Send + Sync {
    fn observe(&self, observation: WindowTiming);
}

/// Whether a decoded window sits on the encoder frame-rate subsampling grid (`Body`), or is the
/// final window of a decoded waveform span, which is widened backward to real audio content and
/// may start off that grid (`Tail`). Without VAD the span is the complete channel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowKind {
    Body,
    Tail,
}

/// Named inputs for constructing one validated window-evidence record. Grouped because the
/// record has more independent fields than a positional constructor can keep unambiguous.
pub(crate) struct WindowEvidenceInput {
    pub(crate) index: usize,
    pub(crate) kind: WindowKind,
    pub(crate) start_sec: f32,
    pub(crate) end_sec: f32,
    pub(crate) first_sample: usize,
    pub(crate) frontend_origin_sample: usize,
    pub(crate) first_feature_frame: usize,
    pub(crate) feature_frames: usize,
    pub(crate) residue: usize,
    pub(crate) origin: usize,
    pub(crate) output_frames: usize,
    pub(crate) keep_until_sec: f32,
    pub(crate) labels: Vec<usize>,
    pub(crate) mask: Vec<bool>,
    pub(crate) tokens: Vec<Token>,
    pub(crate) word_spans: Vec<(usize, usize)>,
}

/// Immutable per-window collapse-and-geometry evidence, emitted after a successful decode when a
/// window-evidence observer is installed. It carries the CTC decoder's own raw per-frame argmax
/// labels, its collapsed pre-word-merge tokens, and the token-index span behind each decoded
/// word, all in this window's local frame/token coordinates, so an external tool can reproduce
/// word segmentation without re-running recognition or re-implementing the decoder's word rule.
/// Temporal fields and `first_sample` refer to the original channel. Feature frames, `origin`,
/// and `residue` refer to the crop whose original sample offset is `frontend_origin_sample`;
/// token frames remain local to the decoded window. A crop need not start on the source hop grid.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowEvidence {
    index: usize,
    kind: WindowKind,
    start_sec: f32,
    end_sec: f32,
    first_sample: usize,
    frontend_origin_sample: usize,
    first_feature_frame: usize,
    feature_frames: usize,
    residue: usize,
    origin: usize,
    output_frames: usize,
    keep_until_sec: f32,
    labels: Vec<usize>,
    mask: Vec<bool>,
    tokens: Vec<Token>,
    word_spans: Vec<(usize, usize)>,
}

impl WindowEvidence {
    pub(crate) fn new(input: WindowEvidenceInput) -> Result<Self, String> {
        if !input.start_sec.is_finite() || input.start_sec < 0.0 {
            return Err("window evidence start must be finite and nonnegative".into());
        }
        if !input.end_sec.is_finite() || input.end_sec < input.start_sec {
            return Err("window evidence end must be finite and not precede its start".into());
        }
        if !input.keep_until_sec.is_finite() || input.keep_until_sec < 0.0 {
            return Err("window evidence keep-until must be finite and nonnegative".into());
        }
        if input.first_sample < input.frontend_origin_sample {
            return Err("window evidence first sample precedes its frontend origin".into());
        }
        if input.labels.len() != input.output_frames {
            return Err(format!(
                "window evidence labels has {} frames, expected {}",
                input.labels.len(),
                input.output_frames
            ));
        }
        if input.mask.len() != input.output_frames {
            return Err(format!(
                "window evidence mask has {} frames, expected {}",
                input.mask.len(),
                input.output_frames
            ));
        }
        for token in &input.tokens {
            if token.frame() >= input.output_frames {
                return Err(format!(
                    "window evidence token frame {} is outside the decoded window of {} frames",
                    token.frame(),
                    input.output_frames
                ));
            }
        }
        for &(first, last) in &input.word_spans {
            if first > last || last >= input.tokens.len() {
                return Err(format!(
                    "window evidence word span ({first}, {last}) is outside the {}-token decoded window",
                    input.tokens.len()
                ));
            }
        }
        Ok(Self {
            index: input.index,
            kind: input.kind,
            start_sec: input.start_sec,
            end_sec: input.end_sec,
            first_sample: input.first_sample,
            frontend_origin_sample: input.frontend_origin_sample,
            first_feature_frame: input.first_feature_frame,
            feature_frames: input.feature_frames,
            residue: input.residue,
            origin: input.origin,
            output_frames: input.output_frames,
            keep_until_sec: input.keep_until_sec,
            labels: input.labels,
            mask: input.mask,
            tokens: input.tokens,
            word_spans: input.word_spans,
        })
    }

    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn kind(&self) -> WindowKind {
        self.kind
    }

    pub const fn start_sec(&self) -> f32 {
        self.start_sec
    }

    pub const fn end_sec(&self) -> f32 {
        self.end_sec
    }

    pub const fn first_sample(&self) -> usize {
        self.first_sample
    }

    /// Original-channel sample offset of the waveform used to compute this crop's features.
    /// It is zero on the complete-channel path. Add this offset before interpreting local
    /// feature/token geometry on the source timeline; dividing it into frames loses precision
    /// when a VAD boundary does not lie on the frontend hop grid.
    pub const fn frontend_origin_sample(&self) -> usize {
        self.frontend_origin_sample
    }

    pub const fn first_feature_frame(&self) -> usize {
        self.first_feature_frame
    }

    pub const fn feature_frames(&self) -> usize {
        self.feature_frames
    }

    pub const fn residue(&self) -> usize {
        self.residue
    }

    pub const fn origin(&self) -> usize {
        self.origin
    }

    pub const fn output_frames(&self) -> usize {
        self.output_frames
    }

    pub const fn keep_until_sec(&self) -> f32 {
        self.keep_until_sec
    }

    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Returns the token-index span behind each decoded word, in word order. Each span's
    /// indices refer to this record's own `tokens()` list.
    pub fn word_spans(&self) -> &[(usize, usize)] {
        &self.word_spans
    }
}

/// A process adapter for successful-window evidence observations.
pub trait WindowEvidenceObserver: Send + Sync {
    fn observe(&self, observation: WindowEvidence);
}

#[derive(Clone)]
enum TimingSlot {
    Disabled,
    Enabled(Arc<dyn WindowTimingObserver>),
}

#[derive(Clone)]
enum EvidenceSlot {
    Disabled,
    Enabled(Arc<dyn WindowEvidenceObserver>),
}

/// A typed enabled/disabled observation mode selected by a process adapter.
///
/// It holds two independently enabled channels: window timing (the process trace) and window
/// evidence (the offline evidence writer). Each channel defaults to disabled, and enabling one
/// never changes the other.
#[derive(Clone)]
pub struct ObservationMode {
    timing: TimingSlot,
    evidence: EvidenceSlot,
}

impl ObservationMode {
    pub fn disabled() -> Self {
        Self {
            timing: TimingSlot::Disabled,
            evidence: EvidenceSlot::Disabled,
        }
    }

    pub fn enabled(observer: Arc<dyn WindowTimingObserver>) -> Self {
        Self {
            timing: TimingSlot::Enabled(observer),
            evidence: EvidenceSlot::Disabled,
        }
    }

    /// Returns the same mode with its window-evidence channel enabled, preserving whichever
    /// window-timing channel this mode already carries.
    pub fn with_evidence(self, observer: Arc<dyn WindowEvidenceObserver>) -> Self {
        Self {
            evidence: EvidenceSlot::Enabled(observer),
            ..self
        }
    }

    pub(crate) fn emit(&self, observation: WindowTiming) {
        if let TimingSlot::Enabled(observer) = &self.timing {
            observer.observe(observation);
        }
    }

    pub(crate) fn evidence_enabled(&self) -> bool {
        matches!(self.evidence, EvidenceSlot::Enabled(_))
    }

    pub(crate) fn emit_evidence(&self, observation: WindowEvidence) {
        if let EvidenceSlot::Enabled(observer) = &self.evidence {
            observer.observe(observation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WindowEvidence, WindowEvidenceInput, WindowKind};
    use gigaam_recognition::Token;

    fn valid_input() -> WindowEvidenceInput {
        WindowEvidenceInput {
            frontend_origin_sample: 0,
            index: 0,
            kind: WindowKind::Body,
            start_sec: 0.0,
            end_sec: 1.0,
            first_sample: 0,
            first_feature_frame: 0,
            feature_frames: 4,
            residue: 0,
            origin: 0,
            output_frames: 2,
            keep_until_sec: 1.0,
            labels: vec![0, 1],
            mask: vec![true, false],
            tokens: vec![Token::new(1, 1)],
            word_spans: vec![(0, 0)],
        }
    }

    #[test]
    fn valid_evidence_preserves_every_field() {
        let evidence =
            WindowEvidence::new(valid_input()).expect("valid window evidence input must succeed");
        assert_eq!(evidence.index(), 0);
        assert_eq!(evidence.kind(), WindowKind::Body);
        assert_eq!((evidence.start_sec(), evidence.end_sec()), (0.0, 1.0));
        assert_eq!(evidence.feature_frames(), 4);
        assert_eq!(evidence.frontend_origin_sample(), 0);
        assert_eq!((evidence.residue(), evidence.origin()), (0, 0));
        assert_eq!(evidence.output_frames(), 2);
        assert_eq!(evidence.keep_until_sec(), 1.0);
        assert_eq!(evidence.labels(), &[0, 1]);
        assert_eq!(evidence.mask(), &[true, false]);
        assert_eq!(evidence.tokens(), &[Token::new(1, 1)]);
        assert_eq!(evidence.word_spans(), &[(0, 0)]);
    }

    #[test]
    fn word_spans_index_within_the_carried_tokens_list() {
        let mut input = valid_input();
        input.tokens = vec![Token::new(1, 0), Token::new(2, 1)];
        input.word_spans = vec![(0, 1)];
        let evidence =
            WindowEvidence::new(input).expect("a span within the token list must succeed");
        for &(first, last) in evidence.word_spans() {
            assert!(first <= last);
            assert!(last < evidence.tokens().len());
        }
    }

    #[test]
    fn window_evidence_refuses_mismatched_lengths_and_out_of_range_tokens() {
        let mut wrong_labels = valid_input();
        wrong_labels.labels = vec![0];
        assert!(WindowEvidence::new(wrong_labels).is_err());

        let mut wrong_mask = valid_input();
        wrong_mask.mask = vec![true];
        assert!(WindowEvidence::new(wrong_mask).is_err());

        let mut out_of_range_token = valid_input();
        out_of_range_token.tokens = vec![Token::new(1, 2)];
        assert!(WindowEvidence::new(out_of_range_token).is_err());

        let mut inverted_window = valid_input();
        inverted_window.start_sec = 1.0;
        inverted_window.end_sec = 0.0;
        assert!(WindowEvidence::new(inverted_window).is_err());

        let mut negative_keep_until = valid_input();
        negative_keep_until.keep_until_sec = -1.0;
        assert!(WindowEvidence::new(negative_keep_until).is_err());

        let mut preceding_source_sample = valid_input();
        preceding_source_sample.frontend_origin_sample = 1;
        assert!(WindowEvidence::new(preceding_source_sample).is_err());

        let mut out_of_range_word_span = valid_input();
        out_of_range_word_span.word_spans = vec![(0, 1)];
        assert!(
            WindowEvidence::new(out_of_range_word_span).is_err(),
            "a word span whose end exceeds the token list must refuse"
        );

        let mut inverted_word_span = valid_input();
        inverted_word_span.tokens = vec![Token::new(1, 1), Token::new(1, 1)];
        inverted_word_span.word_spans = vec![(0, 0), (1, 0)]; // the second span ends before it starts
        assert!(
            WindowEvidence::new(inverted_word_span).is_err(),
            "a word span whose end precedes its start must refuse"
        );
    }
}
