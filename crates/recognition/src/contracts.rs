// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Capability contracts shared by recognition algorithms and execution adapters.

use gigaam_audio::{ChannelAudioView, FeatureMatrixView};

/// A finite positive number of recognized frames per second.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct FrameRate(f32);

impl FrameRate {
    pub fn new(value: f32) -> Result<Self, String> {
        if !value.is_finite() || value <= 0.0 {
            return Err("recognition frame rate must be finite and positive".into());
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

/// One non-blank token emitted at an encoder frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    id: usize,
    frame: usize,
}

impl Token {
    pub const fn new(id: usize, frame: usize) -> Self {
        Self { id, frame }
    }

    pub const fn id(&self) -> usize {
        self.id
    }

    pub const fn frame(&self) -> usize {
        self.frame
    }
}

/// One recognized word with finite, ordered timestamps relative to a decoded window.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    text: String,
    start: f32,
    end: f32,
}

impl Word {
    pub fn new(text: String, start: f32, end: f32) -> Result<Self, String> {
        if text.trim().is_empty() {
            return Err("recognition word text must not be empty".into());
        }
        if !start.is_finite() || !end.is_finite() {
            return Err("recognition word timestamps must be finite".into());
        }
        if start < 0.0 || end < start {
            return Err("recognition word timestamps must be nonnegative and ordered".into());
        }
        Ok(Self { text, start, end })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub const fn start(&self) -> f32 {
        self.start
    }

    pub const fn end(&self) -> f32 {
        self.end
    }

    /// Returns the same word at a later finite window offset.
    pub fn shifted(self, offset: f32) -> Result<Self, String> {
        if !offset.is_finite() {
            return Err("recognition word offset must be finite".into());
        }
        let start = self.start + offset;
        let end = self.end + offset;
        Self::new(self.text, start, end)
    }

    /// Returns the same word with an end that does not extend beyond a finite window boundary.
    pub fn capped_end(self, maximum: f32) -> Result<Self, String> {
        if !maximum.is_finite() || maximum < 0.0 {
            return Err("recognition word end boundary must be finite and nonnegative".into());
        }
        if maximum < self.start {
            return Err("recognition word end boundary precedes the word start".into());
        }
        Self::new(self.text, self.start, self.end.min(maximum))
    }
}

/// One coherent decoded recognition window.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    words: Vec<Word>,
    silence: Vec<bool>,
    output_frames: usize,
    encoder_seconds: f64,
    labels: Option<Vec<usize>>,
    tokens: Option<Vec<Token>>,
    word_spans: Option<Vec<(usize, usize)>>,
}

impl Decoded {
    pub fn new(
        words: Vec<Word>,
        silence: Vec<bool>,
        output_frames: usize,
        encoder_seconds: f64,
    ) -> Result<Self, String> {
        Self::assemble(
            words,
            silence,
            output_frames,
            encoder_seconds,
            None,
            None,
            None,
        )
    }

    /// Constructs one decoded window that also carries a CTC decoder's raw per-frame argmax
    /// labels (blank included, one entry per output frame), its collapsed pre-word-merge tokens,
    /// and the token-index span behind each of `words`, all in this window's local frame/token
    /// coordinates. `words` and `silence` describe the same decoded window as `new`; each span in
    /// `word_spans` indexes into `tokens`.
    pub fn new_with_labels(
        words: Vec<Word>,
        silence: Vec<bool>,
        output_frames: usize,
        encoder_seconds: f64,
        labels: Vec<usize>,
        tokens: Vec<Token>,
        word_spans: Vec<(usize, usize)>,
    ) -> Result<Self, String> {
        Self::assemble(
            words,
            silence,
            output_frames,
            encoder_seconds,
            Some(labels),
            Some(tokens),
            Some(word_spans),
        )
    }

    fn assemble(
        words: Vec<Word>,
        silence: Vec<bool>,
        output_frames: usize,
        encoder_seconds: f64,
        labels: Option<Vec<usize>>,
        tokens: Option<Vec<Token>>,
        word_spans: Option<Vec<(usize, usize)>>,
    ) -> Result<Self, String> {
        if silence.len() != output_frames {
            return Err(format!(
                "recognition silence mask has {} frames, expected {output_frames}",
                silence.len()
            ));
        }
        if !encoder_seconds.is_finite() || encoder_seconds < 0.0 {
            return Err("recognition encoder duration must be finite and nonnegative".into());
        }
        if let Some(labels) = &labels
            && labels.len() != output_frames
        {
            return Err(format!(
                "recognition label sequence has {} frames, expected {output_frames}",
                labels.len()
            ));
        }
        if let Some(tokens) = &tokens {
            for token in tokens {
                if token.frame() >= output_frames {
                    return Err(format!(
                        "recognition token frame {} is outside the decoded window of {output_frames} frames",
                        token.frame()
                    ));
                }
            }
        }
        if let Some(word_spans) = &word_spans {
            let tokens_len = match &tokens {
                Some(tokens) => tokens.len(),
                None => 0,
            };
            for &(first, last) in word_spans {
                if first > last || last >= tokens_len {
                    return Err(format!(
                        "recognition word span ({first}, {last}) is outside the {tokens_len}-token decoded window"
                    ));
                }
            }
        }
        Ok(Self {
            words,
            silence,
            output_frames,
            encoder_seconds,
            labels,
            tokens,
            word_spans,
        })
    }

    pub fn words(&self) -> &[Word] {
        &self.words
    }

    pub fn into_words(self) -> Vec<Word> {
        self.words
    }

    /// Consumes one coherent recognition result without copying its decoded words or silence mask.
    pub fn into_parts(self) -> (Vec<Word>, Vec<bool>, usize, f64) {
        (
            self.words,
            self.silence,
            self.output_frames,
            self.encoder_seconds,
        )
    }

    pub fn silence(&self) -> &[bool] {
        &self.silence
    }

    /// Returns the raw per-frame argmax labels behind this window's tokens and words, one per
    /// output frame including blank, or `None` for a decoder that exposes no per-frame labels.
    pub fn labels(&self) -> Option<&[usize]> {
        self.labels.as_deref()
    }

    /// Returns the collapsed pre-word-merge tokens behind this window's words, or `None` for a
    /// decoder that exposes no per-frame labels.
    pub fn tokens(&self) -> Option<&[Token]> {
        self.tokens.as_deref()
    }

    /// Returns the token-index span behind each of this window's words, in word order, or `None`
    /// for a decoder that exposes no per-frame labels. Each span's indices refer to this
    /// window's own `tokens()` list.
    pub fn word_spans(&self) -> Option<&[(usize, usize)]> {
        self.word_spans.as_deref()
    }

    pub const fn output_frames(&self) -> usize {
        self.output_frames
    }

    pub const fn encoder_seconds(&self) -> f64 {
        self.encoder_seconds
    }
}

/// Provider-independent encoder port.
///
/// The input is an Audio-owned validated feature matrix. The returned vector keeps the current
/// row-major `[output_frames][output_dimension]` representation used by execution adapters.
pub trait Encoder {
    fn out_dim(&self) -> usize;

    /// Returns `(values, output_frames)` for the supplied feature matrix.
    fn forward(&mut self, features: FeatureMatrixView<'_>) -> Result<(Vec<f32>, usize), String>;
}

/// Decodes one validated feature window into a coherent recognition result.
pub trait WindowDecoder {
    fn frame_rate(&self) -> FrameRate;

    fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String>;
}

/// Dynamically dispatched decoder used by the current CLI composition.
impl WindowDecoder for Box<dyn WindowDecoder> {
    fn frame_rate(&self) -> FrameRate {
        (**self).frame_rate()
    }

    fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
        (**self).decode(features)
    }
}

/// Injected speech-probability execution port over Audio's validated sample value.
///
/// Each call covers one complete independent 16 kHz waveform and returns exactly
/// `ceil(audio.len() / 512)` finite probabilities in `[0, 1]`, one per consecutive
/// 512-sample frame, including a partial final frame. Empty input returns no probabilities.
/// Implementations reset waveform-specific recurrent/context state between calls; reusing a
/// detector for another physical channel must not carry the preceding channel's state.
/// The Audio view carries no rate, so callers must establish the fixed rate before execution.
pub trait SpeechProbabilityDetector {
    fn probabilities(&mut self, audio: ChannelAudioView<'_>) -> Result<Vec<f32>, String>;
}

#[cfg(test)]
mod tests {
    use super::{Decoded, Token, Word};

    fn word() -> Word {
        Word::new("word".into(), 0.0, 0.1).expect("test word timestamps are valid")
    }

    #[test]
    fn plain_construction_leaves_labels_and_tokens_absent() {
        let decoded = Decoded::new(vec![word()], vec![false, true], 2, 0.0)
            .expect("valid plain decoded construction must succeed");
        assert_eq!(decoded.labels(), None);
        assert_eq!(decoded.tokens(), None);
        assert_eq!(decoded.word_spans(), None);
    }

    #[test]
    fn labeled_construction_preserves_labels_tokens_and_word_spans() {
        let decoded = Decoded::new_with_labels(
            vec![word()],
            vec![false, false],
            2,
            0.0,
            vec![1, 1],
            vec![Token::new(1, 0), Token::new(1, 1)],
            vec![(0, 1)],
        )
        .expect("valid labeled decoded construction must succeed");
        assert_eq!(decoded.labels(), Some(&[1, 1][..]));
        assert_eq!(
            decoded.tokens(),
            Some(&[Token::new(1, 0), Token::new(1, 1)][..])
        );
        assert_eq!(decoded.word_spans(), Some(&[(0, 1)][..]));
    }

    #[test]
    fn labeled_construction_refuses_a_wrong_labels_length() {
        assert!(
            Decoded::new_with_labels(
                vec![word()],
                vec![false, false],
                2,
                0.0,
                vec![1],
                Vec::new(),
                Vec::new(),
            )
            .is_err()
        );
    }

    #[test]
    fn labeled_construction_refuses_a_token_frame_outside_the_window() {
        assert!(
            Decoded::new_with_labels(
                vec![word()],
                vec![false, false],
                2,
                0.0,
                vec![1, 1],
                vec![Token::new(1, 2)],
                Vec::new(),
            )
            .is_err()
        );
    }

    #[test]
    fn labeled_construction_refuses_a_word_span_outside_the_token_list() {
        assert!(
            Decoded::new_with_labels(
                vec![word()],
                vec![false, false],
                2,
                0.0,
                vec![1, 1],
                vec![Token::new(1, 0)],
                vec![(0, 1)],
            )
            .is_err(),
            "a word span whose end exceeds the token list must refuse"
        );
        assert!(
            Decoded::new_with_labels(
                vec![word()],
                vec![false, false],
                2,
                0.0,
                vec![1, 1],
                vec![Token::new(1, 0), Token::new(1, 1)],
                vec![(1, 0)],
            )
            .is_err(),
            "a word span whose end precedes its start must refuse"
        );
    }
}
