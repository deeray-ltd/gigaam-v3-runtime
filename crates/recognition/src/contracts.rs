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
    pub(crate) const fn new(id: usize, frame: usize) -> Self {
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
}

impl Decoded {
    pub fn new(
        words: Vec<Word>,
        silence: Vec<bool>,
        output_frames: usize,
        encoder_seconds: f64,
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
        Ok(Self {
            words,
            silence,
            output_frames,
            encoder_seconds,
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
pub trait SpeechProbabilityDetector {
    fn probabilities(&mut self, audio: ChannelAudioView<'_>) -> Result<Vec<f32>, String>;
}
