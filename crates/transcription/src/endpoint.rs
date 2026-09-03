// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Endpoint decisions from a silence mask.

use gigaam_primitives::usize_to_f32;

/// Validated silence durations used to close or reset a streaming utterance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndpointRules {
    after_speech_seconds: f32,
    without_speech_seconds: f32,
}

impl EndpointRules {
    /// Creates endpoint rules from finite, nonnegative silence durations.
    pub fn new(after_speech_seconds: f32, without_speech_seconds: f32) -> Result<Self, String> {
        let rules = Self {
            after_speech_seconds,
            without_speech_seconds,
        };
        rules.validate()?;
        Ok(rules)
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        validate_duration(self.after_speech_seconds, "endpoint after-speech silence")?;
        validate_duration(self.without_speech_seconds, "endpoint no-speech silence")?;
        Ok(())
    }

    /// Returns the silence duration that closes an utterance containing speech.
    pub const fn after_speech_seconds(self) -> f32 {
        self.after_speech_seconds
    }

    /// Returns the silence duration that resets an utterance without speech.
    pub const fn without_speech_seconds(self) -> f32 {
        self.without_speech_seconds
    }
}

impl Default for EndpointRules {
    fn default() -> Self {
        Self {
            after_speech_seconds: 1.2,
            without_speech_seconds: 2.4,
        }
    }
}

/// The action selected by the current endpoint state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Endpoint {
    None,
    AfterSpeech,
    NoSpeech,
}

/// Whether the current endpoint buffer contains recognized speech.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeechState {
    Absent,
    Present,
}

/// Counts trailing silent frames among the real-audio prefix of a silence mask.
pub fn trailing_silence(mask: &[bool], real_frames: usize) -> usize {
    mask[..real_frames.min(mask.len())]
        .iter()
        .rev()
        .take_while(|&&silent| silent)
        .count()
}

/// Determines whether trailing silence closes or resets the current utterance.
///
/// `real_frames` excludes any decoder-padding frames. `frames_per_second` must describe the
/// supplied mask and be finite and positive.
pub fn detect(
    rules: EndpointRules,
    mask: &[bool],
    real_frames: usize,
    frames_per_second: f32,
    speech_state: SpeechState,
) -> Result<Endpoint, String> {
    rules.validate()?;
    if !frames_per_second.is_finite() || frames_per_second <= 0.0 {
        return Err("endpoint frame rate must be finite and positive".into());
    }

    let silence_seconds = usize_to_f32(trailing_silence(mask, real_frames)) / frames_per_second;
    match speech_state {
        SpeechState::Present if silence_seconds >= rules.after_speech_seconds => {
            Ok(Endpoint::AfterSpeech)
        }
        SpeechState::Absent if silence_seconds >= rules.without_speech_seconds => {
            Ok(Endpoint::NoSpeech)
        }
        SpeechState::Absent | SpeechState::Present => Ok(Endpoint::None),
    }
}

/// Splits speech into half-open frame ranges separated by at least `minimum_gap` silent frames.
/// Leading and trailing silence are excluded.
pub fn segments(mask: &[bool], minimum_gap: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = None;
    let mut last_speech = 0_usize;

    for (index, &silent) in mask.iter().enumerate() {
        if !silent {
            match start {
                None => start = Some(index),
                Some(existing_start) if index - last_speech > minimum_gap => {
                    ranges.push((existing_start, last_speech + 1));
                    start = Some(index);
                }
                Some(_) => {}
            }
            last_speech = index;
        }
    }

    if let Some(speech_start) = start {
        ranges.push((speech_start, last_speech + 1));
    }
    ranges
}

fn validate_duration(value: f32, name: &str) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be finite and nonnegative"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Endpoint, EndpointRules, SpeechState, detect, segments, trailing_silence};

    fn silence_mask(pattern: &str) -> Vec<bool> {
        pattern.chars().map(|character| character == '.').collect()
    }

    #[test]
    fn trailing_silence_ignores_padding_frames() {
        let mask = silence_mask("xx....");
        assert_eq!(trailing_silence(&mask, 6), 4);
        assert_eq!(trailing_silence(&mask, 3), 1);
        assert_eq!(trailing_silence(&silence_mask("......"), 6), 6);
        assert_eq!(trailing_silence(&silence_mask("....xx"), 6), 0);
    }

    #[test]
    fn endpoint_rules_distinguish_speech_and_no_speech() {
        let rules = EndpointRules::new(1.0, 2.0)
            .expect("finite endpoint durations define a valid rule set");
        let after_speech = silence_mask(&format!("xx{}", ".".repeat(25)));
        assert_eq!(
            detect(
                rules,
                &after_speech,
                after_speech.len(),
                25.0,
                SpeechState::Present,
            )
            .expect("positive frame rate permits endpoint detection"),
            Endpoint::AfterSpeech
        );
        assert_eq!(
            detect(
                rules,
                &after_speech,
                after_speech.len(),
                25.0,
                SpeechState::Absent,
            )
            .expect("positive frame rate permits endpoint detection"),
            Endpoint::None
        );

        let no_speech = silence_mask(&".".repeat(50));
        assert_eq!(
            detect(
                rules,
                &no_speech,
                no_speech.len(),
                25.0,
                SpeechState::Absent,
            )
            .expect("positive frame rate permits endpoint detection"),
            Endpoint::NoSpeech
        );
        assert_eq!(
            detect(rules, &no_speech, 20, 25.0, SpeechState::Absent)
                .expect("positive frame rate permits endpoint detection"),
            Endpoint::None
        );
    }

    #[test]
    fn segmentation_preserves_speech_ranges_and_gap_boundary() {
        assert_eq!(
            segments(&silence_mask("..xxx...x.xx...."), 3),
            vec![(2, 5), (8, 12)]
        );
        assert_eq!(segments(&silence_mask("..xxx..xx"), 3), vec![(2, 9)]);
        assert!(segments(&silence_mask("......"), 3).is_empty());
    }

    #[test]
    fn invalid_durations_and_frame_rates_refuse() {
        assert!(EndpointRules::new(f32::NAN, 1.0).is_err());
        let rules = EndpointRules::default();
        assert!(detect(rules, &[], 0, 0.0, SpeechState::Absent).is_err());
    }
}
