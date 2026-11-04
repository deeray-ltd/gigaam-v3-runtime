// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Final batch dialogue ordering and turn construction.

use crate::contracts::{ChannelWord, TranscriptWord, Turn};

/// A chronological final transcript belonging to one original audio channel.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelTranscript {
    channel: usize,
    words: Vec<TranscriptWord>,
}

impl ChannelTranscript {
    /// Associates a chronological transcript with its original channel identity.
    pub fn new(channel: usize, words: Vec<TranscriptWord>) -> Result<Self, String> {
        if words
            .windows(2)
            .any(|pair| pair[1].start() < pair[0].start())
        {
            return Err("channel transcript words must be ordered by start time".into());
        }
        Ok(Self { channel, words })
    }

    pub const fn channel(&self) -> usize {
        self.channel
    }

    pub fn words(&self) -> &[TranscriptWord] {
        &self.words
    }

    /// Groups retained words by source window and configured word gap.
    pub fn scored_segments(&self, gap: TurnGap) -> Result<Vec<ScoredSegment>, String> {
        let mut segments = Vec::new();
        let mut first = 0;
        while first < self.words.len() {
            let owner = self.words[first]
                .provenance
                .as_ref()
                .ok_or_else(|| "missing word provenance".to_owned())?;
            let mut end = first + 1;
            while end < self.words.len() {
                let next = self.words[end]
                    .provenance
                    .as_ref()
                    .ok_or_else(|| "missing word provenance".to_owned())?;
                if !std::sync::Arc::ptr_eq(&owner.window, &next.window) {
                    break;
                }
                let previous = self.words[end - 1]
                    .provenance
                    .as_ref()
                    .ok_or_else(|| "missing word provenance".to_owned())?;
                if previous.last >= next.first {
                    return Err("overlapping or out-of-order token spans".into());
                }
                if self.words[end].start() - self.words[end - 1].end() > gap.seconds() {
                    break;
                }
                end += 1;
            }
            let last = self.words[end - 1]
                .provenance
                .as_ref()
                .ok_or_else(|| "missing word provenance".to_owned())?;
            if owner.first > last.last {
                return Err("inconsistent token span".into());
            }
            let tokens = owner
                .window
                .tokens
                .get(owner.first..=last.last)
                .ok_or_else(|| "token span outside window".to_owned())?
                .to_vec();
            let mut sum = 0.0;
            for token in &tokens {
                sum += token
                    .log_probability()
                    .ok_or_else(|| "missing token score".to_owned())?;
            }
            let count =
                u32::try_from(tokens.len()).map_err(|_| "token count overflows".to_owned())?;
            if count == 0 {
                return Err("empty scored token span".into());
            }
            let seek = owner
                .window
                .origin
                .checked_mul(100)
                .ok_or_else(|| "window seek overflows".to_owned())?
                / owner.window.sample_rate;
            segments.push(ScoredSegment {
                start: self.words[first].start(),
                end: self.words[first..end]
                    .iter()
                    .map(TranscriptWord::end)
                    .fold(self.words[first].end(), f32::max),
                text: crate::contracts::words_to_text(&self.words[first..end]),
                tokens,
                seek,
                no_speech_probability: owner.window.no_speech_probability,
                average_log_probability: sum / f64::from(count),
            });
            first = end;
        }
        Ok(segments)
    }

    pub fn into_words(self) -> Vec<TranscriptWord> {
        self.words
    }
}

/// A retained word group with native occurrence scores and source-window diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredSegment {
    start: f32,
    end: f32,
    text: String,
    tokens: Vec<gigaam_recognition::Token>,
    seek: usize,
    no_speech_probability: f64,
    average_log_probability: f64,
}
impl ScoredSegment {
    pub const fn start(&self) -> f32 {
        self.start
    }
    pub const fn end(&self) -> f32 {
        self.end
    }
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn tokens(&self) -> &[gigaam_recognition::Token] {
        &self.tokens
    }
    pub const fn seek(&self) -> usize {
        self.seek
    }
    pub const fn no_speech_probability(&self) -> f64 {
        self.no_speech_probability
    }
    pub const fn average_log_probability(&self) -> f64 {
        self.average_log_probability
    }
}

/// A validated minimum pause that separates final turns within one channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurnGap(f32);

impl TurnGap {
    /// Creates a finite nonnegative turn boundary duration in seconds.
    pub fn new(seconds: f32) -> Result<Self, String> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err("turn gap must be finite and nonnegative".into());
        }
        Ok(Self(seconds))
    }

    pub const fn seconds(self) -> f32 {
        self.0
    }
}

/// Merges final channel transcripts into a deterministic chronological dialogue line.
pub fn merge(channels: &[ChannelTranscript]) -> Result<Vec<ChannelWord>, String> {
    validate_channel_identities(channels)?;
    let mut words: Vec<ChannelWord> = channels
        .iter()
        .flat_map(|channel| {
            channel
                .words()
                .iter()
                .cloned()
                .map(move |word| ChannelWord::new(channel.channel(), word))
        })
        .collect();
    words.sort_by(|left, right| {
        left.word()
            .start()
            .total_cmp(&right.word().start())
            .then(left.channel().cmp(&right.channel()))
    });
    Ok(words)
}

/// Splits every channel transcript into turns and orders the resulting turns chronologically.
pub fn turns(channels: &[ChannelTranscript], gap: TurnGap) -> Result<Vec<Turn>, String> {
    let mut result = channel_segments(channels, gap)?;
    result.sort_by(|left, right| {
        left.start()
            .total_cmp(&right.start())
            .then(left.channel().cmp(&right.channel()))
    });
    Ok(result)
}

/// Splits every channel transcript into channel-grouped final segments.
pub(crate) fn channel_segments(
    channels: &[ChannelTranscript],
    gap: TurnGap,
) -> Result<Vec<Turn>, String> {
    validate_channel_identities(channels)?;
    let mut result = Vec::new();
    for channel in channels {
        let mut current: Vec<ChannelWord> = Vec::new();
        for word in channel.words() {
            if let Some(last) = current.last()
                && word.start() - last.word().end() >= gap.seconds()
            {
                result.push(finish_turn(
                    channel.channel(),
                    std::mem::take(&mut current),
                )?);
            }
            current.push(ChannelWord::new(channel.channel(), word.clone()));
        }
        if !current.is_empty() {
            result.push(finish_turn(channel.channel(), current)?);
        }
    }
    Ok(result)
}

/// Renders one final turn per line using its original channel identity.
pub fn dialog_text(turns: &[Turn]) -> String {
    turns
        .iter()
        .map(|turn| format!("[channel {}] {}", turn.channel(), turn.text()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn finish_turn(channel: usize, words: Vec<ChannelWord>) -> Result<Turn, String> {
    let Some(first) = words.first() else {
        return Err("turn construction requires at least one word".into());
    };
    let start = first.word().start();
    let end = words
        .iter()
        .map(|word| word.word().end())
        .fold(start, f32::max);
    Turn::new(channel, start, end, words)
}

fn validate_channel_identities(channels: &[ChannelTranscript]) -> Result<(), String> {
    for (index, channel) in channels.iter().enumerate() {
        if channels[..index]
            .iter()
            .any(|earlier| earlier.channel() == channel.channel())
        {
            return Err("channel transcript identities must be unique".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ChannelTranscript, TurnGap, dialog_text, merge, turns};
    use crate::contracts::TranscriptWord;

    fn word(text: &str, start: f32, end: f32) -> TranscriptWord {
        TranscriptWord::new(text.into(), start, end)
            .expect("test word timestamps are finite and ordered")
    }

    fn channel(index: usize, words: Vec<TranscriptWord>) -> ChannelTranscript {
        ChannelTranscript::new(index, words).expect("test channel words are ordered by start time")
    }

    fn gap(seconds: f32) -> TurnGap {
        TurnGap::new(seconds).expect("test turn gap is finite and nonnegative")
    }

    #[test]
    fn merge_orders_by_time_then_original_channel_identity() {
        let merged = merge(&[
            channel(0, vec![word("hello", 0.0, 0.4), word("yes", 3.0, 3.2)]),
            channel(1, vec![word("hi", 1.0, 1.5)]),
        ])
        .expect("unique channel identities can be merged");
        assert_eq!(
            merged
                .iter()
                .map(|item| (item.channel(), item.word().text()))
                .collect::<Vec<_>>(),
            vec![(0, "hello"), (1, "hi"), (0, "yes")]
        );
    }

    #[test]
    fn turns_split_on_intra_channel_gap_and_preserve_cross_channel_order() {
        let result = turns(
            &[
                channel(
                    0,
                    vec![
                        word("hello", 0.0, 0.4),
                        word("yes", 0.5, 0.8),
                        word("bye", 3.4, 3.9),
                    ],
                ),
                channel(1, vec![word("hi", 1.0, 1.5)]),
            ],
            gap(1.0),
        )
        .expect("unique ordered channels produce turns");

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].channel(), 0);
        assert_eq!(result[0].text(), "hello yes");
        assert_eq!(result[1].channel(), 1);
        assert_eq!(result[1].text(), "hi");
        assert_eq!(result[2].channel(), 0);
        assert_eq!(result[2].text(), "bye");
        assert_eq!(result[0].end(), 0.8);
        assert_eq!(result[2].start(), 3.4);
    }

    #[test]
    fn overlapping_channels_remain_independent_turns() {
        let result = turns(
            &[
                channel(0, vec![word("speaking", 0.0, 2.0)]),
                channel(1, vec![word("interrupting", 1.0, 1.5)]),
            ],
            gap(1.0),
        )
        .expect("unique ordered channels produce turns");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].channel(), 0);
        assert_eq!(result[1].channel(), 1);
        assert!(result[0].start() < result[1].start());
        assert!(result[0].end() > result[1].start());
    }

    #[test]
    fn turn_end_is_the_maximum_end_of_nonmonotonic_words() {
        let result = turns(
            &[channel(0, vec![word("a", 0.0, 2.0), word("b", 0.5, 1.0)])],
            gap(1.0),
        )
        .expect("one ordered channel produces a turn");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].end(), 2.0);
    }

    #[test]
    fn mono_transcript_splits_into_utterance_turns() {
        let result = turns(
            &[channel(
                0,
                vec![
                    word("one", 0.0, 0.3),
                    word("two", 0.4, 0.7),
                    word("three", 5.0, 5.3),
                ],
            )],
            gap(1.0),
        )
        .expect("one ordered channel produces turns");
        assert_eq!(
            dialog_text(&result),
            "[channel 0] one two\n[channel 0] three"
        );
    }

    #[test]
    fn invalid_channel_inputs_refuse_before_turn_construction() {
        assert!(
            ChannelTranscript::new(0, vec![word("later", 1.0, 1.1), word("early", 0.0, 0.1)])
                .is_err()
        );
        assert!(merge(&[channel(0, vec![]), channel(0, vec![])]).is_err());
        assert!(TurnGap::new(f32::NAN).is_err());
    }
    #[test]
    fn scored_segments_include_internal_boundary_and_split_windows() -> Result<(), String> {
        use crate::contracts::{WindowProvenance, WordProvenance};
        use gigaam_recognition::Token;
        use std::sync::Arc;
        let owner = Arc::new(WindowProvenance {
            origin: 1600,
            end: 3200,
            sample_rate: 16000,
            no_speech_probability: 0.2,
            tokens: vec![
                Token::scored(1, 0, -0.2)?,
                Token::scored(2, 1, -0.4)?,
                Token::scored(3, 2, -0.6)?,
            ],
        });
        let next = Arc::new(WindowProvenance {
            origin: 3200,
            end: 4800,
            sample_rate: 16000,
            no_speech_probability: 0.3,
            tokens: vec![Token::scored(4, 0, -0.8)?],
        });
        let mut a = TranscriptWord::new("a".into(), 0.1, 0.2)?;
        a.provenance = Some(WordProvenance {
            window: Arc::clone(&owner),
            first: 0,
            last: 0,
        });
        let mut b = TranscriptWord::new("b".into(), 0.2, 0.3)?;
        b.provenance = Some(WordProvenance {
            window: owner,
            first: 2,
            last: 2,
        });
        let mut c = TranscriptWord::new("c".into(), 0.3, 0.4)?;
        c.provenance = Some(WordProvenance {
            window: next,
            first: 0,
            last: 0,
        });
        let segments =
            ChannelTranscript::new(0, vec![a, b, c])?.scored_segments(TurnGap::new(1.0)?)?;
        assert_eq!(segments.len(), 2);
        assert_eq!(
            segments[0]
                .tokens()
                .iter()
                .map(Token::id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!((segments[0].average_log_probability() + 0.4).abs() < 1e-12);
        assert_eq!(segments[0].seek(), 10);
        assert_eq!(segments[1].seek(), 20);
        Ok(())
    }
    #[test]
    fn scored_gap_rejects_overlapping_occurrences_and_accepts_disjoint() -> Result<(), String> {
        use crate::contracts::{WindowProvenance, WordProvenance};
        use gigaam_recognition::Token;
        use std::sync::Arc;
        let owner = Arc::new(WindowProvenance {
            origin: 0,
            end: 16000,
            sample_rate: 16000,
            no_speech_probability: 0.0,
            tokens: vec![Token::scored(1, 0, -0.1)?, Token::scored(2, 1, -0.2)?],
        });
        let mut a = TranscriptWord::new("a".into(), 0.0, 0.1)?;
        a.provenance = Some(WordProvenance {
            window: Arc::clone(&owner),
            first: 0,
            last: 0,
        });
        let mut b = TranscriptWord::new("b".into(), 0.5, 0.6)?;
        b.provenance = Some(WordProvenance {
            window: Arc::clone(&owner),
            first: 0,
            last: 0,
        });
        assert!(
            ChannelTranscript::new(0, vec![a.clone(), b.clone()])?
                .scored_segments(TurnGap::new(0.1)?)
                .is_err()
        );
        b.provenance = Some(WordProvenance {
            window: owner,
            first: 1,
            last: 1,
        });
        assert_eq!(
            ChannelTranscript::new(0, vec![a, b])?
                .scored_segments(TurnGap::new(0.1)?)?
                .len(),
            2
        );
        Ok(())
    }
}
