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

    pub fn into_words(self) -> Vec<TranscriptWord> {
        self.words
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
}
