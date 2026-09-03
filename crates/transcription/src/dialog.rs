// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Revision-aware reconstruction of a multi-channel dialogue line.

use crate::contracts::{
    BackchannelMark, ChannelSnapshot, DialogTurn, DialogTurnData, SnapshotWord, TurnsPatch,
    WordFinality, WordStability,
};
use crate::turns::TurnGap;

const APPROXIMATE_TIMESTAMP_SECONDS: f32 = 0.06;

/// A validated positive duration for identifying short overlapping backchannels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackchannelDuration(f32);

impl BackchannelDuration {
    /// Creates a finite positive maximum backchannel duration in seconds.
    pub fn new(seconds: f32) -> Result<Self, String> {
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err("backchannel duration must be finite and positive".into());
        }
        Ok(Self(seconds))
    }

    pub const fn seconds(self) -> f32 {
        self.0
    }
}

/// Whether dialogue reconstruction marks short turns overlapped by a longer peer turn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BackchannelPolicy {
    Disabled,
    MarkShorterThan(BackchannelDuration),
}

enum MergeMode {
    Incremental,
    Terminal,
}

/// Reconstructs an ordered dialogue line and emits minimal tail patches.
pub struct DialogMerger {
    turn_gap: TurnGap,
    backchannel_policy: BackchannelPolicy,
    last: Vec<DialogTurn>,
    final_frontier: f32,
    stable_frontier: f32,
}

impl DialogMerger {
    /// Creates a dialogue merger with backchannel marking disabled.
    pub fn new(turn_gap: TurnGap) -> Self {
        Self {
            turn_gap,
            backchannel_policy: BackchannelPolicy::Disabled,
            last: Vec::new(),
            final_frontier: f32::NEG_INFINITY,
            stable_frontier: f32::NEG_INFINITY,
        }
    }

    /// Returns this merger configured with an explicit backchannel policy.
    pub fn with_backchannel_policy(mut self, backchannel_policy: BackchannelPolicy) -> Self {
        self.backchannel_policy = backchannel_policy;
        self
    }

    /// Updates the dialogue state from immutable per-channel streaming snapshots.
    pub fn update(&mut self, channels: &[ChannelSnapshot]) -> Result<Option<TurnsPatch>, String> {
        self.update_with(channels, MergeMode::Incremental)
    }

    /// Performs terminal dialogue reconstruction after all active sessions have flushed.
    pub fn finalize(&mut self, channels: &[ChannelSnapshot]) -> Result<Option<TurnsPatch>, String> {
        self.update_with(channels, MergeMode::Terminal)
    }

    /// Returns the current complete dialogue line.
    pub fn dialog(&self) -> &[DialogTurn] {
        &self.last
    }

    fn update_with(
        &mut self,
        channels: &[ChannelSnapshot],
        mode: MergeMode,
    ) -> Result<Option<TurnsPatch>, String> {
        validate_channels(channels)?;
        self.advance_frontiers(channels, mode);

        let mut dialogue = Vec::new();
        for channel in channels {
            dialogue.extend(segment(
                channel,
                self.turn_gap,
                self.final_frontier,
                self.stable_frontier,
            )?);
        }
        dialogue.sort_by(|left, right| {
            left.start()
                .total_cmp(&right.start())
                .then(left.channel().cmp(&right.channel()))
                .then(left.index().cmp(&right.index()))
        });
        mark_backchannels(&mut dialogue, self.backchannel_policy);

        let revise_from = first_difference(&self.last, &dialogue);
        if revise_from == dialogue.len() && dialogue.len() == self.last.len() {
            return Ok(None);
        }
        let patch = TurnsPatch::new(
            revise_from,
            dialogue[revise_from..].to_vec(),
            self.final_frontier,
        )?;
        self.last = dialogue;
        Ok(Some(patch))
    }

    fn advance_frontiers(&mut self, channels: &[ChannelSnapshot], mode: MergeMode) {
        let Some(first_channel) = channels.first() else {
            return;
        };
        let committed_fuzz = match mode {
            MergeMode::Incremental => APPROXIMATE_TIMESTAMP_SECONDS,
            MergeMode::Terminal => 0.0,
        };
        let final_candidate = channels.iter().skip(1).fold(
            first_channel.cut_time() - committed_fuzz,
            |minimum, channel| minimum.min(channel.cut_time() - committed_fuzz),
        );
        let stable_candidate = channels
            .iter()
            .skip(1)
            .fold(first_channel.stable_frontier(), |minimum, channel| {
                minimum.min(channel.stable_frontier())
            });
        self.final_frontier = self.final_frontier.max(final_candidate);
        self.stable_frontier = self.stable_frontier.max(stable_candidate);
    }
}

fn segment(
    channel: &ChannelSnapshot,
    gap: TurnGap,
    final_frontier: f32,
    stable_frontier: f32,
) -> Result<Vec<DialogTurn>, String> {
    let mut turns = Vec::new();
    let mut current: Vec<&SnapshotWord> = Vec::new();
    for word in channel.words() {
        if let Some(last) = current.last()
            && word.word().start() - last.word().end() >= gap.seconds()
        {
            turns.push(finish_turn(
                channel.channel(),
                turns.len(),
                std::mem::take(&mut current),
                final_frontier,
                stable_frontier,
            )?);
        }
        current.push(word);
    }
    if !current.is_empty() {
        turns.push(finish_turn(
            channel.channel(),
            turns.len(),
            current,
            final_frontier,
            stable_frontier,
        )?);
    }
    Ok(turns)
}

fn finish_turn(
    channel: usize,
    index: usize,
    words: Vec<&SnapshotWord>,
    final_frontier: f32,
    stable_frontier: f32,
) -> Result<DialogTurn, String> {
    let Some(first) = words.first() else {
        return Err("dialogue turn construction requires at least one word".into());
    };
    let start = first.word().start();
    let end = words
        .iter()
        .map(|word| word.word().end())
        .fold(start, f32::max);
    let text = words
        .iter()
        .map(|word| word.word().text())
        .collect::<Vec<_>>()
        .join(" ");
    let all_final = words
        .iter()
        .all(|word| matches!(word.finality(), WordFinality::Final));
    let all_stable = words
        .iter()
        .all(|word| matches!(word.stability(), WordStability::Stable));
    let finality = if all_final && end <= final_frontier {
        WordFinality::Final
    } else {
        WordFinality::Open
    };
    let stability = match finality {
        WordFinality::Final => WordStability::Stable,
        WordFinality::Open if all_stable && end <= stable_frontier => WordStability::Stable,
        WordFinality::Open => WordStability::Revisable,
    };
    DialogTurn::new(DialogTurnData {
        channel,
        index,
        start,
        end,
        text,
        stability,
        finality,
        backchannel: BackchannelMark::No,
    })
}

fn mark_backchannels(dialogue: &mut [DialogTurn], policy: BackchannelPolicy) {
    let BackchannelPolicy::MarkShorterThan(maximum_duration) = policy else {
        return;
    };

    let marks: Vec<BackchannelMark> = dialogue
        .iter()
        .enumerate()
        .map(|(index, turn)| {
            let duration = turn.end() - turn.start();
            let overlaps_longer_peer = dialogue.iter().enumerate().any(|(peer_index, peer)| {
                peer_index != index
                    && peer.channel() != turn.channel()
                    && peer.start() <= turn.start()
                    && peer.end() >= turn.end()
                    && peer.end() - peer.start() > duration
            });
            if duration <= maximum_duration.seconds() && overlaps_longer_peer {
                BackchannelMark::Yes
            } else {
                BackchannelMark::No
            }
        })
        .collect();
    for (turn, mark) in dialogue.iter_mut().zip(marks) {
        *turn = turn.clone().with_backchannel(mark);
    }
}

fn first_difference(previous: &[DialogTurn], current: &[DialogTurn]) -> usize {
    let shared = previous.len().min(current.len());
    match (0..shared).find(|&index| !same_turn(&previous[index], &current[index])) {
        Some(index) => index,
        None => shared,
    }
}

fn same_turn(left: &DialogTurn, right: &DialogTurn) -> bool {
    left.channel() == right.channel()
        && left.index() == right.index()
        && left.text() == right.text()
        && approximately_equal(left.start(), right.start())
        && approximately_equal(left.end(), right.end())
        && left.stability() == right.stability()
        && left.finality() == right.finality()
        && left.backchannel() == right.backchannel()
}

fn approximately_equal(left: f32, right: f32) -> bool {
    (left - right).abs() <= APPROXIMATE_TIMESTAMP_SECONDS
}

fn validate_channels(channels: &[ChannelSnapshot]) -> Result<(), String> {
    for (index, channel) in channels.iter().enumerate() {
        if channels[..index]
            .iter()
            .any(|earlier| earlier.channel() == channel.channel())
        {
            return Err("dialogue channel identities must be unique".into());
        }
        if channel
            .words()
            .windows(2)
            .any(|pair| pair[1].word().start() < pair[0].word().start())
        {
            return Err("dialogue channel words must be ordered by start time".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BackchannelDuration, BackchannelPolicy, DialogMerger};
    use crate::contracts::{
        BackchannelMark, ChannelSnapshot, SnapshotWord, TranscriptWord, WordFinality, WordStability,
    };
    use crate::turns::TurnGap;

    fn word(
        text: &str,
        start: f32,
        end: f32,
        finality: WordFinality,
        stability: WordStability,
    ) -> SnapshotWord {
        SnapshotWord::new(
            TranscriptWord::new(text.into(), start, end)
                .expect("test word timestamps are finite and ordered"),
            finality,
            stability,
        )
    }

    fn channel(
        index: usize,
        words: Vec<SnapshotWord>,
        cut_time: f32,
        stable_frontier: f32,
    ) -> ChannelSnapshot {
        ChannelSnapshot::new(index, words, cut_time, stable_frontier)
            .expect("test snapshot frontiers are valid")
    }

    fn merger() -> DialogMerger {
        DialogMerger::new(TurnGap::new(1.0).expect("test turn gap is valid"))
    }

    #[test]
    fn mono_input_splits_into_final_turns() {
        let mut merger = merger();
        let patch = merger
            .update(&[channel(
                0,
                vec![
                    word("one", 0.0, 0.3, WordFinality::Final, WordStability::Stable),
                    word("two", 0.4, 0.7, WordFinality::Final, WordStability::Stable),
                    word(
                        "three",
                        5.0,
                        5.3,
                        WordFinality::Final,
                        WordStability::Stable,
                    ),
                ],
                6.0,
                6.0,
            )])
            .expect("valid snapshot permits dialogue reconstruction")
            .expect("initial dialogue input changes the line");
        assert_eq!(patch.turns().len(), 2);
        assert_eq!(patch.turns()[0].text(), "one two");
        assert_eq!(patch.turns()[1].text(), "three");
        assert!(matches!(patch.turns()[0].finality(), WordFinality::Final));
        assert!(matches!(patch.turns()[1].finality(), WordFinality::Final));
    }

    #[test]
    fn turns_order_by_start_then_original_channel_identity() {
        let mut merger = merger();
        let patch = merger
            .update(&[
                channel(
                    0,
                    vec![word(
                        "hello",
                        0.3,
                        2.6,
                        WordFinality::Final,
                        WordStability::Stable,
                    )],
                    3.0,
                    3.0,
                ),
                channel(
                    1,
                    vec![word(
                        "yes",
                        3.0,
                        4.1,
                        WordFinality::Final,
                        WordStability::Stable,
                    )],
                    5.0,
                    5.0,
                ),
            ])
            .expect("valid snapshots permit dialogue reconstruction")
            .expect("initial dialogue input changes the line");
        assert_eq!(
            patch
                .turns()
                .iter()
                .map(|turn| (turn.channel(), turn.text()))
                .collect::<Vec<_>>(),
            vec![(0, "hello"), (1, "yes")]
        );
    }

    #[test]
    fn finality_advances_only_after_committed_frontier() {
        let mut merger = merger();
        let first = merger
            .update(&[channel(
                0,
                vec![word(
                    "hello",
                    6.0,
                    6.4,
                    WordFinality::Open,
                    WordStability::Revisable,
                )],
                5.5,
                6.5,
            )])
            .expect("valid snapshot permits dialogue reconstruction")
            .expect("initial dialogue input changes the line");
        assert!(matches!(first.turns()[0].finality(), WordFinality::Open));

        let second = merger
            .update(&[channel(
                0,
                vec![word(
                    "hello",
                    6.0,
                    6.4,
                    WordFinality::Final,
                    WordStability::Stable,
                )],
                7.0,
                8.0,
            )])
            .expect("valid snapshot permits dialogue reconstruction")
            .expect("finality advancement changes the line");
        assert!(matches!(second.turns()[0].finality(), WordFinality::Final));
    }

    #[test]
    fn late_earlier_onset_revises_from_the_first_position() {
        let mut merger = merger();
        merger
            .update(&[
                channel(0, vec![], 6.0, 6.0),
                channel(
                    1,
                    vec![word(
                        "question",
                        6.1,
                        6.8,
                        WordFinality::Open,
                        WordStability::Revisable,
                    )],
                    5.5,
                    6.0,
                ),
            ])
            .expect("valid snapshots permit dialogue reconstruction")
            .expect("initial dialogue input changes the line");
        let patch = merger
            .update(&[
                channel(
                    0,
                    vec![word(
                        "yes",
                        6.0,
                        6.3,
                        WordFinality::Open,
                        WordStability::Revisable,
                    )],
                    5.5,
                    6.0,
                ),
                channel(
                    1,
                    vec![word(
                        "question",
                        6.1,
                        6.8,
                        WordFinality::Open,
                        WordStability::Revisable,
                    )],
                    5.5,
                    6.0,
                ),
            ])
            .expect("valid snapshots permit dialogue reconstruction")
            .expect("late earlier onset changes the line");
        assert_eq!(patch.revise_from(), 0);
        assert_eq!(
            patch
                .turns()
                .iter()
                .map(|turn| (turn.channel(), turn.text()))
                .collect::<Vec<_>>(),
            vec![(0, "yes"), (1, "question")]
        );
        assert!(
            patch
                .turns()
                .iter()
                .all(|turn| matches!(turn.finality(), WordFinality::Open))
        );
    }

    #[test]
    fn final_turn_remains_outside_later_patch() {
        let mut merger = merger();
        merger
            .update(&[
                channel(
                    0,
                    vec![word(
                        "done",
                        1.0,
                        2.0,
                        WordFinality::Final,
                        WordStability::Stable,
                    )],
                    5.0,
                    5.0,
                ),
                channel(1, vec![], 5.0, 5.0),
            ])
            .expect("valid snapshots permit dialogue reconstruction")
            .expect("initial dialogue input changes the line");
        let patch = merger
            .update(&[
                channel(
                    0,
                    vec![word(
                        "done",
                        1.0,
                        2.0,
                        WordFinality::Final,
                        WordStability::Stable,
                    )],
                    5.0,
                    5.0,
                ),
                channel(
                    1,
                    vec![word(
                        "new",
                        6.0,
                        6.5,
                        WordFinality::Open,
                        WordStability::Revisable,
                    )],
                    5.0,
                    6.0,
                ),
            ])
            .expect("valid snapshots permit dialogue reconstruction")
            .expect("new open turn changes the line");
        assert_eq!(patch.revise_from(), 1);
    }

    #[test]
    fn frontiers_are_monotonic_and_unchanged_input_emits_no_patch() {
        let mut merger = merger();
        let input = || {
            channel(
                0,
                vec![word(
                    "a",
                    1.0,
                    1.5,
                    WordFinality::Final,
                    WordStability::Stable,
                )],
                10.0,
                10.0,
            )
        };
        assert!(
            merger
                .update(&[input()])
                .expect("valid snapshot permits dialogue reconstruction")
                .is_some()
        );
        let high_frontier = merger.final_frontier;
        assert!(
            merger
                .update(&[channel(
                    0,
                    vec![word(
                        "a",
                        1.0,
                        1.5,
                        WordFinality::Final,
                        WordStability::Stable
                    )],
                    3.0,
                    3.0,
                )])
                .expect("valid snapshot permits dialogue reconstruction")
                .is_none()
        );
        assert!(merger.final_frontier >= high_frontier);
    }

    #[test]
    fn backchannel_policy_marks_only_short_overlapped_peer_turns() {
        let policy = BackchannelPolicy::MarkShorterThan(
            BackchannelDuration::new(1.0).expect("test backchannel duration is valid"),
        );
        let mut merger = merger().with_backchannel_policy(policy);
        let patch = merger
            .update(&[
                channel(
                    0,
                    vec![word(
                        "speaking long",
                        0.0,
                        5.0,
                        WordFinality::Final,
                        WordStability::Stable,
                    )],
                    6.0,
                    6.0,
                ),
                channel(
                    1,
                    vec![word(
                        "yes",
                        2.0,
                        2.5,
                        WordFinality::Final,
                        WordStability::Stable,
                    )],
                    6.0,
                    6.0,
                ),
            ])
            .expect("valid snapshots permit dialogue reconstruction")
            .expect("initial dialogue input changes the line");
        assert_eq!(
            patch
                .turns()
                .iter()
                .map(|turn| (turn.channel(), turn.backchannel()))
                .collect::<Vec<_>>(),
            vec![(0, BackchannelMark::No), (1, BackchannelMark::Yes)]
        );
    }

    #[test]
    fn flag_changes_are_visible_and_invalid_snapshots_refuse() {
        let mut merger = merger();
        merger
            .update(&[channel(
                0,
                vec![word(
                    "word",
                    6.0,
                    6.5,
                    WordFinality::Open,
                    WordStability::Revisable,
                )],
                5.0,
                5.0,
            )])
            .expect("valid snapshot permits dialogue reconstruction")
            .expect("initial dialogue input changes the line");
        let changed = merger
            .update(&[channel(
                0,
                vec![word(
                    "word",
                    6.0,
                    6.5,
                    WordFinality::Final,
                    WordStability::Stable,
                )],
                7.0,
                8.0,
            )])
            .expect("valid snapshot permits dialogue reconstruction");
        assert!(changed.is_some());

        assert!(
            merger
                .update(&[channel(0, vec![], 1.0, 1.0), channel(0, vec![], 1.0, 1.0),])
                .is_err()
        );
    }
}
