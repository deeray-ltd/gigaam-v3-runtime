// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Bounded line and chronological turn replay over finalized transcript records.

use crate::{OriginalChannel, StoredWords, TranscriptWord, TurnGap, WordCursor};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{Read, Seek};

/// A line's time span and the position from which its text can be replayed.
pub struct StoredLine {
    channel: usize,
    start: f32,
    end: f32,
    words: usize,
    first: WordCursor,
}

impl StoredLine {
    pub const fn channel(&self) -> usize {
        self.channel
    }

    pub const fn start(&self) -> f32 {
        self.start
    }

    pub const fn end(&self) -> f32 {
        self.end
    }

    /// Replays this line one word at a time without collecting its text.
    pub fn words<F: Read + Seek>(
        self,
        store: &mut StoredWords<F>,
    ) -> impl Iterator<Item = Result<TranscriptWord, String>> + '_ {
        let mut cursor = self.first;
        let mut remaining = self.words;
        std::iter::from_fn(move || {
            if remaining == 0 {
                return None;
            }
            match store.next(&mut cursor) {
                Ok(Some(word)) => {
                    remaining -= 1;
                    Some(Ok(word))
                }
                Ok(None) => {
                    remaining = 0;
                    Some(Err(
                        "transcript line ended before its recorded word count".into()
                    ))
                }
                Err(error) => {
                    remaining = 0;
                    Some(Err(error))
                }
            }
        })
    }
}

/// Finds consecutive lines on one physical channel.
pub struct ChannelLines {
    channel: usize,
    next: WordCursor,
    gap: TurnGap,
}

impl ChannelLines {
    pub fn new(
        channel: OriginalChannel,
        words: &StoredWords<impl Read + Seek>,
        gap: TurnGap,
    ) -> Result<Self, String> {
        let channel = channel.index();
        Ok(Self {
            channel,
            next: words.cursor(channel)?,
            gap,
        })
    }

    /// Measures the next line's maximum end and saves its start cursor for replay.
    pub fn next(
        &mut self,
        words: &mut StoredWords<impl Read + Seek>,
    ) -> Result<Option<StoredLine>, String> {
        let first = self.next.clone();
        let Some(word) = words.next(&mut self.next)? else {
            return Ok(None);
        };
        let mut line = StoredLine {
            channel: self.channel,
            start: word.start(),
            end: word.end(),
            words: 1,
            first,
        };
        let mut previous_end = word.end();
        loop {
            let checkpoint = self.next.clone();
            let Some(word) = words.next(&mut self.next)? else {
                break;
            };
            if word.start() - previous_end >= self.gap.seconds() {
                self.next = checkpoint;
                break;
            }
            previous_end = word.end();
            line.end = line.end.max(previous_end);
            line.words = line
                .words
                .checked_add(1)
                .ok_or_else(|| "transcript line word count overflow".to_owned())?;
        }
        Ok(Some(line))
    }
}

struct PendingTurn {
    line: StoredLine,
    scan: ChannelLines,
}

impl PartialEq for PendingTurn {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for PendingTurn {}

impl PartialOrd for PendingTurn {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingTurn {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .line
            .start
            .total_cmp(&self.line.start)
            .then(other.line.channel.cmp(&self.line.channel))
    }
}

/// Merges channel lines by start time then original channel, with one pending line per channel.
pub struct TurnLines {
    heads: BinaryHeap<PendingTurn>,
}

impl TurnLines {
    pub fn new(
        channels: &[OriginalChannel],
        words: &mut StoredWords<impl Read + Seek>,
        gap: TurnGap,
    ) -> Result<Self, String> {
        let mut heads = BinaryHeap::with_capacity(channels.len());
        for &channel in channels {
            let mut scan = ChannelLines::new(channel, words, gap)?;
            if let Some(line) = scan.next(words)? {
                heads.push(PendingTurn { line, scan });
            }
        }
        Ok(Self { heads })
    }

    pub fn next(
        &mut self,
        words: &mut StoredWords<impl Read + Seek>,
    ) -> Result<Option<StoredLine>, String> {
        let Some(PendingTurn { line, mut scan }) = self.heads.pop() else {
            return Ok(None);
        };
        if let Some(next) = scan.next(words)? {
            self.heads.push(PendingTurn { line: next, scan });
        }
        Ok(Some(line))
    }
}
