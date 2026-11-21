// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Request-local transcript storage with forward replay by physical channel.

use crate::contracts::{TranscriptWord, WindowProvenance, WordProvenance};
use gigaam_primitives::usize_to_u64_checked;
use gigaam_recognition::Token;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::sync::Arc;

const HEADER_BYTES: usize = 8;
const TOKEN_BYTES: usize = 25;
const NO_WINDOW: u8 = 0;
const SAME_WINDOW: u8 = 1;
const NEW_WINDOW: u8 = 2;

/// Writes finalized transcript words to request-owned seekable storage.
pub struct WordRecords<F> {
    file: F,
    channels: Vec<Option<Range<u64>>>,
    last_channel: Option<usize>,
    last_start: f32,
    window: Option<Arc<WindowProvenance>>,
    end: u64,
    record: Vec<u8>,
    largest_record: usize,
    write_failed: bool,
}

impl<F: Read + Write + Seek> WordRecords<F> {
    /// Begins recording at the supplied file's current position.
    pub fn new(mut file: F, channel_count: usize) -> Result<Self, String> {
        if channel_count == 0 {
            return Err("word records require at least one physical channel".into());
        }
        let mut channels = Vec::new();
        channels
            .try_reserve_exact(channel_count)
            .map_err(|error| error.to_string())?;
        channels.resize(channel_count, None);
        let end = file.stream_position().map_err(record_io_error)?;
        Ok(Self {
            file,
            channels,
            last_channel: None,
            last_start: 0.0,
            window: None,
            end,
            record: Vec::new(),
            largest_record: 0,
            write_failed: false,
        })
    }

    /// Appends a word in ascending physical-channel and per-channel time order.
    pub fn push(&mut self, channel: usize, word: TranscriptWord) -> Result<(), String> {
        if self.write_failed {
            return Err("word records cannot continue after a write failure".into());
        }
        if channel >= self.channels.len() {
            return Err("word record channel is outside the physical channel count".into());
        }
        let same_channel = self.last_channel == Some(channel);
        if self.last_channel.is_some_and(|previous| previous > channel)
            || (same_channel && word.start() < self.last_start)
        {
            return Err("word records must be ordered by physical channel and start time".into());
        }

        self.record.clear();
        self.record.resize(HEADER_BYTES, 0);
        encode_word(
            &mut self.record,
            &word,
            self.window.as_ref().filter(|_| same_channel),
        )?;
        let payload = &self.record[HEADER_BYTES..];
        let payload_len = payload.len();
        let length = usize_to_u64_checked(payload_len).map_err(|error| error.to_string())?;
        let next = self
            .end
            .checked_add(8)
            .and_then(|offset| offset.checked_add(length))
            .ok_or_else(|| "word record file offset overflow".to_owned())?;
        self.record[..HEADER_BYTES].copy_from_slice(&length.to_le_bytes());
        if let Err(error) = self.file.write_all(&self.record) {
            self.write_failed = true;
            return Err(record_io_error(error));
        }

        let range = self.channels[channel].get_or_insert(self.end..self.end);
        range.end = next;
        self.end = next;
        self.last_channel = Some(channel);
        self.last_start = word.start();
        self.window = word.provenance.map(|provenance| provenance.window);
        self.largest_record = self.largest_record.max(payload_len);
        Ok(())
    }

    /// Flushes the words and prepares the file for replay.
    pub fn finish(mut self) -> Result<StoredWords<F>, String> {
        if self.write_failed {
            return Err("word records cannot finish after a write failure".into());
        }
        self.file.flush().map_err(record_io_error)?;
        Ok(StoredWords {
            file: self.file,
            identity: Arc::new(()),
            channels: self.channels,
            record: self.record,
            largest_record: self.largest_record,
            control: None,
        })
    }
}

/// Replays finalized words from one request across independent channel cursors.
pub struct StoredWords<F> {
    file: F,
    identity: Arc<()>,
    channels: Vec<Option<Range<u64>>>,
    record: Vec<u8>,
    largest_record: usize,
    control: Option<gigaam_recognition::ExecutionControl>,
}

/// A forward position within one physical channel of the finished transcript.
#[derive(Clone)]
pub struct WordCursor {
    identity: Arc<()>,
    range: Range<u64>,
    last_start: f32,
    window: Option<Arc<WindowProvenance>>,
}

impl<F: Read + Seek> StoredWords<F> {
    /// Carries request cancellation through the projection passes over retained words.
    pub fn with_control(mut self, control: gigaam_recognition::ExecutionControl) -> Self {
        self.control = Some(control);
        self
    }

    /// Starts an independent replay of the physical channel, which may be empty.
    pub fn cursor(&self, channel: usize) -> Result<WordCursor, String> {
        let range = self.channels.get(channel).ok_or_else(|| {
            "word cursor channel is outside the physical channel count".to_owned()
        })?;
        Ok(WordCursor {
            identity: Arc::clone(&self.identity),
            range: range.clone().unwrap_or(0..0),
            last_start: 0.0,
            window: None,
        })
    }

    /// Reads the next word for a cursor created by this store.
    pub fn next(&mut self, cursor: &mut WordCursor) -> Result<Option<TranscriptWord>, String> {
        if let Some(control) = &self.control {
            crate::contracts::checkpoint(control)?;
        }
        if !Arc::ptr_eq(&self.identity, &cursor.identity) {
            return Err("word cursor belongs to another store".into());
        }
        if cursor.range.is_empty() {
            cursor.window = None;
            return Ok(None);
        }
        let payload_start = cursor
            .range
            .start
            .checked_add(8)
            .filter(|offset| *offset <= cursor.range.end)
            .ok_or_else(|| "incomplete word record header".to_owned())?;
        self.file
            .seek(SeekFrom::Start(cursor.range.start))
            .map_err(record_io_error)?;
        let mut header = [0; HEADER_BYTES];
        self.file.read_exact(&mut header).map_err(record_io_error)?;
        let length = u64::from_le_bytes(header);
        let end = payload_start
            .checked_add(length)
            .filter(|offset| *offset <= cursor.range.end)
            .ok_or_else(|| "word record exceeds its channel range".to_owned())?;
        let length = usize::try_from(length).map_err(|error| error.to_string())?;
        // The writer's independently tracked maximum limits allocation even
        // if a corrupt header claims the rest of a long channel.
        if length > self.largest_record || length == 0 {
            return Err("invalid word record length".into());
        }
        self.record.clear();
        self.record
            .try_reserve_exact(length)
            .map_err(|error| error.to_string())?;
        self.record.resize(length, 0);
        self.file
            .read_exact(&mut self.record)
            .map_err(record_io_error)?;

        let word = decode_word(&self.record, cursor.window.as_ref())?;
        if word.start() < cursor.last_start {
            return Err("word records are out of time order".into());
        }
        cursor.range.start = end;
        cursor.last_start = word.start();
        cursor.window = word
            .provenance
            .as_ref()
            .map(|value| Arc::clone(&value.window));
        Ok(Some(word))
    }
}

// A record stores exact timestamp bits, text bytes, and optional token span.
// Only the first word in a contiguous provenance run includes the window.
fn encode_word(
    record: &mut Vec<u8>,
    word: &TranscriptWord,
    previous: Option<&Arc<WindowProvenance>>,
) -> Result<(), String> {
    let window_kind = match &word.provenance {
        None => NO_WINDOW,
        Some(value) if previous.is_some_and(|last| Arc::ptr_eq(last, &value.window)) => SAME_WINDOW,
        Some(_) => NEW_WINDOW,
    };
    record.push(window_kind);
    record.extend_from_slice(&word.start().to_le_bytes());
    record.extend_from_slice(&word.end().to_le_bytes());
    encode_count(record, word.text().len())?;
    record.extend_from_slice(word.text().as_bytes());

    if let Some(provenance) = &word.provenance {
        let window = &provenance.window;
        validate_span(window, provenance.first, provenance.last)?;
        encode_count(record, provenance.first)?;
        encode_count(record, provenance.last)?;
        if window_kind == NEW_WINDOW {
            validate_window(
                window.origin,
                window.end,
                window.sample_rate,
                window.no_speech_probability,
            )?;
            encode_count(record, window.origin)?;
            encode_count(record, window.end)?;
            encode_count(record, window.sample_rate)?;
            record.extend_from_slice(&window.no_speech_probability.to_le_bytes());
            encode_count(record, window.tokens.len())?;
            for token in &window.tokens {
                encode_count(record, token.id())?;
                encode_count(record, token.frame())?;
                match token.log_probability() {
                    Some(score) => {
                        record.push(1);
                        record.extend_from_slice(&score.to_le_bytes());
                    }
                    None => {
                        record.push(0);
                        record.extend_from_slice(&0.0_f64.to_le_bytes());
                    }
                }
            }
        }
    }
    Ok(())
}

fn encode_count(record: &mut Vec<u8>, value: usize) -> Result<(), String> {
    let value = usize_to_u64_checked(value).map_err(|error| error.to_string())?;
    record.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn decode_word(
    bytes: &[u8],
    previous: Option<&Arc<WindowProvenance>>,
) -> Result<TranscriptWord, String> {
    let mut record = RecordDecoder::new(bytes);
    let [window_kind] = record.fixed()?;
    let start = f32::from_le_bytes(record.fixed()?);
    let end = f32::from_le_bytes(record.fixed()?);
    let text_length = record.count()?;
    let text = std::str::from_utf8(record.take(text_length)?).map_err(|error| error.to_string())?;
    let mut word = TranscriptWord::new(text.into(), start, end)?;
    match window_kind {
        NO_WINDOW => {}
        SAME_WINDOW | NEW_WINDOW => {
            let first = record.count()?;
            let last = record.count()?;
            let window = if window_kind == SAME_WINDOW {
                previous
                    .cloned()
                    .ok_or_else(|| "word record refers to an absent provenance window".to_owned())?
            } else {
                Arc::new(decode_window(&mut record)?)
            };
            validate_span(&window, first, last)?;
            word.provenance = Some(WordProvenance {
                window,
                first,
                last,
            });
        }
        _ => return Err("invalid word record provenance tag".into()),
    }
    if !record.remaining.is_empty() {
        return Err("unexpected trailing bytes in word record".into());
    }
    Ok(word)
}

fn decode_window(record: &mut RecordDecoder<'_>) -> Result<WindowProvenance, String> {
    let origin = record.count()?;
    let end = record.count()?;
    let sample_rate = record.count()?;
    let no_speech_probability = f64::from_le_bytes(record.fixed()?);
    validate_window(origin, end, sample_rate, no_speech_probability)?;
    let count = record.count()?;
    if count > record.remaining.len() / TOKEN_BYTES {
        return Err("word record token count exceeds its payload".into());
    }
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(count)
        .map_err(|error| error.to_string())?;
    for _ in 0..count {
        let id = record.count()?;
        let frame = record.count()?;
        let [scored] = record.fixed()?;
        let score = f64::from_le_bytes(record.fixed()?);
        let token = match scored {
            0 if score.to_bits() == 0 => Token::new(id, frame),
            1 => Token::scored(id, frame, score)?,
            _ => return Err("invalid word record token score tag".into()),
        };
        tokens.push(token);
    }
    Ok(WindowProvenance {
        origin,
        end,
        sample_rate,
        no_speech_probability,
        tokens,
    })
}

fn validate_window(
    origin: usize,
    end: usize,
    sample_rate: usize,
    no_speech: f64,
) -> Result<(), String> {
    if origin > end || sample_rate == 0 || !(0.0..=1.0).contains(&no_speech) {
        return Err("invalid word record provenance window".into());
    }
    Ok(())
}

fn validate_span(window: &WindowProvenance, first: usize, last: usize) -> Result<(), String> {
    if first > last || last >= window.tokens.len() {
        return Err("word record token span is outside its provenance window".into());
    }
    Ok(())
}

struct RecordDecoder<'a> {
    remaining: &'a [u8],
}

impl<'a> RecordDecoder<'a> {
    fn new(remaining: &'a [u8]) -> Self {
        Self { remaining }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], String> {
        let (value, rest) = self
            .remaining
            .split_at_checked(length)
            .ok_or_else(|| "incomplete word record payload".to_owned())?;
        self.remaining = rest;
        Ok(value)
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let (value, rest) = self
            .remaining
            .split_first_chunk::<N>()
            .ok_or_else(|| "incomplete word record field".to_owned())?;
        self.remaining = rest;
        Ok(*value)
    }

    fn count(&mut self) -> Result<usize, String> {
        usize::try_from(u64::from_le_bytes(self.fixed()?)).map_err(|error| error.to_string())
    }
}

fn record_io_error(error: std::io::Error) -> String {
    format!("word record I/O: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn word(
        text: &str,
        time: f32,
        provenance: Option<(Arc<WindowProvenance>, usize)>,
    ) -> Result<TranscriptWord, String> {
        let mut word = TranscriptWord::new(text.to_owned(), time, time + 0.125)?;
        word.provenance = provenance.map(|(window, index)| WordProvenance {
            window,
            first: index,
            last: index,
        });
        Ok(word)
    }

    #[test]
    fn cursors_replay_physical_channels_and_preserve_provenance_runs() -> Result<(), String> {
        let window = Arc::new(WindowProvenance {
            origin: 6,
            end: 22,
            sample_rate: 16_000,
            no_speech_probability: 0.123_456_789_123_456,
            tokens: vec![
                Token::scored(11, 1, -0.123_456_789_123_456)?,
                Token::new(12, 2),
            ],
        });
        let first = word("слово", -0.0, Some((window.clone(), 0)))?;
        let second = word("other", 0.125, Some((window.clone(), 1)))?;
        let unscored = word("plain", 0.25, None)?;
        let new_run = word("another", 0.375, Some((window.clone(), 0)))?;
        let other_channel = word("one", 0.0, Some((window, 0)))?;

        let mut file = WordRecords::new(Cursor::new(Vec::new()), 4)?;
        file.push(0, first.clone())?;
        file.push(0, second.clone())?;
        file.push(0, unscored.clone())?;
        file.push(0, new_run.clone())?;
        file.push(2, other_channel.clone())?;
        let mut words = file.finish()?;
        let mut left = words.cursor(0)?;
        let mut right = words.cursor(2)?;
        let mut repeated = words.cursor(0)?;
        assert!(words.next(&mut words.cursor(1)?)?.is_none());
        assert!(words.next(&mut words.cursor(3)?)?.is_none());
        assert!(words.cursor(4).is_err());
        let another = WordRecords::new(Cursor::new(Vec::new()), 4)?.finish()?;
        assert!(words.next(&mut another.cursor(0)?).is_err());

        let ans = words.next(&mut left)?.ok_or("missing first word")?;
        assert_eq!(ans, first);
        assert_eq!(ans.start().to_bits(), first.start().to_bits());
        assert_eq!(words.next(&mut right)?, Some(other_channel));
        assert_eq!(words.next(&mut repeated)?, Some(first));
        let next = words.next(&mut left)?.ok_or("missing second word")?;
        assert_eq!(next, second);
        let a = ans.provenance.ok_or("missing first provenance")?.window;
        let b = next.provenance.ok_or("missing second provenance")?.window;
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(words.next(&mut repeated)?, Some(second));
        assert_eq!(words.next(&mut left)?, Some(unscored));
        let later = words.next(&mut left)?.ok_or("missing last word")?;
        assert_eq!(later, new_run);
        let later = later.provenance.ok_or("missing later provenance")?.window;
        assert!(!Arc::ptr_eq(&a, &later));
        assert!(words.next(&mut left)?.is_none());
        assert!(words.next(&mut right)?.is_none());
        Ok(())
    }

    #[test]
    fn corrupt_headers_payloads_and_truncation_are_refused() -> Result<(), String> {
        let mut records = WordRecords::new(Cursor::new(Vec::new()), 1)?;
        for _ in 0..32 {
            records.push(0, word("small", 0.0, None)?)?;
        }
        let mut store = records.finish()?;
        let mut cursor = store.cursor(0)?;

        let original = store.file.get_ref().clone();
        let original_buffer_len = store.record.len();
        // This claimed payload fits the channel's extent but is much larger than any record.
        store.file.get_mut()[..8].copy_from_slice(&512_u64.to_le_bytes());
        assert!(
            store
                .next(&mut cursor)
                .expect_err("bad length")
                .contains("invalid word record length")
        );
        assert_eq!(store.record.len(), original_buffer_len);

        *store.file.get_mut() = original.clone();
        store.file.get_mut()[HEADER_BYTES] = 0xff;
        assert!(
            store
                .next(&mut cursor)
                .expect_err("bad provenance tag")
                .contains("provenance tag")
        );

        *store.file.get_mut() = original;
        store.file.get_mut().truncate(HEADER_BYTES + 1);
        assert!(
            store
                .next(&mut cursor)
                .expect_err("short payload")
                .contains("word record I/O")
        );
        store.file.get_mut().truncate(3);
        assert!(
            store
                .next(&mut cursor)
                .expect_err("short header")
                .contains("word record I/O")
        );
        Ok(())
    }

    #[test]
    fn writer_rejects_invalid_channels_and_unordered_words() -> Result<(), String> {
        assert!(WordRecords::new(Cursor::new(Vec::new()), 0).is_err());
        let mut records = WordRecords::new(Cursor::new(Vec::new()), 3)?;
        assert!(records.push(3, word("outside", 0.0, None)?).is_err());
        records.push(1, word("accepted", 0.125, None)?)?;
        assert!(records.push(1, word("earlier", 0.0, None)?).is_err());
        assert!(records.push(0, word("wrong channel", 0.0, None)?).is_err());
        let mut store = records.finish()?;
        let mut cursor = store.cursor(1)?;
        assert_eq!(
            store.next(&mut cursor)?,
            Some(word("accepted", 0.125, None)?)
        );
        assert!(store.next(&mut cursor)?.is_none());
        Ok(())
    }
}
