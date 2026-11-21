// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::atoms::limits::*;
use crate::atoms::{Atom, AtomHeader, AtomIterator, ReadAtom, Result, decode_error};

#[derive(Clone, Copy, Debug)]
pub struct StscEntry {
    pub first_chunk: u32,
    pub first_sample: u32,
    pub samples_per_chunk: u32,
    #[allow(dead_code)]
    pub sample_desc_index: u32,
}

/// Sample to Chunk Atom
#[allow(dead_code)]
#[derive(Debug)]
pub struct StscAtom {
    /// Entries.
    pub entries: Vec<StscEntry>,
    pub source: Option<(u64, u32)>,
}

impl StscAtom {
    /// Finds the `StscEntry` for the sample indicated by `sample_num`. Note, `sample_num` is indexed
    /// relative to the `StscAtom`. Complexity is O(log2 N).
    pub fn find_entry_for_sample(&self, sample_num: u32) -> Option<&StscEntry> {
        let mut left = 1;
        let mut right = self.entries.len();

        while left < right {
            let mid = left + (right - left) / 2;

            let entry = self.entries.get(mid).expect("mid is always within entries bounds");

            if entry.first_sample < sample_num {
                left = mid + 1;
            }
            else {
                right = mid;
            }
        }

        // The index found above (left) is the exclusive upper bound of all entries where
        // first_sample < sample_num. Therefore, the entry to return has an index of left-1. The
        // index will never equal 0 so this is safe. If the table were empty, left == 1, thus calling
        // get with an index of 0, and safely returning None.
        self.entries.get(left - 1)
    }
}

impl StscEntry {
    pub fn new(
        first_chunk_raw: u32, samples_per_chunk: u32, sample_desc_index: u32,
    ) -> Result<Self> {
        if first_chunk_raw == 0 {
            return decode_error("isomp4 (stsc): first_chunk must be >= 1");
        }
        if samples_per_chunk == 0 {
            return decode_error("isomp4 (stsc): entry has 0 samples per chunk");
        }
        Ok(Self { first_chunk: first_chunk_raw - 1, first_sample: 0, samples_per_chunk, sample_desc_index })
    }

    pub fn after(mut self, previous: Self) -> Result<Self> {
        if self.first_chunk < previous.first_chunk {
            return decode_error("isomp4 (stsc): entry first chunk not monotonic");
        }
        let count = self.first_chunk - previous.first_chunk;
        let Some(start) = count.checked_mul(previous.samples_per_chunk)
            .and_then(|value| previous.first_sample.checked_add(value)) else {
            return decode_error("isomp4 (stsc): sample count overflow");
        };
        self.first_sample = start;
        Ok(self)
    }
}

impl Atom for StscAtom {
    fn read<R: ReadAtom>(it: &mut AtomIterator<R>, _header: &AtomHeader) -> Result<Self> {
        let (_, _) = it.read_extended_header()?;
        let entry_count = it.read_u32()?;
        let source = it.fixed_table_position(u64::from(entry_count) * 12)?
            .map(|start| (start, entry_count));
        let mut entries = if source.is_some() {
            Vec::new()
        } else {
            Vec::with_capacity(MAX_TABLE_INITIAL_CAPACITY.min(usize::try_from(entry_count).unwrap_or(usize::MAX)))
        };
        let mut previous = None;
        for _ in 0..entry_count {
            let first = it.read_u32()?;
            let samples_per_chunk = it.read_u32()?;
            let description = it.read_u32()?;
            let mut entry = StscEntry::new(first, samples_per_chunk, description)?;
            if let Some(previous) = previous {
                entry = entry.after(previous)?;
            }
            previous = Some(entry);
            if source.is_none() {
                entries.push(entry);
            }
        }
        Ok(StscAtom { entries, source })
    }
}
