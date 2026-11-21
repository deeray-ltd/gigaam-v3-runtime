// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::atoms::{
    Atom, AtomHeader, AtomIterator, AtomType, ReadAtom, Result, TfhdAtom, TrunAtom, decode_error,
};

#[derive(Clone, Copy, Debug)]
pub struct DeferredRuns {
    pub start: u64,
    pub end: u64,
    explicit_duration: u64,
    default_duration_samples: u64,
}

impl DeferredRuns {
    pub fn duration(self, default_duration: u32) -> u64 {
        self.explicit_duration + self.default_duration_samples * u64::from(default_duration)
    }
}

/// Track fragment atom.
#[allow(dead_code)]
#[derive(Debug)]
pub struct TrafAtom {
    /// Track fragment header.
    pub tfhd: TfhdAtom,
    /// Track fragment sample runs.
    pub truns: Vec<TrunAtom>,
    pub deferred: Option<DeferredRuns>,
    /// The total number of samples in this track fragment.
    pub total_sample_count: u32,
}

impl Atom for TrafAtom {
    fn read<R: ReadAtom>(it: &mut AtomIterator<R>, header: &AtomHeader) -> Result<Self> {
        let mut tfhd = None;
        let mut truns = Vec::new();
        let mut deferred = match (it.fixed_table_position(0)?, header.end()) {
            (Some(start), Some(end)) => Some(DeferredRuns {
                start, end, explicit_duration: 0, default_duration_samples: 0,
            }),
            _ => None,
        };

        let mut total_sample_count = 0u32;

        while let Some(header) = it.next_header()? {
            match header.atom_type {
                AtomType::TrackFragmentHeader => {
                    tfhd = Some(it.read_atom::<TfhdAtom>()?);
                }
                AtomType::TrackFragmentRun => {
                    let trun = it.read_atom::<TrunAtom>()?;

                    // Increment the total sample count.
                    let Some(total) = total_sample_count.checked_add(trun.sample_count) else {
                        return decode_error("isomp4 (traf): sample count overflow");
                    };
                    total_sample_count = total;

                    if let Some(runs) = &mut deferred {
                        let duration_without_default = trun.total_duration(0);
                        runs.explicit_duration += duration_without_default;
                        runs.default_duration_samples += trun.total_duration(1) - duration_without_default;
                    } else {
                        truns.push(trun);
                    }
                }
                _ => (),
            }
        }

        // Tfhd is mandatory.
        let Some(tfhd) = tfhd
        else {
            return decode_error("isomp4 (traf): missing tfhd atom");
        };

        Ok(TrafAtom { tfhd, truns, deferred, total_sample_count })
    }
}
