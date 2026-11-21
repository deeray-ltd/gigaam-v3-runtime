// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use symphonia_core::codecs::CodecParameters;
use symphonia_core::support_format;

use symphonia_core::errors::{
    Error, Result, SeekErrorKind, decode_error, seek_error, unsupported_error,
};
use symphonia_core::formats::prelude::*;
use symphonia_core::formats::probe::{ProbeFormatData, ProbeableFormat, Score, Scoreable};
use symphonia_core::formats::well_known::FORMAT_ID_ISOMP4;
use symphonia_core::io::*;
use symphonia_core::meta::{Metadata, MetadataLog};
use symphonia_core::units::Time;

use std::collections::HashMap;
use std::io::{Seek, SeekFrom};
use std::num::NonZero;
use std::sync::Arc;

use crate::atoms::{AtomError, AtomIterator, AtomType, ReadAtom};
use crate::atoms::{FtypAtom, MetaAtom, MoofAtom, MoovAtom, SidxAtom, TrakAtom};
use crate::stream::*;

use log::{debug, info, trace, warn};

const ISOMP4_FORMAT_INFO: FormatInfo = FormatInfo {
    format: FORMAT_ID_ISOMP4,
    short_name: "isomp4",
    long_name: "ISO Base Media File Format",
};

pub struct TrackState {
    /// The track number.
    track_num: usize,
    /// The track ID.
    track_id: u32,
    /// The current segment.
    cur_seg: usize,
    /// The current sample index relative to the track.
    next_sample: u32,
    /// The current sample byte position relative to the start of the track.
    next_sample_pos: u64,
}

impl TrackState {
    pub fn make(track_num: usize, trak: &TrakAtom, timespan: &TimeSpan) -> (Self, Track) {
        let mut track = Track::new(trak.tkhd.id);

        // Create the codec parameters using the sample description atom.
        if let Some(codec_params) = trak.mdia.minf.stbl.stsd.make_codec_params() {
            track.with_codec_params(codec_params);
        }

        // Populate timing information.
        track
            .with_time_base(TimeBase::from_recip(timespan.timescale))
            .with_duration(timespan.duration);

        // If the track is an audio track, and the timescale is equal to the sample rate, then the
        // number of frames is equal to the duration. This is the case for almost all audio tracks.
        // If not, there is no generic, low overhead, & precise way to determine the number of
        // frames.
        if let Some(CodecParameters::Audio(audio)) = &track.codec_params {
            if let Some(sample_rate) = audio.sample_rate {
                if sample_rate == timespan.timescale.get() {
                    track.with_num_frames(timespan.duration.get());
                }
            }
        }

        let state = Self {
            track_num,
            track_id: trak.tkhd.id,
            cur_seg: 0,
            next_sample: 0,
            next_sample_pos: 0,
        };

        (state, track)
    }
}

/// Information regarding the next sample.
#[derive(Debug)]
struct NextSampleInfo {
    /// The track number of the next sample.
    track_num: usize,
    /// The track id.
    track_id: u32,
    /// The timestamp of the next sample.
    ts: Timestamp,
    /// The timestamp expressed in seconds.
    time: Time,
    /// The duration of the next sample.
    dur: Duration,
    /// The segment containing the next sample.
    seg_idx: usize,
}

/// Information regarding a sample.
#[derive(Debug)]
struct SampleDataInfo {
    /// The position of the sample in the track.
    pos: u64,
    /// The length of the sample.
    len: u32,
}

/// A representation of time, defining a duration relative to a specific frequency
#[derive(Debug)]
pub struct TimeSpan {
    pub timescale: NonZero<u32>,
    pub duration: Duration,
}

impl Default for TimeSpan {
    fn default() -> Self {
        Self { timescale: NonZero::new(1).expect("1 is non-zero"), duration: Duration::ZERO }
    }
}

impl TimeSpan {
    pub fn new(timescale: NonZero<u32>, duration: Duration) -> Self {
        TimeSpan { timescale, duration }
    }
}

/// ISO Base Media File Format (MP4, M4A, MOV, etc.) demultiplexer.
///
/// `IsoMp4Reader` implements a demuxer for the ISO Base Media File Format.
pub struct IsoMp4Reader<'s> {
    iter: AtomIterator<MediaSourceStream<'s>>,
    media_info: MediaInfo,
    tracks: Vec<Track>,
    metadata: MetadataLog,
    /// Segments of the movie. Sorted in ascending order by sequence number.
    segs: Vec<Box<dyn StreamSegment>>,
    retain_consumed_segments: bool,
    deferred_tables: bool,
    selected_track: Option<u32>,
    batch_pcm: Option<BatchPcm>,
    pending_pcm: Option<PendingPcm>,
    sample_tables: Vec<TrackTables>,
    /// State tracker for each track.
    track_states: Vec<TrackState>,
    /// Optional, movie extends atom used for fragmented streams.
    moov: Arc<MoovAtom>,
}

struct BatchPcm {
    track_id: u32,
    frame_bytes: u64,
    max_frames: u64,
    require_complete_frames: bool,
}

struct PendingPcm {
    track_id: u32,
    pos: u64,
    packet_len: u64,
    total_frames: u64,
    consumed_frames: u64,
    pts: Timestamp,
    duration: Duration,
}

const SZ_BLOCK: u32 = 1024;

#[derive(Default)]
struct TableBlock {
    offset: Option<u64>,
    bytes: Box<[u8]>,
}

impl TableBlock {
    fn record(&mut self, iter: &mut AtomIterator<MediaSourceStream<'_>>,
        first: u64, index: u32, count: u32, width: u32) -> Result<&[u8]> {
        if index >= count {
            return decode_error("isomp4: missing source-backed table entry");
        }
        let block_first = (index / SZ_BLOCK) * SZ_BLOCK;
        let offset = first + u64::from(block_first) * u64::from(width);
        let width = usize::try_from(width).map_err(|_| Error::DecodeError("table width exceeds usize"))?;
        if self.offset != Some(offset) {
            let count = usize::try_from((count - block_first).min(SZ_BLOCK))
                .map_err(|_| Error::DecodeError("table block count exceeds usize"))?;
            let bytes = count * width;
            self.bytes = iter.read_packet_at(offset, bytes)?;
            self.offset = Some(offset);
        }
        let index = usize::try_from(index - block_first)
            .map_err(|_| Error::DecodeError("table block index exceeds usize"))?;
        let pos = index * width;
        self.bytes.get(pos..pos + width).ok_or(Error::DecodeError("invalid table block"))
    }

    fn read(&mut self, iter: &mut AtomIterator<MediaSourceStream<'_>>,
        first: u64, index: u32, count: u32, width: u32) -> Result<u64> {
        let record = self.record(iter, first, index, count, width)?;
        match width {
            4 => Ok(u64::from(u32::from_be_bytes(record.try_into().expect("read a full u32")))),
            8 => Ok(u64::from_be_bytes(record.try_into().expect("read a full u64"))),
            _ => decode_error("unexpected mp4 table width"),
        }
    }
}

#[derive(Default)]
struct TrackTables {
    blocks: [TableBlock; 2],
    tab: [Option<u64>; 2],
    chunk_run: Option<ChunkRun>,
    timing: TimingCursor,
    active_run: TableBlock,
    run_clock: Option<RunClock>,
    fragment_cursor: Option<FragmentCursor>,

}

impl TrackTables {
    fn read(
        &mut self,
        iter: &mut AtomIterator<MediaSourceStream<'_>>,
        start: u64,
        index: u32,
        count: u32,
        width: u32,
    ) -> Result<u64> {
        let slot = match self.tab.iter().position(|table| *table == Some(start)) {
            Some(slot) => slot,
            None => {
                let slot = self.tab.iter().position(Option::is_none).unwrap_or(0);
                self.tab[slot] = Some(start);
                self.blocks[slot] = TableBlock::default();
                slot
            }
        };
        self.blocks[slot].read(iter, start, index, count, width)
    }
}

struct ChunkRun {
    start: u64,
    index: u32,
    current: crate::atoms::stsc::StscEntry,
    next: Option<crate::atoms::stsc::StscEntry>,
}

struct ReaderTables<'a, 's> {
    tables: &'a mut TrackTables,
    iter: &'a mut AtomIterator<MediaSourceStream<'s>>,
}

fn chunk_run_at(
    iter: &mut AtomIterator<MediaSourceStream<'_>>,
    start: u64,
    index: u32,
) -> Result<crate::atoms::stsc::StscEntry> {
    let location = start + u64::from(index) * 12;
    let bytes = iter.read_packet_at(location, 12)?;
    let get = |offset| u32::from_be_bytes(bytes[offset..offset + 4].try_into().expect("read full entry"));
    Ok(crate::atoms::stsc::StscEntry::new(get(0), get(4), get(8))?)
}

struct RunClock {
    start: u64,
    prefix_sample: u32,
    prefix_time: u64,
}

struct FragmentCursor {
    start: u64,
    next_atom: u64,
    following_sample: u32,
    following_time: u64,
    following_byte: u64,
    current: Option<crate::stream::LocatedRun>,
}

impl crate::stream::TableLookup for ReaderTables<'_, '_> {
    fn fixed(&mut self, start: u64, index: u32, count: u32, width: u32) -> Result<u64> {
        self.tables.read(self.iter, start, index, count, width)
    }

    fn run_value(&mut self, run: crate::atoms::trun::RunFields, sample: u32,
        offset: usize) -> Result<u32> {
        let record = self.tables.active_run.record(self.iter, run.start, sample, run.count, run.width)?;
        let bytes = record.get(offset..offset + 4).ok_or(Error::DecodeError("invalid trun field"))?;
        Ok(u32::from_be_bytes(bytes.try_into().expect("validated field width")))
    }

    fn run_time(&mut self, run: crate::atoms::trun::RunFields, sample: u32) -> Result<(u64, u32)> {
        let offset = run.duration_offset.ok_or(Error::DecodeError("missing trun duration"))?;
        let mut clock = match self.tables.run_clock.take() {
            Some(clock) if clock.start == run.start && clock.prefix_sample <= sample => clock,
            _ => RunClock { start: run.start, prefix_sample: 0, prefix_time: 0 },
        };
        while clock.prefix_sample < sample {
            let duration = self.run_value(run, clock.prefix_sample, offset)?;
            clock.prefix_sample += 1;
            clock.prefix_time = clock.prefix_time.checked_add(u64::from(duration))
                .ok_or(Error::DecodeError("trun duration overflow"))?;
        }
        let time = clock.prefix_time;
        self.tables.run_clock = Some(clock);
        Ok((time, self.run_value(run, sample, offset)?))
    }

    fn fragment_run(&mut self, track: crate::stream::FragmentRuns, sample: u32)
        -> Result<Option<crate::stream::LocatedRun>> {
        let reset = self.tables.fragment_cursor.as_ref().is_none_or(|cursor| {
            cursor.start != track.range.start || cursor.current.as_ref().is_some_and(|run| sample < run.first_sample)
        });
        if reset {
            self.tables.fragment_cursor = Some(FragmentCursor {
                start: track.range.start, next_atom: track.range.start, following_sample: 0,
                following_time: 0, following_byte: track.base_pos, current: None,
            });
        }
        let cursor = self.tables.fragment_cursor.as_mut().expect("initialized");
        while cursor.current.is_none() || sample >= cursor.following_sample {
            let Some((run, next)) = self.iter.next_deferred_run(cursor.next_atom, track.range.end)?
            else { return Ok(None) };
            let first_byte = match run.data_offset {
                Some(offset) => track.base_pos.checked_add_signed(i64::from(offset))
                    .ok_or(Error::DecodeError("isomp4: invalid signed run byte offset"))?,
                None => cursor.following_byte,
            };
            let first_sample = cursor.following_sample;
            let first_time = cursor.following_time;
            cursor.next_atom = next;
            cursor.following_sample += run.sample_count;
            cursor.following_time += run.total_duration(track.default_dur);
            cursor.following_byte = first_byte.checked_add(run.total_size(track.default_size))
                .ok_or(Error::DecodeError("isomp4: run byte offset overflow"))?;
            cursor.current = Some(crate::stream::LocatedRun { run, first_sample, first_time, first_byte });
        }
        Ok(cursor.current.clone())
    }

    fn chunk_run(
        &mut self, start: u64, count: u32, sample: u32,
    ) -> Result<Option<crate::atoms::stsc::StscEntry>> {
        if count == 0 {
            return Ok(None);
        }
        let reset = self.tables.chunk_run.as_ref().is_none_or(|run| {
            run.start != start || (run.index != 0 && sample <= run.current.first_sample)
        });
        if reset {
            let current = chunk_run_at(self.iter, start, 0)?;
            let next = if count > 1 {
                Some(chunk_run_at(self.iter, start, 1)?.after(current)?)
            } else {
                None
            };
            self.tables.chunk_run = Some(ChunkRun { start, index: 0, current, next });
        }
        let run = self.tables.chunk_run.as_mut().expect("initialized above");
        while let Some(next) = run.next {
            if next.first_sample >= sample {
                break;
            }
            run.current = next;
            run.index += 1;
            run.next = if run.index + 1 < count {
                Some(chunk_run_at(self.iter, start, run.index + 1)?.after(next)?)
            } else {
                None
            };
        }
        Ok(Some(run.current))
    }
}

#[derive(Default)]
struct TimingCursor {
    block: TableBlock,
    index: u32,
    first_sample: u64,
    first_ts: u64,
    count: u32,
    delta: u32,
    loaded: bool,
}

impl TimingCursor {
    fn sample(&mut self, iter: &mut AtomIterator<MediaSourceStream<'_>>, start: u64, count: u32, sample: u32) -> Result<Option<crate::stream::SampleTiming>> {
        let sample = u64::from(sample);
        if sample < self.first_sample { *self = Self::default(); }
        loop {
            if !self.loaded {
                if self.index >= count { return Ok(None); }
                let entry = self.block.read(iter, start, self.index, count, 8)?;
                let [a, b, c, d, e, f, g, h] = entry.to_be_bytes();
                self.count = u32::from_be_bytes([a, b, c, d]);
                self.delta = u32::from_be_bytes([e, f, g, h]);
                self.loaded = true;
            }
            let end = self.first_sample.checked_add(u64::from(self.count)).ok_or(Error::DecodeError("isomp4: stts sample overflow"))?;
            if sample < end {
                let ts = self.first_ts.checked_add((sample - self.first_sample) * u64::from(self.delta)).ok_or(Error::DecodeError("isomp4: stts timestamp overflow"))?;
                return Ok(Some(crate::stream::SampleTiming { ts, dur: self.delta }));
            }
            self.first_ts = self.first_ts.checked_add(u64::from(self.count) * u64::from(self.delta)).ok_or(Error::DecodeError("isomp4: stts timestamp overflow"))?;
            self.first_sample = end;
            self.index += 1;
            self.loaded = false;
        }
    }
}


impl<'s> IsoMp4Reader<'s> {
    pub fn try_new(mss: MediaSourceStream<'s>, opts: FormatOptions) -> Result<Self> {
        Self::try_new_with_table_storage(mss, opts, false)
    }

    pub fn try_new_with_table_storage(
        mut mss: MediaSourceStream<'s>,
        opts: FormatOptions,
        defer_sizes: bool,
    ) -> Result<Self> {
        // To get to beginning of the atom.
        mss.seek_buffered_rel(-4);

        let is_seekable = mss.is_seekable();

        let mut ftyp = None;
        let mut moov = None;

        // Get the total length of the stream, if possible.
        let total_len = if is_seekable {
            let pos = mss.pos();
            let len = mss.seek(SeekFrom::End(0))?;
            mss.seek(SeekFrom::Start(pos))?;
            info!("stream is seekable with len={len} bytes.");
            Some(len)
        }
        else {
            None
        };

        let mut metadata = opts.external_data.metadata.unwrap_or_default();

        // Parse all atoms if the stream is seekable, otherwise parse all atoms up-to the mdat atom.
        let mut it = AtomIterator::new(mss, total_len);
        if defer_sizes {
            it.use_deferred_sample_sizes();
        }
        // Maps each track id to its cumulative duration (TimeSpan) as parsed from the segment
        // index.
        let mut sidx_timespans: HashMap<u32, TimeSpan> = HashMap::new();

        while let Some(header) = it.next_header()? {
            // Top-level atoms.
            match header.atom_type() {
                AtomType::FileType => {
                    ftyp = Some(it.read_atom::<FtypAtom>()?);
                }
                AtomType::Movie => {
                    moov = Some(it.read_atom::<MoovAtom>()?);
                }
                AtomType::SegmentIndex => {
                    let sidx = it.read_atom::<SidxAtom>()?;

                    // Calculate the total duration, per track, from the segment index atoms.
                    let sidx_timespan = sidx_timespans
                        .entry(sidx.reference_id)
                        .or_insert(TimeSpan::new(sidx.timescale, Duration::ZERO));

                    if sidx_timespan.timescale != sidx.timescale {
                        return unsupported_error(
                            "isomp4: different sidx timescale for the same track",
                        );
                    }

                    // Don't overflow.
                    // TODO: Duration should maybe be None since SIDX is non-authoritative.
                    sidx_timespan.duration = sidx_timespan
                        .duration
                        .checked_add(Duration::new(sidx.total_duration))
                        .ok_or(Error::DecodeError("isomp4: sidx total duration overflow"))?
                }
                AtomType::MediaData | AtomType::MovieFragment => {
                    // The mdat atom contains the codec bitstream data. For fragmented streams, a
                    // moof + mdat pair is required. If the ftyp and moov atoms have been read, then
                    // the top-level atom scan can exit here and begin playback immediately as an
                    // optimization. If not, then the scan must continue.
                    //
                    // The scan must also exit if the source is unseekable because in that case
                    // the format reader cannot skip past these atoms without dropping packets.
                    let is_playable = moov.is_some() && ftyp.is_some();

                    if is_playable || !is_seekable {
                        if !is_playable {
                            warn!("mp4 is not streamable.");
                        }
                        break;
                    }
                }
                AtomType::Meta => {
                    if it.omit_user_metadata() {
                        // Ancillary metadata is not consumed by the seekable batch adapter.
                        it.skip_atom()?;
                    }
                    else {
                        let mut meta = it.read_atom::<MetaAtom>()?;
                        if let Some(rev) = meta.take_metadata() {
                            metadata.push(rev);
                        }
                    }
                }
                AtomType::Free => (),
                AtomType::Skip => (),
                _ => {
                    info!("skipping top-level atom: {:?}.", header.atom_type());
                }
            }
        }

        if ftyp.is_none() {
            return unsupported_error("isomp4: missing ftyp atom");
        }

        if moov.is_none() {
            return unsupported_error("isomp4: missing moov atom");
        }

        // If the top-level atom scan iterated across the entire source (e.g., if moov was the last
        // atom), then the iterator must return to the first moof or mdat atom. This is only
        // possible if the source is seekable. If it's not, then the media will be effectively
        // unplayable.
        if is_seekable && it.pending().is_none() {
            let mut mss = it.into_inner();
            mss.seek(SeekFrom::Start(0))?;

            it = AtomIterator::new(mss, total_len);
            if defer_sizes {
                it.use_deferred_sample_sizes();
            }

            while let Some(header) = it.next_header()? {
                if let AtomType::MovieFragment | AtomType::MediaData = header.atom_type() {
                    break;
                }
            }
        }

        // Fragments (moof + mdat pairs) are streamed. So if the pending atom is a moof, seek the
        // iterator to the start of the moof atom.
        if let Some(atom) = it.pending() {
            if atom.atom_type() == AtomType::MovieFragment {
                it.seek_atom_start()?;
            }
        }

        let mut moov = moov.expect("moov is checked for None above");

        if moov.is_fragmented() {
            if !sidx_timespans.is_empty() {
                info!("stream is segmented with a segment index.");
            }
            else {
                info!("stream is segmented without a segment index.");
            }
        }

        if let Some(rev) = moov.take_metadata() {
            metadata.push(rev);
        }

        // Create a track and track state for each Track (trak) atom.
        let mut tracks = Vec::with_capacity(moov.traks.len());
        let mut track_states = Vec::with_capacity(moov.traks.len());

        for (t, trak) in moov.traks.iter().enumerate() {
            // Determine the timespan of the track.
            let timespan = if moov.is_fragmented() {
                // If fragmented, prefer the duration from the sidx, if it is provided. Otherwise,
                // fallback to the mdhd.
                sidx_timespans
                    .get(&trak.tkhd.id)
                    .map(|sidx_tspan| TimeSpan::new(sidx_tspan.timescale, sidx_tspan.duration))
                    .unwrap_or_else(|| {
                        TimeSpan::new(trak.mdia.mdhd.timescale, trak.mdia.mdhd.duration.into())
                    })
            }
            else {
                // If non-fragmented, use the total duration (media timescale) from the track's
                // stts atom. Since edits are not currently supported, this is the duration of all
                // samples that will be yielded.
                //
                // TODO: Support edits. Once supported, prefer the tkhd duration.
                let duration = Duration::from(trak.mdia.minf.stbl.stts.total_duration);

                TimeSpan::new(trak.mdia.mdhd.timescale, duration)
            };

            let (track_state, track) = TrackState::make(t, trak, &timespan);

            tracks.push(track);
            track_states.push(track_state);
        }

        // The number of tracks specified in the moov atom must match the number in the mvex atom.
        if let Some(mvex) = &moov.mvex {
            if mvex.trexs.len() != moov.traks.len() {
                return decode_error("isomp4: mvex and moov track number mismatch");
            }
        }

        // The moov atom will be shared among all segments and the demuxer using an Arc.
        let moov = Arc::new(moov);

        let segs: Vec<Box<dyn StreamSegment>> = vec![Box::new(MoovSegment::new(moov.clone()))];

        // Populate media information.
        let mut media_info = MediaInfo::new();
        media_info.with_time_base(TimeBase::from_recip(moov.mvhd.timescale));
        media_info.with_duration(Duration::new(moov.mvhd.duration));

        let sample_tables = (0..tracks.len()).map(|_| TrackTables::default()).collect();
        Ok(IsoMp4Reader {
            iter: it,
            media_info,
            tracks,
            metadata,
            track_states,
            segs,
            retain_consumed_segments: true,
            deferred_tables: defer_sizes && is_seekable,
            selected_track: None,
            batch_pcm: None,
            pending_pcm: None,
            sample_tables,
            moov,
        })
    }

    /// Releases consumed fragment history and disallows timestamp seeks.
    /// Underlying byte seeks remain available for interleaved packets.
    pub fn read_sequentially(&mut self) {
        self.retain_consumed_segments = false;
    }

    /// Walks source-backed fragments for one explicitly chosen track.
    pub fn select_for_batch(&mut self, track_id: u32) -> Result<()> {
        if !self.deferred_tables || self.tracks.iter().all(|track| track.id != track_id) {
            return unsupported_error("isomp4: selected-track batch requires a seekable source");
        }
        self.selected_track = Some(track_id);
        self.read_sequentially();
        Ok(())
    }

    /// Caps batch reads at the frame width consumed by the selected PCM decoder.
    /// The caller verifies the codec and supplies the decoder's complete frame width.
    pub fn enable_batch_pcm(
        &mut self,
        track_id: u32,
        frame_bytes: std::num::NonZeroUsize,
        max_frames: std::num::NonZeroUsize,
    ) -> Result<()> {
        if !self.deferred_tables || self.selected_track != Some(track_id) {
            return unsupported_error("isomp4: PCM batch track is not selected");
        }
        if self.pending_pcm.is_some() || self.track_states.iter().any(|track| track.track_id == track_id && track.next_sample != 0) {
            return decode_error("isomp4: PCM batch mode must be enabled before reading");
        }
        let width = u64::try_from(frame_bytes.get()).map_err(|_| Error::DecodeError("isomp4: PCM frame width exceeds u64"))?;
        if frame_bytes.get().checked_mul(max_frames.get())
            .and_then(|bytes| bytes.checked_add(frame_bytes.get() - 1)).is_none() {
            return decode_error("isomp4: invalid PCM frame geometry");
        }
        let max_frames = u64::try_from(max_frames.get()).map_err(|_| Error::DecodeError("isomp4: PCM frame limit exceeds u64"))?;
        let state = self.track_states.iter().find(|track| track.track_id == track_id)
            .ok_or(Error::DecodeError("isomp4: selected PCM track is missing"))?;
        let declared = self.moov.traks[state.track_num].mdia.minf.stbl.stsd.interleaved_pcm_frame_bytes();
        // Keep the previous alignment policy for verified layouts. Other layouts
        // still use the decoder's frame width to bound reads.
        let require_complete_frames = declared == Some(width);
        self.batch_pcm = Some(BatchPcm { track_id, frame_bytes: width, max_frames, require_complete_frames });
        Ok(())
    }

    fn next_pcm_packet(&mut self) -> Result<Packet> {
        let layout = self.batch_pcm.as_ref().ok_or(Error::DecodeError("isomp4: PCM layout missing"))?;
        let pending = self.pending_pcm.as_mut().ok_or(Error::DecodeError("isomp4: PCM range missing"))?;
        let end_frame = pending.consumed_frames.checked_add(layout.max_frames)
            .unwrap_or(u64::MAX).min(pending.total_frames);
        let offset = pending.consumed_frames.checked_mul(layout.frame_bytes)
            .ok_or(Error::DecodeError("isomp4: PCM offset overflow"))?;
        let pos = pending.pos.checked_add(offset).ok_or(Error::DecodeError("isomp4: PCM position overflow"))?;
        // Include the original packet remainder only in the last read. The decoder
        // consumes complete frames and already discards any terminal remainder.
        let end = if end_frame == pending.total_frames {
            pending.packet_len
        } else {
            end_frame.checked_mul(layout.frame_bytes)
                .ok_or(Error::DecodeError("isomp4: PCM offset overflow"))?
        };
        let len = usize::try_from(end - offset).map_err(|_| Error::DecodeError("isomp4: PCM chunk too large"))?;
        // Both factors are u64, so their product fits u128. Split packets
        // contain more than the configured nonzero number of frames. Ordered
        // frame indices produce monotonic floor-rounded tick edges.
        let edge = |index: u64| -> Result<u64> {
            let ticks = u128::from(pending.duration.get()) * u128::from(index) / u128::from(pending.total_frames);
            u64::try_from(ticks).map_err(|_| Error::DecodeError("isomp4: PCM time overflow"))
        };
        let first_tick = edge(pending.consumed_frames)?;
        let last_tick = edge(end_frame)?;
        let pts = pending.pts.checked_add(Duration::new(first_tick))
            .ok_or(Error::DecodeError("isomp4: PCM timestamp overflow"))?;
        let data = self.iter.read_packet_at(pos, len)?;
        let track_id = pending.track_id;
        pending.consumed_frames = end_frame;
        if end_frame == pending.total_frames { self.pending_pcm = None; }
        Ok(Packet::new(track_id, pts, Duration::new(last_tick - first_tick), data))
    }

    fn release_consumed_segments(&mut self) {
        if !self.retain_consumed_segments && self.segs.len() > 1 {
            // Called only when every track has exhausted the known segments.
            // Keep the latest segment to carry timing into the next fragment.
            let last = self.segs.pop().expect("more than one segment");
            self.segs.clear();
            self.segs.push(last);
            for track in &mut self.track_states {
                track.cur_seg = 0;
            }
        }
    }

    /// Idempotently gets information regarding the next sample of the media stream. This function
    /// selects the next sample with the lowest timestamp of all tracks.
    fn next_sample_info(&mut self) -> Result<Option<NextSampleInfo>> {
        let mut earliest = None;

        // TODO: Consider returning samples based on lowest byte position in the track instead of
        // timestamp. This may be important if video tracks are ever decoded (i.e., DTS vs. PTS).

        for (state, track) in self.track_states.iter().zip(&self.tracks) {
            if self.selected_track.is_some_and(|id| id != track.id) {
                continue;
            }
            // Get the timebase of the track used to calculate the presentation time.
            let tb = track.time_base.expect("track always created with a timebase");

            // Get the next timestamp for the next sample of the current track. The next sample may
            // be in a future segment.
            for (seg_idx_delta, seg) in self.segs[state.cur_seg..].iter().enumerate() {
                // Try to get the timestamp for the next sample of the track from the segment.
                let timing = if let Some((start, count)) = seg.timing_source(state.track_num) {
                    self.sample_tables[state.track_num].timing.sample(&mut self.iter, start, count, state.next_sample)?
                } else {
                    let mut lookup = ReaderTables {
                        tables: &mut self.sample_tables[state.track_num],
                        iter: &mut self.iter,
                    };
                    seg.sample_timing(state.track_num, state.next_sample, &mut lookup)?
                };
                if let Some(timing) = timing {
                    // Calculate the presentation time using the timestamp.
                    let Some(ts) = timing.ts.try_into().ok()
                    else {
                        return Ok(None);
                    };

                    let Some(sample_time) = tb.calc_time(ts)
                    else {
                        return Ok(None);
                    };

                    // Compare the presentation time of the sample from this track to other tracks,
                    // and select the track with the earliest presentation time.
                    match earliest {
                        Some(NextSampleInfo { time, .. }) if time <= sample_time => {
                            // Earliest is less than or equal to the track's next sample
                            // presentation time. No need to update earliest.
                        }
                        _ => {
                            // Earliest was either None, or greater than the track's next sample
                            // presentation time. Update earliest.
                            earliest = Some(NextSampleInfo {
                                track_num: state.track_num,
                                track_id: state.track_id,
                                ts,
                                time: sample_time,
                                dur: Duration::from(timing.dur),
                                seg_idx: seg_idx_delta + state.cur_seg,
                            });
                        }
                    }

                    // Either the next sample of the track had the earliest presentation time seen
                    // thus far, or it was greater than those from other tracks, but there is no
                    // reason to check samples in future segments.
                    break;
                }
            }
        }

        Ok(earliest)
    }

    fn consume_next_sample(&mut self, info: &NextSampleInfo) -> Result<Option<SampleDataInfo>> {
        // Get the track state.
        let track = &mut self.track_states[info.track_num];

        // Get the segment associated with the sample.
        let seg = &self.segs[info.seg_idx];

        // Get the sample data descriptor.
        let mut lookup = ReaderTables { tables: &mut self.sample_tables[track.track_num], iter: &mut self.iter };
        let sample_data_desc =
            seg.sample_data(track.track_num, track.next_sample, false, &mut lookup)?;
        if let Some(layout) = &self.batch_pcm {
            if layout.require_complete_frames
                && info.track_id == layout.track_id
                && u64::from(sample_data_desc.size) % layout.frame_bytes != 0
            {
                return decode_error("isomp4: PCM sample has incomplete interleaved frames");
            }
        }

        // The sample base position in the sample data descriptor remains constant if the sample
        // followed immediately after the previous sample. In this case, the track state's
        // next_sample_pos is the position of the current sample. If the base position has jumped,
        // then the base position is the position of current the sample.
        let pos = if sample_data_desc.base_pos > track.next_sample_pos {
            sample_data_desc.base_pos
        }
        else {
            track.next_sample_pos
        };

        let next_sample_pos = pos.checked_add(u64::from(sample_data_desc.size))
            .ok_or(Error::DecodeError("isomp4: sample extent overflows"))?;

        // Advance the track's current segment to the next sample's segment.
        track.cur_seg = info.seg_idx;

        // Advance the track's next sample number and position.
        track.next_sample += 1;
        track.next_sample_pos = next_sample_pos;

        Ok(Some(SampleDataInfo { pos, len: sample_data_desc.size }))
    }

    fn try_read_more_segments(&mut self) -> Result<bool> {
        // If all tracks ended in the last segment, then do not try to read anymore segments.
        //
        // Note, there will always be one segment because the moov atom was converted into a segment
        // when the reader was instantiated.
        let last_seg = self.segs.last().expect("always at least one segment from moov atom");
        if last_seg.all_tracks_ended() {
            return Ok(false);
        }

        // Continue iterating over atoms until a segment (a moof + mdat atom pair) is found. All
        // other atoms will be ignored.
        loop {
            let header = match self.iter.next_header() {
                Ok(Some(header)) => header,
                Ok(None) => break,
                // If fragmented, an EOF is the only way to truly detect the end of stream.
                Err(AtomError::Other(Error::IoError(err)))
                    if self.moov.is_fragmented()
                        && err.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                // Passthrough other errors.
                Err(err) => return Err(err.into()),
            };

            match header.atom_type() {
                AtomType::MediaData => {
                    if self.selected_track.is_none() {
                        return Ok(true);
                    }
                }
                AtomType::MovieFragment => {
                    let track = self.selected_track;
                    let moof = self.iter.read_atom_using(|iter, header| {
                        MoofAtom::read_for_track(iter, header, track)
                    })?;

                    // A moof segment can only be created if the media is fragmented.
                    if self.moov.is_fragmented() {
                        // Get the last segment.
                        let last_seg =
                            self.segs.last().expect("always at least one segment from moov atom");

                        // Create a new segment for the moof atom.
                        let seg = MoofSegment::new(moof, self.moov.clone(), last_seg.as_ref());

                        // Segments should have a monotonic sequence number.
                        if seg.sequence_num() <= last_seg.sequence_num() {
                            warn!("moof fragment has a non-monotonic sequence number.");
                        }

                        // Push the segment.
                        self.segs.push(Box::new(seg));
                        if self.selected_track.is_some() {
                            return Ok(true);
                        }
                    }
                    else {
                        return decode_error("isomp4: moof atom present without mvex atom");
                    }
                }
                _ => {
                    trace!("skipping atom: {:?}.", header.atom_type());
                }
            }
        }

        // If no atoms were returned above, then the end-of-stream has been reached.
        Ok(false)
    }

    fn seek_track_by_time(&mut self, track_num: usize, time: Time) -> Result<SeekedTo> {
        // Convert time to timestamp for the track.
        if let Some(track) = self.tracks.get(track_num) {
            let tb = track.time_base.expect("track always created with a timebase");
            let ts = tb.calc_timestamp(time).ok_or(Error::SeekError(SeekErrorKind::OutOfRange))?;
            self.seek_track_by_ts(track_num, ts)
        }
        else {
            seek_error(SeekErrorKind::Unseekable)
        }
    }

    fn seek_track_by_ts(&mut self, track_num: usize, ts: Timestamp) -> Result<SeekedTo> {
        debug!("seeking track_num={track_num} to frame_ts={ts}");

        struct SeekLocation {
            seg_idx: usize,
            sample_num: u32,
        }

        // Can only seek to 0 or positive timestamps.
        if ts.is_negative() {
            return seek_error(SeekErrorKind::OutOfRange);
        }

        let mut seg_skip = 0;

        let seek_loc = 'locate: loop {
            // Iterate over all segments and attempt to find the segment and sample number that
            // contains the desired timestamp. Skip segments already examined.
            for (seg_idx, seg) in self.segs.iter().enumerate().skip(seg_skip) {
                if let Some(sample_num) = seg.ts_sample(track_num, ts.get() as u64)? {
                    break 'locate SeekLocation { seg_idx, sample_num };
                }

                // Mark the segment as examined.
                seg_skip = seg_idx + 1;
            }

            // Otherwise, try to read more segments from the stream.
            if !self.try_read_more_segments()? {
                return seek_error(SeekErrorKind::OutOfRange);
            }
        };

        let seg = &self.segs[seek_loc.seg_idx];

        // Get the sample timing.
        let mut timing_lookup = ReaderTables {
            tables: &mut self.sample_tables[track_num],
            iter: &mut self.iter,
        };
        let timing = match seg.sample_timing(track_num, seek_loc.sample_num, &mut timing_lookup)? {
            Some(t) => t,
            None => return seek_error(SeekErrorKind::OutOfRange),
        };

        // Try to convert the sample timing to a timestamp.
        let actual_ts = match Timestamp::try_from(timing.ts) {
            Ok(ts) => ts,
            _ => return seek_error(SeekErrorKind::OutOfRange),
        };

        // Get the sample information.
        let mut lookup = ReaderTables { tables: &mut self.sample_tables[track_num], iter: &mut self.iter };
        let data_desc = seg.sample_data(track_num, seek_loc.sample_num, true, &mut lookup)?;

        // Update the track's next sample information to point to the seeked sample.
        let track = &mut self.track_states[track_num];

        track.cur_seg = seek_loc.seg_idx;
        track.next_sample = seek_loc.sample_num;
        track.next_sample_pos = data_desc.base_pos
            + match data_desc.offset {
                Some(o) => o,
                None => return seek_error(SeekErrorKind::OutOfRange),
            };

        debug!(
            "seeked track_num={} (track_id={}) to packet_ts={} (delta={})",
            track_num,
            track.track_id,
            actual_ts,
            actual_ts.saturating_delta(ts),
        );

        Ok(SeekedTo { track_id: track.track_id, required_ts: ts, actual_ts })
    }
}

impl Scoreable for IsoMp4Reader<'_> {
    fn score(_src: ScopedStream<&mut MediaSourceStream<'_>>) -> Result<Score> {
        Ok(Score::Supported(255))
    }
}

impl ProbeableFormat<'_> for IsoMp4Reader<'_> {
    fn try_probe_new(
        mss: MediaSourceStream<'_>,
        opts: FormatOptions,
    ) -> Result<Box<dyn FormatReader + '_>> {
        Ok(Box::new(IsoMp4Reader::try_new(mss, opts)?))
    }

    fn probe_data() -> &'static [ProbeFormatData] {
        &[support_format!(
            ISOMP4_FORMAT_INFO,
            &["mp4", "m4a", "m4p", "m4b", "m4r", "m4v", "mov"],
            &["video/mp4", "audio/m4a"],
            &[b"ftyp"] // Top-level atoms
        )]
    }
}

impl FormatReader for IsoMp4Reader<'_> {
    fn format_info(&self) -> &FormatInfo {
        &ISOMP4_FORMAT_INFO
    }

    fn media_info(&self) -> &MediaInfo {
        &self.media_info
    }

    fn next_packet(&mut self) -> Result<Option<Packet>> {
        if self.pending_pcm.is_some() {
            return self.next_pcm_packet().map(Some);
        }
        // Get the index of the track with the next-nearest (minimum) timestamp.
        let next_sample_info = loop {
            // Using the current set of segments, try to get the next sample info.
            if let Some(info) = self.next_sample_info()? {
                break info;
            }
            else {
                // The inner reader of the atom iterator has been used/seeked around to read
                // packets, so resync the reader and iterator by seeking to the end of the current
                // pending atom. Under regular circumstances, no actual expensive seek operation is
                // performed since the reader should be at the end of the last iterated atom if we
                // are trying to read another.
                match self.iter.seek_atom_end() {
                    Ok(_) | Err(AtomError::NoPendingAtom) => (),
                    Err(_) => return decode_error("sync lost"),
                };

                // No more segments. If the stream is unseekable, it may be the case that there are
                // more segments coming. If the stream is seekable it might be fragmented and no
                // segments are found in the moov atom. Iterate atoms until a new segment is found
                // or the end-of-stream is reached
                self.release_consumed_segments();
                if !self.try_read_more_segments()? {
                    return Ok(None);
                }
            }
        };

        // Get the position and length information of the next sample.
        let sample_info = match self.consume_next_sample(&next_sample_info)? {
            Some(s) => s,
            None => return decode_error("isomp4: failed to get next sample data"),
        };

        if let Some(layout) = &self.batch_pcm {
            let packet_len = u64::from(sample_info.len);
            let total_frames = packet_len / layout.frame_bytes;
            if next_sample_info.track_id == layout.track_id && total_frames > layout.max_frames {
                self.iter.validate_packet_extent(sample_info.pos, packet_len)?;
                self.pending_pcm = Some(PendingPcm {
                    track_id: next_sample_info.track_id,
                    pos: sample_info.pos,
                    packet_len,
                    total_frames,
                    consumed_frames: 0,
                    pts: next_sample_info.ts,
                    duration: next_sample_info.dur,
                });
                return self.next_pcm_packet().map(Some);
            }
        }
        let len = usize::try_from(sample_info.len)
            .map_err(|_| Error::DecodeError("isomp4: packet size exceeds usize"))?;
        let data = if self.selected_track.is_some() {
            self.iter.read_packet_at(sample_info.pos, len)?
        } else {
            self.iter.read_raw_boxed_slice_exact(sample_info.pos, len)?
        };

        Ok(Some(Packet::new(
            next_sample_info.track_id,
            next_sample_info.ts,
            next_sample_info.dur,
            data,
        )))
    }

    fn metadata(&mut self) -> Metadata<'_> {
        self.metadata.metadata()
    }

    fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    fn seek(&mut self, _mode: SeekMode, to: SeekTo) -> Result<SeekedTo> {
        if self.deferred_tables || !self.retain_consumed_segments || self.tracks.is_empty() {
            return seek_error(SeekErrorKind::Unseekable);
        }

        match to {
            SeekTo::Timestamp { ts, track_id } => {
                // The seek timestamp is in timebase units specific to the selected track. Get the
                // selected track and use the timebase to convert the timestamp into time units so
                // that the other tracks can be seeked.
                if let Some((track_num, track)) =
                    self.tracks.iter().enumerate().find(|(_, track)| track.id == track_id)
                {
                    // Convert to time units.
                    let tb = match track.time_base {
                        Some(tb) => tb,
                        None => return seek_error(SeekErrorKind::Unseekable),
                    };
                    let time =
                        tb.calc_time(ts).ok_or(Error::SeekError(SeekErrorKind::Unseekable))?;

                    // Seek all tracks excluding the primary track to the desired time.
                    for t in 0..self.track_states.len() {
                        if t != track_num {
                            self.seek_track_by_time(t, time)?;
                        }
                    }

                    // Seek the primary track and return the result.
                    self.seek_track_by_ts(track_num, ts)
                }
                else {
                    seek_error(SeekErrorKind::InvalidTrack)
                }
            }
            SeekTo::Time { time, track_id } => {
                // If provided, find the track number of the track with the desired track_id, or
                // default to the first track.
                let track_num = match track_id {
                    Some(id) => self
                        .tracks
                        .iter()
                        .position(|track| track.id == id)
                        .ok_or(Error::SeekError(SeekErrorKind::InvalidTrack))?,
                    None => 0,
                };

                // Seek all tracks excluding the selected track and discard the result.
                for t in 0..self.track_states.len() {
                    if t != track_num {
                        self.seek_track_by_time(t, time)?;
                    }
                }

                // Seek the primary track and return the result.
                self.seek_track_by_time(track_num, time)
            }
        }
    }

    fn into_inner<'s>(self: Box<Self>) -> MediaSourceStream<'s>
    where
        Self: 's,
    {
        self.iter.into_inner()
    }
}

impl ReadAtom for MediaSourceStream<'_> {}

impl From<AtomError> for Error {
    fn from(value: AtomError) -> Self {
        // Map all atom iteration errors to decode errors.
        let msg = match value {
            AtomError::InvalidAtomSize => "isomp4: invalid atom size",
            AtomError::InvalidUtf8 => "isomp4: invalid utf-8 string",
            AtomError::MaximumDepthReached => "isomp4: maximum recursion depth reached",
            AtomError::NoParentAtom => "isomp4: no parent atom",
            AtomError::NoPendingAtom => "isomp4: no atom pending read",
            AtomError::Overrun => "isomp4: overrun while reading atom",
            AtomError::SeekOutOfRange => "isomp4: out-of-bounds seek for a non-seekable stream",
            AtomError::UnexpectedEndOfAtom => "isomp4: unexpected end of atom",
            AtomError::UnexpectedPosition => "isomp4: unexpected position",
            AtomError::UnexpectedUnknownSizeAtom => "isomp4: unknown size atom has sized parent",
            AtomError::UnexpectedReadOperation => "isomp4: unexpected read operation",
            AtomError::UnknownAtomSize => "isomp4: unknown atom size",
            AtomError::Other(err) => return err,
        };
        Error::DecodeError(msg)
    }
}

// fn convert_timescale(
//     duration: u64,
//     src_timescale: NonZero<u32>,
//     dst_timescale: NonZero<u32>,
// ) -> Duration {
//     if src_timescale == dst_timescale {
//         return Duration::from(duration);
//     }
//     Duration::from(
//         ((duration as u128 * dst_timescale.get() as u128) / src_timescale.get() as u128) as u64,
//     )
// }
