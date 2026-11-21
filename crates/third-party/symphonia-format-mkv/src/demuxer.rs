// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::{HashMap, VecDeque};
use std::convert::TryFrom;
use std::num::NonZero;

use symphonia_core::codecs::audio::{CODEC_ID_NULL_AUDIO, well_known::CODEC_ID_VORBIS};
use symphonia_core::errors::{Error, Result, SeekErrorKind, seek_error, unsupported_error};
use symphonia_core::formats::prelude::*;
use symphonia_core::formats::probe::{ProbeFormatData, ProbeableFormat, Score, Scoreable};
use symphonia_core::formats::well_known::FORMAT_ID_MKV;
use symphonia_core::io::*;
use symphonia_core::meta::{Metadata, MetadataLog};
use symphonia_core::support_format;
use symphonia_core::units::TimeBase;

use log::{info, warn};

use crate::codecs::{CodecPrivateMode, is_known_audio_track, make_track_codec_params};
use crate::ebml::{EbmlElementInfo, EbmlError, EbmlIterator, EbmlSchema, ReadEbml};
use crate::lacing::{Frame, FrameRange, extract_frames, extract_frame_ranges};
use crate::schema::{MkvElement, MkvSchema};
use crate::segment::{
    AttachmentsElement, AudioSeekHead, BlockGroupElement, BlockGroupRange, ChaptersElement, CuesElement, EbmlHeaderElement,
    InfoElement, MatroskaTicks, NonZeroMatroskaTicks, SeekHeadElement, SegmentTicks,
    SignedTrackTicks, TagsElement, TargetTagsMap, TrackTicks, TracksElement,
};

const MKV_FORMAT_INFO: FormatInfo =
    FormatInfo { format: FORMAT_ID_MKV, short_name: "matroska", long_name: "Matroska / WebM" };

pub struct TrackState {
    /// The Matroska track number (Symphonia's track ID).
    track_num: u32,
    /// The default frame duration.
    pub(crate) default_frame_duration: Option<NonZeroMatroskaTicks>,
    /// The codec delay.
    pub(crate) codec_delay: MatroskaTicks,
    /// The track's timebase.
    pub(crate) track_time_base: TimeBase,
    /// The track's timestamp scale.
    pub(crate) track_timestamp_scale: f64,
}

/// Matroska (MKV) and WebM demultiplexer.
///
/// `MkvReader` implements a demuxer for the Matroska and WebM formats.
pub struct MkvReader<'s> {
    /// Iterator over EBML element headers
    iter: EbmlIterator<MediaSourceStream<'s>, MkvSchema>,
    media_info: MediaInfo,
    tracks: Vec<Track>,
    track_states: HashMap<u32, TrackState>,
    attachments: Vec<Attachment>,
    chapters: Option<ChapterGroup>,
    metadata: MetadataLog,
    cues: Option<CuesElement>,
    retain_cues: bool,
    current_cluster: Option<ClusterState>,
    frames: VecDeque<Frame>,
    frame_ranges: VecDeque<FrameRange>,
    defer_blocks: bool,
    selected_track: Option<u32>,
    batch_pcm: Option<BatchPcm>,
    batch_vorbis_prefix_len: Option<u64>,
    pending_pcm: Option<PendingPcm>,
}

struct BatchPcm {
    track_id: u32,
    frame_bytes: u64,
    max_frames: u64,
}

struct PendingPcm {
    frame: FrameRange,
    total_frames: u64,
    consumed_frames: u64,
}

#[derive(Copy, Clone, Debug)]
struct ClusterState {
    /// The cluster timestamp in Segment ticks..
    timestamp: Option<SegmentTicks>,
    /// The start position in bytes of the cluster.
    start: u64,
}

impl<'s> MkvReader<'s> {
    pub fn try_new(mss: MediaSourceStream<'s>, opts: FormatOptions) -> Result<Self> {
        Self::try_new_with_cues(mss, opts, true)
    }

    pub fn try_new_with_cues(
        mss: MediaSourceStream<'s>,
        opts: FormatOptions,
        retain_cues: bool,
    ) -> Result<Self> {
        Self::try_new_impl(mss, opts, retain_cues, false)
    }

    /// Open seekable audio batch input with codec data for the first known audio track.
    ///
    /// Other tracks keep their timing and identifiers so interleaved blocks can be validated.
    pub fn try_new_batch_audio(mss: MediaSourceStream<'s>, opts: FormatOptions) -> Result<Self> {
        Self::try_new_impl(mss, opts, false, true)
    }

    fn try_new_impl(
        mss: MediaSourceStream<'s>,
        opts: FormatOptions,
        retain_cues: bool,
        batch_audio: bool,
    ) -> Result<Self> {
        // Get the total length of the stream, if possible.
        let (is_seekable, total_len) = (mss.is_seekable(), mss.byte_len());

        match total_len {
            Some(len) if is_seekable => info!("stream is seekable with len={len} bytes."),
            _ => (),
        }

        let defer_blocks = is_seekable && !retain_cues && total_len.is_some();
        if batch_audio && !defer_blocks {
            return unsupported_error("mkv: batch audio requires a seekable source with known length");
        }
        let mut it = EbmlIterator::new(mss, MkvSchema, total_len, defer_blocks);

        // Read the EBML header.
        let ebml = it.next_element::<EbmlHeaderElement>()?;

        if !matches!(ebml.doc_type.as_str(), "matroska" | "webm") {
            return unsupported_error("mkv: not a matroska / webm file");
        }

        // Read the root element, Segment.
        let segment_pos = match it.next_header()? {
            Some(elem) if elem.element_type() == MkvElement::Segment => elem.data_pos(),
            _ => return unsupported_error("mkv: missing segment element"),
        };

        // Descend into the Segment element.
        it.push_element()?;

        let mut segment_tracks = None;
        let mut info = None;
        let mut cues = None;
        let mut current_cluster = None;
        let mut seek_positions = Vec::new();
        let mut tags = Vec::new();
        let mut batch_seek = [None::<u64>; 2];
        let mut attachments = None;
        let mut chapters = None;

        while let Ok(Some(header)) = it.next_header() {
            match header.element_type() {
                MkvElement::SeekHead if !retain_cues => {
                    let seek_head = it.read_master_element::<AudioSeekHead>()?;
                    for (position, found) in batch_seek.iter_mut().zip([seek_head.info, seek_head.tracks]) {
                        *position = (*position).max(found);
                    }
                }
                MkvElement::SeekHead => {
                    let seek_head = it.read_master_element::<SeekHeadElement>()?;
                    for element in seek_head.seeks.into_vec() {
                        let element_type = match it.schema().get_element_info(element.id as u32) {
                            Some(info) => info.element_type(),
                            None => continue,
                        };
                        seek_positions.push((element_type, element.position));
                    }
                }
                MkvElement::Tracks => {
                    segment_tracks = Some(it.read_master_element::<TracksElement>()?);
                }
                MkvElement::Info => {
                    info = Some(it.read_master_element::<InfoElement>()?);
                }
                MkvElement::Cues if retain_cues => {
                    cues = Some(it.read_master_element::<CuesElement>()?);
                }
                MkvElement::Tags if retain_cues => {
                    // Multiple tags element per segment allowed.
                    tags.push(it.read_master_element::<TagsElement>()?);
                }
                MkvElement::Cluster => {
                    // Set state for current cluster for the first call of `next_element`.
                    current_cluster =
                        Some(ClusterState { timestamp: None, start: header.pos() - segment_pos });

                    // Don't look forward into the stream since we can't be sure that we'll
                    // find anything useful.
                    break;
                }
                MkvElement::Attachments if retain_cues => {
                    // Only one attachments element per segment is expected.
                    if attachments.is_some() {
                        log::warn!("unexpected attachments element");
                    }
                    attachments = Some(it.read_master_element::<AttachmentsElement>()?);
                }
                MkvElement::Chapters if retain_cues => {
                    // Only one chapters element per segment is expected.
                    if chapters.is_some() {
                        log::warn!("unexpected chapters element");
                    }
                    chapters = Some(it.read_master_element::<ChaptersElement>()?);
                }
                other => {
                    log::debug!("top-level scan ignored element {other:?}");
                }
            }
        }

        if is_seekable {
            for (kind, pos) in [MkvElement::Info, MkvElement::Tracks].into_iter().zip(batch_seek) {
                if let Some(pos) = pos {
                    seek_positions.push((kind, pos));
                }
            }
            // All elements preceeding the element iterator's current position have already been
            // read and do not need to be revisited.
            seek_positions.retain(|sp| sp.1 >= it.pos());
            // Make sure we don't jump backwards unnecessarily.
            seek_positions.sort_by_key(|sp| sp.1);

            for (_, pos) in seek_positions {
                // Ascend back to the segment element.
                it.pop_elements_upto(MkvElement::Segment)?;

                // Seek the iterator to the child element.
                it.seek_to_child(pos)?;

                // Resume iteration.
                let element_type = match it.next_header()? {
                    Some(header) => header.element_type(),
                    _ => continue,
                };

                // Safety: The element type or position may be incorrect. The element iterator will
                // validate the type (as declared in the header) of the element at the seeked
                // position against the element type asked to be read.
                match element_type {
                    MkvElement::Tracks => {
                        segment_tracks = Some(it.read_master_element::<TracksElement>()?);
                    }
                    MkvElement::Info => {
                        info = Some(it.read_master_element::<InfoElement>()?);
                    }
                    MkvElement::Tags if retain_cues => {
                        // Multiple tags element per segment allowed.
                        tags.push(it.read_master_element::<TagsElement>()?);
                    }
                    MkvElement::Cues if retain_cues => {
                        cues = Some(it.read_master_element::<CuesElement>()?);
                    }
                    MkvElement::Attachments if retain_cues => {
                        // Only one attachments element per segment is expected.
                        if attachments.is_some() {
                            log::warn!("unexpected attachments element after meta seek");
                        }
                        attachments = Some(it.read_master_element::<AttachmentsElement>()?);
                    }
                    MkvElement::Chapters if retain_cues => {
                        // Only one chapters element per segment is expected.
                        if chapters.is_some() {
                            log::warn!("unexpected chapters element after meta seek");
                        }
                        chapters = Some(it.read_master_element::<ChaptersElement>()?)
                    }
                    _ => (),
                }
            }
        }

        let segment_tracks =
            segment_tracks.ok_or(Error::DecodeError("mkv: missing Tracks element"))?;

        // If seekable, seek to the start of the first cluster, if known, or the start of the
        // segment. If unseekable, the element iterator is already positioned at the start of the
        // first cluster.
        if is_seekable {
            let cluster_pos = current_cluster.as_ref().map(|cluster| cluster.start).unwrap_or(0);
            it.seek_to_child(cluster_pos)?;
            let _ = it.next_header()?;
        }

        // Descend into the cluster.
        it.push_element()?;

        let info = info.ok_or(Error::DecodeError("mkv: missing Info element"))?;

        // Create a hashmap of all per-target tags (edition, chapter, & attachment tags).
        let mut per_target_tags: TargetTagsMap = Default::default();

        segment_tracks.get_target_uids(&mut per_target_tags);

        if let Some(chapters) = &chapters {
            chapters.get_target_uids(&mut per_target_tags);
        }
        if let Some(attachments) = &attachments {
            attachments.get_target_uids(&mut per_target_tags);
        }

        // Begin with externally provided metadata and chapters.
        let mut metadata = opts.external_data.metadata.unwrap_or_default();

        // Post-process all tag elements into metadata revisions, while also collecting per-target
        // tags.
        let is_video = segment_tracks.tracks.as_ref().iter().any(|t| t.video.is_some());
        for tag in tags {
            metadata.push(tag.into_metadata(&mut per_target_tags, is_video));
        }

        // Post-process chapters element.
        let chapters = chapters
            .map(|chapters| chapters.into_chapter_group(&mut per_target_tags))
            .unwrap_or(opts.external_data.chapters);

        // Process attachments element.
        let attachments = attachments
            .map(|attachments| attachments.into_attachments(&mut per_target_tags))
            .unwrap_or_default();

        // Should TimeBase use a u64/u64 rational?
        // Reduce the timebase to reduce the chance of overflows later.
        let time_base = TimeBase::new(
            info.timestamp_scale
                .try_into()
                .map_err(|_| Error::Unsupported("mkv: timestamp scale too large (report this)"))?,
            NonZero::new(1_000_000_000).expect("1_000_000_000 is non-zero"),
        )
        .reduce();

        let mut tracks = Vec::new();
        let mut track_states = HashMap::new();
        let mut selected_audio = false;

        for track in segment_tracks.tracks {
            // The track's timebase is scaled by the track timestamp scale.
            let track_time_base = time_base
                .scale(track.track_timestamp_scale)
                .ok_or(Error::DecodeError("mkv: track timebase is invalid"))?;

            // Create the track state.
            let state = TrackState {
                // TODO: This should be 64-bit, but track IDs are 32-bit.
                track_num: u32::try_from(track.number.get())
                    .map_err(|_| Error::Unsupported("mkv: track number too large (report this)"))?,
                default_frame_duration: track.default_duration,
                codec_delay: track.codec_delay,
                track_time_base,
                track_timestamp_scale: track.track_timestamp_scale,
            };

            // Create the track.
            let mut tr = Track::new(state.track_num);

            tr.with_time_base(track_time_base);

            if let Some(lang_bcp47) = &track.lang_bcp47 {
                tr.with_language(lang_bcp47);
            }
            else {
                tr.with_language(&track.lang);
            }

            tr.with_flags(track.flags);

            let private_mode = if batch_audio && (selected_audio || !is_known_audio_track(&track)) {
                CodecPrivateMode::Omit
            }
            else if defer_blocks && !selected_audio {
                CodecPrivateMode::CompactVorbisSetup
            }
            else {
                CodecPrivateMode::Full
            };
            if let Some(codec_params) = make_track_codec_params(track, &mut it, private_mode)? {
                selected_audio |= codec_params.audio().is_some_and(|p| p.codec != CODEC_ID_NULL_AUDIO);
                tr.with_codec_params(codec_params);
            }

            tracks.push(tr);
            track_states.insert(state.track_num, state);
        }

        // Populate media information.
        let mut media_info = MediaInfo::new();

        media_info.with_time_base(time_base);

        if let Some(duration) = info.duration {
            media_info.with_duration(Duration::new(duration.get().round() as u64));
        }

        Ok(Self {
            iter: it,
            media_info,
            tracks,
            track_states,
            attachments,
            chapters,
            metadata,
            cues,
            retain_cues,
            current_cluster,
            frames: VecDeque::new(),
            frame_ranges: VecDeque::new(),
            defer_blocks,
            selected_track: None,
            batch_pcm: None,
            batch_vorbis_prefix_len: None,
            pending_pcm: None,
        })
    }

    /// Selects the first usable audio track for seekable batch consumption.
    pub fn select_for_batch(&mut self, track_id: u32) -> Result<()> {
        use symphonia_core::formats::TrackType;
        if !self.defer_blocks || self.pending_pcm.is_some() {
            return Err(Error::DecodeError("mkv: batch selection requires seekable batch input"));
        }
        let selected = self.first_track_known_codec(TrackType::Audio)
            .ok_or(Error::DecodeError("mkv: no selected audio track"))?;
        if selected.id != track_id {
            return Err(Error::DecodeError("mkv: requested batch track is not selected"));
        }
        self.batch_vorbis_prefix_len = selected
            .codec_params
            .as_ref()
            .and_then(|params| params.audio())
            .filter(|params| params.codec == CODEC_ID_VORBIS)
            .and_then(|params| params.extra_data.as_deref())
            .and_then(crate::vorbis::audio_packet_prefix_len);
        self.selected_track = Some(track_id);
        Ok(())
    }

    /// Enables bounded reads for the selected track on a seekable batch input.
    /// The caller supplies validated interleaved PCM frame geometry for that track.
    pub fn enable_batch_pcm(
        &mut self,
        track_id: u32,
        frame_bytes: NonZero<usize>,
        max_frames: NonZero<usize>,
    ) -> Result<()> {
        use symphonia_core::formats::TrackType;
        if !self.defer_blocks || self.pending_pcm.is_some() || self.selected_track != Some(track_id) {
            return Err(Error::DecodeError("mkv: bounded PCM mode requires seekable batch input"));
        }
        let selected = self.first_track_known_codec(TrackType::Audio)
            .ok_or(Error::DecodeError("mkv: no selected audio track"))?;
        if selected.id != track_id {
            return Err(Error::DecodeError("mkv: bounded PCM track is not selected"));
        }
        let channels = selected.codec_params.as_ref().and_then(|params| params.audio())
            .and_then(|params| params.channels.as_ref())
            .ok_or(Error::DecodeError("mkv: bounded PCM channel layout is missing"))?
            .count();
        if channels == 0 || frame_bytes.get() % channels != 0 {
            return Err(Error::DecodeError("mkv: invalid PCM frame geometry"));
        }
        let frame_bytes = u64::try_from(frame_bytes.get())
            .map_err(|_| Error::DecodeError("mkv: PCM frame width exceeds u64"))?;
        let max_frames = u64::try_from(max_frames.get())
            .map_err(|_| Error::DecodeError("mkv: PCM frame limit exceeds u64"))?;
        self.batch_pcm = Some(BatchPcm { track_id, frame_bytes, max_frames });
        Ok(())
    }

    fn next_pcm_packet(&mut self) -> Result<Packet> {
        let layout = self.batch_pcm.as_ref().ok_or(Error::DecodeError("mkv: PCM layout missing"))?;
        let pending = self.pending_pcm.as_mut().ok_or(Error::DecodeError("mkv: PCM range missing"))?;
        let end_frame = pending.consumed_frames.saturating_add(layout.max_frames)
            .min(pending.total_frames);
        let frames = end_frame.checked_sub(pending.consumed_frames)
            .ok_or(Error::DecodeError("mkv: PCM frame range is invalid"))?;
        let byte_offset = pending.consumed_frames.checked_mul(layout.frame_bytes)
            .ok_or(Error::DecodeError("mkv: PCM offset overflow"))?;
        let start = pending.frame.start.checked_add(byte_offset)
            .ok_or(Error::DecodeError("mkv: PCM position overflow"))?;
        let len = frames.checked_mul(layout.frame_bytes)
            .ok_or(Error::DecodeError("mkv: PCM length overflow"))?;
        let count = usize::try_from(len)
            .map_err(|_| Error::DecodeError("mkv: PCM chunk too large"))?;
        // Cumulative edges distribute the original duration without accumulating rounding error.
        let edge = |index: u64| -> Result<u64> {
            let value = u128::from(pending.frame.dur.get())
                .checked_mul(u128::from(index))
                .and_then(|total| total.checked_div(u128::from(pending.total_frames)))
                .ok_or(Error::DecodeError("mkv: PCM time range is invalid"))?;
            u64::try_from(value).map_err(|_| Error::DecodeError("mkv: PCM time overflow"))
        };
        let start_tick = edge(pending.consumed_frames)?;
        let end_tick = edge(end_frame)?;
        let pts = pending.frame.pts.checked_add_unsigned(TrackTicks::from(start_tick))
            .ok_or(Error::DecodeError("mkv: PCM timestamp overflow"))?;
        let duration = end_tick.checked_sub(start_tick)
            .ok_or(Error::DecodeError("mkv: PCM time range is invalid"))?;
        let track_id = pending.frame.track_num;
        let data = self.iter.with_binary_range(start, len, |mut reader| {
            use symphonia_core::io::ReadBytes;
            Ok(reader.read_boxed_slice_exact(count)?)
        })?;
        pending.consumed_frames = end_frame;
        if end_frame == pending.total_frames {
            self.pending_pcm = None;
        }
        Ok(Packet::new(
            track_id,
            pts.into_ts(),
            TrackTicks::from(duration).into_dur(),
            data,
        ))
    }

    fn seek_track_by_ts_forward(&mut self, track_id: u32, ts: Timestamp) -> Result<SeekedTo> {
        let actual_ts = 'out: loop {
            // Skip frames from the buffer until the given timestamp
            while let Some(frame) = self.frames.front() {
                let next_frame_pts = frame
                    .pts
                    .into_ts()
                    .checked_add(frame.dur.into_dur())
                    .ok_or(Error::SeekError(SeekErrorKind::OutOfRange))?;

                if next_frame_pts >= ts && frame.track_num == track_id {
                    break 'out frame.pts.into_ts();
                }
                else {
                    self.frames.pop_front();
                }
            }

            if !self.next_element()? {
                // There are no more elements.
                return Err(Error::SeekError(SeekErrorKind::OutOfRange));
            }
        };

        Ok(SeekedTo { track_id, required_ts: ts, actual_ts })
    }

    fn seek_track_by_ts_atomic(
        &mut self,
        id: u32,
        tb: TimeBase,
        ts: Timestamp,
    ) -> Result<SeekedTo> {
        // Save the iterator and cluster states to restore in-case of and error.
        let iter_state = self.iter.save_state();
        let cluster_state = self.current_cluster;

        match self.seek_track_by_ts(id, tb, ts) {
            Err(err) => {
                // Restore saved iterator and cluster states.
                self.iter.restore_state(iter_state)?;
                self.current_cluster = cluster_state;
                Err(err)
            }
            value => value,
        }
    }

    fn seek_track_by_ts(&mut self, id: u32, tb: TimeBase, ts: Timestamp) -> Result<SeekedTo> {
        log::debug!("seeking track_id={id} to ts={ts}");

        // If cues exist, seek to the nearest cue point.
        if let Some(cues) = &self.cues {
            let mut target_cue_point = None;

            // Cue points store timestamps in Matroska ticks while the timestamp being seeked to is
            // in signed Track ticks. Convert to unsigned Matroska ticks for iterating the cue
            // points. If the timestamp is negative, then this is an error because cue points only
            // contain unsigned Matroska ticks.
            let ts = SignedTrackTicks::from(ts)
                .try_into_matroska_ticks(tb)
                .ok_or(Error::SeekError(SeekErrorKind::OutOfRange))?;

            for cue_point in &cues.points {
                if cue_point.time > ts {
                    break;
                }
                target_cue_point = Some(cue_point);
            }

            let target_cue_point =
                target_cue_point.ok_or(Error::SeekError(SeekErrorKind::OutOfRange))?;

            log::debug!(
                "found cue point: track_id={}, ts={}, seg_pos={}",
                target_cue_point.positions.track.get(),
                target_cue_point.time,
                target_cue_point.positions.cluster_pos
            );

            // Ascend back to the segment element.
            self.iter.pop_elements_upto(MkvElement::Segment)?;

            // Seek to the specific cluster element.
            self.iter.seek_to_child(target_cue_point.positions.cluster_pos)?;

            // Resume iteration.
            let cluster = match self.iter.next_header()? {
                // The seeked element is a cluster.
                Some(header) if header.element_type() == MkvElement::Cluster => header,
                // The seeked element is not a cluster or there were no more elements at the cue
                // position. The cue point was malformed.
                _ => return seek_error(SeekErrorKind::Unseekable),
            };

            // Convert the cue point's timestamp (Matroska ticks) into Segment ticks for the cluster
            // state.
            let timestamp = target_cue_point.time.into_segment_ticks(
                self.media_info.time_base.expect("media info time base always populated"),
            );

            // Update the current cluster metadata.
            self.current_cluster =
                Some(ClusterState { timestamp: Some(timestamp), start: cluster.pos() });

            // Descend into the cluster element.
            self.iter.push_element()?;

            // If a cluster relative position is available, seek to the exact simple block or
            // block group element.
            if let Some(cluster_rel_pos) = target_cue_point.positions.cluster_rel_pos {
                self.iter.seek_to_child(cluster_rel_pos)?;
            }
        }

        // Seek to exact block.
        self.seek_track_by_ts_forward(id, ts)
    }

    fn next_element(&mut self) -> Result<bool> {
        match self.iter.next_header()? {
            None => {
                // The EBML iterator has consumed all child elements at the current level of the
                // document.
                match self.iter.parent() {
                    None => {
                        // The parent is the document. The media has ended.
                        return Ok(false);
                    }
                    Some(parent) if parent.element_type() == MkvElement::Cluster => {
                        // The parent was a cluster element. Reset the cluster state.
                        self.current_cluster = None;
                    }
                    _ => (),
                }

                // Ascend to its parent.
                self.iter.pop_element()?;
            }
            Some(child) => {
                match child.element_type() {
                    // Cluster element.
                    MkvElement::Cluster => {
                        self.current_cluster =
                            Some(ClusterState { timestamp: None, start: child.data_pos() });

                        // Descend into the cluster.
                        self.iter.push_element()?;
                    }
                    // Children of a cluster element.
                    MkvElement::Timestamp => {
                        // Cluster timestamp element.
                        match self.current_cluster.as_mut() {
                            Some(cc) => {
                                cc.timestamp = self.iter.read_u64()?.map(SegmentTicks::from)
                            }
                            _ => log::warn!("expected to have cluster"),
                        }
                    }
                    block_type @ (MkvElement::SimpleBlock | MkvElement::BlockGroup) => {
                        // Get the current cluster information.
                        let Some(cluster) = self.current_cluster.as_ref()
                        else {
                            log::warn!("expected to have cluster");
                            return Ok(true);
                        };

                        // Get the cluster timestamp.
                        let Some(cluster_ts) = cluster.timestamp
                        else {
                            log::warn!("missing cluster timestamp");
                            return Ok(true);
                        };

                        if self.defer_blocks {
                            let (start, len, duration) = if block_type == MkvElement::SimpleBlock {
                                (child.data_pos(), child.data_size().ok_or(EbmlError::UnknownElementDataSize)?, None)
                            } else {
                                let group = self.iter.read_master_element::<BlockGroupRange>()?;
                                (group.start, group.len, group.duration)
                            };
                            let end = start.checked_add(len).ok_or(Error::DecodeError("mkv: block end overflow"))?;
                            let ranges = &mut self.frame_ranges;
                            let tracks = &self.track_states;
                            return self.iter.with_binary_range(start, len, |reader| {
                                extract_frame_ranges(reader, end, duration, cluster_ts, tracks, ranges)
                            });
                        }

                        // Get block data and duration.
                        let (data, duration) = match block_type {
                            MkvElement::SimpleBlock => (self.iter.read_binary()?, None),
                            MkvElement::BlockGroup => {
                                let group = self.iter.read_master_element::<BlockGroupElement>()?;
                                (group.data, group.duration)
                            }
                            _ => unreachable!(),
                        };

                        // Extract frames.
                        if !extract_frames(
                            &data,
                            duration,
                            cluster_ts,
                            &self.track_states,
                            &mut self.frames,
                        )? {
                            warn!("pts for block is too large");
                            return Ok(false);
                        }
                    }
                    // All other elements.
                    other => {
                        log::debug!("ignored element {other:?}");
                    }
                }
            }
        }

        Ok(true)
    }
}

impl ProbeableFormat<'_> for MkvReader<'_> {
    fn try_probe_new(
        mss: MediaSourceStream<'_>,
        opts: FormatOptions,
    ) -> Result<Box<dyn FormatReader + '_>>
    where
        Self: Sized,
    {
        Ok(Box::new(MkvReader::try_new(mss, opts)?))
    }

    fn probe_data() -> &'static [ProbeFormatData] {
        &[support_format!(
            MKV_FORMAT_INFO,
            &["webm", "mkv"],
            &["video/webm", "video/x-matroska"],
            &[b"\x1A\x45\xDF\xA3"] // Top-level element Ebml element
        )]
    }
}

impl FormatReader for MkvReader<'_> {
    fn format_info(&self) -> &FormatInfo {
        &MKV_FORMAT_INFO
    }

    fn media_info(&self) -> &MediaInfo {
        &self.media_info
    }

    fn attachments(&self) -> &[Attachment] {
        &self.attachments
    }

    fn chapters(&self) -> Option<&ChapterGroup> {
        self.chapters.as_ref()
    }

    fn metadata(&mut self) -> Metadata<'_> {
        self.metadata.metadata()
    }

    fn seek(&mut self, _mode: SeekMode, to: SeekTo) -> Result<SeekedTo> {
        if !self.retain_cues || self.tracks.is_empty() {
            return seek_error(SeekErrorKind::Unseekable);
        }

        match to {
            SeekTo::Time { time, track_id } => {
                let track = match track_id {
                    Some(id) => self.tracks.iter().find(|track| track.id == id),
                    None => self.tracks.first(),
                };
                let track = track.ok_or(Error::SeekError(SeekErrorKind::InvalidTrack))?;
                let tb = track.time_base.expect("track always has a timebase");
                let ts = match tb.calc_timestamp(time) {
                    Some(ts) => ts,
                    None => {
                        warn!("seek ts is too large");
                        return seek_error(SeekErrorKind::OutOfRange);
                    }
                };
                let track_id = track.id;
                self.seek_track_by_ts_atomic(track_id, tb, ts)
            }
            SeekTo::Timestamp { ts, track_id } => {
                match self.tracks.iter().find(|t| t.id == track_id) {
                    Some(track) => {
                        let tb = track.time_base.expect("track always has a timebase");
                        self.seek_track_by_ts_atomic(track_id, tb, ts)
                    }
                    None => seek_error(SeekErrorKind::InvalidTrack),
                }
            }
        }
    }

    fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    fn next_packet(&mut self) -> Result<Option<Packet>> {
        loop {
            if self.pending_pcm.is_some() {
                return self.next_pcm_packet().map(Some);
            }
            if let Some(frame) = self.frame_ranges.pop_front() {
                if self.selected_track.is_some_and(|selected| selected != frame.track_num) {
                    continue;
                }
                if let Some(layout) = &self.batch_pcm {
                    if frame.track_num == layout.track_id && frame.len != 0 {
                        if frame.len % layout.frame_bytes != 0 {
                            return Err(Error::DecodeError(
                                "mkv: PCM frame has incomplete interleaved samples",
                            ));
                        }
                        let total_frames = frame.len / layout.frame_bytes;
                        self.pending_pcm = Some(PendingPcm {
                            frame,
                            total_frames,
                            consumed_frames: 0,
                        });
                        return self.next_pcm_packet().map(Some);
                    }
                }
                let prefix_len = frame.len.min(self.batch_vorbis_prefix_len.unwrap_or(frame.len));
                let size = usize::try_from(prefix_len)
                    .map_err(|_| Error::DecodeError("mkv: frame too large"))?;
                let data = self.iter.with_binary_range(frame.start, frame.len, |mut reader| {
                    use symphonia_core::io::ReadBytes;
                    Ok(reader.read_boxed_slice_exact(size)?)
                })?;
                return Ok(Some(Packet::new(frame.track_num, frame.pts.into_ts(), frame.dur.into_dur(), data)));
            }
            if let Some(frame) = self.frames.pop_front() {
                return Ok(Some(Packet::new(
                    frame.track_num,
                    frame.pts.into_ts(),
                    frame.dur.into_dur(),
                    frame.data,
                )));
            }

            if !self.next_element()? {
                // Reached the end of stream.
                return Ok(None);
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

impl Scoreable for MkvReader<'_> {
    fn score(_src: ScopedStream<&mut MediaSourceStream<'_>>) -> Result<Score> {
        Ok(Score::Supported(255))
    }
}

impl ReadEbml for MediaSourceStream<'_> {}

impl From<EbmlError> for Error {
    fn from(value: EbmlError) -> Self {
        // All non-IO EBML errors are mapped to a decode error.
        let msg = match value {
            EbmlError::IoError(err) => return Error::IoError(err),
            EbmlError::InvalidEbmlElementIdLength => "mkv (ebml): invalid ebml element id length",
            EbmlError::InvalidEbmlDataLength => "mkv (ebml): invalid ebml vint length",
            EbmlError::UnknownElement => "mkv (ebml): the element is unknown",
            EbmlError::UnknownElementDataSize => "mkv (ebml): the element data size is unknown",
            EbmlError::UnexpectedElement => "mkv (ebml): encountered an unexpected element",
            EbmlError::UnexpectedElementDataType => {
                "mkv (ebml): unexpected element data type for the operation"
            }
            EbmlError::UnexpectedElementDataSize => {
                "mkv (ebml): unexpected data size for the element's data type"
            }
            EbmlError::NoElement => "mkv (ebml): no current element",
            EbmlError::NoParent => "mkv (ebml): no parent element",
            EbmlError::NotAnAncestor => {
                "mkv (ebml): the element is not an ancestor of the current element"
            }
            EbmlError::Overrun => "mkv (ebml): the element was overrun when read",
            EbmlError::ExpectedMasterElement => "mkv (ebml): expected a master element",
            EbmlError::ExpectedNonMasterElement => "mkv (ebml): expected a non-master element",
            EbmlError::SeekOutOfRange => "mkv (ebml): the seek was out of range",
            EbmlError::BufferTooSmall => "mkv (ebml): the buffer is too small",
            EbmlError::MaximumDepthReached => "mkv (ebml): maximum ebml document depth reached",
            EbmlError::ElementError(err) => err,
        };
        Error::DecodeError(msg)
    }
}
