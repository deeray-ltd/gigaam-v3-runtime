// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

use std::sync::LazyLock;
use symphonia::core::{
    errors::Result,
    formats::{
        Attachment, FormatInfo, FormatOptions, FormatReader, MediaInfo, SeekMode, SeekTo, SeekedTo,
        Track, TrackType,
        probe::{Probe, ProbeFormatData, ProbeableFormat, Score, Scoreable},
    },
    io::{MediaSourceStream, ScopedStream},
    meta::{ChapterGroup, Metadata},
    packet::Packet,
};
use symphonia::default::formats::{
    AdtsReader, FlacReader, IsoMp4Reader, MkvReader, MpaReader, OggReader, WavReader,
};

struct Sequential<R>(R);

impl<R: FormatReader> FormatReader for Sequential<R> {
    fn format_info(&self) -> &FormatInfo {
        self.0.format_info()
    }
    fn media_info(&self) -> &MediaInfo {
        self.0.media_info()
    }
    fn attachments(&self) -> &[Attachment] {
        self.0.attachments()
    }
    fn chapters(&self) -> Option<&ChapterGroup> {
        self.0.chapters()
    }
    fn metadata(&mut self) -> Metadata<'_> {
        self.0.metadata()
    }
    fn seek(&mut self, mode: SeekMode, to: SeekTo) -> Result<SeekedTo> {
        self.0.seek(mode, to)
    }
    fn tracks(&self) -> &[Track] {
        self.0.tracks()
    }
    fn next_packet(&mut self) -> Result<Option<Packet>> {
        self.0.next_packet()
    }
    fn into_inner<'s>(self: Box<Self>) -> MediaSourceStream<'s>
    where
        Self: 's,
    {
        Box::new(self.0).into_inner()
    }
}

impl<R: Scoreable> Scoreable for Sequential<R> {
    fn score(src: ScopedStream<&mut MediaSourceStream<'_>>) -> Result<Score> {
        R::score(src)
    }
}

impl<'input> ProbeableFormat<'input> for Sequential<IsoMp4Reader<'_>> {
    fn try_probe_new(
        mss: MediaSourceStream<'input>,
        opts: FormatOptions,
    ) -> Result<Box<dyn FormatReader + 'input>> {
        let mut reader = IsoMp4Reader::try_new_with_table_storage(mss, opts, true)?;
        let track = reader
            .first_track_known_codec(TrackType::Audio)
            .ok_or(symphonia::core::errors::Error::Unsupported(
                "no audio track",
            ))?
            .id;
        reader.select_for_batch(track)?;
        Ok(Box::new(reader))
    }
    fn probe_data() -> &'static [ProbeFormatData] {
        IsoMp4Reader::probe_data()
    }
}

impl<'input> ProbeableFormat<'input> for Sequential<MkvReader<'_>> {
    fn try_probe_new(
        mss: MediaSourceStream<'input>,
        opts: FormatOptions,
    ) -> Result<Box<dyn FormatReader + 'input>> {
        Ok(Box::new(MkvReader::try_new_with_cues(mss, opts, false)?))
    }
    fn probe_data() -> &'static [ProbeFormatData] {
        MkvReader::probe_data()
    }
}

pub(crate) static BATCH_PROBE: LazyLock<Probe> = LazyLock::new(|| {
    let mut probe = Probe::default();
    probe.register_metadata::<symphonia::default::meta::Id3v2Reader<'_>>();
    probe.register_format::<AdtsReader<'_>>();
    probe.register_format::<FlacReader<'_>>();
    probe.register_format::<MpaReader<'_>>();
    probe.register_format::<WavReader<'_>>();
    probe.register_format::<OggReader<'_>>();
    probe.register_format::<Sequential<IsoMp4Reader<'_>>>();
    probe.register_format::<Sequential<MkvReader<'_>>>();
    probe
});
