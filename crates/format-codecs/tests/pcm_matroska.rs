// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

use gigaam_audio::{FileDecoderProvider, FileDecoders};
use std::{
    io::{self, Cursor, Read, Seek, SeekFrom},
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use symphonia::core::{
    formats::{FormatOptions, FormatReader},
    io::MediaSourceStream,
};
use symphonia::default::formats::MkvReader;

struct CountedSource {
    cursor: Cursor<Vec<u8>>,
    bytes_read: Arc<AtomicUsize>,
}
impl Read for CountedSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let count = self.cursor.read(buf)?;
        self.bytes_read.fetch_add(count, Ordering::Relaxed);
        Ok(count)
    }
}
impl Seek for CountedSource {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(position)
    }
}

fn element(id: &[u8], payload: Vec<u8>) -> Result<Vec<u8>, String> {
    let len = u64::try_from(payload.len()).map_err(|error| error.to_string())?;
    let width = (1_usize..=8)
        .find(|width| {
            let shift = u32::try_from(7 * width).expect("EBML size width fits u32");
            len < (1_u64 << shift) - 1
        })
        .ok_or("EBML size too large")?;
    let marker = 1_u64 << u32::try_from(7 * width).map_err(|error| error.to_string())?;
    let bytes = (len | marker).to_be_bytes();
    let mut out = Vec::with_capacity(id.len() + width + payload.len());
    out.extend_from_slice(id);
    out.extend_from_slice(&bytes[8 - width..]);
    out.extend_from_slice(&payload);
    Ok(out)
}
fn uint(id: &[u8], value: u64) -> Result<Vec<u8>, String> {
    let bytes = value.to_be_bytes();
    let first = bytes.iter().position(|byte| *byte != 0).unwrap_or(7);
    element(id, bytes[first..].to_vec())
}
fn samples(frame: usize) -> (i16, i16) {
    (
        if frame.is_multiple_of(2) {
            i16::MIN
        } else {
            i16::MAX
        },
        if frame.is_multiple_of(3) {
            8192
        } else {
            -16384
        },
    )
}
fn fixture(
    frames: usize,
    incomplete: bool,
    duration: Option<u64>,
    float: bool,
) -> Result<Vec<u8>, String> {
    let mut ebml = element(&[0x42, 0x82], b"matroska".to_vec())?;
    ebml.extend(uint(&[0x42, 0xf7], 1)?);
    ebml.extend(uint(&[0x42, 0x87], 2)?);
    ebml.extend(uint(&[0x42, 0x85], 2)?);
    let mut file = element(&[0x1a, 0x45, 0xdf, 0xa3], ebml)?;
    let mut info_data = uint(&[0x2a, 0xd7, 0xb1], 1_000_000)?;
    info_data.extend(element(&[0x4d, 0x80], b"fixture".to_vec())?);
    info_data.extend(element(&[0x57, 0x41], b"fixture".to_vec())?);
    let info = element(&[0x15, 0x49, 0xa9, 0x66], info_data)?;
    let mut audio = element(&[0xb5], 8000_f64.to_be_bytes().to_vec())?;
    audio.extend(uint(&[0x9f], 2)?);
    audio.extend(uint(&[0x62, 0x64], if float { 32 } else { 16 })?);
    let mut track = uint(&[0xd7], 1)?;
    track.extend(uint(&[0x73, 0xc5], 1)?);
    track.extend(uint(&[0x83], 2)?);
    track.extend(element(
        &[0x86],
        if float {
            b"A_PCM/FLOAT/IEEE".to_vec()
        } else {
            b"A_PCM/INT/LIT".to_vec()
        },
    )?);
    track.extend(element(&[0xe1], audio)?);
    let tracks = element(&[0x16, 0x54, 0xae, 0x6b], element(&[0xae], track)?)?;
    let mut block = vec![0x81, 0, 0, 0];
    for frame in 0..frames {
        let (left, right) = samples(frame);
        if float {
            block.extend_from_slice(&(f32::from(left) / 32768.0).to_le_bytes());
            block.extend_from_slice(&(f32::from(right) / 32768.0).to_le_bytes());
        } else {
            block.extend_from_slice(&left.to_le_bytes());
            block.extend_from_slice(&right.to_le_bytes());
        }
    }
    if incomplete {
        block.push(7);
    }
    let mut group = element(&[0xa1], block)?;
    if let Some(duration) = duration {
        group.extend(uint(&[0x9b], duration)?);
    }
    let mut cluster = uint(&[0xe7], 100)?;
    cluster.extend(element(&[0xa0], group)?);
    let mut segment = info;
    segment.extend(tracks);
    segment.extend(element(&[0x1f, 0x43, 0xb6, 0x75], cluster)?);
    file.extend(element(&[0x18, 0x53, 0x80, 0x67], segment)?);
    Ok(file)
}
fn reader(bytes: Vec<u8>) -> Result<MkvReader<'static>, Box<dyn std::error::Error>> {
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut reader = MkvReader::try_new_with_cues(source, FormatOptions::default(), false)?;
    reader.select_for_batch(1)?;
    Ok(reader)
}

#[test]
fn large_pcm_block_is_split_before_packet_allocation_and_preserves_pcm_and_time()
-> Result<(), Box<dyn std::error::Error>> {
    const FRAMES: usize = 2_097_152;
    const BOUND: usize = 4097;
    let bytes = fixture(FRAMES, false, Some(262_144), false)?;
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes.clone())), Default::default());
    let mut original = MkvReader::try_new(source, FormatOptions::default())?;
    assert!(
        original
            .enable_batch_pcm(
                1,
                NonZeroUsize::new(4).ok_or("frame")?,
                NonZeroUsize::new(BOUND).ok_or("bound")?
            )
            .is_err()
    );
    let old = original.next_packet()?.ok_or("original packet")?;
    assert_eq!(old.data.len(), FRAMES * 4);
    assert_eq!(old.pts.get(), 100);
    assert_eq!(old.dur.get(), 262_144);
    assert!(original.next_packet()?.is_none());

    let mut split = reader(bytes.clone())?;
    split.enable_batch_pcm(
        1,
        NonZeroUsize::new(4).ok_or("frame")?,
        NonZeroUsize::new(BOUND).ok_or("bound")?,
    )?;
    let mut frame = 0_usize;
    while let Some(packet) = split.next_packet()? {
        assert!(packet.data.len() <= BOUND * 4);
        assert!(packet.data.len().is_multiple_of(4));
        let count = packet.data.len() / 4;
        let start = u64::try_from(frame)? * 262_144 / u64::try_from(FRAMES)?;
        let end = u64::try_from(frame + count)? * 262_144 / u64::try_from(FRAMES)?;
        assert_eq!(packet.pts.get(), 100 + i64::try_from(start)?);
        assert_eq!(packet.dts, packet.pts);
        assert_eq!(packet.dur.get(), end - start);
        assert_eq!(&packet.data[..], &old.data[frame * 4..(frame + count) * 4]);
        frame += count;
    }
    assert_eq!(frame, FRAMES);
    assert_eq!(FRAMES % BOUND, 3585);
    assert!(split.next_packet()?.is_none());

    let providers: Vec<Arc<dyn FileDecoderProvider>> =
        vec![Arc::new(gigaam_format_codecs::Containers)];
    let input_len = bytes.len();
    let bytes_read = Arc::new(AtomicUsize::new(0));
    let source = CountedSource {
        cursor: Cursor::new(bytes),
        bytes_read: Arc::clone(&bytes_read),
    };
    let mut decoded = FileDecoders::new(providers)?.open(Box::new(source), Some("mkv"))?;
    assert!(
        bytes_read.load(Ordering::Relaxed) < input_len / 2,
        "opening must not read the whole large block"
    );
    assert_eq!(decoded.metadata().sample_rate().hertz(), 8000);
    assert_eq!(decoded.metadata().channel_count().get(), 2);
    let mut position = 0_usize;
    while let Some(chunk) = decoded.next_chunk(NonZeroUsize::new(BOUND).ok_or("bound")?)? {
        assert_eq!(chunk.first_frame(), u64::try_from(position)?);
        for local in 0..chunk.audio().frames() {
            let (left, right) = samples(position + local);
            assert_eq!(
                chunk.audio().channels()[0].samples()[local],
                f32::from(left) / 32768.0
            );
            assert_eq!(
                chunk.audio().channels()[1].samples()[local],
                f32::from(right) / 32768.0
            );
        }
        position += chunk.audio().frames();
        assert_eq!(chunk.end_frame(), u64::try_from(position)?);
    }
    assert_eq!(position, FRAMES);
    Ok(())
}

#[test]
fn pcm_tail_refuses_and_other_codec_keeps_original_packet() -> Result<(), Box<dyn std::error::Error>>
{
    let malformed = fixture(17, true, Some(10), false)?;
    let providers: Vec<Arc<dyn FileDecoderProvider>> =
        vec![Arc::new(gigaam_format_codecs::Containers)];
    assert!(
        FileDecoders::new(providers)?
            .open(Box::new(Cursor::new(malformed.clone())), Some("mkv"))
            .is_err()
    );
    let mut incomplete = reader(malformed)?;
    assert!(
        incomplete
            .enable_batch_pcm(
                2,
                NonZeroUsize::new(4).ok_or("frame")?,
                NonZeroUsize::new(4).ok_or("bound")?
            )
            .is_err()
    );
    assert!(
        incomplete
            .enable_batch_pcm(
                1,
                NonZeroUsize::new(3).ok_or("frame")?,
                NonZeroUsize::new(4).ok_or("bound")?
            )
            .is_err()
    );
    incomplete.enable_batch_pcm(
        1,
        NonZeroUsize::new(4).ok_or("frame")?,
        NonZeroUsize::new(4).ok_or("bound")?,
    )?;
    assert!(incomplete.next_packet().is_err());
    let mut truncated_bytes = fixture(17, false, Some(10), false)?;
    truncated_bytes.truncate(truncated_bytes.len() - 1);
    if let Ok(mut truncated) = reader(truncated_bytes) {
        truncated.enable_batch_pcm(
            1,
            NonZeroUsize::new(4).ok_or("frame")?,
            NonZeroUsize::new(4).ok_or("bound")?,
        )?;
        assert!(truncated.next_packet().is_err());
    }
    let mut overflowing_time = reader(fixture(3, false, Some(u64::MAX), false)?)?;
    overflowing_time.enable_batch_pcm(
        1,
        NonZeroUsize::new(4).ok_or("frame")?,
        NonZeroUsize::new(1).ok_or("bound")?,
    )?;
    assert!(overflowing_time.next_packet()?.is_some());
    assert!(overflowing_time.next_packet()?.is_some());
    assert!(overflowing_time.next_packet().is_err());
    let mut unknown_duration = reader(fixture(17, false, None, false)?)?;
    unknown_duration.enable_batch_pcm(
        1,
        NonZeroUsize::new(4).ok_or("frame")?,
        NonZeroUsize::new(4).ok_or("bound")?,
    )?;
    while let Some(packet) = unknown_duration.next_packet()? {
        assert_eq!(packet.pts.get(), 100);
        assert_eq!(packet.dur.get(), 0);
    }
    let mut other = reader(fixture(17, false, Some(10), true)?)?;
    // An unconfigured reader retains the original full float packet.

    assert_eq!(
        other
            .next_packet()?
            .ok_or("original float packet")?
            .data
            .len(),
        17 * 8
    );
    assert!(other.next_packet()?.is_none());
    // A distinct four-byte sample width produces eight-byte stereo frames.
    let mut float = reader(fixture(17, false, Some(10), true)?)?;
    float.enable_batch_pcm(
        1,
        NonZeroUsize::new(8).ok_or("frame")?,
        NonZeroUsize::new(4).ok_or("bound")?,
    )?;
    let mut frames = 0;
    while let Some(packet) = float.next_packet()? {
        assert!(packet.data.len() <= 4 * 8);
        assert!(packet.data.len().is_multiple_of(8));
        frames += packet.data.len() / 8;
    }
    assert_eq!(frames, 17);
    let providers: Vec<Arc<dyn FileDecoderProvider>> =
        vec![Arc::new(gigaam_format_codecs::Containers)];
    const FLOAT_FRAMES: usize = 262_145;
    let bytes = fixture(FLOAT_FRAMES, false, Some(32_768), true)?;
    let input_len = bytes.len();
    let bytes_read = Arc::new(AtomicUsize::new(0));
    let source = CountedSource {
        cursor: Cursor::new(bytes),
        bytes_read: Arc::clone(&bytes_read),
    };
    let mut decoded = FileDecoders::new(providers)?.open(Box::new(source), Some("mkv"))?;
    assert!(bytes_read.load(Ordering::Relaxed) < input_len / 2);
    let mut position = 0;
    while let Some(chunk) = decoded.next_chunk(NonZeroUsize::new(4097).ok_or("bound")?)? {
        assert_eq!(chunk.first_frame(), u64::try_from(position)?);
        for i in 0..chunk.audio().frames() {
            let (left, right) = samples(position + i);
            assert_eq!(
                chunk.audio().channels()[0].samples()[i],
                f32::from(left) / 32768.0
            );
            assert_eq!(
                chunk.audio().channels()[1].samples()[i],
                f32::from(right) / 32768.0
            );
        }
        position += chunk.audio().frames();
    }
    assert_eq!(position, FLOAT_FRAMES);
    Ok(())
}

#[test]
fn batch_skips_unselected_track_before_loading_its_payload()
-> Result<(), Box<dyn std::error::Error>> {
    let mut ebml = element(&[0x42, 0x82], b"matroska".to_vec())?;
    ebml.extend(uint(&[0x42, 0x87], 2)?);
    ebml.extend(uint(&[0x42, 0x85], 2)?);
    let mut file = element(&[0x1a, 0x45, 0xdf, 0xa3], ebml)?;
    let mut info = uint(&[0x2a, 0xd7, 0xb1], 1_000_000)?;
    info.extend(element(&[0x4d, 0x80], b"fixture".to_vec())?);
    info.extend(element(&[0x57, 0x41], b"fixture".to_vec())?);
    let mut tracks = Vec::new();
    for id in [1_u64, 2] {
        let mut audio = element(&[0xb5], 8000_f64.to_be_bytes().to_vec())?;
        audio.extend(uint(&[0x9f], 2)?);
        audio.extend(uint(&[0x62, 0x64], 16)?);
        let mut track = uint(&[0xd7], id)?;
        track.extend(uint(&[0x73, 0xc5], id)?);
        track.extend(uint(&[0x83], 2)?);
        track.extend(element(&[0x86], b"A_PCM/INT/LIT".to_vec())?);
        track.extend(element(&[0xe1], audio)?);
        tracks.extend(element(&[0xae], track)?);
    }
    let mut ignored = vec![0x82, 0, 0, 0];
    ignored.resize(4 + 1024 * 1024, 7);
    let mut selected = vec![0x81, 0, 0, 0];
    for frame in 0..17 {
        let (left, right) = samples(frame);
        selected.extend(left.to_le_bytes());
        selected.extend(right.to_le_bytes());
    }
    let mut cluster = uint(&[0xe7], 0)?;
    cluster.extend(element(&[0xa3], ignored)?);
    cluster.extend(element(&[0xa3], selected)?);
    let mut segment = element(&[0x15, 0x49, 0xa9, 0x66], info)?;
    segment.extend(element(&[0x16, 0x54, 0xae, 0x6b], tracks)?);
    segment.extend(element(&[0x1f, 0x43, 0xb6, 0x75], cluster)?);
    file.extend(element(&[0x18, 0x53, 0x80, 0x67], segment)?);
    let input_len = file.len();
    let mut ordinary = MkvReader::try_new(
        MediaSourceStream::new(Box::new(Cursor::new(file.clone())), Default::default()),
        FormatOptions::default(),
    )?;
    assert_eq!(
        ordinary
            .next_packet()?
            .ok_or("ordinary first packet")?
            .track_id,
        2
    );
    assert_eq!(
        ordinary
            .next_packet()?
            .ok_or("ordinary second packet")?
            .track_id,
        1
    );
    let read = Arc::new(AtomicUsize::new(0));
    let source = CountedSource {
        cursor: Cursor::new(file),
        bytes_read: Arc::clone(&read),
    };
    let providers: Vec<Arc<dyn FileDecoderProvider>> =
        vec![Arc::new(gigaam_format_codecs::Containers)];
    let mut decoder = FileDecoders::new(providers)?.open(Box::new(source), Some("mkv"))?;
    let chunk = decoder
        .next_chunk(NonZeroUsize::new(32).ok_or("bound")?)?
        .ok_or("selected audio")?;
    assert_eq!((chunk.first_frame(), chunk.end_frame()), (0, 17));
    for i in 0..17 {
        let (left, right) = samples(i);
        assert_eq!(
            chunk.audio().channels()[0].samples()[i],
            f32::from(left) / 32768.0
        );
        assert_eq!(
            chunk.audio().channels()[1].samples()[i],
            f32::from(right) / 32768.0
        );
    }
    assert!(
        decoder
            .next_chunk(NonZeroUsize::new(32).ok_or("bound")?)?
            .is_none()
    );
    assert!(
        read.load(Ordering::Relaxed) < input_len / 2,
        "unselected block must not be read"
    );
    Ok(())
}
