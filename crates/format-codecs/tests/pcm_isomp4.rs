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

struct CountedSource {
    data: Cursor<Vec<u8>>,
    bytes_read: Arc<AtomicUsize>,
}

impl Read for CountedSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let count = self.data.read(buf)?;
        self.bytes_read.fetch_add(count, Ordering::Relaxed);
        Ok(count)
    }
}

impl Seek for CountedSource {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.data.seek(position)
    }
}

fn atom(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = u32::try_from(payload.len() + 8).expect("test atoms fit in a u32");
    [&size.to_be_bytes(), kind, payload].concat()
}

fn words(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect()
}

fn sample(frame: usize) -> (i16, i16) {
    let left = if frame.is_multiple_of(2) {
        -16384
    } else {
        32767
    };
    let right = if frame.is_multiple_of(3) {
        8192
    } else {
        -32768
    };
    (left, right)
}

#[derive(Clone, Copy)]
enum MediaPacket {
    Complete,
    LegacyRemainder,
    TruncatedExtent,
}

#[derive(Default)]
struct MovieHeaders<'a> {
    edits: &'a [u8],
    compatible_brands: Option<&'a [u8]>,
}

fn nonmixable_mov(frames: usize, packet: MediaPacket, headers: MovieHeaders<'_>) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(frames * 4);
    if matches!(packet, MediaPacket::LegacyRemainder) {
        pcm.push(0x31);
    }
    for frame in 0..frames {
        let (left, right) = sample(frame);
        pcm.extend(left.to_le_bytes());
        pcm.extend(right.to_le_bytes());
    }
    if matches!(packet, MediaPacket::LegacyRemainder) {
        pcm.push(0x52);
    }
    let bytes = u32::try_from(pcm.len()).expect("fixture PCM size fits in a u32");
    let frames = u32::try_from(frames).expect("fixture frame count fits in a u32");
    let brands = headers.compatible_brands.unwrap_or(b"qt  ");
    let ftyp = atom(b"ftyp", &[b"qt  \0\0\0\0".as_slice(), brands].concat());
    let offset = u32::try_from(ftyp.len() + 8).expect("media offset fits in a u32");
    let mdat = atom(b"mdat", &pcm);

    let entry = [
        &words(&[0, 1, 2 << 16, 0, (3 << 16) | 16, 0xfffe0000, 65536, 72])[..],
        &8000.0_f64.to_be_bytes(),
        // Two channels, S16, signed + packed + nonmixable flags, one frame per packet.
        &words(&[2, 0x7f000000, 16, 0x4c, 4, 1]),
    ]
    .concat();
    let (stts, stsc, stsz) = match packet {
        MediaPacket::Complete => (
            words(&[0, 1, 1, frames]),
            words(&[0, 1, 1, 1, 1]),
            words(&[0, bytes, 1]),
        ),
        MediaPacket::LegacyRemainder => (
            words(&[0, 2, 1, 0, 1, frames]),
            words(&[0, 1, 1, 2, 1]),
            words(&[0, 0, 2, 1, bytes - 1]),
        ),
        MediaPacket::TruncatedExtent => (
            words(&[0, 1, 1, frames]),
            words(&[0, 1, 1, 1, 1]),
            words(&[0, bytes + 1024, 1]),
        ),
    };
    let stbl = atom(
        b"stbl",
        &[
            atom(b"stsd", &[words(&[0, 1]), atom(b"lpcm", &entry)].concat()),
            atom(b"stts", &stts),
            atom(b"stsc", &stsc),
            atom(b"stsz", &stsz),
            atom(b"stco", &words(&[0, 1, offset])),
        ]
        .concat(),
    );
    let mdia = atom(
        b"mdia",
        &[
            atom(b"mdhd", &words(&[0, 0, 0, 8000, frames, 0])),
            atom(b"hdlr", b"\0\0\0\0\0\0\0\0soun\0\0\0\0\0\0\0\0\0\0\0\0"),
            atom(b"minf", &stbl),
        ]
        .concat(),
    );
    let tkhd = atom(b"tkhd", &words(&[0, 0, 0, 1, 0, frames, 0, 0, 0, 0]));
    let moov = atom(
        b"moov",
        &[
            atom(b"mvhd", &words(&[0, 0, 0, 8000, frames, 65536, 1 << 24])),
            atom(b"trak", &[tkhd.as_slice(), headers.edits, &mdia].concat()),
        ]
        .concat(),
    );
    [ftyp, mdat, moov].concat()
}

const FRAMES: usize = 131_072;

#[test]
fn opening_large_nonmixable_lpcm_reads_only_part_of_its_packet() -> Result<(), String> {
    verify_stereo_pcm(MediaPacket::Complete, MovieHeaders::default())
}

#[test]
fn bounded_read_preserves_legacy_packets_with_partial_frames() -> Result<(), String> {
    verify_stereo_pcm(MediaPacket::LegacyRemainder, MovieHeaders::default())
}

#[test]
fn rejects_a_declared_packet_outside_the_source_before_emitting_pcm() -> Result<(), String> {
    let providers: Vec<Arc<dyn FileDecoderProvider>> =
        vec![Arc::new(gigaam_format_codecs::Containers)];
    let source = Cursor::new(nonmixable_mov(
        FRAMES,
        MediaPacket::TruncatedExtent,
        MovieHeaders::default(),
    ));
    assert!(
        FileDecoders::new(providers)?
            .open(Box::new(source), Some("mov"))
            .is_err()
    );
    Ok(())
}

fn edit_list(version: u8, declared_count: u32, records: &[u8]) -> Vec<u8> {
    let payload = [
        &[version, 0, 0, 0],
        &declared_count.to_be_bytes()[..],
        records,
    ]
    .concat();
    atom(b"edts", &atom(b"elst", &payload))
}

#[test]
fn batch_skips_large_edit_lists_and_preserves_pcm_positions() -> Result<(), String> {
    let frames_per_edit = 2;
    let entry_count =
        u32::try_from(FRAMES).expect("fixture frame count fits u32") / frames_per_edit;
    for version in 0..=1 {
        let mut records = Vec::new();
        for entry in 0..entry_count {
            let media_time = entry * frames_per_edit;
            if version == 0 {
                records.extend(frames_per_edit.to_be_bytes());
                records.extend(media_time.to_be_bytes());
            } else {
                records.extend(u64::from(frames_per_edit).to_be_bytes());
                records.extend(u64::from(media_time).to_be_bytes());
            }
            records.extend((1_u32 << 16).to_be_bytes()); // Media rate 1.0.
        }
        let edits = edit_list(version, entry_count, &records);
        let headers = MovieHeaders {
            edits: &edits,
            ..MovieHeaders::default()
        };
        verify_stereo_pcm(MediaPacket::Complete, headers)?;
    }
    Ok(())
}

#[test]
fn batch_rejects_incomplete_or_unsupported_edit_lists() -> Result<(), String> {
    let one_short_entry = words(&[1, 0, 1 << 16]);
    for (version, declared_count) in [(0, 2), (1, 1), (2, 1)] {
        let edits = edit_list(version, declared_count, &one_short_entry);
        let headers = MovieHeaders {
            edits: &edits,
            ..MovieHeaders::default()
        };
        let source = Cursor::new(nonmixable_mov(FRAMES, MediaPacket::Complete, headers));
        let providers: Vec<Arc<dyn FileDecoderProvider>> =
            vec![Arc::new(gigaam_format_codecs::Containers)];
        assert!(
            FileDecoders::new(providers)?
                .open(Box::new(source), Some("mov"))
                .is_err(),
            "version {version}, declared count {declared_count}"
        );
    }
    Ok(())
}

#[test]
fn batch_skips_large_compatible_brand_lists_and_preserves_pcm_positions() -> Result<(), String> {
    let brands = b"qt  ".repeat(FRAMES);
    let headers = MovieHeaders {
        compatible_brands: Some(&brands),
        ..MovieHeaders::default()
    };
    verify_stereo_pcm(MediaPacket::Complete, headers)
}

fn verify_stereo_pcm(packet: MediaPacket, headers: MovieHeaders<'_>) -> Result<(), String> {
    // The top-level scan may reach the end and then seek back to the first media atom.
    // Even with both scans, opening cannot read all entries of a large brand list.
    let open_read_limit = headers.compatible_brands.map_or(FRAMES * 2, <[u8]>::len);
    let bytes_read = Arc::new(AtomicUsize::new(0));
    let source = CountedSource {
        data: Cursor::new(nonmixable_mov(FRAMES, packet, headers)),
        bytes_read: Arc::clone(&bytes_read),
    };
    let providers: Vec<Arc<dyn FileDecoderProvider>> =
        vec![Arc::new(gigaam_format_codecs::Containers)];
    let mut decoder = FileDecoders::new(providers)?.open(Box::new(source), Some("mov"))?;
    let opening_bytes = bytes_read.load(Ordering::Relaxed);
    assert!(
        opening_bytes < open_read_limit,
        "opening read {opening_bytes} bytes"
    );
    assert_eq!(decoder.metadata().sample_rate().hertz(), 8000);
    assert_eq!(decoder.metadata().channel_count().get(), 2);

    let limit = NonZeroUsize::new(4097).expect("the consumer chunk size is nonzero");
    let mut position = 0;
    while let Some(chunk) = decoder.next_chunk(limit)? {
        assert_eq!(
            chunk.first_frame(),
            u64::try_from(position).expect("fixture position fits u64")
        );
        let audio = chunk.audio();
        assert!(audio.frames() <= limit.get());
        for (i, (&left, &right)) in audio.channels()[0]
            .samples()
            .iter()
            .zip(audio.channels()[1].samples())
            .enumerate()
        {
            let (expected_left, expected_right) = sample(position + i);
            assert_eq!(left, f32::from(expected_left) / 32768.0);
            assert_eq!(right, f32::from(expected_right) / 32768.0);
        }
        position += audio.frames();
    }
    assert_eq!(position, FRAMES);
    Ok(())
}
