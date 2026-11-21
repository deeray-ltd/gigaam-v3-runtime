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

mod vorbis {
    use super::*;
    use std::error::Error;

    // A generated 48 kHz mono sine encoded with FFmpeg/libvorbis, with optional
    // indexing and metadata removed so the two payloads can be varied independently.
    const CONTROL: &[u8] = include_bytes!("fixtures/vorbis-sine.mkv");
    const SEGMENT: &[u8] = b"\x18\x53\x80\x67";
    const INFO: &[u8] = b"\x15\x49\xA9\x66";
    const TITLE: &[u8] = b"\x7B\xA9";
    const MUXING_APP: &[u8] = b"\x4D\x80";
    const WRITING_APP: &[u8] = b"\x57\x41";
    const TRACKS: &[u8] = b"\x16\x54\xAE\x6B";
    const TRACK_ENTRY: &[u8] = b"\xAE";
    const CODEC_PRIVATE: &[u8] = b"\x63\xA2";
    const CLUSTER: &[u8] = b"\x1F\x43\xB6\x75";
    const SIMPLE_BLOCK: &[u8] = b"\xA3";
    const SMALL_PADDING: usize = 256 * 1024;
    const LARGE_PADDING: usize = 2 * 1024 * 1024;

    type Element<'a> = (&'a [u8], &'a [u8]);

    fn children(mut input: &[u8]) -> Result<Vec<Element<'_>>, Box<dyn Error>> {
        let mut children = Vec::new();
        while let Some(&first) = input.first() {
            if first == 0 {
                return Err("invalid test element ID".into());
            }
            let id_width = usize::try_from(first.leading_zeros() + 1)?;
            let (id, rest) = input
                .split_at_checked(id_width)
                .ok_or("truncated test ID")?;
            let first_size = *rest
                .first()
                .filter(|&&byte| byte != 0)
                .ok_or("invalid test size")?;
            let size_width = first_size.leading_zeros() + 1;
            let (size, rest) = rest
                .split_at_checked(usize::try_from(size_width)?)
                .ok_or("truncated test size")?;
            let marked = size
                .iter()
                .fold(0_u64, |value, &byte| value << 8 | u64::from(byte));
            let marker = 1_u64 << (size_width * 7);
            let length = marked - marker;
            if length == marker - 1 {
                return Err("unknown-size test element".into());
            }
            let (body, tail) = rest
                .split_at_checked(usize::try_from(length)?)
                .ok_or("truncated test element")?;
            children.push((id, body));
            input = tail;
        }
        Ok(children)
    }

    fn read_xiph_length(input: &mut &[u8]) -> Result<usize, Box<dyn Error>> {
        let mut length = 0;
        loop {
            let (&byte, rest) = input.split_first().ok_or("truncated test packet lengths")?;
            *input = rest;
            length += usize::from(byte);
            if byte != 255 {
                return Ok(length);
            }
        }
    }

    fn write_xiph_length(output: &mut Vec<u8>, length: usize) -> Result<(), Box<dyn Error>> {
        output.resize(output.len() + length / 255, 255);
        output.push(u8::try_from(length % 255)?);
        Ok(())
    }

    fn with_headers(
        tracks: &[u8],
        description_bytes: usize,
        setup_padding: usize,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        let entries = children(tracks)?;
        let [(TRACK_ENTRY, track)] = entries.as_slice() else {
            return Err("expected one test track".into());
        };
        let fields = children(track)?;
        let mut private = fields
            .iter()
            .find(|(id, _)| *id == CODEC_PRIVATE)
            .map(|(_, body)| *body)
            .ok_or("missing test CodecPrivate")?;
        let Some((&2, rest)) = private.split_first() else {
            return Err("expected three test Vorbis headers".into());
        };
        private = rest;
        let ident_size = read_xiph_length(&mut private)?;
        let comment_size = read_xiph_length(&mut private)?;
        let (ident, rest) = private
            .split_at_checked(ident_size)
            .ok_or("truncated test identification header")?;
        let (_, setup) = rest
            .split_at_checked(comment_size)
            .ok_or("truncated test comment")?;
        assert!(ident.starts_with(b"\x01vorbis"));
        assert!(setup.starts_with(b"\x05vorbis"));

        let mut comment = b"\x03vorbis".to_vec();
        comment.extend(0_u32.to_le_bytes()); // Empty vendor string.
        comment.extend(1_u32.to_le_bytes()); // One comment field.
        comment.extend(u32::try_from(b"DESCRIPTION=".len() + description_bytes)?.to_le_bytes());
        comment.extend(b"DESCRIPTION=");
        comment.resize(comment.len() + description_bytes, b'x');
        comment.push(1); // Vorbis framing flag.

        let mut private = vec![2];
        write_xiph_length(&mut private, ident.len())?;
        write_xiph_length(&mut private, comment.len())?;
        private.extend(ident);
        private.extend(comment);
        private.extend(setup);
        private.resize(private.len() + setup_padding, 0);

        // Put CodecPrivate before CodecID to exercise field-order independence.
        let mut updated = element(CODEC_PRIVATE, private)?;
        for (id, body) in fields {
            if id != CODEC_PRIVATE {
                updated.extend(element(id, body.to_vec())?);
            }
        }
        Ok(element(TRACK_ENTRY, updated)?)
    }

    fn fixture(
        comment: Option<usize>,
        packet_padding: usize,
        setup_padding: usize,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut result = Vec::new();
        let mut packet = 0;
        for (id, body) in children(CONTROL)? {
            if id != SEGMENT {
                result.extend(element(id, body.to_vec())?);
                continue;
            }
            let mut segment = Vec::new();
            for (id, body) in children(body)? {
                let updated = match (id, comment) {
                    (TRACKS, Some(count)) => with_headers(body, count, setup_padding)?,
                    (CLUSTER, _) if packet_padding != 0 => {
                        let mut cluster = Vec::new();
                        for (id, body) in children(body)? {
                            let mut payload = body.to_vec();
                            if id == SIMPLE_BLOCK {
                                packet += 1;
                                if packet == 2 {
                                    // Track 1, a 16-bit relative timestamp, and an unlaced block.
                                    assert_eq!(payload[0], 0x81);
                                    assert_eq!(payload[3] & 6, 0);
                                    payload.resize(payload.len() + packet_padding, 0);
                                }
                            }
                            cluster.extend(element(id, payload)?);
                        }
                        cluster
                    }
                    _ => body.to_vec(),
                };
                segment.extend(element(id, updated)?);
            }
            result.extend(element(id, segment)?);
        }
        assert!(packet_padding == 0 || packet >= 2);
        Ok(result)
    }

    fn fixture_with_segment_element(
        target: &[u8],
        replace: impl Fn(&[u8]) -> Result<Vec<u8>, Box<dyn Error>>,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut file = Vec::new();
        for (id, body) in children(CONTROL)? {
            if id == SEGMENT {
                let mut segment = Vec::new();
                for (child, payload) in children(body)? {
                    let updated = if child == target {
                        replace(payload)?
                    } else {
                        payload.to_vec()
                    };
                    segment.extend(element(child, updated)?);
                }
                file.extend(element(id, segment)?);
            } else {
                file.extend(element(id, body.to_vec())?);
            }
        }
        Ok(file)
    }

    struct Decoded {
        pcm: Vec<u32>,
        open_bytes: usize,
        total_bytes: usize,
    }

    fn decode(input: Vec<u8>) -> Result<Decoded, Box<dyn Error>> {
        let bytes_read = Arc::new(AtomicUsize::new(0));
        let source = CountedSource {
            cursor: Cursor::new(input),
            bytes_read: Arc::clone(&bytes_read),
        };
        let providers: Vec<Arc<dyn FileDecoderProvider>> =
            vec![Arc::new(gigaam_format_codecs::Containers)];
        let mut decoder = FileDecoders::new(providers)?.open(Box::new(source), Some("mkv"))?;
        let open_bytes = bytes_read.load(Ordering::Relaxed);
        assert_eq!(decoder.metadata().sample_rate().hertz(), 48_000);
        assert_eq!(decoder.metadata().channel_count().get(), 1);
        let mut pcm = Vec::new();
        let chunk_size = NonZeroUsize::new(2048).ok_or("invalid chunk maximum")?;
        while let Some(chunk) = decoder.next_chunk(chunk_size)? {
            assert_eq!(chunk.first_frame(), u64::try_from(pcm.len())?);
            pcm.extend(
                chunk.audio().channels()[0]
                    .samples()
                    .iter()
                    .map(|sample| sample.to_bits()),
            );
            assert_eq!(chunk.end_frame(), u64::try_from(pcm.len())?);
        }
        assert!(decoder.next_chunk(chunk_size)?.is_none());
        assert_eq!(pcm.len(), 14_912);
        assert!(pcm.iter().any(|&bits| bits != 0));
        Ok(Decoded {
            pcm,
            open_bytes,
            total_bytes: bytes_read.load(Ordering::Relaxed),
        })
    }

    #[test]
    fn batch_ignores_vorbis_comment_body_before_codec_id() -> Result<(), Box<dyn Error>> {
        let baseline = decode(CONTROL.to_vec())?;
        let small = decode(fixture(Some(SMALL_PADDING), 0, 0)?)?;
        let large = decode(fixture(Some(LARGE_PADDING), 0, 0)?)?;
        assert_eq!(baseline.pcm, small.pcm);
        assert_eq!(baseline.pcm, large.pcm);
        // A longer comment requires more lacing bytes, plus buffered reads at
        // the ends. Its body must be skipped without a proportional read.
        let expected_growth = (LARGE_PADDING - SMALL_PADDING).div_ceil(255) + 64 * 1024;
        assert!(large.open_bytes <= small.open_bytes + expected_growth);
        assert!(large.total_bytes < SMALL_PADDING);
        Ok(())
    }

    #[test]
    fn batch_ignores_vorbis_audio_packet_suffix() -> Result<(), Box<dyn Error>> {
        let baseline = decode(CONTROL.to_vec())?;
        let small = decode(fixture(None, SMALL_PADDING, 0)?)?;
        let large = decode(fixture(None, LARGE_PADDING, 0)?)?;
        assert_eq!(baseline.pcm, small.pcm);
        assert_eq!(baseline.pcm, large.pcm);
        assert!(large.open_bytes <= small.open_bytes + 64 * 1024);
        assert!(large.total_bytes < SMALL_PADDING);
        Ok(())
    }

    #[test]
    fn batch_reads_only_the_required_vorbis_setup() -> Result<(), Box<dyn Error>> {
        let baseline = decode(CONTROL.to_vec())?;
        let small = decode(fixture(Some(0), 0, SMALL_PADDING)?)?;
        let large = decode(fixture(Some(0), 0, LARGE_PADDING)?)?;
        assert_eq!(baseline.pcm, small.pcm);
        assert_eq!(baseline.pcm, large.pcm);
        assert!(large.open_bytes <= small.open_bytes + 64 * 1024);
        assert!(large.total_bytes < SMALL_PADDING);
        Ok(())
    }

    #[test]
    fn batch_initializes_only_the_selected_vorbis_setup() -> Result<(), Box<dyn Error>> {
        const TRACK_NUMBER: &[u8] = b"\xD7";
        const TRACK_UID: &[u8] = b"\x73\xC5";
        let baseline = decode(CONTROL.to_vec())?;
        let input = fixture_with_segment_element(TRACKS, |body| {
            let entries = children(body)?;
            let [(TRACK_ENTRY, track)] = entries.as_slice() else {
                return Err("expected one original test track".into());
            };
            let mut second = Vec::new();
            for (id, value) in children(track)? {
                match id {
                    TRACK_NUMBER | TRACK_UID => second.extend(element(id, vec![2])?),
                    CODEC_PRIVATE => {
                        let mut packets = value.get(1..).ok_or("missing test Vorbis headers")?;
                        let ident_len = read_xiph_length(&mut packets)?;
                        let comment_len = read_xiph_length(&mut packets)?;
                        let setup_start = value.len() - packets.len() + ident_len + comment_len;
                        assert!(value[setup_start..].starts_with(b"\x05vorbis"));
                        // The secondary track has never been decoded. Its abbreviated
                        // setup must not prevent transcription of the first track.
                        second.extend(element(id, value[..setup_start + 7].to_vec())?);
                    }
                    _ => second.extend(element(id, value.to_vec())?),
                }
            }
            let mut tracks = body.to_vec();
            tracks.extend(element(TRACK_ENTRY, second)?);
            Ok(tracks)
        })?;
        assert_eq!(decode(input)?.pcm, baseline.pcm);
        Ok(())
    }

    #[test]
    fn batch_skips_segment_info_text_and_preserves_pcm() -> Result<(), Box<dyn Error>> {
        let baseline = decode(CONTROL.to_vec())?;
        for length in [0, SMALL_PADDING, LARGE_PADDING] {
            let input = fixture_with_segment_element(INFO, |body| {
                let mut info = Vec::new();
                for (id, payload) in children(body)? {
                    if ![TITLE, MUXING_APP, WRITING_APP].contains(&id) {
                        info.extend(element(id, payload.to_vec())?);
                    }
                }
                for id in [TITLE, MUXING_APP, WRITING_APP] {
                    info.extend(element(id, vec![b'x'; length])?);
                }
                Ok(info)
            })?;
            let result = decode(input)?;
            assert_eq!(result.pcm, baseline.pcm);
            assert!(result.total_bytes < SMALL_PADDING);
        }
        Ok(())
    }

    #[test]
    fn batch_refuses_missing_or_overrunning_required_segment_info_text()
    -> Result<(), Box<dyn Error>> {
        for missing in [MUXING_APP, WRITING_APP] {
            let input = fixture_with_segment_element(INFO, |body| {
                let mut info = Vec::new();
                for (id, payload) in children(body)? {
                    if id != missing {
                        info.extend(element(id, payload.to_vec())?);
                    }
                }
                Ok(info)
            })?;
            assert!(decode(input).is_err());
        }

        let input = fixture_with_segment_element(INFO, |body| {
            let mut info = Vec::new();
            for (id, payload) in children(body)? {
                if id != WRITING_APP {
                    info.extend(element(id, payload.to_vec())?);
                }
            }
            let mut writing_app = element(WRITING_APP, vec![b'x'; 64])?;
            writing_app.pop().ok_or("nonempty writing app")?;
            info.extend(writing_app);
            Ok(info)
        })?;
        assert!(decode(input).is_err());
        Ok(())
    }
}

fn uint(id: &[u8], value: u64) -> Result<Vec<u8>, String> {
    let bytes = value.to_be_bytes();
    let first = bytes.iter().position(|byte| *byte != 0).unwrap_or(7);
    element(id, bytes[first..].to_vec())
}

mod flac {
    use super::*;
    use gigaam_audio::FileDecoder;

    // A self-contained 16-sample, mono FLAC frame with a verbatim subframe
    // and valid header and frame CRCs. The source samples are given in the test.
    const FRAME: [u8; 42] = [
        0xff, 0xf8, 0x60, 0x08, 0x00, 0x0f, 0x96, 0x02, 0x00, 0x00, 0x20, 0x00, 0xe0, 0x00, 0x40,
        0x00, 0xc0, 0x00, 0x7f, 0xff, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0xe0, 0x00,
        0x40, 0x00, 0xc0, 0x00, 0x7f, 0xff, 0x80, 0x00, 0x00, 0x00, 0x14, 0x45,
    ];

    fn codec_private(padding_blocks: &[usize]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut bytes = b"fLaC".to_vec();
        bytes.extend([if padding_blocks.is_empty() { 0x80 } else { 0 }, 0, 0, 34]);
        bytes.extend([0, 16, 0, 16]); // Minimum and maximum block size.
        bytes.extend([0; 6]); // Unknown minimum and maximum frame size.
        let stream_format = (16_000_u64 << 44) | (15 << 36) | 16;
        bytes.extend(stream_format.to_be_bytes()); // 16 kHz, mono, 16 bits, 16 samples.
        bytes.extend([0; 16]); // No STREAMINFO MD5 signature.
        for (index, &length) in padding_blocks.iter().enumerate() {
            let [high, a, b, c] = u32::try_from(length)?.to_be_bytes();
            assert_eq!(high, 0);
            let kind = if index + 1 == padding_blocks.len() {
                0x81
            } else {
                1
            };
            bytes.extend([kind, a, b, c]);
            bytes.resize(bytes.len() + length, 0);
        }
        Ok(bytes)
    }

    fn fixture(private: Vec<u8>) -> Result<Vec<u8>, String> {
        let mut ebml = element(&[0x42, 0x82], b"matroska".to_vec())?;
        ebml.extend(uint(&[0x42, 0xf7], 1)?);
        ebml.extend(uint(&[0x42, 0x87], 2)?);
        ebml.extend(uint(&[0x42, 0x85], 2)?);
        let mut file = element(&[0x1a, 0x45, 0xdf, 0xa3], ebml)?;
        let mut info = element(&[0x4d, 0x80], b"fixture".to_vec())?;
        info.extend(element(&[0x57, 0x41], b"fixture".to_vec())?);
        let mut segment = element(&[0x15, 0x49, 0xa9, 0x66], info)?;

        let mut track = uint(&[0xd7], 1)?;
        track.extend(uint(&[0x73, 0xc5], 1)?);
        track.extend(uint(&[0x83], 2)?);
        track.extend(element(&[0x86], b"A_FLAC".to_vec())?);
        track.extend(element(&[0x63, 0xa2], private)?);
        let mut audio = element(&[0xb5], 16_000_f64.to_be_bytes().to_vec())?;
        audio.extend(uint(&[0x9f], 1)?);
        audio.extend(uint(&[0x62, 0x64], 16)?);
        track.extend(element(&[0xe1], audio)?);
        segment.extend(element(
            &[0x16, 0x54, 0xae, 0x6b],
            element(&[0xae], track)?,
        )?);

        let mut cluster = uint(&[0xe7], 0)?;
        let mut packet = vec![0x81, 0, 0, 0x80];
        packet.extend(FRAME);
        cluster.extend(element(&[0xa3], packet)?);
        segment.extend(element(&[0x1f, 0x43, 0xb6, 0x75], cluster)?);
        file.extend(element(&[0x18, 0x53, 0x80, 0x67], segment)?);
        Ok(file)
    }

    fn open(bytes: Vec<u8>, bytes_read: &Arc<AtomicUsize>) -> Result<FileDecoder, String> {
        let providers: Vec<Arc<dyn FileDecoderProvider>> =
            vec![Arc::new(gigaam_format_codecs::Containers)];
        let source = CountedSource {
            cursor: Cursor::new(bytes),
            bytes_read: Arc::clone(bytes_read),
        };
        FileDecoders::new(providers)?.open(Box::new(source), Some("mkv"))
    }

    #[test]
    fn batch_flac_reads_streaminfo_without_loading_optional_metadata()
    -> Result<(), Box<dyn std::error::Error>> {
        let samples = [0_i16, 8192, -8192, 16384, -16384, i16::MAX, i16::MIN, 0].repeat(2);
        let expected: Vec<_> = samples
            .into_iter()
            .map(|sample| f32::from(sample) / 32768.0)
            .collect();
        let chunk_limit = NonZeroUsize::new(7).ok_or("positive chunk limit")?;
        let small = 256 * 1024;
        let large = 2 * 1024 * 1024;
        for padding in [&[][..], &[small], &[large], &[large, large, large]] {
            let bytes_read = Arc::new(AtomicUsize::new(0));
            let mut decoder = open(fixture(codec_private(padding)?)?, &bytes_read)?;
            assert_eq!(decoder.metadata().sample_rate().hertz(), 16_000);
            assert_eq!(decoder.metadata().channel_count().get(), 1);
            let mut pcm = Vec::new();
            while let Some(chunk) = decoder.next_chunk(chunk_limit)? {
                assert_eq!(chunk.first_frame(), u64::try_from(pcm.len())?);
                pcm.extend_from_slice(chunk.audio().channels()[0].samples());
                assert_eq!(chunk.end_frame(), u64::try_from(pcm.len())?);
            }
            assert_eq!(pcm, expected);
            assert!(decoder.next_chunk(chunk_limit)?.is_none());
            assert!(
                bytes_read.load(Ordering::Relaxed) < small,
                "decoder must not read the body of any large optional FLAC metadata block"
            );
        }
        Ok(())
    }

    #[test]
    fn batch_flac_refuses_truncated_declared_extents() -> Result<(), Box<dyn std::error::Error>> {
        // A preceding PADDING header claims bytes outside CodecPrivate.
        let private = b"fLaC\x01\x00\x01\x00\x00".to_vec();
        let bytes_read = Arc::new(AtomicUsize::new(0));
        assert!(open(fixture(private)?, &bytes_read).is_err());

        // STREAMINFO is complete, but the containing CodecPrivate ends past the file.
        let mut truncated = fixture(codec_private(&[64])?)?;
        let start = truncated
            .windows(4)
            .position(|span| span == b"fLaC")
            .ok_or("FLAC marker")?;
        truncated.truncate(start + 4 + 4 + 34 + 4 + 1);
        assert!(open(truncated, &bytes_read).is_err());
        Ok(())
    }
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
