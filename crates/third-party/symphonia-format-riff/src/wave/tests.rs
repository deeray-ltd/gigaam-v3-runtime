// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use symphonia_core::codecs::audio::AudioCodecId;
use symphonia_core::codecs::audio::well_known::{CODEC_ID_PCM_F64LE, CODEC_ID_PCM_S16LE};
use symphonia_core::errors::Error;
use symphonia_core::formats::{FormatOptions, FormatReader};
use symphonia_core::io::{MediaSource, MediaSourceStream};

use super::WavReader;

fn chunk(tag: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let length = u32::try_from(body.len()).expect("test chunk fits a RIFF length");
    let mut bytes = tag.to_vec();
    bytes.extend(length.to_le_bytes());
    bytes.extend(body);
    if length & 1 != 0 {
        bytes.push(0);
    }
    bytes
}

fn wave(format: u16, sample_bits: u16, audio: &[u8], info: &[u8]) -> Vec<u8> {
    let sample_rate = 48_000_u32;
    let sample_bytes = sample_bits / 8;
    let mut fmt = format.to_le_bytes().to_vec();
    fmt.extend(1_u16.to_le_bytes());
    fmt.extend(sample_rate.to_le_bytes());
    fmt.extend((sample_rate * u32::from(sample_bytes)).to_le_bytes());
    fmt.extend(sample_bytes.to_le_bytes());
    fmt.extend(sample_bits.to_le_bytes());

    let mut body = b"WAVE".to_vec();
    body.extend(chunk(b"fmt ", &fmt));
    body.extend(chunk(b"LIST", info));
    body.extend(chunk(b"data", audio));
    chunk(b"RIFF", &body)
}

struct Counted {
    inner: Cursor<Vec<u8>>,
    bytes: Arc<AtomicUsize>,
}

impl Read for Counted {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read(output)?;
        self.bytes.fetch_add(count, Ordering::Relaxed);
        Ok(count)
    }
}

impl Seek for Counted {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}

impl MediaSource for Counted {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(u64::try_from(self.inner.get_ref().len()).expect("test source fits u64"))
    }
}

fn source(bytes: Vec<u8>) -> MediaSourceStream<'static> {
    MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default())
}

fn assert_audio(mut reader: WavReader<'_>, expected: &[u8], codec: AudioCodecId) {
    let track = &reader.tracks()[0];
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).expect("one audio track");
    assert_eq!(params.codec, codec);
    assert_eq!(params.sample_rate, Some(48_000));
    assert_eq!(track.num_frames, Some(2));
    let packet = reader.next_packet().expect("packet is valid").expect("there is an audio packet");
    assert_eq!(packet.pts.get(), 0);
    assert_eq!(packet.dur.get(), 2);
    assert_eq!(packet.data.as_ref(), expected);
    assert!(reader.next_packet().expect("reached data end").is_none());
}

#[test]
fn skip_info_preserves_pcm16_and_float64_and_bounds_source_reads() {
    for (format, bits, audio, codec) in [
        (
            1,
            16,
            [i16::MIN, i16::MAX].into_iter().flat_map(i16::to_le_bytes).collect::<Vec<_>>(),
            CODEC_ID_PCM_S16LE,
        ),
        (
            3,
            64,
            [-0.25_f64, 0.75].into_iter().flat_map(f64::to_le_bytes).collect(),
            CODEC_ID_PCM_F64LE,
        ),
    ] {
        let mut info = b"INFO".to_vec();
        // The first value is odd-sized, so the next subchunk verifies its padding.
        info.extend(chunk(b"INAM", b"ab\0"));
        assert_audio(
            WavReader::try_new(source(wave(format, bits, &audio, &info)), FormatOptions::default())
                .expect("ordinary reader opens control"),
            &audio,
            codec,
        );

        for length in [256 * 1024, 2 * 1024 * 1024] {
            let mut large_info = info.clone();
            let mut comment = vec![b'a'; length];
            comment.push(0);
            large_info.extend(chunk(b"ICMT", &comment));
            large_info.extend(chunk(b"IART", b"xy\0"));
            let bytes = Arc::new(AtomicUsize::new(0));
            let input = Counted {
                inner: Cursor::new(wave(format, bits, &audio, &large_info)),
                bytes: Arc::clone(&bytes),
            };
            let stream = MediaSourceStream::new(Box::new(input), Default::default());
            let reader = WavReader::try_new_without_info(stream, FormatOptions::default())
                .expect("metadata skipping opens the valid WAVE");
            assert_audio(reader, &audio, codec);
            assert!(bytes.load(Ordering::Relaxed) < 256 * 1024);
        }
    }
}

#[test]
fn skip_info_rejects_an_invalid_list_or_subchunk_extent() {
    let mut overrun = b"INFOINAM".to_vec();
    overrun.extend(32_u32.to_le_bytes());
    overrun.extend(b"x\0");
    let audio = [0_u8; 4];
    for info in [b"INF".as_slice(), &overrun] {
        let wav = wave(1, 16, &audio, info);
        assert!(matches!(
            WavReader::try_new(source(wav.clone()), FormatOptions::default()),
            Err(Error::DecodeError(_))
        ));
        assert!(matches!(
            WavReader::try_new_without_info(source(wav), FormatOptions::default()),
            Err(Error::DecodeError(_))
        ));
    }

    let mut info = b"INFO".to_vec();
    info.extend(chunk(b"INAM", b"ab\0"));
    let mut cut = wave(1, 16, &audio, &info);
    let info_pos = cut.windows(4).position(|bytes| bytes == b"INFO").expect("INFO list exists");
    cut.truncate(info_pos + info.len() - 2);
    assert!(matches!(
        WavReader::try_new_without_info(source(cut), FormatOptions::default()),
        Err(Error::DecodeError("wav: INFO list exceeds the source"))
    ));
}
