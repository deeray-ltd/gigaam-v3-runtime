// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

use gigaam_audio::{FileDecoderProvider, FileDecoders};
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct CountedSource {
    cursor: Cursor<Vec<u8>>,
    read: Arc<AtomicUsize>,
}

impl Read for CountedSource {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let count = self.cursor.read(output)?;
        self.read.fetch_add(count, Ordering::Relaxed);
        Ok(count)
    }
}

impl Seek for CountedSource {
    fn seek(&mut self, offset: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(offset)
    }
}

fn chunk(
    output: &mut Vec<u8>,
    kind: &[u8; 4],
    body: &[u8],
) -> Result<(), std::num::TryFromIntError> {
    output.extend(kind);
    output.extend(u32::try_from(body.len())?.to_le_bytes());
    output.extend(body);
    if !body.len().is_multiple_of(2) {
        output.push(0);
    }
    Ok(())
}

#[test]
fn batch_wave_after_id3_keeps_float64_samples_without_reading_info()
-> Result<(), Box<dyn std::error::Error>> {
    let mut wave = b"WAVE".to_vec();
    let format = [
        3_u16.to_le_bytes().as_slice(),
        &1_u16.to_le_bytes(),
        &16_000_u32.to_le_bytes(),
        &(16_000_u32 * 8).to_le_bytes(),
        &8_u16.to_le_bytes(),
        &64_u16.to_le_bytes(),
    ]
    .concat();
    chunk(&mut wave, b"fmt ", &format)?;
    let mut list = b"INFO".to_vec();
    chunk(&mut list, b"ICMT", &vec![b'x'; 2 * 1024 * 1024 + 1])?;
    chunk(&mut wave, b"LIST", &list)?;
    let samples = [0.0_f64, 0.25, -0.5, 0.75];
    let data = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    chunk(&mut wave, b"data", &data)?;
    let mut bytes = b"ID3\x04\x00\x00\x00\x00\x00\x00".to_vec();
    chunk(&mut bytes, b"RIFF", &wave)?;
    let read = Arc::new(AtomicUsize::new(0));
    let source = CountedSource {
        cursor: Cursor::new(bytes),
        read: Arc::clone(&read),
    };
    let providers: Vec<Arc<dyn FileDecoderProvider>> =
        vec![Arc::new(gigaam_format_codecs::Containers)];
    let mut decoder = FileDecoders::new(providers)?.open(Box::new(source), Some("wav"))?;
    assert_eq!(decoder.metadata().sample_rate().hertz(), 16_000);
    assert_eq!(decoder.metadata().channel_count().get(), 1);
    let limit = NonZeroUsize::new(8).ok_or("positive chunk limit")?;
    let audio = decoder.next_chunk(limit)?.ok_or("missing audio")?;
    assert_eq!((audio.first_frame(), audio.end_frame()), (0, 4));
    assert_eq!(
        audio.audio().channels()[0].samples(),
        &[0.0, 0.25, -0.5, 0.75]
    );
    assert!(decoder.next_chunk(limit)?.is_none());
    assert!(read.load(Ordering::Relaxed) < 256 * 1024);
    Ok(())
}
