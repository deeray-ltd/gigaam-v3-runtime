// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

use std::io::{self, Read, Seek, SeekFrom};
use symphonia::core::io::MediaSource;

const STREAMINFO_BYTES: usize = 34;
const PREFIX_BYTES: usize = 8 + STREAMINFO_BYTES;

pub(crate) struct FlacAudioLayout {
    prefix: [u8; PREFIX_BYTES],
    audio_start: u64,
    length: u64,
}

/// Recognizes the native FLAC stream header and locates audio by seeking past
/// the declared metadata extents. The decoder receives the original STREAMINFO
/// and frame bytes and sees one final metadata block.
pub(crate) fn audio_layout(
    source: &mut (impl Read + Seek),
    start: u64,
    length: u64,
) -> io::Result<Option<FlacAudioLayout>> {
    if length.saturating_sub(start) < 4 {
        return Ok(None);
    }
    source.seek(SeekFrom::Start(start))?;
    let mut prefix = [0; PREFIX_BYTES];
    source.read_exact(&mut prefix[..4])?;
    if prefix[..4] != *b"fLaC" {
        return Ok(None);
    }

    source.read_exact(&mut prefix[4..8])?;
    if prefix[4] & 0x7f != 0 || prefix[5..8] != [0, 0, 34] {
        return Err(invalid_data("FLAC first metadata block must be STREAMINFO"));
    }
    if prefix[4] & 0x80 != 0 {
        return Ok(None);
    }
    source.read_exact(&mut prefix[8..])?;
    prefix[4] |= 0x80;

    let mut position = source.stream_position()?;
    loop {
        let mut header = [0; 4];
        source.seek(SeekFrom::Start(position))?;
        source.read_exact(&mut header)?;
        let [flags, a, b, c] = header;
        if matches!(flags & 0x7f, 0 | 127) {
            return Err(invalid_data("invalid optional FLAC metadata block type"));
        }
        let bytes = u64::from(u32::from_be_bytes([0, a, b, c]));
        position = position
            .checked_add(4 + bytes)
            .filter(|&end| end <= length)
            .ok_or_else(|| invalid_data("FLAC metadata block ends beyond its audio source"))?;
        if flags & 0x80 != 0 {
            break;
        }
    }

    let prefix_bytes = u64::try_from(PREFIX_BYTES).expect("the FLAC prefix fits u64");
    Ok(Some(FlacAudioLayout {
        prefix,
        audio_start: position,
        length: prefix_bytes + (length - position),
    }))
}

pub(crate) struct FlacAudioSource<R> {
    input: R,
    layout: FlacAudioLayout,
    position: u64,
}

impl<R> FlacAudioSource<R> {
    pub(crate) const fn new(input: R, layout: FlacAudioLayout) -> Self {
        Self {
            input,
            layout,
            position: 0,
        }
    }
}

impl<R: Read + Seek> Read for FlacAudioSource<R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() || self.position >= self.layout.length {
            return Ok(0);
        }

        let prefix_bytes = u64::try_from(PREFIX_BYTES).expect("the FLAC prefix fits u64");
        let count = if self.position < prefix_bytes {
            let start = usize::try_from(self.position).expect("position is inside the FLAC prefix");
            let count = (PREFIX_BYTES - start).min(bytes.len());
            bytes[..count].copy_from_slice(&self.layout.prefix[start..start + count]);
            count
        } else {
            let start = self.layout.audio_start + (self.position - prefix_bytes);
            let available =
                usize::try_from(self.layout.length - self.position).unwrap_or(usize::MAX);
            let count = bytes.len().min(available);
            self.input.seek(SeekFrom::Start(start))?;
            self.input.read(&mut bytes[..count])?
        };
        self.position += u64::try_from(count).map_err(|_| io::ErrorKind::InvalidData)?;
        Ok(count)
    }
}

impl<R: Read + Seek> Seek for FlacAudioSource<R> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.position = match position {
            SeekFrom::Start(offset) => Some(offset),
            SeekFrom::Current(offset) => self.position.checked_add_signed(offset),
            SeekFrom::End(offset) => self.layout.length.checked_add_signed(offset),
        }
        .ok_or(io::ErrorKind::InvalidInput)?;
        Ok(self.position)
    }
}

impl<R: Read + Seek + Send + Sync> MediaSource for FlacAudioSource<R> {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.layout.length)
    }
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::{FlacAudioSource, PREFIX_BYTES, audio_layout};
    use std::io::{Cursor, Read, Seek, SeekFrom};

    #[test]
    fn preserves_streaminfo_and_frame_bytes_across_seek_and_refuses_truncation()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut data = b"fLaC\x00\x00\x00\x22".to_vec();
        data.extend_from_slice(&[0x69; 34]);
        data.extend_from_slice(b"\x06\x00\x00\x04junk\x84\x00\x00\x03tag");
        data.extend_from_slice(b"unchanged encoded frames");
        let length = u64::try_from(data.len()).map_err(|error| error.to_string())?;
        let layout = audio_layout(&mut Cursor::new(&data), 0, length)?.ok_or("FLAC layout")?;
        let mut source = FlacAudioSource::new(Cursor::new(&data), layout);

        let mut expected = data[..PREFIX_BYTES].to_vec();
        expected[4] |= 0x80;
        expected.extend_from_slice(b"unchanged encoded frames");
        let mut actual = Vec::new();
        source
            .read_to_end(&mut actual)
            .map_err(|error| error.to_string())?;
        assert_eq!(actual, expected);
        source
            .seek(SeekFrom::End(-6))
            .map_err(|error| error.to_string())?;
        let mut end = [0; 6];
        source
            .read_exact(&mut end)
            .map_err(|error| error.to_string())?;
        assert_eq!(&end, b"frames");

        let short_body = u64::try_from(PREFIX_BYTES + 8 + 4 + 2).expect("fixture length fits u64");
        assert!(audio_layout(&mut Cursor::new(&data), 0, short_body).is_err());
        assert!(audio_layout(&mut Cursor::new(b"RIFF"), 0, 4)?.is_none());
        Ok(())
    }
}
