// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

use std::io::{Read, Seek, SeekFrom};

const HEADER_LEN: u64 = 10;

/// Locates the end of contiguous leading ID3v2 tags without loading their metadata.
pub(crate) fn audio_offset(source: &mut (impl Read + Seek), length: u64) -> Result<u64, String> {
    let mut position = 0;
    while length.saturating_sub(position) >= 3 {
        source.seek(SeekFrom::Start(position)).map_err(io_error)?;
        let mut marker = [0; 3];
        source.read_exact(&mut marker).map_err(io_error)?;
        if marker != *b"ID3" {
            break;
        }

        let mut fields = [0; 7];
        source.read_exact(&mut fields).map_err(io_error)?;
        let [major, revision, flags, a, b, c, d] = fields;
        if !(2..=4).contains(&major) || revision == u8::MAX {
            return Err("unsupported ID3v2 version".into());
        }
        if major == 2 && flags & 0x40 != 0 {
            return Err("ID3v2.2 compression is unsupported".into());
        }
        let mut body_len = 0_u64;
        for byte in [a, b, c, d] {
            if byte & 0x80 != 0 {
                return Err("ID3v2 tag size is not synchsafe".into());
            }
            body_len = (body_len << 7) | u64::from(byte);
        }

        let has_footer = major == 4 && flags & 0x10 != 0;
        let footer_len = if has_footer { HEADER_LEN } else { 0 };
        let body_end = position
            .checked_add(HEADER_LEN + body_len)
            .ok_or_else(|| "ID3v2 tag extent overflows".to_owned())?;
        let end = body_end
            .checked_add(footer_len)
            .filter(|&end| end <= length)
            .ok_or_else(|| "ID3v2 tag ends beyond its audio source".to_owned())?;
        if has_footer {
            source.seek(SeekFrom::Start(body_end)).map_err(io_error)?;
            let mut footer = [0; 10];
            source.read_exact(&mut footer).map_err(io_error)?;
            if footer[..3] != *b"3DI" || footer[3..] != fields {
                return Err("ID3v2 footer does not match its header".into());
            }
        }
        position = end;
    }
    Ok(position)
}

fn io_error(error: std::io::Error) -> String {
    format!("reading ID3v2 tag: {error}")
}

#[cfg(test)]
mod tests {
    use super::audio_offset;
    use std::io::Cursor;

    #[test]
    fn skips_only_complete_leading_tags_with_a_valid_optional_footer() -> Result<(), String> {
        let tag3 = b"ID3\x03\x00\x00\x00\x00\x00\x04fLaC";
        let tag4 = b"ID3\x04\x00\x10\x00\x00\x00\x02ab3DI\x04\x00\x10\x00\x00\x00\x02";
        let mut data = [tag3.as_slice(), tag4.as_slice(), b"audio packets"].concat();
        let length = u64::try_from(data.len()).map_err(|error| error.to_string())?;
        assert_eq!(audio_offset(&mut Cursor::new(&data), length)?, 36);
        assert_eq!(audio_offset(&mut Cursor::new(b"no tag"), 6)?, 0);

        data[20] = 0x80;
        assert!(audio_offset(&mut Cursor::new(&data), length).is_err());
        assert!(audio_offset(&mut Cursor::new(tag3), 13).is_err());
        Ok(())
    }
}
