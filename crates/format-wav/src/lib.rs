// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

mod wav;
use gigaam_audio::{
    AudioFile, DecodedAudio, EncodedAudio, FileDecoder, FileDecoderProvider, FileFormatMatch,
};
use std::path::Path;
pub use wav::parse_wav;
pub fn read_wav(path: &Path) -> Result<DecodedAudio, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    parse_wav(&bytes)
}
pub fn open_encoded(encoded: EncodedAudio) -> Result<FileDecoder, String> {
    let (bytes, _) = encoded.into_parts();
    Ok(FileDecoder::from_reader(Box::new(wav::WavReader::open(
        std::io::Cursor::new(bytes),
    )?)))
}

/// Format provider supplied by the build composition.
pub struct Wave;

impl FileDecoderProvider for Wave {
    fn name(&self) -> &'static str {
        "wav"
    }

    fn identify(&self, prefix: &[u8], _hint: Option<&str>) -> FileFormatMatch {
        if wav::is_riff_wave(prefix) {
            FileFormatMatch::Exact
        } else {
            FileFormatMatch::None
        }
    }

    fn open(&self, source: Box<dyn AudioFile>, _hint: Option<&str>) -> Result<FileDecoder, String> {
        Ok(FileDecoder::from_reader(Box::new(wav::WavReader::open(
            source,
        )?)))
    }
}
