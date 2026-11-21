// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

mod batch_probe;
mod codecs;
mod id3v2;
pub use codecs::decode_bytes;
use gigaam_audio::{AudioFile, FileDecoder, FileDecoderProvider, FileFormatMatch};

/// Format provider supplied by the build composition.
pub struct Containers;

impl FileDecoderProvider for Containers {
    fn name(&self) -> &'static str {
        "container-codecs"
    }

    fn identify(&self, prefix: &[u8], _hint: Option<&str>) -> FileFormatMatch {
        if prefix.len() >= 12 && &prefix[..4] == b"RIFF" && &prefix[8..12] == b"WAVE" {
            FileFormatMatch::None
        } else {
            FileFormatMatch::Fallback
        }
    }

    fn open(&self, source: Box<dyn AudioFile>, hint: Option<&str>) -> Result<FileDecoder, String> {
        Ok(FileDecoder::from_reader(Box::new(
            codecs::CodecFileReader::open(source, hint)?,
        )))
    }
}

/// Loads a codec-supported file and preserves the legacy Opus delivery rate.
pub fn load(path: &std::path::Path) -> Result<gigaam_audio::DecodedAudio, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let hint = path
        .extension()
        .map(|extension| {
            extension
                .to_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{}: audio extension is not Unicode", path.display()))
        })
        .transpose()?;
    decode_bytes(gigaam_audio::EncodedAudio::new(bytes, hint)?)
}
