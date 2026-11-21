// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

use gigaam_audio::{DecodedAudio, EncodedAudio};
use std::path::Path;

pub fn load(path: &Path) -> Result<DecodedAudio, String> {
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
    decode(EncodedAudio::new(bytes, hint)?)
}

fn decode(encoded: EncodedAudio) -> Result<DecodedAudio, String> {
    if encoded.bytes().len() >= 12
        && &encoded.bytes()[..4] == b"RIFF"
        && &encoded.bytes()[8..12] == b"WAVE"
    {
        return gigaam_format_wav::parse_wav(encoded.bytes());
    }
    #[cfg(feature = "decoders")]
    {
        gigaam_format_codecs::decode_bytes(encoded)
    }
    #[cfg(not(feature = "decoders"))]
    {
        Err(match encoded.format_hint() {
        Some(extension) => format!("audio extension {extension:?} is unsupported because this build has no `decoders` feature"),
        None => "audio format has no extension hint and is unsupported because this build has no `decoders` feature".into(),
    })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_riff_keeps_strict_wav_error() -> Result<(), String> {
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&99_u32.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        let error = decode(EncodedAudio::new(wav, Some("mp3".into()))?).unwrap_err();
        assert!(
            error.contains("RIFF size 99 does not match payload size"),
            "{error}"
        );
        Ok(())
    }
    #[cfg(not(feature = "decoders"))]
    #[test]
    fn unsupported_non_wave_preserves_the_feature_diagnostic() -> Result<(), String> {
        let error = decode(EncodedAudio::new(vec![1, 2], Some("mp3".into()))?).unwrap_err();
        assert_eq!(
            error,
            "audio extension \"mp3\" is unsupported because this build has no `decoders` feature"
        );
        Ok(())
    }
}
