// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

use gigaam_audio::{DecodedAudio, EncodedAudio, FileDecoder, FileDecoderProvider, FileDecoders};
use std::path::Path;
use std::sync::Arc;

/// Opens a seekable input using the file formats linked into this CLI build.
pub fn open(path: &Path) -> Result<FileDecoder, String> {
    let file = std::fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let hint = path
        .extension()
        .map(|extension| {
            extension
                .to_str()
                .ok_or_else(|| format!("{}: audio extension is not Unicode", path.display()))
        })
        .transpose()?;
    let providers: Vec<Arc<dyn FileDecoderProvider>> = vec![
        Arc::new(gigaam_format_wav::Wave),
        #[cfg(feature = "decoders")]
        Arc::new(gigaam_format_codecs::Containers),
    ];
    let result = FileDecoders::new(providers)?.open(Box::new(file), hint);
    #[cfg(not(feature = "decoders"))]
    return result.map_err(|error| {
        if error == "audio format is unavailable on this route" {
            unsupported_format(hint)
        } else {
            error
        }
    });
    #[cfg(feature = "decoders")]
    result
}

#[cfg(not(feature = "decoders"))]
fn unsupported_format(hint: Option<&str>) -> String {
    match hint {
        Some(extension) => format!("audio extension {extension:?} is unsupported because this build has no `decoders` feature"),
        None => "audio format has no extension hint and is unsupported because this build has no `decoders` feature".into(),
    }
}

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
        Err(unsupported_format(encoded.format_hint()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct TestInput(std::path::PathBuf);
    impl Drop for TestInput {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn input(bytes: &[u8], extension: &str) -> Result<TestInput, String> {
        use std::io::Write;
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "gigaam-cli-audio-{}-{}.{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            extension
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| e.to_string())?;
        let input = TestInput(path);
        file.write_all(bytes).map_err(|e| e.to_string())?;
        Ok(input)
    }
    #[test]
    fn seekable_open_preserves_strict_wave_refusal() -> Result<(), String> {
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&99_u32.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        let file = input(&wav, "mp3")?;
        let error = match open(&file.0) {
            Err(error) => error,
            Ok(_) => return Err("malformed WAV unexpectedly decoded".into()),
        };
        assert!(
            error.contains("RIFF size 99 does not match payload size"),
            "{error}"
        );
        Ok(())
    }
    #[cfg(not(feature = "decoders"))]
    #[test]
    fn seekable_open_preserves_disabled_codec_diagnostic() -> Result<(), String> {
        let file = input(&[1, 2], "mp3")?;
        let error = match open(&file.0) {
            Err(error) => error,
            Ok(_) => return Err("unsupported format unexpectedly decoded".into()),
        };
        assert_eq!(
            error,
            "audio extension \"mp3\" is unsupported because this build has no `decoders` feature"
        );
        Ok(())
    }
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
