// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! OpenAI-compatible batch request validation and response projection.

use crate::json::{JSON_CONTENT_TYPE, Json, obj};
use axum::{
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use std::path::Path;

use crate::openai_response::Format;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Engine {
    Ctc,
    Rnnt,
}
#[derive(Clone, Debug)]
pub(crate) struct Input {
    pub(crate) audio: Vec<u8>,
    pub(crate) extension: Option<String>,
    pub(crate) engine: Engine,
    pub(crate) format: Format,
    pub(crate) word: bool,
    pub(crate) segment: bool,
    pub(crate) vad: Option<gigaam_transcription::VadPolicyConfig>,
}

#[derive(Debug)]
pub(crate) struct ApiError {
    status: StatusCode,
    message: String,
    param: Option<&'static str>,
}
impl ApiError {
    pub(crate) fn new(
        status: StatusCode,
        message: impl Into<String>,
        param: Option<&'static str>,
    ) -> Self {
        Self {
            status,
            message: message.into(),
            param,
        }
    }
    pub(crate) fn bad(message: impl Into<String>, param: Option<&'static str>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            param,
        }
    }
    pub(crate) fn response(self) -> Response {
        (
            self.status,
            [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
            obj(vec![(
                "error",
                obj(vec![
                    ("message", Json::str(self.message)),
                    (
                        "type",
                        Json::str(if self.status.is_client_error() {
                            "invalid_request_error"
                        } else {
                            "server_error"
                        }),
                    ),
                    ("param", self.param.map(Json::str).unwrap_or(Json::Null)),
                    ("code", Json::Null),
                ]),
            )])
            .to_string(),
        )
            .into_response()
    }
}

fn parse_model(s: &str, rnnt: bool) -> Result<Engine, ApiError> {
    match s {
        "gigaam-v3" | "gigaam-v3-ctc" => Ok(Engine::Ctc),
        "gigaam-v3-rnnt" if rnnt => Ok(Engine::Rnnt),
        "gigaam-v3-rnnt" => Err(ApiError::bad("RNN-T model is not loaded", Some("model"))),
        _ => Err(ApiError::bad(
            format!("Unsupported model: {s}"),
            Some("model"),
        )),
    }
}
fn extension(filename: Option<&str>) -> Option<String> {
    let base = filename
        .and_then(|s| Path::new(s).file_name())
        .and_then(|s| s.to_str())?;
    let ext = Path::new(base).extension()?.to_str()?.to_ascii_lowercase();
    [
        "wav", "flac", "ogg", "opus", "mp3", "aac", "m4a", "mp4", "mkv", "vorbis", "pcm",
    ]
    .contains(&ext.as_str())
    .then_some(ext)
}
pub(crate) fn parse_fields(
    mut fields: Vec<(String, Option<String>, Vec<u8>)>,
    rnnt: bool,
) -> Result<Input, ApiError> {
    let mut file = None;
    let mut model = None;
    let mut format = Format::Json;
    let mut fmt_seen = false;
    let mut word = false;
    let mut segment = false;
    let mut granularities = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut vad_fields = Vec::new();
    for (name, filename, bytes) in fields.drain(..) {
        if name != "timestamp_granularities[]" && !seen.insert(name.clone()) {
            return Err(ApiError::bad(
                format!("{name} must occur once"),
                Some(match name.as_str() {
                    "file" => "file",
                    "model" => "model",
                    "response_format" => "response_format",
                    "language" => "language",
                    "temperature" => "temperature",
                    "stream" => "stream",
                    "prompt" => "prompt",
                    _ => "field",
                }),
            ));
        }
        match name.as_str() {
            "file" => {
                if file.is_some() {
                    return Err(ApiError::bad("file must occur once", Some("file")));
                }
                if bytes.is_empty() {
                    return Err(ApiError::bad("file must not be empty", Some("file")));
                }
                file = Some((bytes, extension(filename.as_deref())));
            }
            "model" => {
                if model.is_some() {
                    return Err(ApiError::bad("model must occur once", Some("model")));
                }
                let s = String::from_utf8(bytes)
                    .map_err(|_| ApiError::bad("model must be UTF-8", Some("model")))?;
                if s.is_empty() {
                    return Err(ApiError::bad("model must not be empty", Some("model")));
                }
                model = Some(parse_model(&s, rnnt)?);
            }
            "response_format" => {
                if fmt_seen {
                    return Err(ApiError::bad(
                        "response_format must occur once",
                        Some("response_format"),
                    ));
                }
                fmt_seen = true;
                let s = String::from_utf8(bytes).map_err(|_| {
                    ApiError::bad("response_format must be UTF-8", Some("response_format"))
                })?;
                format = match s.as_str() {
                    "json" => Format::Json,
                    "text" => Format::Text,
                    "verbose_json" => Format::Verbose,
                    _ => {
                        return Err(ApiError::bad(
                            format!("Unsupported response_format: {s}"),
                            Some("response_format"),
                        ));
                    }
                }
            }
            "timestamp_granularities[]" => {
                let s = String::from_utf8(bytes).map_err(|_| {
                    ApiError::bad(
                        "timestamp_granularities[] must be UTF-8",
                        Some("timestamp_granularities[]"),
                    )
                })?;
                if granularities.contains(&s) {
                    return Err(ApiError::bad(
                        "duplicate timestamp granularity",
                        Some("timestamp_granularities[]"),
                    ));
                }
                match s.as_str() {
                    "word" => word = true,
                    "segment" => segment = true,
                    _ => {
                        return Err(ApiError::bad(
                            "timestamp granularity must be word or segment",
                            Some("timestamp_granularities[]"),
                        ));
                    }
                }
                granularities.push(s);
            }
            "language" => {
                let s = String::from_utf8(bytes)
                    .map_err(|_| ApiError::bad("language must be UTF-8", Some("language")))?;
                if s != "ru" {
                    return Err(ApiError::bad(
                        "Only language=ru is supported",
                        Some("language"),
                    ));
                }
            }
            "temperature" => {
                let s = String::from_utf8(bytes)
                    .map_err(|_| ApiError::bad("temperature must be UTF-8", Some("temperature")))?;
                if s.parse::<f64>()
                    .map_or(true, |value| !value.is_finite() || value != 0.0)
                {
                    return Err(ApiError::bad(
                        "Only temperature=0 is supported",
                        Some("temperature"),
                    ));
                }
            }
            "stream" => {
                let s = String::from_utf8(bytes)
                    .map_err(|_| ApiError::bad("stream must be UTF-8", Some("stream")))?;
                if s != "false" {
                    return Err(ApiError::bad("streaming is not supported", Some("stream")));
                }
            }
            "vad" | "vad_threshold" | "vad_neg_threshold" | "vad_min_speech_ms"
            | "vad_min_silence_ms" | "vad_speech_pad_ms" => {
                let value = String::from_utf8(bytes)
                    .map_err(|_| ApiError::bad("VAD field must be UTF-8", None))?;
                vad_fields.push((name, value));
            }
            "prompt" if bytes.is_empty() => {}
            "prompt" => return Err(ApiError::bad("prompt is not supported", Some("prompt"))),
            _ => return Err(ApiError::bad(format!("Unsupported field: {name}"), None)),
        }
    }
    if format != Format::Verbose && !granularities.is_empty() {
        return Err(ApiError::bad(
            "timestamp granularities require verbose_json",
            Some("timestamp_granularities[]"),
        ));
    }
    let (audio, extension) = file.ok_or_else(|| ApiError::bad("file is required", Some("file")))?;
    let engine = model.ok_or_else(|| ApiError::bad("model is required", Some("model")))?;
    if format == Format::Verbose && granularities.is_empty() {
        segment = true;
    }
    let vad_parameters = crate::protocol::QueryParameters::from_fields(vad_fields)
        .map_err(|error| ApiError::bad(error, None))?;
    let vad = crate::http::bind_vad_policy(&vad_parameters, crate::response::ResponseVersion::V2)
        .map_err(|error| ApiError::bad(error, None))?;
    Ok(Input {
        vad,
        audio,
        extension,
        engine,
        format,
        word,
        segment,
    })
}

#[cfg(test)]
mod tests {
    use super::{Format, parse_fields};

    fn fields(extra: &[(&str, &str)]) -> Vec<(String, Option<String>, Vec<u8>)> {
        let mut values = vec![
            ("model".into(), None, b"gigaam-v3".to_vec()),
            ("file".into(), Some("speech.wav".into()), b"wave".to_vec()),
        ];
        values.extend(
            extra
                .iter()
                .map(|(key, value)| (String::from(*key), None, value.as_bytes().to_vec())),
        );
        values
    }

    #[test]
    fn validates_sdk_fields_and_shared_vad_controls() {
        let input = parse_fields(
            fields(&[
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("vad", "true"),
                ("vad_threshold", "0.7"),
            ]),
            false,
        )
        .expect("valid multipart fields");
        assert_eq!(input.format, Format::Verbose);
        assert!(input.word);
        assert!(input.vad.is_some());
        assert!(parse_fields(fields(&[("prompt", "")]), false).is_ok());
        for extra in [
            vec![("vad", "false"), ("vad_threshold", "0.7")],
            vec![("vad", "true"), ("vad_threshold", "NaN")],
            vec![("vad", "true"), ("vad", "false")],
            vec![("language", "en")],
            vec![("temperature", "0.5")],
            vec![("stream", "true")],
            vec![("prompt", " ")],
            vec![("prompt", "speech")],
            vec![("unknown", "x")],
        ] {
            assert!(parse_fields(fields(&extra), false).is_err());
        }
    }
}
