// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! OpenAI-compatible batch response projection over retained transcription results.

use std::io::{Read, Seek, Write};

#[cfg(test)]
use axum::{
    http::header,
    response::{IntoResponse, Response},
};
use flate2::{Compression, write::ZlibEncoder};
use gigaam_transcription::{ScoredSegment, StoredWords, TranscriptWord, TurnGap};

#[cfg(test)]
use gigaam_transcription::{ChannelTranscript, words_to_text};

use gigaam_service_host::wire::json::{JSON_CONTENT_TYPE, Json, obj};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Format {
    Json,
    Text,
    Verbose,
}

impl Format {
    pub(crate) const fn content_type(self) -> &'static str {
        match self {
            Self::Text => "text/plain; charset=utf-8",
            Self::Json | Self::Verbose => JSON_CONTENT_TYPE,
        }
    }
}

pub(crate) struct StoredProjectionInput {
    pub duration: f32,
    pub format: Format,
    pub word: bool,
    pub segment: bool,
    pub gap: TurnGap,
}

/// Writes OpenAI's mono batch result, replaying channel zero for each requested field.
pub(crate) fn write_stored(
    words: &mut StoredWords<impl Read + Seek>,
    input: StoredProjectionInput,
    writer: &mut impl Write,
) -> Result<(), String> {
    let StoredProjectionInput {
        duration,
        format,
        word,
        segment,
        gap,
    } = input;
    validate_duration(duration)?;
    match format {
        Format::Json => write_bytes(writer, b"{\"text\":\"")?,
        Format::Verbose => {
            write_bytes(
                writer,
                b"{\"task\":\"transcribe\",\"language\":\"russian\",\"duration\":",
            )?;
            write_json(writer, Json::Num(f64::from(duration)))?;
            write_bytes(writer, b",\"text\":\"")?;
        }
        Format::Text => {}
    }

    let mut cursor = words.cursor(0)?;
    let mut separator = b"".as_slice();
    while let Some(item) = words.next(&mut cursor)? {
        write_bytes(writer, separator)?;
        if format == Format::Text {
            write_bytes(writer, item.text().as_bytes())?;
        } else {
            // Reuse the wire serializer's escaping for each word, without its quote delimiters.
            let quoted = Json::str(item.text()).to_string();
            write_bytes(writer, &quoted.as_bytes()[1..quoted.len() - 1])?;
        }
        separator = b" ";
    }
    if format == Format::Text {
        return Ok(());
    }
    write_bytes(writer, b"\"")?;

    if format == Format::Verbose && word {
        write_bytes(writer, b",\"words\":[")?;
        let mut cursor = words.cursor(0)?;
        let mut separator = b"".as_slice();
        while let Some(item) = words.next(&mut cursor)? {
            write_bytes(writer, separator)?;
            write_json(writer, word_json(&item)?)?;
            separator = b",";
        }
        write_bytes(writer, b"]")?;
    }
    if format == Format::Verbose && segment {
        write_bytes(writer, b",\"segments\":[")?;
        let mut id = 0_usize;
        words.replay_scored_segments(0, gap, |item| {
            if id != 0 {
                write_bytes(writer, b",")?;
            }
            write_json(writer, segment_json(id, &item)?)?;
            id = id.checked_add(1).ok_or("OpenAI segment id overflow")?;
            Ok(())
        })?;
        write_bytes(writer, b"]")?;
    }
    write_bytes(writer, b"}")
}

fn write_bytes(writer: &mut impl Write, bytes: &[u8]) -> Result<(), String> {
    writer
        .write_all(bytes)
        .map_err(|error| format!("OpenAI response I/O: {error}"))
}

fn write_json(writer: &mut impl Write, json: Json) -> Result<(), String> {
    write_bytes(writer, json.to_string().as_bytes())
}

#[cfg(test)]
pub(crate) fn project_channel(
    channel: &ChannelTranscript,
    duration: f32,
    format: Format,
    word: bool,
    segment: bool,
    gap: TurnGap,
) -> Result<Response, String> {
    validate_duration(duration)?;
    let text = words_to_text(channel.words());
    if format == Format::Text {
        return Ok(([(header::CONTENT_TYPE, format.content_type())], text).into_response());
    }
    let mut fields = vec![("text", Json::str(text))];
    if format == Format::Verbose {
        fields.insert(0, ("task", Json::str("transcribe")));
        fields.insert(1, ("language", Json::str("russian")));
        fields.insert(2, ("duration", Json::Num(f64::from(duration))));
        if word {
            let words = channel
                .words()
                .iter()
                .map(word_json)
                .collect::<Result<_, _>>()?;
            fields.push(("words", Json::Array(words)));
        }
        if segment {
            let segments = channel
                .scored_segments(gap)?
                .iter()
                .enumerate()
                .map(|(id, item)| segment_json(id, item))
                .collect::<Result<_, _>>()?;
            fields.push(("segments", Json::Array(segments)));
        }
    }
    Ok((
        [(header::CONTENT_TYPE, format.content_type())],
        obj(fields).to_string(),
    )
        .into_response())
}

fn validate_duration(duration: f32) -> Result<(), String> {
    if !duration.is_finite() || duration < 0.0 {
        return Err("invalid audio duration".into());
    }
    Ok(())
}

fn word_json(item: &TranscriptWord) -> Result<Json, String> {
    if !item.start().is_finite() || !item.end().is_finite() {
        return Err("invalid word timing".into());
    }
    Ok(obj(vec![
        ("word", Json::str(item.text())),
        ("start", Json::Num(f64::from(item.start()))),
        ("end", Json::Num(f64::from(item.end()))),
    ]))
}

fn segment_json(id: usize, item: &ScoredSegment) -> Result<Json, String> {
    let average = item.average_log_probability();
    let no_speech = item.no_speech_probability();
    if !item.start().is_finite()
        || !item.end().is_finite()
        || !average.is_finite()
        || !no_speech.is_finite()
        || !(0.0..=1.0).contains(&no_speech)
    {
        return Err("invalid scored segment metadata".into());
    }
    let tokens = item
        .tokens()
        .iter()
        .map(|token| Json::Usize(token.id()))
        .collect();
    Ok(obj(vec![
        ("id", Json::Usize(id)),
        ("seek", Json::Usize(item.seek())),
        ("start", Json::Num(f64::from(item.start()))),
        ("end", Json::Num(f64::from(item.end()))),
        ("text", Json::str(item.text())),
        ("tokens", Json::Array(tokens)),
        ("temperature", Json::Num(0.0)),
        ("avg_logprob", Json::Num(average)),
        (
            "compression_ratio",
            Json::Num(compression_ratio(item.text())?),
        ),
        ("no_speech_prob", Json::Num(no_speech)),
    ]))
}

fn compressed(text: &str) -> Result<Vec<u8>, String> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
    encoder
        .write_all(text.as_bytes())
        .map_err(|error| format!("compression failed: {error}"))?;
    encoder
        .finish()
        .map_err(|error| format!("compression failed: {error}"))
}

fn compression_ratio(text: &str) -> Result<f64, String> {
    if text.is_empty() {
        return Ok(0.0);
    }
    let bytes = compressed(text)?;
    let input = u32::try_from(text.len()).map_err(|_| "text length overflows".to_owned())?;
    let output =
        u32::try_from(bytes.len()).map_err(|_| "compressed length overflows".to_owned())?;
    if output == 0 {
        return Err("empty compressed output".into());
    }
    Ok(f64::from(input) / f64::from(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    #[tokio::test]
    async fn stored_projection_matches_aggregate_wire_bytes() -> Result<(), String> {
        use gigaam_transcription::WordRecords;
        use std::io::Cursor;
        let gap = TurnGap::new(1.0)?;
        let transcript = vec![
            TranscriptWord::new("Привет".into(), 0.125, 0.25)?,
            TranscriptWord::new("quote:\"\\\n\t\u{001f}".into(), 0.5, 0.75)?,
            TranscriptWord::new("мир".into(), 1.0, 1.25)?,
        ];
        for items in [vec![], transcript] {
            let channel = ChannelTranscript::new(0, items)?;
            let mut records = WordRecords::new(Cursor::new(Vec::new()), 1)?;
            for item in channel.words() {
                records.push(0, item.clone())?;
            }
            let mut stored = records.finish()?;
            for format in [Format::Json, Format::Text, Format::Verbose] {
                for (word, segment) in [(false, false), (true, false), (false, true), (true, true)]
                {
                    let input = StoredProjectionInput {
                        duration: 1.375,
                        format,
                        word,
                        segment,
                        gap,
                    };
                    let mut actual = Vec::new();
                    let projected = write_stored(&mut stored, input, &mut actual);
                    match project_channel(&channel, 1.375, format, word, segment, gap) {
                        Ok(response) => {
                            projected?;
                            assert_eq!(
                                response
                                    .headers()
                                    .get(header::CONTENT_TYPE)
                                    .ok_or("response has a content type")?,
                                format.content_type()
                            );
                            let expected = axum::body::to_bytes(response.into_body(), usize::MAX)
                                .await
                                .map_err(|error| error.to_string())?;
                            assert_eq!(actual, expected);
                        }
                        Err(error) => {
                            assert_eq!(projected, Err(error));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn requested_verbose_shape_and_missing_scores() -> Result<(), String> {
        use gigaam_transcription::TranscriptWord;
        let channel =
            ChannelTranscript::new(0, vec![TranscriptWord::new("Привет".into(), 0.25, 0.75)?])?;
        let gap = TurnGap::new(1.0)?;
        let response = project_channel(&channel, 1.0, Format::Verbose, true, false, gap)?;
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .map_err(|e| e.to_string())?;
        assert_eq!(
            std::str::from_utf8(&body).map_err(|e| e.to_string())?,
            "{\"task\":\"transcribe\",\"language\":\"russian\",\"duration\":1,\"text\":\"Привет\",\"words\":[{\"word\":\"Привет\",\"start\":0.25,\"end\":0.75}]}"
        );
        assert!(project_channel(&channel, 1.0, Format::Verbose, true, true, gap).is_err());
        let empty = ChannelTranscript::new(0, vec![])?;
        let response = project_channel(&empty, 0.0, Format::Verbose, true, true, gap)?;
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .map_err(|e| e.to_string())?;
        assert_eq!(
            std::str::from_utf8(&body).map_err(|e| e.to_string())?,
            "{\"task\":\"transcribe\",\"language\":\"russian\",\"duration\":0,\"text\":\"\",\"words\":[],\"segments\":[]}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn ordinary_formats_have_only_their_documented_body() -> Result<(), String> {
        use gigaam_transcription::TranscriptWord;
        let channel =
            ChannelTranscript::new(0, vec![TranscriptWord::new("hello".into(), 0.0, 0.5)?])?;
        let gap = TurnGap::new(1.0)?;
        for (format, expected) in [
            (Format::Json, "{\"text\":\"hello\"}"),
            (Format::Text, "hello"),
        ] {
            let response = project_channel(&channel, 0.5, format, false, false, gap)?;
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .map_err(|e| e.to_string())?;
            assert_eq!(
                std::str::from_utf8(&body).map_err(|e| e.to_string())?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn compression_roundtrips_and_uses_utf8_byte_lengths() -> Result<(), String> {
        for (text, expected) in [
            ("", 0.0),
            ("hello", 5.0 / 13.0),
            ("Привет, мир!", 21.0 / 31.0),
            ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 40.0 / 12.0),
        ] {
            let encoded = compressed(text)?;
            let mut decoded = String::new();
            ZlibDecoder::new(encoded.as_slice())
                .read_to_string(&mut decoded)
                .map_err(|error| error.to_string())?;
            assert_eq!(decoded, text);
            assert_eq!(compression_ratio(text)?, expected);
        }
        assert_eq!(compressed("")?, vec![120, 156, 3, 0, 0, 0, 0, 1]);
        assert_eq!(
            compressed("Привет, мир!")?,
            vec![
                120, 156, 187, 48, 255, 98, 195, 133, 29, 23, 54, 93, 216, 122, 177, 73, 71, 225,
                194, 158, 11, 59, 46, 54, 40, 2, 0, 157, 147, 13, 117
            ]
        );
        assert_eq!(
            compressed("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")?,
            vec![120, 156, 75, 76, 36, 14, 0, 0, 54, 235, 15, 41]
        );
        Ok(())
    }
}
