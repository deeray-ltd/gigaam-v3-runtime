// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! OpenAI-compatible batch response projection.
use crate::json::{JSON_CONTENT_TYPE, Json, obj};
use axum::{
    http::header,
    response::{IntoResponse, Response},
};
use gigaam_transcription::{MultiChannelBatchResult, TurnGap, words_to_text};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Format {
    Json,
    Text,
    Verbose,
}
pub(crate) struct ProjectionInput<'a> {
    pub result: &'a MultiChannelBatchResult,
    pub duration: f32,
    pub format: Format,
    pub word: bool,
    pub segment: bool,
    pub gap: TurnGap,
}
pub(crate) fn project(input: ProjectionInput<'_>) -> Result<Response, String> {
    let [channel] = input.result.channels() else {
        return Err("OpenAI output requires one selected channel".into());
    };
    let text = words_to_text(channel.words());
    if input.format == Format::Text {
        return Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response());
    }
    let mut fields = vec![("text", Json::str(text))];
    if input.format == Format::Verbose {
        fields.insert(0, ("task", Json::str("transcribe")));
        fields.insert(1, ("duration", Json::round(input.duration, 2)));
        if input.word {
            fields.push((
                "words",
                Json::Array(
                    channel
                        .words()
                        .iter()
                        .map(|word| {
                            obj(vec![
                                ("word", Json::str(word.text())),
                                ("start", Json::round(word.start(), 2)),
                                ("end", Json::round(word.end(), 2)),
                            ])
                        })
                        .collect(),
                ),
            ));
        }
        if input.segment {
            let _ = input.gap;
            fields.push((
                "segments",
                Json::Array(
                    input
                        .result
                        .segments()
                        .iter()
                        .enumerate()
                        .map(|(id, segment)| {
                            obj(vec![
                                ("id", Json::Usize(id)),
                                ("start", Json::round(segment.start(), 2)),
                                ("end", Json::round(segment.end(), 2)),
                                ("text", Json::str(segment.text())),
                            ])
                        })
                        .collect(),
                ),
            ));
        }
    }
    Ok((
        [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
        obj(fields).to_string(),
    )
        .into_response())
}
