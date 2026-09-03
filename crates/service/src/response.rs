// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Build `POST /v1/transcribe` JSON responses to the contract schema with pure functions over
//! Transcription application results and HTTP wire envelopes using the service JSON writer.
use crate::json::{JSON_CONTENT_TYPE, Json, error_object, obj};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use gigaam_recognition::Device;
use gigaam_transcription::{
    ChannelTranscript, DuplicateClassification, SourceChannelCount, TranscriptWord, Turn,
    words_to_text,
};

/// Query request parameters that affect response shape.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Options {
    pub words: bool,
    pub segments: bool,
    pub turns: bool,
    pub turn_gap: f32,
    pub split: bool,
}

/// Fully named input to response shaping. It prevents positional ambiguity at the
/// public HTTP boundary and keeps the response constructor below the argument-count limit.
pub(crate) struct ResponseInput<'a> {
    pub channels: &'a [ChannelTranscript],
    pub segments: &'a [Turn],
    pub turns: &'a [Turn],
    pub options: Options,
    pub duration_sec: f32,
    pub sample_rate_in: u32,
    pub source_channels: SourceChannelCount,
    pub duplicate: DuplicateClassification,
    pub model: &'a str,
    pub rtf: f32,
}

/// Validated immutable capability text retained by the health endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HealthProjection {
    provider: Device,
    has_rnnt: bool,
}

impl HealthProjection {
    pub(crate) const fn new(provider: Device, has_rnnt: bool) -> Self {
        Self { provider, has_rnnt }
    }
}

impl Default for Options {
    fn default() -> Self {
        Options {
            words: false,
            segments: false,
            turns: false,
            turn_gap: 1.0,
            split: false,
        }
    }
}

fn word_json(w: &TranscriptWord, channel: Option<usize>) -> Json {
    let mut f = vec![
        ("start", Json::round(w.start(), 2)),
        ("end", Json::round(w.end(), 2)),
        ("text", Json::str(w.text().to_owned())),
    ];
    if let Some(c) = channel {
        f.push(("channel", Json::Int(channel_index(c))));
    }
    obj(f)
}

fn turn_json(t: &Turn) -> Json {
    obj(vec![
        ("channel", Json::Int(channel_index(t.channel()))),
        ("start", Json::round(t.start(), 2)),
        ("end", Json::round(t.end(), 2)),
        ("text", Json::str(t.text())),
    ])
}

fn segment_json(segment: &Turn) -> Json {
    obj(vec![
        ("start", Json::round(segment.start(), 2)),
        ("end", Json::round(segment.end(), 2)),
        ("channel", Json::Int(channel_index(segment.channel()))),
        ("text", Json::str(segment.text())),
    ])
}

fn channel_index(value: usize) -> i64 {
    match i64::try_from(value) {
        Ok(value) => value,
        Err(_) => panic!("a channel index from one addressable response collection must fit i64"),
    }
}

/// Precomputed Transcription projections are serialized without reconstructing application state.
pub(crate) fn transcribe_response(input: ResponseInput<'_>) -> Json {
    let ResponseInput {
        channels,
        segments,
        turns,
        options: opt,
        duration_sec,
        sample_rate_in,
        source_channels,
        duplicate,
        model,
        rtf,
    } = input;
    let mut f: Vec<(&str, Json)> = vec![
        ("duration_sec", Json::round(duration_sec, 2)),
        ("sample_rate_in", Json::Int(i64::from(sample_rate_in))),
        ("channels", Json::Int(channel_index(source_channels.get()))),
        (
            "dual_mono",
            Json::Bool(matches!(duplicate, DuplicateClassification::DualMono)),
        ),
        ("model", Json::str(model)),
        ("rtf", Json::round(rtf, 4)),
    ];

    let multi = channels.len() > 1;
    // Text is shared for one channel/merge and an array by channel for split.
    if opt.split && multi {
        let texts: Vec<Json> = channels
            .iter()
            .map(|channel| Json::str(words_to_text(channel.words())))
            .collect();
        f.push(("channels_text", Json::Array(texts)));
    } else {
        let channel = channels
            .first()
            .expect("a validated multi-channel batch result always retains one channel");
        let text = words_to_text(channel.words());
        f.push(("text", Json::str(text)));
    }

    if opt.words {
        let mut arr: Vec<Json> = Vec::new();
        for channel in channels {
            for word in channel.words() {
                arr.push(word_json(
                    word,
                    if multi { Some(channel.channel()) } else { None },
                ));
            }
        }
        f.push(("words", Json::Array(arr)));
    }
    if opt.segments {
        f.push((
            "segments",
            Json::Array(segments.iter().map(segment_json).collect()),
        ));
    }
    if opt.turns && multi {
        let ts: Vec<Json> = turns.iter().map(turn_json).collect();
        f.push(("turns", Json::Array(ts)));
    }
    obj(f)
}

pub(crate) fn json_ok(body: String) -> Response {
    ([(header::CONTENT_TYPE, JSON_CONTENT_TYPE)], body).into_response()
}

pub(crate) fn json_err(code: StatusCode, message: impl Into<String>) -> Response {
    (
        code,
        [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
        error_object(message),
    )
        .into_response()
}

pub(crate) fn health_response(projection: HealthProjection) -> Response {
    let mut models = vec![Json::str("ctc")];
    if projection.has_rnnt {
        models.push(Json::str("rnnt"));
    }
    json_ok(
        obj(vec![
            ("status", Json::str("ok")),
            ("models", Json::Array(models)),
            ("provider", Json::str(projection.provider.as_str())),
        ])
        .to_string(),
    )
}

pub(crate) fn liveness_response() -> Response {
    json_ok(obj(vec![("status", Json::str("alive"))]).to_string())
}

/// Typed not-ready reason. The wire body names the actual cause instead of a generic refusal,
/// so a stopped recognition worker is never reported as an in-progress shutdown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadinessOutcome {
    Ready,
    Draining,
    WorkerStopped,
}

pub(crate) fn readiness_response(outcome: ReadinessOutcome) -> Response {
    match outcome {
        ReadinessOutcome::Ready => json_ok(obj(vec![("status", Json::str("ready"))]).to_string()),
        ReadinessOutcome::Draining => json_err(StatusCode::SERVICE_UNAVAILABLE, "draining"),
        ReadinessOutcome::WorkerStopped => {
            json_err(StatusCode::SERVICE_UNAVAILABLE, "worker stopped")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::{StatusCode, header};
    use gigaam_transcription::{TurnGap, turns};

    fn w(t: &str, s: f32, e: f32) -> TranscriptWord {
        TranscriptWord::new(t.into(), s, e).expect("test word timestamps are ordered and finite")
    }

    fn channel(index: usize, words: Vec<TranscriptWord>) -> ChannelTranscript {
        ChannelTranscript::new(index, words).expect("test channel words are ordered by start time")
    }

    fn projections(channels: &[ChannelTranscript], gap_seconds: f32) -> (Vec<Turn>, Vec<Turn>) {
        let gap = TurnGap::new(gap_seconds).expect("test turn gap is valid");
        let segments = channels
            .iter()
            .flat_map(|channel| {
                turns(std::slice::from_ref(channel), gap).expect("test channel transcript is valid")
            })
            .collect();
        let turns = turns(channels, gap).expect("test channel transcripts are valid");
        (segments, turns)
    }

    #[test]
    fn mono_minimal() {
        let ch = vec![channel(0, vec![w("hello", 0.0, 0.4), w("world", 0.5, 0.8)])];
        let (segments, turns) = projections(&ch, 1.0);
        let j = transcribe_response(ResponseInput {
            channels: &ch,
            segments: &segments,
            turns: &turns,
            options: Options::default(),
            duration_sec: 0.8,
            sample_rate_in: 16000,
            source_channels: SourceChannelCount::new(1).expect("one source channel is valid"),
            duplicate: DuplicateClassification::NotDualMono,
            model: "ctc",
            rtf: 0.01,
        });
        assert_eq!(
            j.to_string(),
            r#"{"duration_sec":0.8,"sample_rate_in":16000,"channels":1,"dual_mono":false,"model":"ctc","rtf":0.01,"text":"hello world"}"#
        );
    }
    #[test]
    fn stereo_turns_and_words() {
        let ch = vec![
            channel(0, vec![w("hello", 0.0, 0.4)]),
            channel(1, vec![w("yes", 1.0, 1.3)]),
        ];
        let opt = Options {
            words: true,
            turns: true,
            split: true,
            ..Options::default()
        };
        let (segments, turns) = projections(&ch, 1.0);
        let j = transcribe_response(ResponseInput {
            channels: &ch,
            segments: &segments,
            turns: &turns,
            options: opt,
            duration_sec: 1.3,
            sample_rate_in: 8000,
            source_channels: SourceChannelCount::new(2).expect("two source channels are valid"),
            duplicate: DuplicateClassification::NotDualMono,
            model: "ctc",
            rtf: 0.02,
        });
        assert_eq!(
            j.to_string(),
            r#"{"duration_sec":1.3,"sample_rate_in":8000,"channels":2,"dual_mono":false,"model":"ctc","rtf":0.02,"channels_text":["hello","yes"],"words":[{"start":0,"end":0.4,"text":"hello","channel":0},{"start":1,"end":1.3,"text":"yes","channel":1}],"turns":[{"channel":0,"start":0,"end":0.4,"text":"hello"},{"channel":1,"start":1,"end":1.3,"text":"yes"}]}"#
        );
    }

    #[test]
    fn response_serializes_supplied_segments_and_turns_without_reconstructing_them() {
        let channels = vec![
            channel(0, vec![w("first", 0.0, 0.1), w("later", 2.0, 2.1)]),
            channel(1, vec![w("middle", 0.5, 0.6)]),
        ];
        let (segments, turns) = projections(&channels, 1.0);
        let options = Options {
            segments: true,
            turns: true,
            split: true,
            turn_gap: 5.0,
            ..Options::default()
        };
        let rendered = transcribe_response(ResponseInput {
            channels: &channels,
            segments: &segments,
            turns: &turns,
            options,
            duration_sec: 2.1,
            sample_rate_in: 16_000,
            source_channels: SourceChannelCount::new(2).expect("two source channels are valid"),
            duplicate: DuplicateClassification::NotDualMono,
            model: "ctc",
            rtf: 0.01,
        })
        .to_string();

        assert_eq!(
            rendered,
            r#"{"duration_sec":2.1,"sample_rate_in":16000,"channels":2,"dual_mono":false,"model":"ctc","rtf":0.01,"channels_text":["first later","middle"],"segments":[{"start":0,"end":0.1,"channel":0,"text":"first"},{"start":2,"end":2.1,"channel":0,"text":"later"},{"start":0.5,"end":0.6,"channel":1,"text":"middle"}],"turns":[{"channel":0,"start":0,"end":0.1,"text":"first"},{"channel":1,"start":0.5,"end":0.6,"text":"middle"},{"channel":0,"start":2,"end":2.1,"text":"later"}]}"#
        );
    }

    #[tokio::test]
    async fn http_errors_use_the_shared_json_content_type_and_error_object() {
        let response = json_err(StatusCode::BAD_REQUEST, "invalid audio");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&axum::http::HeaderValue::from_static(
                "application/json; charset=utf-8",
            ))
        );
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("HTTP error response body is available"),
            "{\"error\":\"invalid audio\"}"
        );
    }
}
