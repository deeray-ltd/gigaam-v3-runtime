// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Pure WebSocket wire projections over committed Transcription events.

use crate::json::{JSON_CONTENT_TYPE, Json, error_object, obj};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use gigaam_transcription::{
    DialogTurn, EndpointSource, FinalReason, MultiChannelStep, SourceChannelCount, StreamConfig,
    StreamEvent, StreamLockPolicy, StreamWord, TurnsPatch, WordFinality, WordStability,
};

/// Confirms the immutable policy bound to an active session without claiming channel activation.
pub(crate) fn applied_config_event(config: &StreamConfig) -> String {
    let (endpoint, vad) = match config.endpoint_source() {
        EndpointSource::Manual => ("manual", obj(vec![("enabled", Json::Bool(false))])),
        EndpointSource::Blank => ("blank", obj(vec![("enabled", Json::Bool(false))])),
        EndpointSource::Vad => (
            "vad",
            obj(vec![
                ("enabled", Json::Bool(true)),
                (
                    "threshold",
                    Json::Num(f64::from(config.vad_threshold().get())),
                ),
            ]),
        ),
    };
    let lock = match config.lock_policy() {
        StreamLockPolicy::Advisory => "advisory",
        StreamLockPolicy::CommitStable => "commit_stable",
    };
    obj(vec![
        ("type", Json::str("applied_stream_config")),
        ("schema_version", Json::Usize(1)),
        (
            "model_sample_rate",
            Json::Num(f64::from(config.sample_rate().hertz())),
        ),
        ("endpoint", Json::str(endpoint)),
        ("window_sec", Json::Num(f64::from(config.window_sec()))),
        ("overlap_sec", Json::Num(f64::from(config.overlap_sec()))),
        ("step_sec", Json::Num(f64::from(config.step_sec()))),
        ("horizon_sec", Json::Num(f64::from(config.horizon_sec()))),
        ("lock", Json::str(lock)),
        (
            "seam_after_sec",
            Json::Num(f64::from(config.seam_after_sec())),
        ),
        (
            "keep_silence_sec",
            Json::Num(f64::from(config.keep_silence_sec())),
        ),
        (
            "endpoint_rules",
            obj(vec![
                (
                    "after_speech_sec",
                    Json::Num(f64::from(config.rules().after_speech_seconds())),
                ),
                (
                    "without_speech_sec",
                    Json::Num(f64::from(config.rules().without_speech_seconds())),
                ),
            ]),
        ),
        ("vad", vad),
    ])
    .to_string()
}

fn word_json(word: &StreamWord) -> Json {
    obj(vec![
        ("start", Json::round(word.start(), 2)),
        ("end", Json::round(word.end(), 2)),
        ("text", Json::str(word.text().to_owned())),
        (
            "stable",
            Json::Bool(word.stability() == WordStability::Stable),
        ),
    ])
}

fn event_json(event: &StreamEvent, channel: Option<usize>) -> String {
    let mut fields = match event {
        StreamEvent::Words(words) => vec![
            ("type", Json::str("words")),
            ("at", Json::round(words.at(), 2)),
            ("revise_from", Json::Usize(words.revise_from())),
            (
                "words",
                Json::Array(words.words().iter().map(word_json).collect()),
            ),
        ],
        StreamEvent::Stable(frontier) => vec![
            ("type", Json::str("stable")),
            ("at", Json::round(frontier.at(), 2)),
            ("upto", Json::Usize(frontier.upto())),
        ],
        StreamEvent::Final(final_event) => vec![
            ("type", Json::str("final")),
            ("at", Json::round(final_event.at(), 2)),
            ("upto", Json::Usize(final_event.upto())),
            (
                "endpoint",
                Json::Bool(final_event.reason() == FinalReason::Endpoint),
            ),
        ],
    };
    if let Some(channel) = channel {
        fields.push(("channel", Json::Usize(channel)));
    }
    obj(fields).to_string()
}

fn turn_json(turn: &DialogTurn) -> Json {
    obj(vec![
        ("channel", Json::Usize(turn.channel())),
        ("k", Json::Usize(turn.index())),
        ("start", Json::round(turn.start(), 2)),
        ("end", Json::round(turn.end(), 2)),
        ("text", Json::str(turn.text().to_owned())),
        (
            "stable",
            Json::Bool(turn.stability() == WordStability::Stable),
        ),
        ("final", Json::Bool(turn.finality() == WordFinality::Final)),
        (
            "backchannel",
            Json::Bool(turn.backchannel() == gigaam_transcription::BackchannelMark::Yes),
        ),
    ])
}

fn turns_json(patch: &TurnsPatch) -> String {
    obj(vec![
        ("type", Json::str("turns")),
        ("revise_from", Json::Usize(patch.revise_from())),
        ("frontier", Json::round(patch.frontier(), 2)),
        (
            "turns",
            Json::Array(patch.turns().iter().map(turn_json).collect()),
        ),
    ])
    .to_string()
}

/// Serializes each already-committed group in its established channel-before-dialogue order.
pub(crate) fn serialize_step(
    step: &MultiChannelStep,
    source_channels: SourceChannelCount,
) -> Vec<String> {
    let mut messages = Vec::new();
    for group in step.emission_groups() {
        for channel_event in group.channel_events() {
            let channel = match source_channels.get() {
                1 => None,
                _ => Some(channel_event.channel().index()),
            };
            messages.push(event_json(channel_event.event(), channel));
        }
        if let Some(patch) = group.dialog_patch() {
            messages.push(turns_json(patch));
        }
    }
    messages
}

/// Projects a WebSocket handshake refusal without coupling the stream adapter to HTTP responses.
pub(crate) fn pre_upgrade_refusal(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
        error_object(message),
    )
        .into_response()
}

/// The exact in-band terminal failure envelope shared by every stream path.
pub(crate) fn error_event(message: impl Into<String>) -> String {
    obj(vec![
        ("type", Json::str("error")),
        ("message", Json::str(message)),
    ])
    .to_string()
}

/// The exact successful terminal event.
pub(crate) fn end_event() -> String {
    obj(vec![("type", Json::str("end"))]).to_string()
}

#[cfg(test)]
mod tests {
    use super::{applied_config_event, end_event, error_event, pre_upgrade_refusal};
    use axum::body::to_bytes;
    use axum::http::{StatusCode, header};
    use gigaam_audio::SampleRate;
    use gigaam_transcription::{
        EndpointRules, EndpointSource, StreamConfig, StreamLockPolicy, VadProbability,
    };

    #[test]
    fn applied_stream_policy_preserves_exact_bound_floats_and_active_detector_scope()
    -> Result<(), String> {
        let config = StreamConfig::timing_changes()
            .with_window_sec(1.0)?
            .with_overlap_sec(0.25)?
            .with_step_sec(0.125)?
            .with_horizon_sec(0.75)?
            .with_seam_after_sec(0.375)?
            .with_keep_silence_sec(0.0625)?
            .apply(StreamConfig::checked_default(SampleRate::new(16_000)?)?)?
            .with_rules(EndpointRules::new(0.0, 0.25)?)?
            .with_lock_policy(StreamLockPolicy::CommitStable)?
            .with_endpoint_source(EndpointSource::Vad)?
            .with_vad_threshold(VadProbability::new(0.7)?)?;
        assert_eq!(
            applied_config_event(&config),
            concat!(
                "{\"type\":\"applied_stream_config\",\"schema_version\":1,",
                "\"model_sample_rate\":16000,\"endpoint\":\"vad\",\"window_sec\":1,",
                "\"overlap_sec\":0.25,\"step_sec\":0.125,\"horizon_sec\":0.75,",
                "\"lock\":\"commit_stable\",\"seam_after_sec\":0.375,\"keep_silence_sec\":0.0625,",
                "\"endpoint_rules\":{\"after_speech_sec\":0,\"without_speech_sec\":0.25},",
                "\"vad\":{\"enabled\":true,\"threshold\":0.699999988079071}}"
            )
        );
        let blank = applied_config_event(&config.with_endpoint_source(EndpointSource::Blank)?);
        assert!(blank.contains("\"endpoint\":\"blank\""));
        assert!(blank.ends_with("\"vad\":{\"enabled\":false}}"));
        assert!(!blank.contains("threshold"));
        Ok(())
    }

    #[tokio::test]
    async fn stream_error_and_end_projections_share_the_exact_json_envelope() {
        assert_eq!(
            error_event("draining"),
            "{\"type\":\"error\",\"message\":\"draining\"}"
        );
        assert_eq!(end_event(), "{\"type\":\"end\"}");
        let refusal = pre_upgrade_refusal(StatusCode::SERVICE_UNAVAILABLE, "draining");
        assert_eq!(refusal.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            refusal.headers().get(header::CONTENT_TYPE),
            Some(&axum::http::HeaderValue::from_static(
                "application/json; charset=utf-8",
            ))
        );
        assert_eq!(
            to_bytes(refusal.into_body(), usize::MAX)
                .await
                .expect("pre-upgrade response body is available"),
            "{\"error\":\"draining\"}"
        );
    }
}
