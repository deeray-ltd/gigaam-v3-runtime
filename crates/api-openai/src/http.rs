// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! OpenAI batch HTTP routes and multipart adaptation.

use axum::{
    Router,
    extract::{DefaultBodyLimit, FromRequest, Multipart, Request, State},
    http::StatusCode,
    response::Response,
    routing::{get, post},
};
use gigaam_service_host::{
    AdmissionRefusal,
    batch::{
        AdmittedWorkResult, BatchModel, BatchRequest, BatchService, TranscribeError, UploadError,
    },
    wire::{RequestId, json_ok, with_request_id},
};
use gigaam_transcription::{BatchChannelPolicy, TurnGap};
use std::time::Instant;

pub(crate) fn routes(state: BatchService) -> Router {
    let limit = state.upload_limit_bytes();
    Router::new()
        .route("/v1/audio/transcriptions", post(openai_transcribe))
        .route("/v1/models", get(openai_models))
        .layer(DefaultBodyLimit::max(limit))
        .with_state(state)
}

fn complete_early_response(
    state: &BatchService,
    id: &RequestId,
    started: Instant,
    response: Response,
) -> Response {
    state.record_access(id.text(), response.status().as_u16(), started, 0, "unknown");
    with_request_id(response, id)
}

async fn openai_models(State(state): State<BatchService>) -> Response {
    use gigaam_service_host::wire::json::{Json, obj};
    let variant = state.variant();
    let mut names = vec![variant.as_str()];
    match variant.family() {
        gigaam_model_package::ModelFamily::Ctc => names.extend(["gigaam-v3", "gigaam-v3-ctc"]),
        gigaam_model_package::ModelFamily::Rnnt => names.push("gigaam-v3-rnnt"),
    }
    let data = names
        .into_iter()
        .map(|id| {
            obj(vec![
                ("id", Json::str(id)),
                ("object", Json::str("model")),
                ("created", Json::Int(0)),
                ("owned_by", Json::str("local")),
            ])
        })
        .collect();
    json_ok(
        obj(vec![
            ("object", Json::str("list")),
            ("data", Json::Array(data)),
        ])
        .to_string(),
    )
}

pub(super) async fn parse_openai_multipart(
    state: &BatchService,
    request: Request,
) -> Result<crate::openai::Input, crate::openai::ApiError> {
    let mut multipart = Multipart::from_request(request, state)
        .await
        .map_err(|error| crate::openai::ApiError::new(error.status(), error.to_string(), None))?;
    let mut fields = Vec::new();
    while let Some(mut field) = multipart
        .next_field()
        .await
        .map_err(|error| crate::openai::ApiError::new(error.status(), error.to_string(), None))?
    {
        if fields.len() >= 16 {
            return Err(crate::openai::ApiError::bad(
                "too many multipart fields",
                None,
            ));
        }
        let name = field
            .name()
            .ok_or_else(|| crate::openai::ApiError::bad("field name is required", None))?
            .to_owned();
        let filename = field.file_name().map(str::to_owned);
        let mut bytes = Vec::new();
        while let Some(chunk) = field.chunk().await.map_err(|error| {
            crate::openai::ApiError::new(error.status(), error.to_string(), None)
        })? {
            if name != "file" && chunk.len() > 16 * 1024 - bytes.len() {
                return Err(crate::openai::ApiError::bad(
                    "multipart text field exceeds 16 KiB",
                    None,
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        fields.push((name, filename, bytes));
    }
    crate::openai::parse_fields_for_variant(fields, state.variant())
}

pub(super) async fn openai_transcribe(
    State(state): State<BatchService>,
    request: Request,
) -> Response {
    state.record_request();
    let started = Instant::now();
    let request_id = match RequestId::from_headers(request.headers()) {
        Ok(id) => id,
        Err(error) => {
            state.record_error();
            return crate::openai::ApiError::bad(error, None).response();
        }
    };
    let (mut input, reader) = match state
        .read_upload(|| parse_openai_multipart(&state, request))
        .await
    {
        Ok(input) => input,
        Err(UploadError::Refused(refusal)) => {
            state.record_refusal(refusal);
            return complete_early_response(&state, &request_id, started, openai_refusal(refusal));
        }
        Err(UploadError::Parse(error)) => {
            state.record_error();
            return complete_early_response(&state, &request_id, started, error.response());
        }
        Err(UploadError::TimedOut) => {
            state.record_timeout();
            return complete_early_response(
                &state,
                &request_id,
                started,
                crate::openai::ApiError::new(StatusCode::REQUEST_TIMEOUT, "upload timeout", None)
                    .response(),
            );
        }
    };
    let bytes = input.audio.len();
    let model = match input.engine {
        crate::openai::Engine::Ctc => "ctc",
        crate::openai::Engine::Rnnt => "rnnt",
    };
    let mut reader = Some(reader);
    let admission = match state.admit(&mut reader) {
        Ok(admission) => admission,
        Err(refusal) => {
            drop(input);
            drop(reader);
            state.record_refusal(refusal);
            return complete_early_response(&state, &request_id, started, openai_refusal(refusal));
        }
    };
    let audio = std::mem::take(&mut input.audio);
    let gap = TurnGap::new(1.0).expect("one second is a valid OpenAI segment grouping gap");
    let request = BatchRequest {
        model: match input.engine {
            crate::openai::Engine::Ctc => BatchModel::Ctc,
            crate::openai::Engine::Rnnt => BatchModel::Rnnt,
        },
        extension: input.extension.clone(),
        audio,
        channel_policy: BatchChannelPolicy::single_output(),
        turn_gap: gap,
        vad: input.vad,
        scored: input.segment,
        downmix: true,
    };
    let result = state
        .execute(
            admission,
            request_id.text().to_owned(),
            started,
            request,
            move |output| {
                crate::openai_response::project(crate::openai_response::ProjectionInput {
                    result: &output.result,
                    duration: output.duration,
                    format: input.format,
                    word: input.word,
                    segment: input.segment,
                    gap,
                })
                .map_err(TranscribeError::Unavailable)
            },
        )
        .await;
    let response = match result {
        AdmittedWorkResult::Completed(Ok(response)) => response,
        AdmittedWorkResult::Completed(Err(error)) => {
            state.record_error();
            let (status, message) = match error {
                TranscribeError::BadRequest(message)
                | TranscribeError::UnsupportedCapability(message) => {
                    (StatusCode::BAD_REQUEST, message)
                }
                TranscribeError::Unavailable(message) => (StatusCode::SERVICE_UNAVAILABLE, message),
            };
            crate::openai::ApiError::new(status, message, None).response()
        }
        AdmittedWorkResult::WorkerFailure => {
            state.record_error();
            crate::openai::ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "internal worker error",
                None,
            )
            .response()
        }
        AdmittedWorkResult::TimedOut => {
            state.record_timeout();
            crate::openai::ApiError::new(StatusCode::GATEWAY_TIMEOUT, "processing timeout", None)
                .response()
        }
    };
    state.record_access(
        request_id.text(),
        response.status().as_u16(),
        started,
        bytes,
        model,
    );
    with_request_id(response, &request_id)
}

fn openai_refusal(refusal: AdmissionRefusal) -> Response {
    let message = match refusal {
        AdmissionRefusal::Overloaded => "service overloaded, retry later",
        AdmissionRefusal::Draining => "draining",
    };
    crate::openai::ApiError::new(StatusCode::SERVICE_UNAVAILABLE, message, None).response()
}
