// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Native batch HTTP adaptation and response projection.

use crate::response::{
    Options, ResponseContract, ResponseVersion, StoredResponseInput, TranscribeErrorCode, json_err,
    transcribe_error_response, write_stored,
};
use axum::{
    Router,
    body::{Body, HttpBody},
    extract::{RawQuery, Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use gigaam_service_host::AdmissionRefusal;
use gigaam_service_host::batch::{
    AdmittedWorkResult, BatchModel, BatchRequest, BatchService as HttpState, TranscribeError,
    UploadError, UploadReservation,
};
use gigaam_service_host::upload::BatchInput;
use gigaam_service_host::wire::{
    QueryParameters, RequestId, bind_vad_policy, query_bool, query_finite_f32, with_request_id,
};
use gigaam_transcription::{BatchChannelPolicy, TurnGap, VadPolicyConfig};
use std::future::poll_fn;
use std::pin::Pin;
use std::time::Instant;

/// Registers the bounded native batch upload route.
pub(crate) fn routes(state: HttpState) -> Router {
    Router::new()
        .route("/v1/transcribe", post(upload_transcribe))
        .with_state(state)
}

pub(super) async fn upload_transcribe(
    State(state): State<HttpState>,
    request: Request,
) -> Response {
    let started = Instant::now();
    let request_id = match RequestId::from_headers(request.headers()) {
        Ok(id) => id,
        Err(error) => {
            state.record_request();
            state.record_error();
            return json_err(StatusCode::BAD_REQUEST, error);
        }
    };
    let query = match TranscribeQuery::parse_for_variant(request.uri().query(), state.variant()) {
        Ok(query) => query,
        Err(error) => {
            state.record_request();
            state.record_error();
            return complete_early_response(&state, &request_id, started, error.response());
        }
    };
    let (parts, body) = request.into_parts();
    let headers = parts.headers;
    let raw_query = parts.uri.query().map(str::to_owned);
    let (body, reader) = match state
        .read_upload(|reader| read_native_body(&state, body, reader))
        .await
    {
        Ok(upload) => upload,
        Err(UploadError::Refused(refusal)) => {
            state.record_request();
            let response =
                native_admission_refusal(refusal, &state, &request_id, query.response_version);
            return complete_early_response(&state, &request_id, started, response);
        }
        Err(UploadError::Parse(rejection)) => {
            state.record_request();
            state.record_error();
            return complete_early_response(
                &state,
                &request_id,
                started,
                rejection.into_response(),
            );
        }
        Err(UploadError::TimedOut) => {
            state.record_request();
            state.record_timeout();
            return complete_early_response(
                &state,
                &request_id,
                started,
                json_err(StatusCode::REQUEST_TIMEOUT, "upload timeout"),
            );
        }
    };
    transcribe_inner(
        State(state),
        headers,
        RawQuery(raw_query),
        body,
        Some(reader),
    )
    .await
}

enum NativeUploadError {
    Read(axum::Error),
    TooLarge,
    Storage,
}

impl IntoResponse for NativeUploadError {
    fn into_response(self) -> Response {
        match self {
            Self::Read(error) => (
                StatusCode::BAD_REQUEST,
                format!("Failed to buffer the request body: {error}"),
            )
                .into_response(),
            Self::TooLarge => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "Failed to buffer the request body: length limit exceeded",
            )
                .into_response(),
            Self::Storage => json_err(
                StatusCode::SERVICE_UNAVAILABLE,
                "upload storage unavailable",
            ),
        }
    }
}

async fn read_native_body(
    state: &HttpState,
    mut body: Body,
    reader: UploadReservation,
) -> Result<BatchInput, NativeUploadError> {
    let limit = state.upload_limit_bytes();
    let mut received = 0_usize;
    let mut upload = None;
    while let Some(frame) = poll_fn(|context| Pin::new(&mut body).poll_frame(context)).await {
        let frame = frame.map_err(NativeUploadError::Read)?;
        let Ok(chunk) = frame.into_data() else {
            continue;
        };
        if chunk.is_empty() {
            continue;
        }
        received = received
            .checked_add(chunk.len())
            .filter(|&size| size <= limit)
            .ok_or(NativeUploadError::TooLarge)?;
        let scratch = match upload.take() {
            Some(scratch) => scratch,
            None => state
                .start_upload(&reader)
                .map_err(|_| NativeUploadError::Storage)?,
        };
        upload = Some(
            scratch
                .append(chunk)
                .await
                .map_err(|_| NativeUploadError::Storage)?,
        );
    }
    Ok(match upload {
        Some(upload) => upload.into(),
        None => Vec::new().into(),
    })
}

/// Adapts a validated batch request while the admission owner retains capacity through completion.
pub(super) async fn transcribe_inner(
    State(state): State<HttpState>,
    headers: HeaderMap,
    RawQuery(query_text): RawQuery,
    body: impl Into<BatchInput>,
    mut reader: Option<UploadReservation>,
) -> Response {
    state.record_request();
    let started = Instant::now();
    let body = body.into();
    let body_len = body.len();
    let request_id = match RequestId::from_headers(&headers) {
        Ok(request_id) => request_id,
        Err(error) => {
            state.record_error();
            drop(body);
            drop(reader);
            return json_err(StatusCode::BAD_REQUEST, error);
        }
    };
    if body.is_empty() {
        state.record_error();
        drop(body);
        drop(reader);
        return with_request_id(
            json_err(StatusCode::BAD_REQUEST, "empty body: audio expected"),
            &request_id,
        );
    }
    let query = match TranscribeQuery::parse_for_variant(query_text.as_deref(), state.variant()) {
        Ok(query) => query,
        Err(error) => {
            state.record_error();
            drop(body);
            drop(reader);
            return with_request_id(error.response(), &request_id);
        }
    };
    let model = query.model.as_str().to_owned();
    let response_version = query.response_version;
    let admission = match state.admit(&mut reader) {
        Ok(admission) => admission,
        Err(refusal) => {
            drop(body);
            drop(reader);
            return native_admission_refusal(refusal, &state, &request_id, response_version);
        }
    };
    let request = match native_batch_request(&query, body) {
        Ok(request) => request,
        Err(error) => {
            state.record_error();
            return with_request_id(
                transcribe_error_response(
                    response_version,
                    error.status(),
                    error_code(&error),
                    error.message(),
                ),
                &request_id,
            );
        }
    };
    let variant = state.variant();
    let response = match state
        .execute_file(
            admission,
            request_id.text().to_owned(),
            started,
            request,
            move |mut result, mut writer| {
                let duration = result.duration;
                let source_rate = result.source_rate;
                let rtf = result.rtf;
                let (selection, words) = result.replay();
                write_stored(
                    StoredResponseInput {
                        contract: ResponseContract {
                            version: query.response_version,
                            vad: query.vad,
                        },
                        selection,
                        options: query.options,
                        duration_sec: duration,
                        sample_rate_in: source_rate,
                        model: query.model.as_str(),
                        variant: variant.as_str(),
                        rtf,
                    },
                    words,
                    &mut writer,
                )
            },
        )
        .await
    {
        AdmittedWorkResult::Completed(Ok(file)) => {
            let length = file.len();
            Response::builder()
                .header(
                    axum::http::header::CONTENT_TYPE,
                    gigaam_service_host::wire::json::JSON_CONTENT_TYPE,
                )
                .header(axum::http::header::CONTENT_LENGTH, length)
                .body(file.into_body())
                .expect("static response headers are valid")
        }
        AdmittedWorkResult::Refused(refusal) => {
            native_admission_refusal(refusal, &state, &request_id, response_version)
        }
        AdmittedWorkResult::Completed(Err(error)) => {
            state.record_error();
            let status = error.status();
            transcribe_error_response(
                response_version,
                status,
                error_code(&error),
                error.message(),
            )
        }
        AdmittedWorkResult::WorkerFailure => {
            state.record_error();
            transcribe_error_response(
                response_version,
                StatusCode::SERVICE_UNAVAILABLE,
                TranscribeErrorCode::Unavailable,
                "internal worker error",
            )
        }
        AdmittedWorkResult::TimedOut => {
            state.record_timeout();
            transcribe_error_response(
                response_version,
                StatusCode::GATEWAY_TIMEOUT,
                TranscribeErrorCode::Timeout,
                "processing timeout",
            )
        }
    };
    state.record_access(
        request_id.text(),
        response.status().as_u16(),
        started,
        body_len,
        &model,
    );
    with_request_id(response, &request_id)
}

pub(super) fn native_batch_request(
    query: &TranscribeQuery,
    bytes: impl Into<BatchInput>,
) -> Result<BatchRequest, TranscribeError> {
    Ok(BatchRequest {
        model: query.model,
        extension: query.extension.clone(),
        audio: bytes.into(),
        channel_policy: http_batch_channel_policy(query.options),
        turn_gap: TurnGap::new(query.options.turn_gap).map_err(TranscribeError::BadRequest)?,
        vad: query.vad,
        scored: false,
        downmix: false,
    })
}

pub(super) fn native_admission_refusal(
    refusal: AdmissionRefusal,
    state: &HttpState,
    request_id: &RequestId,
    version: ResponseVersion,
) -> Response {
    state.record_refusal(refusal);
    match refusal {
        AdmissionRefusal::Overloaded => with_request_id(
            transcribe_error_response(
                version,
                StatusCode::SERVICE_UNAVAILABLE,
                TranscribeErrorCode::Overloaded,
                "service overloaded, retry later",
            ),
            request_id,
        ),
        AdmissionRefusal::Draining => with_request_id(
            transcribe_error_response(
                version,
                StatusCode::SERVICE_UNAVAILABLE,
                TranscribeErrorCode::Draining,
                "draining",
            ),
            request_id,
        ),
    }
}

fn parse_batch_model(value: Option<&str>) -> Result<BatchModel, String> {
    match value {
        None | Some("ctc") => Ok(BatchModel::Ctc),
        Some("rnnt") => Ok(BatchModel::Rnnt),
        Some(value) => match value.parse::<gigaam_model_package::VariantId>() {
            Ok(variant) => Ok(match variant.family() {
                gigaam_model_package::ModelFamily::Ctc => BatchModel::Ctc,
                gigaam_model_package::ModelFamily::Rnnt => BatchModel::Rnnt,
            }),
            Err(_) => Err(format!("unknown model {value:?}")),
        },
    }
}

#[derive(Clone, Debug)]
pub(super) struct TranscribeQuery {
    pub(super) model: BatchModel,
    pub(super) extension: Option<String>,
    pub(super) options: Options,
    pub(super) response_version: ResponseVersion,
    pub(super) vad: Option<VadPolicyConfig>,
}

#[derive(Debug)]
pub(super) enum QueryBindError {
    Invalid(String),
    Unavailable {
        version: ResponseVersion,
        message: &'static str,
    },
}
impl QueryBindError {
    fn response(self) -> Response {
        match self {
            Self::Invalid(message) => json_err(StatusCode::BAD_REQUEST, message),
            Self::Unavailable { version, message } => transcribe_error_response(
                version,
                StatusCode::BAD_REQUEST,
                TranscribeErrorCode::UnsupportedCapability,
                message,
            ),
        }
    }
}

impl TranscribeQuery {
    pub(super) fn parse_for_variant(
        raw: Option<&str>,
        variant: gigaam_model_package::VariantId,
    ) -> Result<Self, QueryBindError> {
        let parameters = QueryParameters::parse(raw).map_err(QueryBindError::Invalid)?;
        let value = parameters.get("model");
        let mut query = Self::parse(raw).map_err(QueryBindError::Invalid)?;
        let selected = match variant.family() {
            gigaam_model_package::ModelFamily::Ctc => BatchModel::Ctc,
            gigaam_model_package::ModelFamily::Rnnt => BatchModel::Rnnt,
        };
        if value.is_none() {
            query.model = selected;
        }
        if let Some(value) = value
            && let Ok(requested) = value.parse::<gigaam_model_package::VariantId>()
            && requested != variant
        {
            return Err(QueryBindError::Unavailable {
                version: query.response_version,
                message: "requested variant is not loaded",
            });
        }
        if query.model != selected {
            return Err(QueryBindError::Unavailable {
                version: query.response_version,
                message: "requested model family is not loaded",
            });
        }
        Ok(query)
    }
    pub(super) fn parse(raw: Option<&str>) -> Result<Self, String> {
        let parameters = QueryParameters::parse(raw)?;
        parameters.reject_unknown(&[
            "model",
            "ext",
            "words",
            "segments",
            "turns",
            "turn_gap",
            "channels",
            "response_version",
            "vad",
            "vad_threshold",
            "vad_neg_threshold",
            "vad_min_speech_ms",
            "vad_min_silence_ms",
            "vad_speech_pad_ms",
        ])?;
        let response_version = match parameters.get("response_version") {
            None | Some("1") => ResponseVersion::V1,
            Some("2") => ResponseVersion::V2,
            Some(value) => return Err(format!("response_version must be 1 or 2, got {value:?}")),
        };
        let vad = bind_vad_policy(&parameters, response_version == ResponseVersion::V2)?;
        let model = parse_batch_model(parameters.get("model"))?;
        let extension = match parameters.get("ext") {
            None => None,
            Some(value) if valid_extension(value) => Some(value.to_owned()),
            Some(value) => {
                return Err(format!(
                    "ext must be one of wav|flac|ogg|opus|mp3|aac|m4a|mp4|mkv|vorbis|pcm, got {value:?}"
                ));
            }
        };
        let options = Options {
            words: query_bool(&parameters, "words", false)?,
            segments: query_bool(&parameters, "segments", false)?,
            turns: query_bool(&parameters, "turns", false)?,
            turn_gap: query_positive_f32(&parameters, "turn_gap", 1.0)?,
            split: match parameters.get("channels") {
                None => false,
                Some("split") => true,
                Some(value) => return Err(format!("channels must be split, got {value:?}")),
            },
        };
        Ok(Self {
            model,
            extension,
            options,
            response_version,
            vad,
        })
    }
}

fn valid_extension(value: &str) -> bool {
    matches!(
        value,
        "wav" | "flac" | "ogg" | "opus" | "mp3" | "aac" | "m4a" | "mp4" | "mkv" | "vorbis" | "pcm"
    )
}

fn query_positive_f32(
    parameters: &QueryParameters,
    key: &str,
    default: f32,
) -> Result<f32, String> {
    let value = query_finite_f32(parameters, key, default)?;
    if value <= 0.0 {
        return Err(format!("{key} must be greater than zero"));
    }
    Ok(value)
}

/// Maps the HTTP response-shape request to the application channel-selection policy.
pub(super) fn http_batch_channel_policy(options: Options) -> BatchChannelPolicy {
    match (options.split, options.turns) {
        (false, false) => BatchChannelPolicy::single_output(),
        (false, true) | (true, false) | (true, true) => BatchChannelPolicy::separate_channels(),
    }
}

fn error_code(error: &TranscribeError) -> TranscribeErrorCode {
    match error {
        TranscribeError::BadRequest(_) => TranscribeErrorCode::InvalidRequest,
        TranscribeError::UnsupportedCapability(_) => TranscribeErrorCode::UnsupportedCapability,
        TranscribeError::Unavailable(_) => TranscribeErrorCode::Unavailable,
    }
}
fn complete_early_response(
    state: &HttpState,
    request_id: &RequestId,
    started: Instant,
    response: Response,
) -> Response {
    state.record_access(
        request_id.text(),
        response.status().as_u16(),
        started,
        0,
        "unknown",
    );
    with_request_id(response, request_id)
}
