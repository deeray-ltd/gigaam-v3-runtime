// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! HTTP adaptation for batch transcription. The body is audio; parameters are strict query text.

use crate::admission::{
    AdmissionRefusal, AdmissionState, HttpAdmission, RequestBodyLimit, RequestWorkOwner,
};
use crate::context::ApplicationContext;
use crate::metrics::Metrics;
use crate::protocol::{QueryParameters, query_bool, query_finite_f32};
use crate::response::{Options, ResponseInput, json_err, json_ok, transcribe_response};
use crate::telemetry::TelemetryProducer;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, RawQuery, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use axum::routing::post;
use gigaam_audio::{EncodedAudio, load_bytes, resample_audio};
use gigaam_recognition::{ExecutionControl, ExecutionState};
use gigaam_transcription::{
    BatchChannelPolicy, BatchSetup, MultiChannelBatchError, MultiChannelBatchOptions,
    MultiChannelBatchResult, MultiChannelBatchSetup, MultiChannelBatchTranscriber, TurnGap,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Narrow state of the byte-consuming HTTP protocol adapter.
#[derive(Clone)]
pub(crate) struct HttpState {
    context: Arc<ApplicationContext>,
    admission: Arc<AdmissionState>,
    metrics: Arc<Metrics>,
    telemetry: TelemetryProducer,
    request_timeout: Duration,
    request_body_limit: RequestBodyLimit,
}

/// Cohesive private inputs assembled by the facade for the HTTP adapter state.
pub(crate) struct HttpStateParameters {
    pub(crate) context: Arc<ApplicationContext>,
    pub(crate) admission: Arc<AdmissionState>,
    pub(crate) metrics: Arc<Metrics>,
    pub(crate) telemetry: TelemetryProducer,
    pub(crate) request_timeout: Duration,
    pub(crate) request_body_limit: RequestBodyLimit,
}

impl HttpState {
    pub(crate) fn new(parameters: HttpStateParameters) -> Self {
        let HttpStateParameters {
            context,
            admission,
            metrics,
            telemetry,
            request_timeout,
            request_body_limit,
        } = parameters;
        Self {
            context,
            admission,
            metrics,
            telemetry,
            request_timeout,
            request_body_limit,
        }
    }
}

/// Opaque route group that guarantees the route-local request-body boundary.
pub(crate) struct HttpRoutes(Router);

impl HttpRoutes {
    pub(crate) fn into_router(self) -> Router {
        self.0
    }
}

/// Builds the only byte-consuming route group and attaches its validated body limit.
pub(crate) fn routes(state: HttpState) -> HttpRoutes {
    let limit = state.request_body_limit.bytes();
    body_limited_routes(
        Router::new()
            .route("/v1/transcribe", post(transcribe))
            .with_state(state),
        limit,
    )
}

fn body_limited_routes(routes: Router, limit: usize) -> HttpRoutes {
    HttpRoutes(routes.layer(DefaultBodyLimit::max(limit)))
}

#[cfg(test)]
pub(crate) fn body_limited_test_routes(routes: Router, limit: RequestBodyLimit) -> HttpRoutes {
    body_limited_routes(routes, limit.bytes())
}

enum TranscribeError {
    BadRequest(String),
    Unavailable(String),
}

impl TranscribeError {
    fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    fn message(self) -> String {
        match self {
            Self::BadRequest(message) | Self::Unavailable(message) => message,
        }
    }
}

fn multi_channel_batch_error(error: MultiChannelBatchError) -> TranscribeError {
    match error {
        MultiChannelBatchError::Input(message) | MultiChannelBatchError::Selection(message) => {
            TranscribeError::BadRequest(message)
        }
        MultiChannelBatchError::Transcription(message) => TranscribeError::Unavailable(message),
    }
}

/// Maps the HTTP response-shape request to the application channel-selection policy.
fn http_batch_channel_policy(options: Options) -> BatchChannelPolicy {
    match (options.split, options.turns) {
        (false, false) => BatchChannelPolicy::single_output(),
        (false, true) | (true, false) | (true, true) => BatchChannelPolicy::separate_channels(),
    }
}

/// Publishes aggregate stage timings only after a complete multi-channel batch result exists.
fn complete_multi_channel_batch(
    metrics: &Metrics,
    operation: impl FnOnce() -> Result<MultiChannelBatchResult, MultiChannelBatchError>,
) -> Result<MultiChannelBatchResult, TranscribeError> {
    let result = operation().map_err(multi_channel_batch_error)?;
    let timings = result.timings();
    metrics.observe_frontend(timings.frontend_seconds());
    metrics.observe_encoder(timings.encoder_seconds());
    metrics.observe_decode(timings.decode_seconds());
    Ok(result)
}

/// A request identifier kept as both log text and a validated response header.
#[derive(Clone)]
struct RequestId {
    text: String,
    header: HeaderValue,
}

impl RequestId {
    fn from_headers(headers: &HeaderMap) -> Result<Self, String> {
        let Some(header) = headers.get("x-request-id") else {
            let text = crate::log::new_req_id();
            let header = HeaderValue::from_bytes(text.as_bytes())
                .expect("generated request identifiers contain only visible ASCII bytes");
            return Ok(Self { text, header });
        };
        let text = header
            .to_str()
            .map_err(|_| "x-request-id must contain visible ASCII text")?;
        if text.is_empty() {
            return Err("x-request-id must not be empty".into());
        }
        Ok(Self {
            text: text.to_owned(),
            header: header.clone(),
        })
    }
}

fn with_request_id(mut response: Response, request_id: &RequestId) -> Response {
    response
        .headers_mut()
        .insert("x-request-id", request_id.header.clone());
    response
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchModel {
    Ctc,
    Rnnt,
}

impl BatchModel {
    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("ctc") => Ok(Self::Ctc),
            Some("rnnt") => Ok(Self::Rnnt),
            Some(value) => Err(format!("model must be ctc or rnnt, got {value:?}")),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Ctc => "ctc",
            Self::Rnnt => "rnnt",
        }
    }
}

#[derive(Clone, Debug)]
struct TranscribeQuery {
    model: BatchModel,
    extension: Option<String>,
    options: Options,
}

impl TranscribeQuery {
    fn parse(raw: Option<&str>) -> Result<Self, String> {
        let parameters = QueryParameters::parse(raw)?;
        parameters.reject_unknown(&[
            "model", "ext", "words", "segments", "turns", "turn_gap", "channels",
        ])?;
        let model = BatchModel::parse(parameters.get("model"))?;
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

enum HttpAdmissionDecision {
    Admitted(HttpAdmission),
    Refused(AdmissionRefusal),
}

enum AdmittedWorkResult<T> {
    Completed(Result<T, TranscribeError>),
    WorkerFailure,
    TimedOut,
}

/// The production execution seam retains the admission owner inside blocking work until it
/// terminalizes, even after the HTTP deadline has returned a timeout response.
async fn execute_admitted_work<T, F>(
    owner: RequestWorkOwner,
    control: ExecutionControl,
    timeout: Duration,
    work: F,
) -> AdmittedWorkResult<T>
where
    T: Send + 'static,
    F: FnOnce(&ExecutionControl) -> Result<T, TranscribeError> + Send + 'static,
{
    let work_control = control.clone();
    let job = tokio::task::spawn_blocking(move || match work(&work_control) {
        Ok(value) => match owner.complete() {
            ExecutionState::Completed => Ok(value),
            ExecutionState::Cancelled => {
                Err(TranscribeError::Unavailable("processing cancelled".into()))
            }
            state => Err(TranscribeError::Unavailable(format!(
                "processing ended in unexpected state {state:?}"
            ))),
        },
        Err(error) => {
            owner.fail();
            Err(error)
        }
    });
    match tokio::time::timeout(timeout, job).await {
        Ok(Ok(result)) => AdmittedWorkResult::Completed(result),
        Ok(Err(_)) => AdmittedWorkResult::WorkerFailure,
        Err(_) => {
            control.request_cancellation();
            AdmittedWorkResult::TimedOut
        }
    }
}

/// Resolves capacity before response projection so no rejected request can start work.
fn resolve_http_admission(admission: &crate::admission::AdmissionState) -> HttpAdmissionDecision {
    match admission.admit_http() {
        Ok(admission) => HttpAdmissionDecision::Admitted(admission),
        Err(refusal) => HttpAdmissionDecision::Refused(refusal),
    }
}

/// Projects typed capacity refusal at the HTTP boundary with its preserved public history.
fn http_admission_refusal(
    refusal: AdmissionRefusal,
    metrics: &Metrics,
    request_id: &RequestId,
) -> Response {
    match refusal {
        AdmissionRefusal::Overloaded => {
            metrics.overload();
            with_request_id(
                json_err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "service overloaded, retry later",
                ),
                request_id,
            )
        }
        AdmissionRefusal::Draining => {
            metrics.transcribe_error();
            with_request_id(
                json_err(StatusCode::SERVICE_UNAVAILABLE, "draining"),
                request_id,
            )
        }
    }
}

/// Completes one already-admitted HTTP history while telemetry remains a nonblocking side effect.
fn complete_admitted_response(
    response: Response,
    telemetry: &TelemetryProducer,
    request_id: &RequestId,
    access: crate::log::TranscribeAccess,
) -> Response {
    telemetry.offer_access(crate::log::render(crate::log::AccessEvent::Transcribe(
        access,
    )));
    with_request_id(response, request_id)
}

/// Adapts a validated batch request while the admission owner retains capacity through completion.
pub(crate) async fn transcribe(
    State(state): State<HttpState>,
    headers: HeaderMap,
    RawQuery(query_text): RawQuery,
    body: Bytes,
) -> Response {
    state.metrics.transcribe();
    let started = Instant::now();
    let metrics = Arc::clone(&state.metrics);
    let body_len = body.len();
    let request_id = match RequestId::from_headers(&headers) {
        Ok(request_id) => request_id,
        Err(error) => {
            metrics.transcribe_error();
            return json_err(StatusCode::BAD_REQUEST, error);
        }
    };
    if body.is_empty() {
        metrics.transcribe_error();
        return with_request_id(
            json_err(StatusCode::BAD_REQUEST, "empty body: audio expected"),
            &request_id,
        );
    }
    let query = match TranscribeQuery::parse(query_text.as_deref()) {
        Ok(query) => query,
        Err(error) => {
            metrics.transcribe_error();
            return with_request_id(json_err(StatusCode::BAD_REQUEST, error), &request_id);
        }
    };
    let model = query.model.as_str().to_owned();
    let admission = match resolve_http_admission(state.admission.as_ref()) {
        HttpAdmissionDecision::Admitted(admission) => admission,
        HttpAdmissionDecision::Refused(refusal) => {
            return http_admission_refusal(refusal, metrics.as_ref(), &request_id);
        }
    };
    let control = ExecutionControl::for_request();
    let owner = RequestWorkOwner::new(admission, control.clone());
    let timeout = state.request_timeout;
    let work_context = Arc::clone(&state.context);
    let work_metrics = Arc::clone(&metrics);
    let work = move |work_control: &ExecutionControl| {
        do_transcribe(
            &work_context,
            &work_metrics,
            &query,
            body.to_vec(),
            work_control,
        )
    };
    let response = match execute_admitted_work(owner, control.clone(), timeout, work).await {
        AdmittedWorkResult::Completed(Ok(json)) => json_ok(json),
        AdmittedWorkResult::Completed(Err(error)) => {
            metrics.transcribe_error();
            let status = error.status();
            json_err(status, error.message())
        }
        AdmittedWorkResult::WorkerFailure => {
            metrics.transcribe_error();
            json_err(StatusCode::SERVICE_UNAVAILABLE, "internal worker error")
        }
        AdmittedWorkResult::TimedOut => {
            metrics.timeout();
            json_err(StatusCode::GATEWAY_TIMEOUT, "processing timeout")
        }
    };
    metrics.observe_latency(started.elapsed().as_secs_f64());
    let status = response.status().as_u16();
    complete_admitted_response(
        response,
        &state.telemetry,
        &request_id,
        crate::log::TranscribeAccess {
            request_id: request_id.text.clone(),
            status,
            milliseconds: started.elapsed().as_millis(),
            bytes: body_len,
            model,
        },
    )
}

/// Synchronous request path: decode -> channels -> scheduled recognition windows -> JSON.
fn do_transcribe(
    context: &ApplicationContext,
    metrics: &Metrics,
    query: &TranscribeQuery,
    bytes: Vec<u8>,
    control: &ExecutionControl,
) -> Result<String, TranscribeError> {
    let scheduler = match query.model {
        BatchModel::Ctc => &context.capabilities.ctc,
        BatchModel::Rnnt => context
            .capabilities
            .rnnt
            .as_ref()
            .ok_or_else(|| TranscribeError::BadRequest("rnnt model is not loaded".into()))?,
    };
    let encoded =
        EncodedAudio::new(bytes, query.extension.clone()).map_err(TranscribeError::BadRequest)?;
    let loaded =
        load_bytes(encoded).map_err(|error| TranscribeError::BadRequest(error.to_string()))?;
    let source_rate = loaded.sample_rate().hertz();
    if !(4000..=192000).contains(&source_rate) {
        return Err(TranscribeError::BadRequest(format!(
            "invalid sample rate {source_rate} Hz (4000..192000)"
        )));
    }
    if loaded.channels().is_empty() {
        return Err(TranscribeError::BadRequest(
            "input has no audio channels".into(),
        ));
    }
    let audio = resample_audio(loaded, context.capabilities.frontend.sample_rate())
        .map_err(TranscribeError::BadRequest)?;
    let options = query.options;
    let channel_policy = http_batch_channel_policy(options);
    let turn_gap = TurnGap::new(options.turn_gap).map_err(TranscribeError::BadRequest)?;
    let batch_options = MultiChannelBatchOptions::new(channel_policy, turn_gap);
    let transcriber = MultiChannelBatchTranscriber::new(MultiChannelBatchSetup::new(
        BatchSetup {
            frontend: Arc::clone(&context.capabilities.frontend),
            decoder: scheduler.window_channel(control.clone()),
            config: context.policy.http_batch,
            control: control.clone(),
            observations: context.policy.observations.clone(),
        },
        batch_options,
    ))
    .map_err(TranscribeError::Unavailable)?;
    let started = Instant::now();
    let result = complete_multi_channel_batch(metrics, || transcriber.transcribe(&audio))?;
    let duration = audio.duration_seconds();
    let rtf = started.elapsed().as_secs_f32() / duration.max(1e-6);
    let response = transcribe_response(ResponseInput {
        channels: result.channels(),
        segments: result.segments(),
        turns: result.turns(),
        options,
        duration_sec: duration,
        sample_rate_in: source_rate,
        source_channels: result.source_channels(),
        duplicate: result.duplicate(),
        model: query.model.as_str(),
        rtf,
    });
    Ok(response.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::RequestWorkOwner;
    use crate::admission::ServiceAdmission;
    use crate::context::ApplicationContext;
    use crate::telemetry::test_support;
    use crate::{
        RequestBodyLimit, ServiceCapabilities, ServiceCapabilitiesParameters, ServicePolicy,
        ServicePolicyParameters,
    };
    use axum::body::to_bytes;
    use gigaam_audio::FeatureMatrix;
    use gigaam_recognition::{Decoded, Device, ExecutionScheduler, FrameRate, WindowDecoder};
    use std::cell::Cell;
    use std::sync::Arc;
    use std::time::Duration;

    fn histogram_count(metrics: &Metrics, name: &str) -> u64 {
        let rendered = metrics.render(0, 0, 0, None);
        let prefix = format!("{name}_count ");
        let line = rendered
            .lines()
            .find(|line| line.starts_with(&prefix))
            .expect("rendered metrics must contain every histogram count");
        let value = line
            .strip_prefix(&prefix)
            .expect("located histogram count must retain its metric prefix");
        value
            .parse::<u64>()
            .expect("rendered histogram count must be an unsigned integer")
    }

    struct ControlledDecoder {
        started: Option<tokio::sync::oneshot::Sender<()>>,
        release: std::sync::mpsc::Receiver<()>,
    }

    struct TerminalDecoder;

    impl WindowDecoder for TerminalDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(25.0).expect("the controlled terminal decoder frame rate is valid")
        }

        fn decode(
            &mut self,
            features: gigaam_audio::FeatureMatrixView<'_>,
        ) -> Result<Decoded, String> {
            Decoded::new(
                Vec::new(),
                vec![false; features.frames()],
                features.frames(),
                0.0,
            )
        }
    }

    fn state_with_telemetry(
        fixture: &crate::tests::AssemblyFixture,
        admission: Arc<crate::admission::AdmissionState>,
        metrics: Arc<Metrics>,
        telemetry: TelemetryProducer,
    ) -> Result<HttpState, String> {
        let capabilities = ServiceCapabilities::new(ServiceCapabilitiesParameters {
            pack: Arc::clone(&fixture.pack),
            frontend: Arc::clone(&fixture.frontend),
            ctc: Arc::new(ExecutionScheduler::spawn(TerminalDecoder)),
            rnnt: None,
            provider: Device::Cpu,
            intra_threads: None,
        })?;
        let policy = ServicePolicy::new(ServicePolicyParameters {
            model_sample_rate: fixture.frontend.sample_rate(),
            window_seconds: 1.0,
            overlap_seconds: 0.0,
            dedup_default: true,
            dedup_window_samples: 1,
            dedup_threshold: 0.99,
            observations: gigaam_transcription::ObservationMode::disabled(),
            backchannel_max_seconds: 0.0,
        })?;
        Ok(HttpState::new(HttpStateParameters {
            context: Arc::new(ApplicationContext::new(capabilities, policy)?),
            admission,
            metrics,
            telemetry,
            request_timeout: Duration::from_secs(1),
            request_body_limit: RequestBodyLimit::new(8_192)?,
        }))
    }

    #[derive(Debug, Eq, PartialEq)]
    struct ResponseProjection {
        status: StatusCode,
        request_id: Option<HeaderValue>,
        body: Bytes,
    }

    async fn project(response: Response) -> ResponseProjection {
        let status = response.status();
        let request_id = response.headers().get("x-request-id").cloned();
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("controlled terminal response body is available");
        ResponseProjection {
            status,
            request_id,
            body,
        }
    }

    async fn admitted_terminal_histories(producer: &TelemetryProducer) -> Vec<ResponseProjection> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-request-id",
            HeaderValue::from_static("telemetry-history"),
        );
        let request_id = RequestId::from_headers(&headers)
            .expect("controlled terminal request identifier is valid");
        let mut histories = Vec::new();
        for (response, status, model) in [
            (json_ok("{\"text\":\"ok\"}".to_owned()), 200_u16, "ctc"),
            (
                json_err(StatusCode::BAD_REQUEST, "invalid audio"),
                400_u16,
                "ctc",
            ),
            (
                json_err(StatusCode::GATEWAY_TIMEOUT, "deadline elapsed"),
                504_u16,
                "rnnt",
            ),
        ] {
            histories.push(
                project(complete_admitted_response(
                    response,
                    producer,
                    &request_id,
                    crate::log::TranscribeAccess {
                        request_id: request_id.text.clone(),
                        status,
                        milliseconds: 7,
                        bytes: 11,
                        model: model.to_owned(),
                    },
                ))
                .await,
            );
        }
        histories
    }

    impl WindowDecoder for ControlledDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(100.0).expect("the controlled decoder frame rate is valid")
        }

        fn decode(
            &mut self,
            features: gigaam_audio::FeatureMatrixView<'_>,
        ) -> Result<Decoded, String> {
            let started = self
                .started
                .take()
                .ok_or_else(|| "controlled decoder may run exactly once".to_owned())?;
            started
                .send(())
                .map_err(|_| "the test must observe the controlled decoder starting".to_owned())?;
            match self.release.recv() {
                Ok(()) => Decoded::new(
                    vec![],
                    vec![false; features.frames()],
                    features.frames(),
                    0.0,
                ),
                Err(_) => Err("the test must release the controlled decoder".into()),
            }
        }
    }

    #[test]
    fn transcribe_error_statuses_are_typed() {
        assert_eq!(
            TranscribeError::BadRequest("invalid audio".into()).status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            TranscribeError::Unavailable("worker unavailable".into()).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn transcribe_query_percent_decodes_and_preserves_typed_options() {
        let query = TranscribeQuery::parse(Some(
            "model=ctc&words=true&turn_gap=1%2E5&channels=split&ext=wav",
        ))
        .expect("a complete documented query must parse");
        assert_eq!(query.model, BatchModel::Ctc);
        assert_eq!(query.extension.as_deref(), Some("wav"));
        assert!(query.options.words);
        assert!(query.options.split);
        assert_eq!(query.options.turn_gap, 1.5);
    }

    #[test]
    fn transcribe_query_refuses_malformed_duplicate_unknown_and_nonfinite_values() {
        for raw in [
            "words=true&words=false",
            "unknown=value",
            "words",
            "words=%ZZ",
            "turn_gap=NaN",
            "turn_gap=0",
            "channels=merge",
            "words=on",
        ] {
            assert!(
                TranscribeQuery::parse(Some(raw)).is_err(),
                "query {raw:?} must be refused"
            );
        }
    }

    #[test]
    fn http_output_choices_exhaustively_map_to_the_typed_batch_policy() {
        for (query, expected) in [
            (None, BatchChannelPolicy::single_output()),
            (Some("turns=true"), BatchChannelPolicy::separate_channels()),
            (
                Some("channels=split"),
                BatchChannelPolicy::separate_channels(),
            ),
            (
                Some("channels=split&turns=true"),
                BatchChannelPolicy::separate_channels(),
            ),
        ] {
            let query = TranscribeQuery::parse(query)
                .expect("each documented HTTP output-choice combination must parse");
            assert_eq!(http_batch_channel_policy(query.options), expected);
        }
    }

    #[test]
    fn failed_multi_channel_batch_publishes_no_aggregate_stage_samples() {
        let metrics = Metrics::new(1, 1);
        let error = match complete_multi_channel_batch(&metrics, || {
            Err(MultiChannelBatchError::Transcription(
                "test decoder failure".into(),
            ))
        }) {
            Ok(_) => panic!(
                "a failed multi-channel batch operation must remain failed at the Service boundary"
            ),
            Err(error) => error,
        };

        assert_eq!(error.status(), StatusCode::SERVICE_UNAVAILABLE);
        for name in [
            "asr_frontend_seconds",
            "asr_encoder_seconds",
            "asr_decode_seconds",
        ] {
            assert_eq!(
                histogram_count(&metrics, name),
                0,
                "a failed multi-channel batch operation must publish no {name} histogram sample"
            );
        }
    }

    #[test]
    fn multi_channel_batch_input_refusal_projects_to_bad_request() {
        let message = "test input sample rate is incompatible with the frontend";
        let error = multi_channel_batch_error(MultiChannelBatchError::Input(message.into()));
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(error.message(), message);
    }

    #[test]
    fn request_identifier_is_validated_before_the_handler_returns_a_response() {
        let mut invalid = HeaderMap::new();
        let invalid_value = HeaderValue::from_bytes(&[0xFF])
            .expect("an opaque HTTP header value can be constructed for this boundary test");
        invalid.insert("x-request-id", invalid_value);
        let error = match RequestId::from_headers(&invalid) {
            Ok(_) => panic!("a non-text request identifier must be refused"),
            Err(error) => error,
        };
        assert_eq!(
            json_err(StatusCode::BAD_REQUEST, error).status(),
            StatusCode::BAD_REQUEST
        );

        let mut valid = HeaderMap::new();
        valid.insert("x-request-id", HeaderValue::from_static("request-123"));
        let request_id = RequestId::from_headers(&valid)
            .expect("a visible request identifier must retain its typed header value");
        let response = with_request_id(json_ok("{}".into()), &request_id);
        assert_eq!(
            response.headers().get("x-request-id"),
            valid.get("x-request-id")
        );
    }

    #[tokio::test]
    async fn deadline_cancellation_retains_admission_until_request_work_terminalizes() {
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
            .expect("test admission settings are valid");
        let admission = crate::admission::AdmissionState::new(&settings)
            .expect("test admission state is valid");
        let token = admission
            .admit_http()
            .expect("the test starts with one available HTTP admission permit");
        let control = ExecutionControl::for_request();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let (terminal_tx, terminal_rx) = tokio::sync::oneshot::channel();
        let scheduler = Arc::new(ExecutionScheduler::spawn(ControlledDecoder {
            started: Some(started_tx),
            release: release_rx,
        }));
        let mut scheduled = scheduler.window_channel(control.clone());
        let owner = RequestWorkOwner::new(token, control.clone());
        let features = FeatureMatrix::from_values(1, 1, vec![0.0])
            .expect("the controlled request feature matrix is valid");

        let job = tokio::task::spawn_blocking(move || {
            let terminal = match scheduled.decode(features.view()) {
                Ok(_) => owner.complete(),
                Err(_) => owner.fail(),
            };
            if terminal_tx.send(terminal).is_err() {
                panic!("the test must observe controlled request work terminalization");
            }
        });

        started_rx
            .await
            .expect("controlled request work must be known running before its deadline");
        assert_eq!(
            control.state(),
            ExecutionState::Running,
            "the decoder start event must put the request under running control"
        );
        let deadline = tokio::time::timeout(Duration::ZERO, job).await;
        assert!(
            deadline.is_err(),
            "the deadline must win while the controlled blocking work is running"
        );
        assert_eq!(
            control.request_cancellation(),
            ExecutionState::CancelRequested,
            "the request owner must cancel exactly the running request"
        );
        assert_eq!(
            admission.available_http(),
            0,
            "admission remains unavailable while cancelled blocking work owns its permit"
        );

        release_tx
            .send(())
            .expect("controlled request work must still wait for its release");
        let terminal = tokio::time::timeout(Duration::from_secs(5), terminal_rx)
            .await
            .expect("controlled request work must finish after release")
            .expect("controlled request work must report its terminal state");
        assert_eq!(
            terminal,
            ExecutionState::Cancelled,
            "late completion must acknowledge cancellation rather than revive success"
        );
        assert_eq!(control.state(), ExecutionState::Cancelled);
        assert_eq!(
            admission.available_http(),
            1,
            "admission becomes available only after request work terminalizes"
        );
    }

    #[tokio::test]
    async fn production_execution_seam_keeps_the_committed_permit_after_its_timeout_response() {
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
            .expect("test admission settings are valid");
        let admission = Arc::new(
            crate::admission::AdmissionState::new(&settings)
                .expect("validated test admission settings create state"),
        );
        let control = ExecutionControl::for_request();
        let owner = RequestWorkOwner::new(
            admission
                .admit_http()
                .expect("one running request owns the only HTTP permit"),
            control.clone(),
        );
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let observed_admission = Arc::clone(&admission);
        let task = tokio::spawn(execute_admitted_work(
            owner,
            control.clone(),
            Duration::from_millis(1),
            move |_| {
                started_tx.send(()).map_err(|_| {
                    TranscribeError::Unavailable("test start observer dropped".into())
                })?;
                release_rx.recv().map_err(|_| {
                    TranscribeError::Unavailable("test release channel closed".into())
                })?;
                Ok::<_, TranscribeError>(())
            },
        ));
        started_rx
            .await
            .expect("the production work seam must start its blocking closure");
        assert_eq!(observed_admission.available_http(), 0);
        assert!(matches!(
            task.await.expect("the execution seam task must not panic"),
            AdmittedWorkResult::TimedOut
        ));
        assert_eq!(
            observed_admission.available_http(),
            0,
            "the timeout result must return while the blocking owner retains admission"
        );
        release_tx
            .send(())
            .expect("the controlled blocking closure must receive its release");
        tokio::time::timeout(Duration::from_secs(5), async {
            while observed_admission.available_http() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the late blocking terminalization must release admission");
        assert_eq!(control.state(), ExecutionState::Cancelled);
    }

    fn metric_counter(metrics: &Metrics, name: &str) -> u64 {
        let rendered = metrics.render(1, 1, 0, None);
        let prefix = format!("{name} ");
        let line = match rendered.lines().find(|line| line.starts_with(&prefix)) {
            Some(line) => line,
            None => panic!("rendered metrics must include the {name} counter"),
        };
        let value = match line.strip_prefix(&prefix) {
            Some(value) => value,
            None => panic!("matched metrics line must retain its counter value"),
        };
        value
            .parse::<u64>()
            .expect("a Prometheus counter projection must be an unsigned integer")
    }

    #[tokio::test]
    async fn valid_post_drain_http_is_exactly_refused_without_starting_work() {
        let metrics = Metrics::new(1, 1);
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
            .expect("test admission settings are valid");
        let admission = crate::admission::AdmissionState::new(&settings)
            .expect("validated test admission settings create state");
        admission.begin_draining();
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-request-id",
            HeaderValue::from_static("post-drain-request"),
        );
        let request_id =
            RequestId::from_headers(&headers).expect("a visible test request identifier is valid");
        let work_invocations = Cell::new(0_u8);
        let response = match resolve_http_admission(&admission) {
            HttpAdmissionDecision::Admitted(token) => {
                work_invocations.set(
                    work_invocations
                        .get()
                        .checked_add(1)
                        .expect("one controlled work invocation fits u8"),
                );
                drop(token);
                json_ok("{}".to_owned())
            }
            HttpAdmissionDecision::Refused(refusal) => {
                http_admission_refusal(refusal, &metrics, &request_id)
            }
        };
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.headers().get("x-request-id"),
            headers.get("x-request-id"),
            "the valid post-drain refusal preserves the caller request identifier"
        );
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("the bounded test response body is available"),
            "{\"error\":\"draining\"}"
        );
        assert_eq!(
            work_invocations.get(),
            0,
            "a post-drain application refusal must not invoke batch work"
        );
        assert_eq!(metric_counter(&metrics, "asr_transcribe_errors_total"), 1);
        assert_eq!(metric_counter(&metrics, "asr_overload_total"), 0);
        assert_eq!(metric_counter(&metrics, "asr_timeout_total"), 0);
    }

    #[tokio::test]
    async fn telemetry_outcomes_preserve_admitted_http_histories_and_pre_admission_silence()
    -> Result<(), String> {
        let baseline = test_support::prepare(1, 1)?;
        let baseline_producer = baseline.producer();
        let expected = admitted_terminal_histories(&baseline_producer).await;
        assert_eq!(baseline.snapshot().entered, 3);
        drop(baseline_producer);

        let saturated = test_support::prepare(1, 1)?;
        let saturated_producer = saturated.producer();
        for number in 0..4 {
            saturated_producer.offer_access(crate::log::render(crate::log::AccessEvent::WsError(
                crate::log::WsErrorAccess {
                    phase: "telemetry-fill",
                    message: format!("fill-{number}"),
                },
            )));
        }
        assert_eq!(saturated.snapshot().queued, 4);
        assert_eq!(
            admitted_terminal_histories(&saturated_producer).await,
            expected
        );
        let saturated_snapshot = saturated.snapshot();
        assert_eq!(saturated_snapshot.entered, 7);
        assert_eq!(saturated_snapshot.queued, 4);
        assert_eq!(saturated_snapshot.undelivered, 4);
        assert_eq!(saturated_snapshot.queue_full, 3);
        assert_eq!(saturated_snapshot.sink_closed, 0);

        let receiver_dropped = test_support::prepare(1, 1)?.into_receiver_dropped();
        let receiver_dropped_producer = receiver_dropped.producer();
        assert_eq!(
            admitted_terminal_histories(&receiver_dropped_producer).await,
            expected
        );
        let receiver_dropped_snapshot = receiver_dropped.snapshot();
        assert_eq!(receiver_dropped_snapshot.entered, 3);
        assert_eq!(receiver_dropped_snapshot.queued, 0);
        assert_eq!(receiver_dropped_snapshot.undelivered, 0);
        assert_eq!(receiver_dropped_snapshot.sink_closed, 3);

        let failing = test_support::prepare(1, 1)?.into_failing_stdout()?;
        let failing_producer = failing.producer();
        assert_eq!(
            admitted_terminal_histories(&failing_producer).await,
            expected
        );
        failing.wait_until_terminal();
        let failing_snapshot = failing.snapshot();
        assert_eq!(failing_snapshot.entered, 3);
        assert_eq!(failing_snapshot.queued, 0);
        assert_eq!(failing_snapshot.undelivered, 0);
        assert_eq!(failing_snapshot.sink_closed, 3);
        assert!(
            failing
                .metrics()
                .render(1, 1, 0, None)
                .contains("asr_telemetry_write_failures_total{destination=\"stdout\"} 1\n")
        );
        drop(failing_producer);
        failing.finish();

        let fixture = crate::tests::assembly_fixture()?;
        let pre_admission = test_support::prepare(1, 1)?;
        let pre_admission_metrics = pre_admission.metrics_arc();
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))?;
        let admission = Arc::new(crate::admission::AdmissionState::new(&settings)?);
        admission.begin_draining();
        let state = state_with_telemetry(
            &fixture,
            Arc::clone(&admission),
            pre_admission_metrics,
            pre_admission.producer(),
        )?;
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("pre-admission"));
        let response = transcribe(
            State(state),
            headers,
            RawQuery(Some("ext=wav".to_owned())),
            Bytes::from_static(b"not-a-wave-file"),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .map_err(|error| format!("read pre-admission refusal body: {error}"))?,
            "{\"error\":\"draining\"}"
        );
        let pre_admission_snapshot = pre_admission.snapshot();
        assert_eq!(pre_admission_snapshot.entered, 0);
        assert_eq!(pre_admission_snapshot.queued, 0);
        assert_eq!(pre_admission_snapshot.undelivered, 0);
        assert_eq!(pre_admission_snapshot.queue_full, 0);
        assert_eq!(pre_admission_snapshot.sink_closed, 0);
        Ok(())
    }
}
