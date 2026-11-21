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
use crate::log::{BatchProgress, BatchStage, BatchStageOutcome, TranscribeProgress};
use crate::metrics::Metrics;

#[cfg(test)]
use crate::response::{ResponseVersion, json_err, json_ok};
use crate::telemetry::TelemetryProducer;
#[cfg(test)]
use crate::wire::{RequestId, with_request_id};
use axum::http::StatusCode;
#[cfg(test)]
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, RawQuery, Request, State},
    http::{HeaderMap, HeaderValue},
    response::Response,
};
#[cfg(test)]
pub(crate) use native_http::routes as native_test_routes;
#[cfg(test)]
use native_http::{
    TranscribeQuery, http_batch_channel_policy, native_batch_request, project_native, routes,
    upload_transcribe,
};
#[cfg(test)]
use openai_http::{openai_transcribe, parse_openai_multipart, routes as openai_routes};
#[cfg(test)]
#[path = "../../api-native/src/http.rs"]
mod native_http;
#[cfg(test)]
#[path = "../../api-openai/src/http.rs"]
mod openai_http;

use gigaam_audio::{
    ChannelAudio, ChannelAudioView, DecodedAudio, EncodedAudio, load_bytes, resample_audio,
};
use gigaam_recognition::{ExecutionControl, ExecutionState, SpeechProbabilityDetector};
use gigaam_transcription::{
    BatchChannelPolicy, BatchSetup, MultiChannelBatchError, MultiChannelBatchOptions,
    MultiChannelBatchResult, MultiChannelBatchSetup, MultiChannelBatchTranscriber, TurnGap,
    VadPolicyConfig,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(test)]
pub(crate) async fn transcribe(
    State(state): State<HttpState>,
    headers: HeaderMap,
    RawQuery(query_text): RawQuery,
    body: Bytes,
) -> Response {
    native_http::transcribe_inner(State(state), headers, RawQuery(query_text), body, None).await
}

/// Narrow state of the byte-consuming HTTP protocol adapter.
#[derive(Clone)]
pub struct BatchService {
    context: Arc<ApplicationContext>,
    admission: Arc<AdmissionState>,
    metrics: Arc<Metrics>,
    telemetry: TelemetryProducer,
    request_timeout: Duration,
    upload_timeout: Duration,
    request_body_limit: RequestBodyLimit,
}

/// Cohesive private inputs assembled by the facade for the HTTP adapter state.
pub(crate) struct HttpStateParameters {
    pub(crate) context: Arc<ApplicationContext>,
    pub(crate) admission: Arc<AdmissionState>,
    pub(crate) metrics: Arc<Metrics>,
    pub(crate) telemetry: TelemetryProducer,
    pub(crate) request_timeout: Duration,
    pub(crate) upload_timeout: Duration,
    pub(crate) request_body_limit: RequestBodyLimit,
}

#[cfg(test)]
pub(crate) type HttpState = BatchService;

impl BatchService {
    pub(crate) fn new(parameters: HttpStateParameters) -> Self {
        let HttpStateParameters {
            context,
            admission,
            metrics,
            telemetry,
            request_timeout,
            upload_timeout,
            request_body_limit,
        } = parameters;
        Self {
            context,
            admission,
            metrics,
            telemetry,
            request_timeout,
            upload_timeout,
            request_body_limit,
        }
    }
}

/// A body reader retained until work admission completes.
pub struct UploadReservation(tokio::sync::OwnedSemaphorePermit);

pub enum UploadError<E> {
    Refused(AdmissionRefusal),
    Parse(E),
    TimedOut,
}

pub struct BatchReservation(HttpAdmission);

pub struct BatchOutput {
    pub result: MultiChannelBatchResult,
    pub duration: f32,
    pub source_rate: u32,
    pub rtf: f32,
}

impl BatchService {
    pub fn upload_limit_bytes(&self) -> usize {
        self.request_body_limit.bytes()
    }

    pub fn record_request(&self) {
        self.metrics.transcribe();
    }

    pub fn record_error(&self) {
        self.metrics.transcribe_error();
    }

    pub fn record_timeout(&self) {
        self.metrics.timeout();
    }

    pub fn record_refusal(&self, refusal: AdmissionRefusal) {
        match refusal {
            AdmissionRefusal::Overloaded => self.metrics.overload(),
            AdmissionRefusal::Draining => self.metrics.transcribe_error(),
        }
    }

    pub fn record_access(
        &self,
        id: &str,
        status: u16,
        started: Instant,
        bytes: usize,
        model: &str,
    ) {
        self.metrics
            .observe_latency(started.elapsed().as_secs_f64());
        offer_batch_access(
            &self.telemetry,
            crate::log::TranscribeAccess {
                request_id: id.to_owned(),
                status,
                milliseconds: started.elapsed().as_millis(),
                bytes,
                model: model.to_owned(),
            },
        );
    }

    pub fn variant(&self) -> gigaam_model_package::VariantId {
        self.context.capabilities.pack.variant()
    }

    pub async fn read_upload<T, E, F>(
        &self,
        parse: impl FnOnce() -> F,
    ) -> Result<(T, UploadReservation), UploadError<E>>
    where
        F: std::future::Future<Output = Result<T, E>>,
    {
        let permit = self
            .admission
            .admit_upload()
            .map_err(UploadError::Refused)?;
        let value = tokio::time::timeout(self.upload_timeout, parse())
            .await
            .map_err(|_| UploadError::TimedOut)?
            .map_err(UploadError::Parse)?;
        Ok((value, UploadReservation(permit)))
    }

    pub fn admit(
        &self,
        reader: &mut Option<UploadReservation>,
    ) -> Result<BatchReservation, AdmissionRefusal> {
        let admitted = self.admission.admit_http()?;
        if let Some(reader) = reader.take() {
            drop(reader.0);
        }
        Ok(BatchReservation(admitted))
    }

    pub async fn execute<T: Send + 'static>(
        &self,
        reservation: BatchReservation,
        request_id: String,
        started: Instant,
        request: BatchRequest,
        project: impl FnOnce(BatchOutput) -> Result<T, TranscribeError> + Send + 'static,
    ) -> AdmittedWorkResult<T> {
        let progress = RequestProgress {
            telemetry: self.telemetry.clone(),
            request_id,
            started,
        };
        progress.observe(BatchProgress::Admitted);
        let control = ExecutionControl::for_request();
        let execution = AdmittedExecution {
            owner: RequestWorkOwner::new(reservation.0, control.clone()),
            control,
            timeout: self.request_timeout,
            progress: progress.clone(),
        };
        let context = Arc::clone(&self.context);
        let metrics = Arc::clone(&self.metrics);
        execute_admitted_work(execution, move |control| {
            let capabilities = &context.capabilities;
            let (result, duration, source_rate, rtf) = run_batch(
                BatchExecution {
                    context: &context,
                    metrics: &metrics,
                    request,
                    control,
                    progress: &progress,
                },
                || capabilities.vad_provider.create(capabilities.vad_context()),
            )?;
            progress.stage(BatchStage::ResponseProjection, || {
                project(BatchOutput {
                    result,
                    duration,
                    source_rate,
                    rtf,
                })
            })
        })
        .await
    }
}

#[cfg(test)]
/// Opaque route group that guarantees the route-local request-body boundary.
pub(crate) struct HttpRoutes(Router);

#[cfg(test)]
impl HttpRoutes {
    pub(crate) fn into_router(self) -> Router {
        self.0
    }
}

#[cfg(test)]
fn body_limited_routes(routes: Router, limit: usize) -> HttpRoutes {
    HttpRoutes(routes.layer(DefaultBodyLimit::max(limit)))
}

#[cfg(test)]
pub(crate) fn body_limited_test_routes(routes: Router, limit: RequestBodyLimit) -> HttpRoutes {
    body_limited_routes(routes, limit.bytes())
}

pub enum TranscribeError {
    BadRequest(String),
    UnsupportedCapability(String),
    Unavailable(String),
}

impl TranscribeError {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) | Self::UnsupportedCapability(_) => StatusCode::BAD_REQUEST,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    pub fn message(self) -> String {
        match self {
            Self::BadRequest(message)
            | Self::UnsupportedCapability(message)
            | Self::Unavailable(message) => message,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchModel {
    Ctc,
    Rnnt,
}

impl BatchModel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ctc => "ctc",
            Self::Rnnt => "rnnt",
        }
    }
}

pub enum AdmittedWorkResult<T> {
    Completed(Result<T, TranscribeError>),
    WorkerFailure,
    TimedOut,
}

/// Request-local observations use the existing bounded producer; they own no execution state.
#[derive(Clone)]
struct RequestProgress {
    telemetry: TelemetryProducer,
    request_id: String,
    started: Instant,
}

impl RequestProgress {
    fn observe(&self, progress: BatchProgress) {
        self.telemetry.offer_access(crate::log::render(
            crate::log::AccessEvent::TranscribeProgress(TranscribeProgress {
                request_id: self.request_id.clone(),
                elapsed_milliseconds: self.started.elapsed().as_millis(),
                progress,
            }),
        ));
    }

    fn stage<T>(
        &self,
        stage: BatchStage,
        operation: impl FnOnce() -> Result<T, TranscribeError>,
    ) -> Result<T, TranscribeError> {
        let started = Instant::now();
        self.observe(BatchProgress::StageStarted(stage));
        let mut observation = StageObservation {
            progress: self,
            stage,
            started,
            outcome: BatchStageOutcome::Interrupted,
        };
        let result = operation();
        observation.outcome = if result.is_ok() {
            BatchStageOutcome::Completed
        } else {
            BatchStageOutcome::Failed
        };
        result
    }
}

/// An unwind finishes only the current observed stage; it does not change request ownership.
struct StageObservation<'a> {
    progress: &'a RequestProgress,
    stage: BatchStage,
    started: Instant,
    outcome: BatchStageOutcome,
}

impl Drop for StageObservation<'_> {
    fn drop(&mut self) {
        self.progress.observe(BatchProgress::StageFinished {
            stage: self.stage,
            outcome: self.outcome,
            milliseconds: self.started.elapsed().as_millis(),
        });
    }
}

/// Drops outside the request owner's scope, after completion, failure, or unwind acknowledgement.
struct WorkObservation {
    progress: RequestProgress,
    control: ExecutionControl,
}

impl Drop for WorkObservation {
    fn drop(&mut self) {
        self.progress
            .observe(BatchProgress::WorkFinished(self.control.state()));
    }
}

struct AdmittedExecution {
    owner: RequestWorkOwner,
    control: ExecutionControl,
    timeout: Duration,
    progress: RequestProgress,
}

/// Cancels detached native work when its requesting future disappears.
struct RequestCancellationGuard(Option<ExecutionControl>);

impl Drop for RequestCancellationGuard {
    fn drop(&mut self) {
        if let Some(control) = self.0.take() {
            control.request_cancellation();
        }
    }
}

/// The production execution seam retains the admission owner inside blocking work until it
/// terminalizes, even after the HTTP deadline has returned a timeout response.
async fn execute_admitted_work<T, F>(execution: AdmittedExecution, work: F) -> AdmittedWorkResult<T>
where
    T: Send + 'static,
    F: FnOnce(&ExecutionControl) -> Result<T, TranscribeError> + Send + 'static,
{
    let AdmittedExecution {
        owner,
        control,
        timeout,
        progress,
    } = execution;
    let mut cancellation_guard = RequestCancellationGuard(Some(control.clone()));
    let work_control = control.clone();
    let work_progress = progress.clone();
    let job = tokio::task::spawn_blocking(move || {
        let _observation = WorkObservation {
            progress: work_progress.clone(),
            control: work_control.clone(),
        };
        {
            // This inner owner scope ends before the terminal observation, including on unwind.
            let owner = owner;
            work_progress.observe(BatchProgress::WorkStarted);
            match work(&work_control) {
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
            }
        }
    });
    let result = match tokio::time::timeout(timeout, job).await {
        Ok(Ok(result)) => AdmittedWorkResult::Completed(result),
        Ok(Err(_)) => AdmittedWorkResult::WorkerFailure,
        Err(_) => {
            let state = control.request_cancellation();
            progress.observe(BatchProgress::DeadlineElapsed(state));
            AdmittedWorkResult::TimedOut
        }
    };
    cancellation_guard.0 = None;
    result
}

fn offer_batch_access(telemetry: &TelemetryProducer, access: crate::log::TranscribeAccess) {
    telemetry.offer_access(crate::log::render(crate::log::AccessEvent::Transcribe(
        access,
    )));
}

/// Synchronous request path: decode -> channels -> scheduled recognition windows -> JSON.
#[cfg(test)]
struct BatchWork<'a> {
    context: &'a ApplicationContext,
    metrics: &'a Metrics,
    query: &'a TranscribeQuery,
    bytes: Vec<u8>,
    control: &'a ExecutionControl,
    progress: &'a RequestProgress,
}

/// The factory is request-local and invoked only inside admitted blocking work when enabled.
/// Tests replace native construction without adding a process-owned factory or session pool.
#[cfg(test)]
fn do_transcribe_with_vad_factory<D: SpeechProbabilityDetector>(
    work: BatchWork<'_>,
    factory: impl FnOnce() -> Result<D, String>,
) -> Result<String, TranscribeError> {
    let query = work.query;
    let work_variant = work.context.capabilities.pack.variant().as_str();
    let progress = work.progress;
    let request = native_batch_request(query, work.bytes)?;
    let (result, duration, source_rate, rtf) = run_batch(
        BatchExecution {
            context: work.context,
            metrics: work.metrics,
            request,
            control: work.control,
            progress,
        },
        factory,
    )?;
    progress.stage(BatchStage::ResponseProjection, || {
        Ok(project_native(
            query,
            work_variant,
            BatchOutput {
                result,
                duration,
                source_rate,
                rtf,
            },
        ))
    })
}

// Transport-independent input to the shared decode and transcription path.
pub struct BatchRequest {
    pub model: BatchModel,
    pub extension: Option<String>,
    pub audio: Vec<u8>,
    pub channel_policy: BatchChannelPolicy,
    pub turn_gap: TurnGap,
    pub vad: Option<VadPolicyConfig>,
    pub scored: bool,
    pub downmix: bool,
}

struct BatchExecution<'a> {
    context: &'a ApplicationContext,
    metrics: &'a Metrics,
    request: BatchRequest,
    control: &'a ExecutionControl,
    progress: &'a RequestProgress,
}

fn run_batch<D: SpeechProbabilityDetector>(
    work: BatchExecution<'_>,
    factory: impl FnOnce() -> Result<D, String>,
) -> Result<(MultiChannelBatchResult, f32, u32, f32), TranscribeError> {
    let BatchExecution {
        context,
        metrics,
        request:
            BatchRequest {
                model,
                extension,
                audio,
                channel_policy,
                turn_gap,
                vad,
                scored,
                downmix,
            },
        control,
        progress,
    } = work;
    let selected = match context.capabilities.pack.variant().family() {
        gigaam_model_package::ModelFamily::Ctc => BatchModel::Ctc,
        gigaam_model_package::ModelFamily::Rnnt => BatchModel::Rnnt,
    };
    if model != selected {
        return Err(TranscribeError::UnsupportedCapability(
            "requested model is not loaded".into(),
        ));
    }
    let scheduler = &context.capabilities.scheduler;
    if vad.is_some() || scored {
        if context
            .capabilities
            .frontend
            .sample_rate()
            .as_usize()
            .map_err(TranscribeError::UnsupportedCapability)?
            != gigaam_recognition::vad::VAD_SR
        {
            return Err(TranscribeError::UnsupportedCapability(
                "batch VAD requires a 16000 Hz model sample rate".into(),
            ));
        }
        context
            .capabilities
            .vad_provider
            .preflight(context.capabilities.vad_context())
            .map_err(TranscribeError::UnsupportedCapability)?;
    }
    let loaded = progress.stage(BatchStage::AudioDecode, || {
        let encoded = EncodedAudio::new(audio, extension).map_err(TranscribeError::BadRequest)?;
        load_bytes(encoded).map_err(|error| TranscribeError::BadRequest(error.to_string()))
    })?;
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
    let audio = progress.stage(BatchStage::Resample, || {
        resample_audio(loaded, context.capabilities.frontend.sample_rate())
            .map_err(TranscribeError::BadRequest)
    })?;
    let audio = if downmix { mono_downmix(audio)? } else { audio };
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
    let result = match (vad, scored) {
        (policy, true) => {
            let mut detector = request_vad(factory, control, progress)?;
            let result = progress.stage(BatchStage::BatchTranscription, || {
                complete_multi_channel_batch(metrics, || {
                    transcriber.transcribe_scored(&audio, &mut detector, policy)
                })
            })?;
            metrics.observe_vad(result.timings().vad_seconds());
            result
        }
        (None, false) => progress.stage(BatchStage::BatchTranscription, || {
            complete_multi_channel_batch(metrics, || transcriber.transcribe(&audio))
        })?,
        (Some(policy), false) => {
            let mut detector = request_vad(factory, control, progress)?;
            let result = progress.stage(BatchStage::BatchTranscription, || {
                complete_multi_channel_batch(metrics, || {
                    transcriber.transcribe_with_vad(&audio, &mut detector, policy)
                })
            })?;
            metrics.observe_vad(result.timings().vad_seconds());
            result
        }
    };
    let duration = audio.duration_seconds();
    let rtf = started.elapsed().as_secs_f32() / duration.max(1e-6);
    Ok((result, duration, source_rate, rtf))
}

fn mono_downmix(audio: DecodedAudio) -> Result<DecodedAudio, TranscribeError> {
    if audio.channels().len() == 1 {
        return Ok(audio);
    }
    let count = u16::try_from(audio.channels().len())
        .map_err(|_| TranscribeError::BadRequest("too many source channels".into()))?;
    let divisor = f32::from(count);
    let mut samples = vec![0.0; audio.frames()];
    for channel in audio.channels() {
        for (mixed, sample) in samples.iter_mut().zip(channel.samples()) {
            *mixed += *sample / divisor;
        }
    }
    let mono = ChannelAudio::new(samples).map_err(TranscribeError::BadRequest)?;
    DecodedAudio::new(audio.sample_rate(), vec![mono]).map_err(TranscribeError::BadRequest)
}

fn vad_checkpoint(control: &ExecutionControl) -> Result<(), TranscribeError> {
    match control.state() {
        ExecutionState::Ready | ExecutionState::Queued | ExecutionState::Running => Ok(()),
        ExecutionState::CancelRequested | ExecutionState::Cancelled => {
            Err(TranscribeError::Unavailable("request cancelled".into()))
        }
        ExecutionState::Completed | ExecutionState::Failed => Err(TranscribeError::Unavailable(
            "request is already terminal".into(),
        )),
    }
}

struct ObservedVad<'a, D> {
    detector: D,
    control: &'a ExecutionControl,
    progress: &'a RequestProgress,
}

fn request_vad<'a, D: SpeechProbabilityDetector>(
    factory: impl FnOnce() -> Result<D, String>,
    control: &'a ExecutionControl,
    progress: &'a RequestProgress,
) -> Result<ObservedVad<'a, D>, TranscribeError> {
    vad_checkpoint(control)?;
    let detector = progress.stage(BatchStage::VadInitialization, || {
        let detector = factory().map_err(TranscribeError::Unavailable)?;
        vad_checkpoint(control)?;
        Ok(detector)
    })?;
    Ok(ObservedVad {
        detector,
        control,
        progress,
    })
}

impl<D: SpeechProbabilityDetector> SpeechProbabilityDetector for ObservedVad<'_, D> {
    fn probabilities(&mut self, samples: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
        vad_checkpoint(self.control).map_err(TranscribeError::message)?;
        self.progress
            .stage(BatchStage::VadDetection, || {
                let probabilities = self
                    .detector
                    .probabilities(samples)
                    .map_err(TranscribeError::Unavailable)?;
                vad_checkpoint(self.control)?;
                Ok(probabilities)
            })
            .map_err(TranscribeError::message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_variant_binds_native_model_and_rejects_other_variants() {
        use gigaam_model_package::VariantId;
        for (variant, family) in [
            (VariantId::V3Ctc, BatchModel::Ctc),
            (VariantId::V3E2eCtc, BatchModel::Ctc),
            (VariantId::V3Rnnt, BatchModel::Rnnt),
            (VariantId::V3E2eRnnt, BatchModel::Rnnt),
        ] {
            assert_eq!(
                TranscribeQuery::parse_for_variant(None, variant)
                    .expect("omitted selects loaded")
                    .model,
                family
            );
            assert_eq!(
                TranscribeQuery::parse_for_variant(
                    Some(&format!("model={}", variant.as_str())),
                    variant
                )
                .expect("matching variant")
                .model,
                family
            );
            let other = if variant == VariantId::V3Ctc {
                VariantId::V3E2eCtc
            } else {
                VariantId::V3Ctc
            };
            assert!(
                TranscribeQuery::parse_for_variant(
                    Some(&format!("model={}", other.as_str())),
                    variant
                )
                .is_err()
            );
        }
    }

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

    struct SilentVad;

    impl SpeechProbabilityDetector for SilentVad {
        fn probabilities(&mut self, samples: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
            Ok(vec![0.0; samples.samples().len().div_ceil(512)])
        }
    }

    struct InjectedProvider;
    impl gigaam_recognition::VadProvider for InjectedProvider {
        fn id(&self) -> &'static str {
            "injected"
        }
        fn preflight(&self, context: gigaam_recognition::VadContext<'_>) -> Result<(), String> {
            match context.package {
                gigaam_recognition::VadPackage::Variant(pack) => {
                    pack.vad_model()
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "missing test asset".to_owned())?;
                    Ok(())
                }
                _ => Err("unexpected test package".into()),
            }
        }
        fn create(
            &self,
            _: gigaam_recognition::VadContext<'_>,
        ) -> Result<Box<dyn SpeechProbabilityDetector + Send>, String> {
            Ok(Box::new(SilentVad))
        }
    }

    #[test]
    fn vad_query_is_versioned_strict_and_bound_to_the_shared_policy() {
        let legacy = TranscribeQuery::parse(None).expect("the legacy default query is valid");
        assert_eq!(legacy.response_version, ResponseVersion::V1);
        assert!(legacy.vad.is_none());
        for raw in ["response_version=2", "response_version=2&vad=false"] {
            let query =
                TranscribeQuery::parse(Some(raw)).expect("explicit v2 without VAD is valid");
            assert_eq!(query.response_version, ResponseVersion::V2);
            assert!(query.vad.is_none());
        }
        let defaults = TranscribeQuery::parse(Some("response_version=2&vad=true"))
            .expect("enabled VAD uses the shared defaults");
        assert_eq!(defaults.vad, Some(VadPolicyConfig::default()));
        let custom = TranscribeQuery::parse(Some("response_version=2&vad=true&vad_threshold=0.1234567&vad_neg_threshold=0&vad_min_speech_ms=0&vad_min_silence_ms=0.03125&vad_speech_pad_ms=0.0625"))
            .expect("finite zero and fractional policy values are valid");
        let policy = custom.vad.expect("the query enabled VAD");
        assert_eq!(policy.thresholds().speech().get(), 0.1234567);
        assert_eq!(policy.durations().minimum_speech_milliseconds(), 0.0);
        assert_eq!(policy.durations().minimum_silence_milliseconds(), 0.03125);
        assert_eq!(policy.padding().milliseconds(), 0.0625);
        for raw in [
            "response_version=0",
            "response_version=02",
            "response_version=3",
            "response_version=2&response_version=2",
            "vad=false",
            "vad=true",
            "vad_threshold=0.5",
            "response_version=1&vad=false",
            "response_version=2&vad=1",
            "response_version=2&vad=True",
            "response_version=2&vad=true&vad=true",
            "response_version=2&vad_threshold=0.5",
            "response_version=2&vad=false&vad_speech_pad_ms=0",
            "response_version=2&vad=true&vad_threshold=NaN",
            "response_version=2&vad=true&vad_threshold=1.1",
            "response_version=2&vad=true&vad_neg_threshold=-0.1",
            "response_version=2&vad=true&vad_neg_threshold=0.6",
            "response_version=2&vad=true&vad_min_speech_ms=-1",
            "response_version=2&vad=true&vad_min_silence_ms=inf",
            "response_version=2&vad=true&vad_speech_pad_ms=3e38",
        ] {
            assert!(
                TranscribeQuery::parse(Some(raw)).is_err(),
                "must refuse {raw}"
            );
        }
    }

    #[test]
    fn injected_batch_vad_uses_bound_policy_resampling_and_no_off_asset() -> Result<(), String> {
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(64, 1)?;
        let admission = Arc::new(AdmissionState::new(&ServiceAdmission::new(
            1,
            1,
            Duration::from_secs(5),
        )?)?);
        let state = state_with_telemetry(
            &fixture,
            admission,
            telemetry.metrics_arc(),
            telemetry.producer(),
        )?;
        let progress = RequestProgress {
            telemetry: telemetry.producer(),
            request_id: "injected-vad".into(),
            started: Instant::now(),
        };
        let calls = Cell::new(0);
        let run = |raw: &str| {
            let query = TranscribeQuery::parse(Some(raw))?;
            do_transcribe_with_vad_factory(
                BatchWork {
                    context: &state.context,
                    metrics: &state.metrics,
                    query: &query,
                    bytes: crate::tests::pcm16_wav_silence(8000, 640)?,
                    control: &ExecutionControl::for_request(),
                    progress: &progress,
                },
                || {
                    calls.set(calls.get() + 1);
                    Ok(SilentVad)
                },
            )
            .map_err(TranscribeError::message)
        };
        assert!(!run("ext=wav")?.contains("schema_version"));
        assert!(
            run("ext=wav&response_version=2&vad=false")?
                .contains("\"applied\":{\"vad\":{\"enabled\":false}}")
        );
        let v2 = run("ext=wav&response_version=2&vad=false")?;
        assert!(v2.contains("\"model\":\"ctc\""));
        assert!(v2.contains("\"variant\":\"v3_ctc\""));
        assert_eq!(calls.get(), 0);
        assert!(
            run("ext=wav&response_version=2&vad=true").is_err(),
            "selected missing asset must refuse before factory"
        );
        assert_eq!(calls.get(), 0);
        fixture.write_injected_vad_asset()?;
        let result = run(
            "ext=wav&response_version=2&vad=true&vad_threshold=0.1234567&vad_neg_threshold=0&vad_min_speech_ms=0&vad_speech_pad_ms=0.0625",
        )?;
        assert_eq!(calls.get(), 1);
        assert!(result.contains("\"sample_rate_in\":8000"));
        assert!(result.contains("\"sample_rate\":16000"));
        assert!(result.contains(&format!("\"threshold\":{}", f64::from(0.1234567_f32))));
        assert!(result.contains("\"speech_pad_ms\":0.0625"));
        assert!(result.contains("\"text\":\"\""));
        assert_eq!(histogram_count(&state.metrics, "asr_vad_seconds"), 1);
        let lines = telemetry.drain_lines();
        assert!(
            lines
                .iter()
                .any(|line| line.contains("\"phase\":\"vad_initialization\"")
                    && line.contains("\"state\":\"completed\""))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("\"phase\":\"vad_detection\"")
                    && line.contains("\"state\":\"completed\""))
        );
        Ok(())
    }

    #[test]
    fn unsupported_vad_model_rate_refuses_before_audio_and_factory_but_off_works()
    -> Result<(), String> {
        let fixture = crate::tests::assembly_fixture_at_rate(8000)?;
        let telemetry = test_support::prepare(32, 1)?;
        let admission = Arc::new(AdmissionState::new(&ServiceAdmission::new(
            1,
            1,
            Duration::from_secs(5),
        )?)?);
        let state = state_with_telemetry(
            &fixture,
            admission,
            telemetry.metrics_arc(),
            telemetry.producer(),
        )?;
        let progress = RequestProgress {
            telemetry: telemetry.producer(),
            request_id: "rate-vad".into(),
            started: Instant::now(),
        };
        let query = TranscribeQuery::parse(Some("response_version=2&vad=true"))?;
        let result = do_transcribe_with_vad_factory(
            BatchWork {
                context: &state.context,
                metrics: &state.metrics,
                query: &query,
                bytes: vec![0],
                control: &ExecutionControl::for_request(),
                progress: &progress,
            },
            || -> Result<SilentVad, String> {
                panic!("unsupported model rate must not construct a detector")
            },
        );
        assert!(matches!(
            result,
            Err(TranscribeError::UnsupportedCapability(_))
        ));
        assert!(
            telemetry.drain_lines().is_empty(),
            "pure rate refusal precedes audio decoding"
        );
        let query = TranscribeQuery::parse(Some("ext=wav"))?;
        do_transcribe_with_vad_factory(
            BatchWork {
                context: &state.context,
                metrics: &state.metrics,
                query: &query,
                bytes: crate::tests::pcm16_wav_silence(8000, 640)?,
                control: &ExecutionControl::for_request(),
                progress: &progress,
            },
            || -> Result<SilentVad, String> { panic!("VAD-off must not construct a detector") },
        )
        .map_err(TranscribeError::message)?;
        Ok(())
    }

    struct BlockingVad {
        started: Option<tokio::sync::oneshot::Sender<()>>,
        release: Option<std::sync::mpsc::Receiver<()>>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
        dropped: Arc<std::sync::atomic::AtomicBool>,
    }

    impl SpeechProbabilityDetector for BlockingVad {
        fn probabilities(&mut self, samples: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if let Some(started) = self.started.take() {
                started
                    .send(())
                    .map_err(|_| "detection start observer closed")?;
            }
            if let Some(release) = self.release.take() {
                release.recv().map_err(|_| "detection release closed")?;
            }
            Ok(vec![0.0; samples.samples().len().div_ceil(512)])
        }
    }

    impl Drop for BlockingVad {
        fn drop(&mut self) {
            self.dropped
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn vad_constructor_and_detection_retain_admission_until_cancelled_native_work_drops()
    -> Result<(), String> {
        for block_constructor in [true, false] {
            let fixture = crate::tests::assembly_fixture()?;
            fixture.write_injected_vad_asset()?;
            let telemetry = test_support::prepare(32, 1)?;
            let admission = Arc::new(AdmissionState::new(&ServiceAdmission::new(
                1,
                1,
                Duration::from_secs(5),
            )?)?);
            let state = state_with_telemetry(
                &fixture,
                Arc::clone(&admission),
                telemetry.metrics_arc(),
                telemetry.producer(),
            )?;
            let control = ExecutionControl::for_request();
            let owner = RequestWorkOwner::new(
                admission
                    .admit_http()
                    .expect("fresh admission owns one permit"),
                control.clone(),
            );
            let progress = RequestProgress {
                telemetry: telemetry.producer(),
                request_id: "blocked-vad".into(),
                started: Instant::now(),
            };
            let execution = AdmittedExecution {
                owner,
                control: control.clone(),
                timeout: Duration::from_millis(100),
                progress: progress.clone(),
            };
            let (started_tx, started_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let work_calls = Arc::clone(&calls);
            let work_dropped = Arc::clone(&dropped);
            let metrics = Arc::clone(&state.metrics);
            let query = TranscribeQuery::parse(Some("ext=wav&response_version=2&vad=true"))?;
            let audio = crate::tests::pcm16_wav_silence(16000, 640)?;
            let task = tokio::spawn(execute_admitted_work(execution, move |control| {
                do_transcribe_with_vad_factory(
                    BatchWork {
                        context: &state.context,
                        metrics: &state.metrics,
                        query: &query,
                        bytes: audio,
                        control,
                        progress: &progress,
                    },
                    move || {
                        if block_constructor {
                            started_tx
                                .send(())
                                .map_err(|_| "construction start observer closed")?;
                            release_rx
                                .recv()
                                .map_err(|_| "construction release closed")?;
                            Ok(BlockingVad {
                                started: None,
                                release: None,
                                calls: work_calls,
                                dropped: work_dropped,
                            })
                        } else {
                            Ok(BlockingVad {
                                started: Some(started_tx),
                                release: Some(release_rx),
                                calls: work_calls,
                                dropped: work_dropped,
                            })
                        }
                    },
                )
            }));
            tokio::time::timeout(Duration::from_secs(5), started_rx)
                .await
                .expect("the admitted VAD operation starts")
                .expect("the VAD start sender remains live");
            assert!(matches!(
                task.await.expect("controlled VAD work does not panic"),
                AdmittedWorkResult::TimedOut
            ));
            assert_eq!(admission.available_http(), 0);
            assert!(matches!(
                admission.admit_http(),
                Err(AdmissionRefusal::Overloaded)
            ));
            assert!(!dropped.load(std::sync::atomic::Ordering::SeqCst));
            release_tx
                .send(())
                .expect("the blocked VAD operation waits for release");
            tokio::time::timeout(Duration::from_secs(5), async {
                while admission.available_http() == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("native completion eventually releases the request permit");
            assert_eq!(control.state(), ExecutionState::Cancelled);
            assert!(
                dropped.load(std::sync::atomic::Ordering::SeqCst),
                "the session drops before permit release"
            );
            assert_eq!(
                calls.load(std::sync::atomic::Ordering::SeqCst),
                usize::from(!block_constructor)
            );
            assert_eq!(histogram_count(&metrics, "asr_frontend_seconds"), 0);
            assert_eq!(histogram_count(&metrics, "asr_vad_seconds"), 0);
            let lines = telemetry.drain_lines();
            let phase = if block_constructor {
                "vad_initialization"
            } else {
                "vad_detection"
            };
            assert!(
                lines
                    .iter()
                    .any(|line| line.contains(&format!("\"phase\":\"{phase}\""))
                        && line.contains("\"state\":\"failed\""))
            );
            assert!(
                lines
                    .iter()
                    .all(|line| !line.contains("response_projection"))
            );
        }
        Ok(())
    }

    #[test]
    fn cancellation_before_vad_construction_prevents_initialization() -> Result<(), String> {
        let telemetry = test_support::prepare(16, 1)?;
        let control = ExecutionControl::for_request();
        control.request_cancellation();
        let progress = RequestProgress {
            telemetry: telemetry.producer(),
            request_id: "cancel-before-vad".into(),
            started: Instant::now(),
        };
        let refused = request_vad(
            || -> Result<SilentVad, String> { panic!("a cancelled request cannot initialize VAD") },
            &control,
            &progress,
        );
        assert!(matches!(refused, Err(TranscribeError::Unavailable(_))));
        assert!(telemetry.drain_lines().is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn v2_errors_begin_only_after_complete_binding_and_keep_request_id() -> Result<(), String>
    {
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(64, 1)?;
        let admission = Arc::new(AdmissionState::new(&ServiceAdmission::new(
            1,
            1,
            Duration::from_secs(5),
        )?)?);
        let state = state_with_telemetry(
            &fixture,
            Arc::clone(&admission),
            telemetry.metrics_arc(),
            telemetry.producer(),
        )?;
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-request-id",
            HeaderValue::from_static("versioned-refusal"),
        );
        for (raw, code) in [
            ("response_version=2&ext=wav", Some("invalid_request")),
            (
                "response_version=2&model=rnnt",
                Some("unsupported_capability"),
            ),
            (
                "response_version=2&model=v3_e2e_ctc",
                Some("unsupported_capability"),
            ),
            (
                "response_version=2&vad=true",
                Some("unsupported_capability"),
            ),
            ("response_version=2&vad=true&vad_threshold=NaN", None),
            ("response_version=3", None),
            ("response_version=2&response_version=2", None),
        ] {
            let response = project(
                transcribe(
                    State(state.clone()),
                    headers.clone(),
                    RawQuery(Some(raw.into())),
                    Bytes::from_static(b"invalid audio"),
                )
                .await,
            )
            .await;
            assert_eq!(response.status, StatusCode::BAD_REQUEST);
            assert_eq!(
                response.request_id,
                Some(HeaderValue::from_static("versioned-refusal"))
            );
            let body =
                std::str::from_utf8(&response.body).expect("all error responses are UTF-8 JSON");
            match code {
                Some(code) => assert!(
                    body.starts_with(&format!(
                        "{{\"schema_version\":2,\"error\":{{\"code\":\"{code}\",\"message\":"
                    )),
                    "{body}"
                ),
                None => assert!(body.starts_with("{\"error\":\""), "{body}"),
            }
        }
        for refusal in [AdmissionRefusal::Overloaded, AdmissionRefusal::Draining] {
            let code = match refusal {
                AdmissionRefusal::Overloaded => "overloaded",
                AdmissionRefusal::Draining => "draining",
            };
            let request_id = RequestId::from_headers(&headers)?;
            let response = project(native_http::native_admission_refusal(
                refusal,
                &state,
                &request_id,
                ResponseVersion::V2,
            ))
            .await;
            assert_eq!(response.status, StatusCode::SERVICE_UNAVAILABLE);
            assert!(
                std::str::from_utf8(&response.body)
                    .expect("refusals are UTF-8 JSON")
                    .contains(&format!("\"code\":\"{code}\""))
            );
        }
        Ok(())
    }

    fn histogram_count(metrics: &Metrics, name: &str) -> u64 {
        let rendered = metrics.render(0, 0, Some(0), None);
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
        fn silence_capability(&self) -> gigaam_recognition::SilenceCapability {
            gigaam_recognition::SilenceCapability::LegacyBlank
        }
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
            native_runtime: None,
            pack: Arc::clone(&fixture.pack),
            frontend: Arc::clone(&fixture.frontend),
            scheduler: Arc::new(ExecutionScheduler::spawn(TerminalDecoder)),
            provider: Device::Cpu,
            intra_threads: None,
        })?;
        let mut registry = gigaam_recognition::VadRegistry::with_silero();
        registry.register(Arc::new(InjectedProvider))?;
        let capabilities = capabilities.with_vad_provider(registry.select("injected")?);
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
            upload_timeout: Duration::from_secs(1),
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

    fn assert_phase(line: &str, phase: &str, state: &str) {
        assert!(line.contains("\"event\":\"transcribe_phase\""), "{line}");
        assert!(line.contains(&format!("\"phase\":\"{phase}\"")), "{line}");
        assert!(line.contains(&format!("\"state\":\"{state}\"")), "{line}");
    }

    #[tokio::test]
    async fn batch_phases_follow_actual_work_and_preserve_the_terminal_access_record()
    -> Result<(), String> {
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(16, 1)?;
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(5))?;
        let admission = Arc::new(AdmissionState::new(&settings)?);
        let mut state = state_with_telemetry(
            &fixture,
            Arc::clone(&admission),
            telemetry.metrics_arc(),
            telemetry.producer(),
        )?;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        Arc::get_mut(&mut state.context)
            .expect("the test has not shared its application context yet")
            .capabilities
            .scheduler = Arc::new(ExecutionScheduler::spawn(ControlledDecoder {
            started: Some(started_tx),
            release: release_rx,
        }));
        state.request_timeout = Duration::from_secs(5);
        let audio = crate::tests::pcm16_wav_silence(fixture.frontend.sample_rate().hertz(), 640)?;
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("phase-history"));
        let response = tokio::spawn(transcribe(
            State(state),
            headers,
            RawQuery(Some("ext=wav".to_owned())),
            Bytes::from(audio),
        ));
        tokio::time::timeout(Duration::from_secs(5), started_rx)
            .await
            .expect("the real batch path must reach the controlled decoder")
            .expect("the decoder must report that native-like work started");
        let mut lines = telemetry.drain_lines();
        assert_eq!(lines.len(), 7);
        assert_phase(&lines[6], "batch_transcription", "started");
        assert!(!response.is_finished());
        assert_eq!(admission.available_http(), 0);

        release_tx
            .send(())
            .expect("the running decoder waits for release");
        let response = project(response.await.expect("the batch handler must finish")).await;
        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(
            response.request_id,
            Some(HeaderValue::from_static("phase-history"))
        );
        let body =
            std::str::from_utf8(&response.body).expect("the public batch response is UTF-8 JSON");
        assert!(body.contains("\"model\":\"ctc\""));
        assert!(!body.contains("transcribe_phase"));
        lines.extend(telemetry.drain_lines());
        let expected = [
            ("admission", "admitted"),
            ("work", "started"),
            ("audio_decode", "started"),
            ("audio_decode", "completed"),
            ("resample", "started"),
            ("resample", "completed"),
            ("batch_transcription", "started"),
            ("batch_transcription", "completed"),
            ("response_projection", "started"),
            ("response_projection", "completed"),
            ("work", "terminal"),
        ];
        assert_eq!(lines.len(), expected.len() + 1);
        for (line, (phase, outcome)) in lines.iter().zip(expected) {
            assert_phase(line, phase, outcome);
            assert!(line.contains("\"req_id\":\"phase-history\""));
            assert!(line.contains("\"elapsed_ms\":"));
            if outcome == "completed" {
                assert!(line.contains("\"phase_ms\":"));
            }
        }
        assert!(lines[10].contains("\"execution_state\":\"completed\""));
        assert!(lines[11].contains("\"event\":\"transcribe\""));
        assert!(lines[11].contains("\"status\":200"));
        assert_eq!(admission.available_http(), 1);
        assert_eq!(telemetry.snapshot().queue_full, 0);
        Ok(())
    }

    #[tokio::test]
    async fn failed_audio_reports_only_executed_phases_and_telemetry_cannot_change_its_response()
    -> Result<(), String> {
        let fixture = crate::tests::assembly_fixture()?;
        let baseline = test_support::prepare(16, 1)?;
        let saturated = test_support::prepare(1, 1)?;
        let disconnected = test_support::prepare(1, 1)?.into_receiver_dropped();
        for number in 0..4 {
            saturated.producer().offer_access(crate::log::render(
                crate::log::AccessEvent::WsError(crate::log::WsErrorAccess {
                    phase: "fill",
                    message: number.to_string(),
                }),
            ));
        }
        let mut responses = Vec::new();
        for producer in [
            baseline.producer(),
            saturated.producer(),
            disconnected.producer(),
        ] {
            let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))?;
            let admission = Arc::new(AdmissionState::new(&settings)?);
            let state = state_with_telemetry(
                &fixture,
                Arc::clone(&admission),
                Arc::new(Metrics::new(1, 1)),
                producer,
            )?;
            let mut headers = HeaderMap::new();
            headers.insert("x-request-id", HeaderValue::from_static("invalid-audio"));
            responses.push(
                project(
                    transcribe(
                        State(state),
                        headers,
                        RawQuery(Some("ext=wav".to_owned())),
                        Bytes::from_static(b"not-a-wave-file"),
                    )
                    .await,
                )
                .await,
            );
            assert_eq!(admission.available_http(), 1);
        }
        assert_eq!(responses[0].status, StatusCode::BAD_REQUEST);
        assert_eq!(responses[0], responses[1]);
        assert_eq!(responses[0], responses[2]);
        let lines = baseline.drain_lines();
        assert_eq!(lines.len(), 6);
        assert_phase(&lines[0], "admission", "admitted");
        assert_phase(&lines[1], "work", "started");
        assert_phase(&lines[2], "audio_decode", "started");
        assert_phase(&lines[3], "audio_decode", "failed");
        assert_phase(&lines[4], "work", "terminal");
        assert!(lines[4].contains("\"execution_state\":\"failed\""));
        assert!(lines[5].contains("\"event\":\"transcribe\""));
        assert!(lines[5].contains("\"status\":400"));
        assert!(lines.iter().all(|line| !line.contains("not-a-wave-file")));
        assert_eq!(saturated.snapshot().queue_full, 6);
        assert_eq!(disconnected.snapshot().sink_closed, 6);
        Ok(())
    }

    #[tokio::test]
    async fn panicking_work_reports_interruption_after_releasing_its_owner() -> Result<(), String> {
        let telemetry = test_support::prepare(16, 1)?;
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(5))?;
        let admission = AdmissionState::new(&settings)?;
        let control = ExecutionControl::for_request();
        let progress = RequestProgress {
            telemetry: telemetry.producer(),
            request_id: "panic-history".to_owned(),
            started: Instant::now(),
        };
        let work_progress = progress.clone();
        let execution = AdmittedExecution {
            owner: RequestWorkOwner::new(
                admission
                    .admit_http()
                    .map_err(|_| "test admission refused")?,
                control.clone(),
            ),
            control: control.clone(),
            timeout: Duration::from_secs(5),
            progress,
        };
        let result = execute_admitted_work(execution, move |_| {
            work_progress.stage(
                BatchStage::BatchTranscription,
                || -> Result<(), TranscribeError> { panic!("controlled work unwind") },
            )
        })
        .await;
        assert!(matches!(result, AdmittedWorkResult::WorkerFailure));
        assert_eq!(control.state(), ExecutionState::Failed);
        assert_eq!(admission.available_http(), 1);
        let lines = telemetry.drain_lines();
        assert_eq!(lines.len(), 4);
        assert_phase(&lines[0], "work", "started");
        assert_phase(&lines[1], "batch_transcription", "started");
        assert_phase(&lines[2], "batch_transcription", "interrupted");
        assert_phase(&lines[3], "work", "terminal");
        assert!(lines[3].contains("\"execution_state\":\"failed\""));
        assert!(
            lines
                .iter()
                .all(|line| !line.contains("controlled work unwind"))
        );
        Ok(())
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
            offer_batch_access(
                producer,
                crate::log::TranscribeAccess {
                    request_id: request_id.text.clone(),
                    status,
                    milliseconds: 7,
                    bytes: 11,
                    model: model.to_owned(),
                },
            );
            histories.push(project(with_request_id(response, &request_id)).await);
        }
        histories
    }

    impl WindowDecoder for ControlledDecoder {
        fn silence_capability(&self) -> gigaam_recognition::SilenceCapability {
            gigaam_recognition::SilenceCapability::LegacyBlank
        }
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
        let telemetry =
            test_support::prepare(16, 1).expect("the bounded test telemetry settings are valid");
        let execution = AdmittedExecution {
            owner,
            control: control.clone(),
            timeout: Duration::from_millis(1),
            progress: RequestProgress {
                telemetry: telemetry.producer(),
                request_id: "deadline-history".to_owned(),
                started: Instant::now(),
            },
        };
        let task = tokio::spawn(execute_admitted_work(execution, move |_| {
            started_tx
                .send(())
                .map_err(|_| TranscribeError::Unavailable("test start observer dropped".into()))?;
            release_rx
                .recv()
                .map_err(|_| TranscribeError::Unavailable("test release channel closed".into()))?;
            Ok::<_, TranscribeError>(())
        }));
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
        let lines = telemetry.drain_lines();
        assert_eq!(lines.len(), 2);
        assert!(lines.iter().any(|line| {
            line.contains("\"phase\":\"deadline\"")
                && line.contains("\"execution_state\":\"cancel_requested\"")
        }));
        assert!(
            lines
                .iter()
                .all(|line| !line.contains("\"state\":\"terminal\""))
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
        let lines = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let lines = telemetry.drain_lines();
                if !lines.is_empty() {
                    break lines;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the actual terminal observation follows owner release");
        assert_eq!(lines.len(), 1);
        assert_phase(&lines[0], "work", "terminal");
        assert!(lines[0].contains("\"execution_state\":\"cancelled\""));
    }

    #[tokio::test]
    async fn openai_processing_timeout_returns_504_and_counts_once() -> Result<(), String> {
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(32, 1)?;
        let admission = Arc::new(AdmissionState::new(&ServiceAdmission::new(
            1,
            1,
            Duration::from_secs(5),
        )?)?);
        let metrics = telemetry.metrics_arc();
        let mut state = state_with_telemetry(
            &fixture,
            Arc::clone(&admission),
            Arc::clone(&metrics),
            telemetry.producer(),
        )?;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let context = Arc::get_mut(&mut state.context).ok_or("test context has an extra owner")?;
        context.capabilities.scheduler = Arc::new(ExecutionScheduler::spawn(ControlledDecoder {
            started: Some(started_tx),
            release: release_rx,
        }));
        state.request_timeout = Duration::from_millis(100);
        let mut multipart = b"--boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ngigaam-v3\r\n--boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"silence.wav\"\r\n\r\n".to_vec();
        multipart.extend(crate::tests::pcm16_wav_silence(8000, 640)?);
        multipart.extend(b"\r\n--boundary--\r\n");
        let request = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", "multipart/form-data; boundary=boundary")
            .body(axum::body::Body::from(multipart))
            .map_err(|e| e.to_string())?;
        let task = tokio::spawn(openai_transcribe(State(state), request));
        started_rx.await.map_err(|e| e.to_string())?;
        let response = task.await.map_err(|e| e.to_string())?;
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(metric_counter(&metrics, "asr_timeout_total"), 1);
        assert_eq!(admission.available_http(), 0);
        release_tx.send(()).map_err(|e| e.to_string())?;
        tokio::time::timeout(Duration::from_secs(5), async {
            while admission.available_http() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    #[tokio::test]
    async fn openai_multipart_runs_through_shared_batch_executor() -> Result<(), String> {
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(32, 1)?;
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(5))?;
        let admission = Arc::new(AdmissionState::new(&settings)?);
        let state = state_with_telemetry(
            &fixture,
            admission,
            telemetry.metrics_arc(),
            telemetry.producer(),
        )?;
        let mut multipart = b"--boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ngigaam-v3\r\n--boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"silence.wav\"\r\nContent-Type: audio/wav\r\n\r\n".to_vec();
        multipart.extend(crate::tests::pcm16_wav_silence(8000, 640)?);
        multipart.extend(b"\r\n--boundary--\r\n");
        let request = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", "multipart/form-data; boundary=boundary")
            .body(axum::body::Body::from(multipart))
            .map_err(|e| e.to_string())?;
        let response = openai_transcribe(State(state), request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .map_err(|e| e.to_string())?;
        assert_eq!(&bytes[..], b"{\"text\":\"\"}");
        Ok(())
    }

    #[tokio::test]
    async fn early_protocol_failures_count_requests_and_errors_once() -> Result<(), String> {
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(32, 1)?;
        let admission = Arc::new(AdmissionState::new(&ServiceAdmission::new(
            1,
            1,
            Duration::from_secs(5),
        )?)?);
        let metrics = telemetry.metrics_arc();
        let state = state_with_telemetry(
            &fixture,
            admission,
            Arc::clone(&metrics),
            telemetry.producer(),
        )?;
        let native = Request::builder()
            .uri("/v1/transcribe?unknown=x")
            .body(axum::body::Body::empty())
            .map_err(|e| e.to_string())?;
        assert_eq!(
            upload_transcribe(State(state.clone()), native)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
        let openai = Request::builder()
            .uri("/v1/audio/transcriptions")
            .body(axum::body::Body::empty())
            .map_err(|e| e.to_string())?;
        assert_eq!(
            openai_transcribe(State(state), openai).await.status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(metric_counter(&metrics, "asr_transcribe_requests_total"), 2);
        assert_eq!(metric_counter(&metrics, "asr_transcribe_errors_total"), 2);
        assert_eq!(
            histogram_count(&metrics, "asr_transcribe_latency_seconds"),
            2
        );
        Ok(())
    }

    #[tokio::test]
    async fn multipart_field_limits_and_full_vad_request() -> Result<(), String> {
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(32, 1)?;
        let admission = Arc::new(AdmissionState::new(&ServiceAdmission::new(
            1,
            1,
            Duration::from_secs(5),
        )?)?);
        let metrics = telemetry.metrics_arc();
        let mut state = state_with_telemetry(
            &fixture,
            admission,
            Arc::clone(&metrics),
            telemetry.producer(),
        )?;
        state.request_body_limit = RequestBodyLimit::new(64 * 1024)?;
        fn request(parts: &[(&str, &str)]) -> Result<Request, String> {
            let mut body = Vec::new();
            for (name, value) in parts {
                body.extend(
                    format!(
                        "--b\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                    )
                    .as_bytes(),
                );
            }
            body.extend(b"--b--\r\n");
            Request::builder()
                .header("content-type", "multipart/form-data; boundary=b")
                .body(axum::body::Body::from(body))
                .map_err(|e| e.to_string())
        }
        async fn error_body(state: &HttpState, request: Request) -> Result<String, String> {
            let error = parse_openai_multipart(state, request)
                .await
                .err()
                .ok_or("expected refusal")?;
            let response = error.response();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let bytes = to_bytes(response.into_body(), 4096)
                .await
                .map_err(|e| e.to_string())?;
            String::from_utf8(bytes.to_vec()).map_err(|e| e.to_string())
        }
        let valid = [
            ("model", "gigaam-v3"),
            ("file", "wave"),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
            ("language", "ru"),
            ("temperature", "0"),
            ("stream", "false"),
            ("prompt", ""),
            ("vad", "true"),
            ("vad_threshold", "0.7"),
            ("vad_neg_threshold", "0.3"),
            ("vad_min_speech_ms", "250"),
            ("vad_min_silence_ms", "100"),
            ("vad_speech_pad_ms", "30"),
        ];
        let input = parse_openai_multipart(&state, request(&valid)?)
            .await
            .map_err(|e| format!("{e:?}"))?;
        assert!(input.vad.is_some());
        assert!(input.word && input.segment);
        let mut sixteen = valid.to_vec();
        sixteen.push(("prompt", ""));
        assert!(
            error_body(&state, request(&sixteen)?)
                .await?
                .contains("prompt must occur once")
        );
        sixteen.push(("prompt", ""));
        assert!(
            error_body(&state, request(&sixteen)?)
                .await?
                .contains("too many multipart fields")
        );
        let exact = "x".repeat(16 * 1024);
        assert!(
            error_body(&state, request(&[("prompt", &exact)])?)
                .await?
                .contains("prompt is not supported")
        );
        let over = "x".repeat(16 * 1024 + 1);
        let response = openai_transcribe(State(state), request(&[("prompt", &over)])?).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 4096)
            .await
            .map_err(|e| e.to_string())?;
        assert!(
            String::from_utf8(body.to_vec())
                .map_err(|e| e.to_string())?
                .contains("multipart text field exceeds 16 KiB")
        );
        assert_eq!(metric_counter(&metrics, "asr_transcribe_requests_total"), 1);
        assert_eq!(metric_counter(&metrics, "asr_transcribe_errors_total"), 1);
        Ok(())
    }

    #[tokio::test]
    async fn multipart_counts_epilogue_and_preserves_file_bytes() -> Result<(), String> {
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(32, 1)?;
        let admission = Arc::new(AdmissionState::new(&ServiceAdmission::new(
            1,
            1,
            Duration::from_secs(5),
        )?)?);
        let mut state = state_with_telemetry(
            &fixture,
            admission,
            telemetry.metrics_arc(),
            telemetry.producer(),
        )?;
        let audio = [0, 1, 127, 128, 255, 13, 10, 42];
        let mut body = b"--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ngigaam-v3\r\n--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n\r\n".to_vec();
        body.extend(audio);
        body.extend(b"\r\n--b--\r\nepilogue");
        let request = |bytes: Vec<u8>| {
            Request::builder()
                .header("content-type", "multipart/form-data; boundary=b")
                .body(axum::body::Body::from(bytes))
                .map_err(|e| e.to_string())
        };
        state.request_body_limit = RequestBodyLimit::new(100 * 1024)?;
        let mut preamble = vec![b'p'; 33 * 1024];
        preamble.extend(b"\r\n");
        preamble.extend(&body);
        let mut long_header = b"--b\r\nX-Long: ".to_vec();
        long_header.extend(vec![b'x'; 65 * 1024]);
        long_header.extend(
            b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n\r\n",
        );
        long_header.extend(audio);
        long_header.extend(b"\r\n--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ngigaam-v3\r\n--b--\r\n");
        let mut padding = b"--b".to_vec();
        padding.extend(vec![b' '; 9 * 1024]);
        padding.extend(
            b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n\r\n",
        );
        padding.extend(audio);
        padding.extend(b"\r\n--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ngigaam-v3\r\n--b--\r\n");
        for candidate in [preamble, long_header, padding] {
            let input = parse_openai_multipart(&state, request(candidate)?)
                .await
                .map_err(|e| format!("{e:?}"))?;
            assert_eq!(input.audio, audio);
        }
        state.request_body_limit = RequestBodyLimit::new(body.len() + 1)?;
        let below = parse_openai_multipart(&state, request(body.clone())?)
            .await
            .map_err(|e| format!("{e:?}"))?;
        assert_eq!(below.audio, audio);
        state.request_body_limit = RequestBodyLimit::new(body.len())?;
        let input = parse_openai_multipart(&state, request(body.clone())?)
            .await
            .map_err(|e| format!("{e:?}"))?;
        assert_eq!(input.audio, audio);
        body.push(b'x');
        let error = parse_openai_multipart(&state, request(body)?)
            .await
            .err()
            .ok_or("expected size refusal")?;
        assert_eq!(error.response().status(), StatusCode::PAYLOAD_TOO_LARGE);
        Ok(())
    }

    #[tokio::test]
    async fn oversized_multipart_text_is_rejected_before_body_finishes() -> Result<(), String> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(32, 1)?;
        let admission = Arc::new(AdmissionState::new(&ServiceAdmission::new(
            1,
            1,
            Duration::from_secs(5),
        )?)?);
        let mut state = state_with_telemetry(
            &fixture,
            admission,
            telemetry.metrics_arc(),
            telemetry.producer(),
        )?;
        state.request_body_limit = RequestBodyLimit::new(128 * 1024)?;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| e.to_string())?;
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        let server = tokio::spawn(async move { axum::serve(listener, openai_routes(state)).await });
        let mut stream = tokio::net::TcpStream::connect(address)
            .await
            .map_err(|e| e.to_string())?;
        let prefix = format!(
            "--b\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n{}",
            "x".repeat(16 * 1024 + 1)
        );
        let headers = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {address}\r\nContent-Type: multipart/form-data; boundary=b\r\nContent-Length: 100000\r\nConnection: close\r\n\r\n"
        );
        stream
            .write_all(headers.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        stream
            .write_all(prefix.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        let mut buffer = vec![0; 4096];
        let count = tokio::time::timeout(Duration::from_secs(3), stream.read(&mut buffer))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let response = std::str::from_utf8(&buffer[..count]).map_err(|e| e.to_string())?;
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
        server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn multipart_body_limits_preserve_413_for_headers_and_file() -> Result<(), String> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(32, 1)?;
        let admission = Arc::new(AdmissionState::new(&ServiceAdmission::new(
            1,
            1,
            Duration::from_secs(5),
        )?)?);
        let mut state = state_with_telemetry(
            &fixture,
            admission,
            telemetry.metrics_arc(),
            telemetry.producer(),
        )?;
        state.request_body_limit = RequestBodyLimit::new(128)?;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| e.to_string())?;
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        let server = tokio::spawn(async move { axum::serve(listener, openai_routes(state)).await });
        async fn send(address: std::net::SocketAddr, body: &[u8]) -> Result<String, String> {
            let mut stream = tokio::net::TcpStream::connect(address)
                .await
                .map_err(|e| e.to_string())?;
            let header = format!(
                "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {address}\r\nContent-Type: multipart/form-data; boundary=b\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream
                .write_all(header.as_bytes())
                .await
                .map_err(|e| e.to_string())?;
            stream.write_all(body).await.map_err(|e| e.to_string())?;
            let mut response = Vec::new();
            stream
                .read_to_end(&mut response)
                .await
                .map_err(|e| e.to_string())?;
            String::from_utf8(response).map_err(|e| e.to_string())
        }
        let header_limit = format!("--b\r\nX-Long: {}\r\n\r\n", "x".repeat(180));
        let file_limit = format!(
            "--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n\r\n{}\r\n--b--\r\n",
            "x".repeat(180)
        );
        for body in [header_limit.as_bytes(), file_limit.as_bytes()] {
            let response = send(address, body).await?;
            assert!(response.starts_with("HTTP/1.1 413"), "{response}");
        }
        let mut chunked = tokio::net::TcpStream::connect(address)
            .await
            .map_err(|e| e.to_string())?;
        chunked.write_all(format!("POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {address}\r\nContent-Type: multipart/form-data; boundary=b\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").as_bytes()).await.map_err(|e| e.to_string())?;
        let prefix = b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n\r\nx\r\n--b--\r\n";
        for bytes in [prefix.as_slice(), &[b'x'; 128][..]] {
            chunked
                .write_all(format!("{:x}\r\n", bytes.len()).as_bytes())
                .await
                .map_err(|e| e.to_string())?;
            chunked.write_all(bytes).await.map_err(|e| e.to_string())?;
            chunked
                .write_all(b"\r\n")
                .await
                .map_err(|e| e.to_string())?;
        }
        chunked
            .write_all(b"0\r\n\r\n")
            .await
            .map_err(|e| e.to_string())?;
        let mut response = Vec::new();
        chunked
            .read_to_end(&mut response)
            .await
            .map_err(|e| e.to_string())?;
        let response = String::from_utf8(response).map_err(|e| e.to_string())?;
        assert!(response.starts_with("HTTP/1.1 413"), "{response}");
        let malformed = send(address, b"--b\r\nmalformed\r\n\r\n").await?;
        assert!(malformed.starts_with("HTTP/1.1 400"), "{malformed}");
        server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn openai_upload_timeout_and_disconnect_release_reader() -> Result<(), String> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(16, 1)?;
        let admission = Arc::new(AdmissionState::new(&ServiceAdmission::new(
            1,
            1,
            Duration::from_secs(5),
        )?)?);
        let mut state = state_with_telemetry(
            &fixture,
            Arc::clone(&admission),
            telemetry.metrics_arc(),
            telemetry.producer(),
        )?;
        state.upload_timeout = Duration::from_millis(100);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| e.to_string())?;
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        let server = tokio::spawn(async move { axum::serve(listener, openai_routes(state)).await });
        let header = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {address}\r\nContent-Type: multipart/form-data; boundary=b\r\nContent-Length: 100\r\nConnection: close\r\n\r\n--b\r\n"
        );
        let mut slow = tokio::net::TcpStream::connect(address)
            .await
            .map_err(|e| e.to_string())?;
        slow.write_all(header.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), slow.read_to_end(&mut response))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        assert!(
            String::from_utf8(response)
                .map_err(|e| e.to_string())?
                .starts_with("HTTP/1.1 408")
        );
        let reader = admission
            .admit_upload()
            .map_err(|_| "reader not released after timeout")?;
        drop(reader);
        server.abort();
        let mut state = state_with_telemetry(
            &fixture,
            Arc::clone(&admission),
            telemetry.metrics_arc(),
            telemetry.producer(),
        )?;
        state.upload_timeout = Duration::from_secs(30);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| e.to_string())?;
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        let server = tokio::spawn(async move { axum::serve(listener, openai_routes(state)).await });
        let header = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {address}\r\nContent-Type: multipart/form-data; boundary=b\r\nContent-Length: 100\r\nConnection: close\r\n\r\n--b\r\n"
        );
        let mut disconnected = tokio::net::TcpStream::connect(address)
            .await
            .map_err(|e| e.to_string())?;
        disconnected
            .write_all(header.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match admission.admit_upload() {
                    Ok(reader) => drop(reader),
                    Err(_) => break,
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|e| e.to_string())?;
        drop(disconnected);
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Ok(reader) = admission.admit_upload() {
                    drop(reader);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|e| e.to_string())?;
        server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn full_upload_capacity_refuses_v2_without_reading_body() -> Result<(), String> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(16, 1)?;
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(5))?;
        let admission = Arc::new(AdmissionState::new(&settings)?);
        let reader = admission.admit_upload().map_err(|_| "reader capacity")?;
        let mut state = state_with_telemetry(
            &fixture,
            Arc::clone(&admission),
            telemetry.metrics_arc(),
            telemetry.producer(),
        )?;
        state.upload_timeout = Duration::from_millis(100);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| e.to_string())?;
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        let server = tokio::spawn(async move { axum::serve(listener, routes(state)).await });
        let mut stream = tokio::net::TcpStream::connect(address)
            .await
            .map_err(|e| e.to_string())?;
        stream.write_all(format!("POST /v1/transcribe?response_version=2 HTTP/1.1\r\nHost: {address}\r\nContent-Length: 100\r\nConnection: close\r\n\r\n").as_bytes()).await.map_err(|e| e.to_string())?;
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut response))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let text = String::from_utf8(response).map_err(|e| e.to_string())?;
        assert!(text.starts_with("HTTP/1.1 503"), "{text}");
        assert!(text.contains("\"code\":\"overloaded\""), "{text}");
        let mut invalid = tokio::net::TcpStream::connect(address)
            .await
            .map_err(|e| e.to_string())?;
        invalid.write_all(format!("POST /v1/transcribe?response_version=2&unknown=x HTTP/1.1\r\nHost: {address}\r\nContent-Length: 100\r\nConnection: close\r\n\r\n").as_bytes()).await.map_err(|e| e.to_string())?;
        let mut invalid_response = Vec::new();
        tokio::time::timeout(
            Duration::from_secs(2),
            invalid.read_to_end(&mut invalid_response),
        )
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
        let invalid_text = String::from_utf8(invalid_response).map_err(|e| e.to_string())?;
        assert!(invalid_text.starts_with("HTTP/1.1 400"), "{invalid_text}");
        assert!(!invalid_text.contains("\"code\""), "{invalid_text}");
        drop(reader);
        let mut slow = tokio::net::TcpStream::connect(address)
            .await
            .map_err(|e| e.to_string())?;
        slow.write_all(format!("POST /v1/transcribe HTTP/1.1\r\nHost: {address}\r\nContent-Length: 100\r\nConnection: close\r\n\r\n").as_bytes()).await.map_err(|e| e.to_string())?;
        let mut timeout_response = Vec::new();
        tokio::time::timeout(
            Duration::from_secs(2),
            slow.read_to_end(&mut timeout_response),
        )
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
        let timeout_text = String::from_utf8(timeout_response).map_err(|e| e.to_string())?;
        assert!(timeout_text.starts_with("HTTP/1.1 408"), "{timeout_text}");
        let mut next = tokio::net::TcpStream::connect(address)
            .await
            .map_err(|e| e.to_string())?;
        next.write_all(format!("POST /v1/transcribe HTTP/1.1\r\nHost: {address}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.map_err(|e| e.to_string())?;
        let mut next_response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), next.read_to_end(&mut next_response))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let next_text = String::from_utf8(next_response).map_err(|e| e.to_string())?;
        assert!(
            next_text.starts_with("HTTP/1.1 400"),
            "reader released after timeout: {next_text}"
        );
        let mut disconnected = tokio::net::TcpStream::connect(address)
            .await
            .map_err(|e| e.to_string())?;
        disconnected.write_all(format!("POST /v1/transcribe HTTP/1.1\r\nHost: {address}\r\nContent-Length: 100\r\n\r\nx").as_bytes()).await.map_err(|e| e.to_string())?;
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match admission.admit_upload() {
                    Ok(reader) => drop(reader),
                    Err(_) => break,
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|e| e.to_string())?;
        drop(disconnected);
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Ok(reader) = admission.admit_upload() {
                    drop(reader);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|e| e.to_string())?;
        server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn dropped_request_cancels_work_but_retains_capacity_until_acknowledged()
    -> Result<(), String> {
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(5))?;
        let admission = Arc::new(AdmissionState::new(&settings)?);
        let control = ExecutionControl::for_request();
        let owner = RequestWorkOwner::new(
            admission.admit_http().map_err(|_| "admit work")?,
            control.clone(),
        );
        let telemetry = test_support::prepare(16, 1)?;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let execution = AdmittedExecution {
            owner,
            control: control.clone(),
            timeout: Duration::from_secs(5),
            progress: RequestProgress {
                telemetry: telemetry.producer(),
                request_id: "dropped-request".into(),
                started: Instant::now(),
            },
        };
        let task = tokio::spawn(execute_admitted_work(execution, move |_| {
            started_tx
                .send(())
                .map_err(|_| TranscribeError::Unavailable("start observer dropped".into()))?;
            release_rx
                .recv()
                .map_err(|_| TranscribeError::Unavailable("release dropped".into()))?;
            Ok::<_, TranscribeError>(())
        }));
        started_rx.await.map_err(|error| error.to_string())?;
        task.abort();
        let _ = task.await;
        assert_eq!(control.state(), ExecutionState::CancelRequested);
        assert_eq!(admission.available_http(), 0);
        release_tx.send(()).map_err(|error| error.to_string())?;
        tokio::time::timeout(Duration::from_secs(5), async {
            while admission.available_http() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|error| error.to_string())?;
        assert_eq!(control.state(), ExecutionState::Cancelled);
        Ok(())
    }

    fn metric_counter(metrics: &Metrics, name: &str) -> u64 {
        let rendered = metrics.render(1, 1, Some(0), None);
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
    async fn valid_post_drain_http_is_exactly_refused_without_starting_work() -> Result<(), String>
    {
        let fixture = crate::tests::assembly_fixture()?;
        let telemetry = test_support::prepare(1, 1)?;
        let metrics = telemetry.metrics_arc();
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))?;
        let admission = Arc::new(AdmissionState::new(&settings)?);
        let state = state_with_telemetry(
            &fixture,
            Arc::clone(&admission),
            Arc::clone(&metrics),
            telemetry.producer(),
        )?;
        admission.begin_draining();
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-request-id",
            HeaderValue::from_static("post-drain-request"),
        );
        let request_id =
            RequestId::from_headers(&headers).expect("a visible test request identifier is valid");
        let work_invocations = Cell::new(0_u8);
        let response = match state.admit(&mut None) {
            Ok(token) => {
                work_invocations.set(
                    work_invocations
                        .get()
                        .checked_add(1)
                        .expect("one controlled work invocation fits u8"),
                );
                drop(token);
                json_ok("{}".to_owned())
            }
            Err(refusal) => native_http::native_admission_refusal(
                refusal,
                &state,
                &request_id,
                ResponseVersion::V1,
            ),
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
        Ok(())
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
                .render(1, 1, Some(0), None)
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
