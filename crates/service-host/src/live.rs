// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Host-owned execution resources for live transcription.

use crate::metrics::Metrics;
use gigaam_audio::{FrontendProcessor, LiveDecoders, SelectedLiveDecoder};
use gigaam_model_package::VariantPackage;
use gigaam_recognition::{ExecutionControl, ExecutionScheduler, ScheduledRecognizer};
use gigaam_transcription::TranscriptionObservation;
use gigaam_transcription::{
    EndpointDetector, EndpointSource, OriginalChannel, SilenceSkipPolicy, StreamChannelFactory,
    StreamConfig, StreamSetup,
};
use std::num::NonZeroUsize;
use std::sync::Arc;

pub(crate) struct ServiceStreamChannelFactory {
    pack: Arc<VariantPackage>,
    frontend: Arc<FrontendProcessor>,
    scheduler: Arc<ExecutionScheduler>,
    intra_threads: Option<NonZeroUsize>,
    native_runtime: Option<gigaam_recognition::NativeRuntime>,
    vad_provider: gigaam_recognition::SelectedVadProvider,
}

impl ServiceStreamChannelFactory {
    pub(crate) fn new(capabilities: &crate::context::ServiceCapabilities) -> Self {
        Self {
            pack: Arc::clone(&capabilities.pack),
            frontend: Arc::clone(&capabilities.frontend),
            scheduler: Arc::clone(&capabilities.scheduler),
            intra_threads: capabilities.intra_threads,
            native_runtime: capabilities.native_runtime.clone(),
            vad_provider: capabilities.vad_provider.clone(),
        }
    }
}

impl StreamChannelFactory for ServiceStreamChannelFactory {
    type Decoder = ScheduledRecognizer;
    type Detector = gigaam_recognition::VadDetector;

    fn create_stream(
        &mut self,
        _channel: OriginalChannel,
        config: StreamConfig,
    ) -> Result<StreamSetup<Self::Decoder, Self::Detector>, String> {
        let control = ExecutionControl::without_deadline();
        let decoder = self.scheduler.tick_channel(control.clone());
        let needs_detector = needs_vad(&config);
        let detector = if needs_detector {
            let vad = self.vad_provider.create(gigaam_recognition::VadContext {
                package: gigaam_recognition::VadPackage::Variant(&self.pack),
                native_runtime: self.native_runtime.as_ref(),
                intra_threads: self.intra_threads,
            })?;
            match config.endpoint_source() {
                EndpointSource::Manual => EndpointDetector::ManualWithVad(vad),
                EndpointSource::Blank => EndpointDetector::BlankWithVad(vad),
                EndpointSource::Vad => EndpointDetector::Vad(vad),
            }
        } else {
            match config.endpoint_source() {
                EndpointSource::Manual => EndpointDetector::Manual,
                EndpointSource::Blank => EndpointDetector::Blank,
                EndpointSource::Vad => return Err("VAD endpoint requires a detector".into()),
            }
        };
        Ok(StreamSetup {
            frontend: Arc::clone(&self.frontend),
            decoder,
            config,
            detector,
            control,
        })
    }
}

pub(crate) fn needs_vad(config: &StreamConfig) -> bool {
    config.endpoint_source() == EndpointSource::Vad
        || config.silence_skip() != SilenceSkipPolicy::Off
}

/// Records every committed domain observation before JSON construction or socket delivery.
pub(crate) fn project_observations(metrics: &Metrics, observations: &[TranscriptionObservation]) {
    for observation in observations {
        match observation {
            TranscriptionObservation::ChannelSelectionCommitted(selection) => {
                let collapsed = selection
                    .source_channels()
                    .get()
                    .checked_sub(selection.active_channels().len())
                    .expect(
                        "a validated channel selection cannot contain more active channels than sources",
                    );
                metrics.dedup_collapsed(collapsed);
            }
            TranscriptionObservation::DialogPatchGenerated => metrics.ws_turn_patch(),
        }
    }
}

/// Typed stream parameters supplied by a protocol adapter after decoding its request.
pub struct StreamInput<F = gigaam_audio::SampleFormat> {
    pub rate: gigaam_audio::SampleRate,
    pub format: F,
    pub channels: usize,
    pub emission: gigaam_transcription::StreamEmissionMode,
    pub config: StreamConfig,
    pub turn_gap: f32,
    pub backchannel_max_sec: f32,
    pub dedup: bool,
}

pub(crate) fn validate_endpoint_capability(
    endpoint: EndpointSource,
    capability: gigaam_recognition::SilenceCapability,
) -> Result<(), String> {
    if endpoint == EndpointSource::Blank
        && capability != gigaam_recognition::SilenceCapability::LegacyBlank
    {
        return Err("selected decoder does not support blank stream endpointing".into());
    }
    Ok(())
}

pub(crate) fn prepare_stream<F>(
    context: &crate::context::ApplicationContext,
    input: StreamInput<F>,
) -> Result<gigaam_transcription::MultiChannelStreamSetup<F>, String> {
    use gigaam_transcription::{
        BackchannelDuration, BackchannelPolicy, CorrelationThreshold, MultiChannelStreamOptions,
        MultiChannelStreamSetup, MultiChannelStreamSetupInput, SelectionWindowSamples,
        SourceChannelCount, StreamingChannelPolicy, TurnGap,
    };
    validate_endpoint_capability(
        input.config.endpoint_source(),
        context.capabilities.silence_capability(),
    )?;
    if needs_vad(&input.config) {
        context
            .capabilities
            .vad_provider
            .preflight(context.capabilities.vad_context())?;
    }
    let source_channels = SourceChannelCount::new(input.channels)?;
    let channel_policy = if input.dedup {
        StreamingChannelPolicy::Deduplicate {
            threshold: CorrelationThreshold::new(f64::from(context.policy.dedup_threshold))?,
            analysis_window: SelectionWindowSamples::new(context.policy.dedup_window_samples)?,
        }
    } else {
        StreamingChannelPolicy::AllChannels
    };
    let backchannel_policy = if input.backchannel_max_sec == 0.0 {
        BackchannelPolicy::Disabled
    } else {
        BackchannelPolicy::MarkShorterThan(BackchannelDuration::new(input.backchannel_max_sec)?)
    };
    MultiChannelStreamSetup::new(MultiChannelStreamSetupInput {
        sample_format: input.format,
        source_sample_rate: input.rate,
        source_channels,
        stream_config: input.config,
        options: MultiChannelStreamOptions::new(
            channel_policy,
            input.emission,
            TurnGap::new(input.turn_gap)?,
            backchannel_policy,
        ),
    })
}

/// Bounded stream observations emitted by a protocol adapter.
#[derive(Clone, Debug)]
pub enum LiveEvent {
    Open {
        request_id: String,
        channels: usize,
        rate: u32,
    },
    Close {
        request_id: String,
        channels: usize,
        frames: u64,
        milliseconds: u128,
    },
    Error {
        phase: &'static str,
        message: String,
    },
}

/// Narrow host projection for stream metrics and nonblocking access telemetry.
#[derive(Clone)]
pub struct LiveTelemetry {
    metrics: Arc<Metrics>,
    producer: crate::telemetry::TelemetryProducer,
}
impl LiveTelemetry {
    pub(crate) fn new(
        metrics: Arc<Metrics>,
        producer: crate::telemetry::TelemetryProducer,
    ) -> Self {
        Self { metrics, producer }
    }
    pub fn request_id() -> String {
        crate::log::new_req_id()
    }
    pub fn open(&self) {
        self.metrics.ws_open();
    }
    pub fn reject(&self) {
        self.metrics.ws_reject();
    }
    pub fn frame(&self) {
        self.metrics.ws_frame();
    }
    pub fn channels(&self, channels: usize) {
        self.metrics.ws_channels(channels);
    }
    pub fn error(&self) {
        self.metrics.ws_error();
    }
    pub fn emit(&self, event: LiveEvent) {
        let event = project_live_event(event);
        self.producer.offer_access(crate::log::render(event));
    }
}

pub(crate) fn project_live_event(event: LiveEvent) -> crate::log::AccessEvent {
    match event {
        LiveEvent::Open {
            request_id,
            channels,
            rate,
        } => crate::log::AccessEvent::WsOpen(crate::log::WsOpenAccess {
            request_id,
            channels,
            rate,
        }),
        LiveEvent::Close {
            request_id,
            channels,
            frames,
            milliseconds,
        } => crate::log::AccessEvent::WsClose(crate::log::WsCloseAccess {
            request_id,
            channels,
            frames,
            milliseconds,
        }),
        LiveEvent::Error { phase, message } => {
            crate::log::AccessEvent::WsError(crate::log::WsErrorAccess { phase, message })
        }
    }
}

/// Host-owned resources and lifecycle for live transcription.
#[derive(Clone)]
pub struct LiveService {
    context: Arc<crate::context::ApplicationContext>,
    admission: Arc<crate::admission::AdmissionState>,
    metrics: Arc<Metrics>,
    telemetry: LiveTelemetry,
    audio_decoders: LiveDecoders,
}

/// Stream defaults configured by the host.
#[derive(Clone, Debug)]
pub struct StreamDefaults {
    pub stream_base: StreamConfig,
    pub dedup_default: bool,
    pub backchannel_max_sec: f32,
}

/// A validated setup, before capacity has been reserved.
pub struct PreparedLiveStream(gigaam_transcription::MultiChannelStreamSetup<SelectedLiveDecoder>);
impl PreparedLiveStream {
    pub fn source_channels(&self) -> gigaam_transcription::SourceChannelCount {
        self.0.source_channels()
    }
    pub fn source_sample_rate(&self) -> gigaam_audio::SampleRate {
        self.0.source_sample_rate()
    }
    pub fn stream_config(&self) -> &StreamConfig {
        self.0.stream_config()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LiveRefusal {
    Overloaded,
    Draining,
}

/// Reserved capacity and the lifecycle subscription captured before upgrade.
pub struct LiveReservation {
    lease: Arc<crate::admission::WsAdmission>,
    phase: tokio::sync::watch::Receiver<crate::admission::AdmissionPhase>,
    service: LiveService,
    setup: PreparedLiveStream,
}

/// Owns the failed initialization's slot until the adapter finishes its terminal exchange.
pub struct LiveStartError {
    message: String,
    _lease: Arc<crate::admission::WsAdmission>,
}

impl LiveStartError {
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// A started stream; each native job holds its own reservation lease.
pub struct LiveSession {
    lease: Arc<crate::admission::WsAdmission>,
    state: Arc<
        std::sync::Mutex<
            Option<
                gigaam_transcription::MultiChannelSession<
                    ScheduledRecognizer,
                    gigaam_recognition::VadDetector,
                >,
            >,
        >,
    >,
    metrics: Arc<Metrics>,
}

impl LiveService {
    pub(crate) fn new(
        context: Arc<crate::context::ApplicationContext>,
        admission: Arc<crate::admission::AdmissionState>,
        metrics: Arc<Metrics>,
        producer: crate::telemetry::TelemetryProducer,
    ) -> Self {
        let telemetry = LiveTelemetry::new(Arc::clone(&metrics), producer);
        Self {
            context,
            admission,
            metrics,
            telemetry,
            audio_decoders: LiveDecoders::default(),
        }
    }
    #[cfg(test)]
    pub(crate) fn test_context(&self) -> Arc<crate::context::ApplicationContext> {
        Arc::clone(&self.context)
    }
    pub fn telemetry(&self) -> LiveTelemetry {
        self.telemetry.clone()
    }
    pub fn defaults(&self) -> StreamDefaults {
        StreamDefaults {
            stream_base: self.context.policy.stream_base.clone(),
            dedup_default: self.context.policy.dedup_default,
            backchannel_max_sec: self.context.policy.backchannel_max_seconds,
        }
    }
    pub fn supports_stream(&self) -> bool {
        self.context.capabilities.pack.variant().family() == gigaam_model_package::ModelFamily::Ctc
    }
    /// Binds the decoder allowlist used by one route.
    pub fn with_audio_decoders(mut self, decoders: LiveDecoders) -> Self {
        self.audio_decoders = decoders;
        self
    }

    pub fn audio_decoder(&self, selector: Option<&str>) -> Result<SelectedLiveDecoder, String> {
        self.audio_decoders.select(selector)
    }

    pub fn validate<F: Into<SelectedLiveDecoder>>(
        &self,
        input: StreamInput<F>,
    ) -> Result<PreparedLiveStream, String> {
        if !self.supports_stream() {
            return Err("stream requires CTC".into());
        }
        Ok(PreparedLiveStream(
            prepare_stream(&self.context, input)?.select_decoder(),
        ))
    }
    pub fn reserve(&self, setup: PreparedLiveStream) -> Result<LiveReservation, LiveRefusal> {
        let permits =
            crate::admission::WsPermitCount::from_source_channels(setup.source_channels())
                .map_err(|_| LiveRefusal::Overloaded)?;
        let lease = self
            .admission
            .admit_ws(permits)
            .map_err(|refusal| match refusal {
                crate::admission::AdmissionRefusal::Overloaded => LiveRefusal::Overloaded,
                crate::admission::AdmissionRefusal::Draining => LiveRefusal::Draining,
            })?;
        Ok(LiveReservation {
            lease: Arc::new(lease),
            phase: self.admission.subscribe(),
            service: self.clone(),
            setup,
        })
    }
}
fn spawn_leased<T: Send + 'static>(
    lease: Arc<crate::admission::WsAdmission>,
    work: impl FnOnce() -> T + Send + 'static,
) -> tokio::task::JoinHandle<T> {
    tokio::task::spawn_blocking(move || {
        let _lease = lease;
        work()
    })
}
impl LiveReservation {
    pub fn is_draining(&self) -> bool {
        *self.phase.borrow() == crate::admission::AdmissionPhase::Draining
    }
    pub async fn wait_for_drain(&mut self) -> Result<(), String> {
        while !self.is_draining() {
            self.phase.changed().await.map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub async fn start(self) -> Result<(LiveSession, LiveDrain), LiveStartError> {
        let Self {
            lease,
            phase,
            service,
            setup,
        } = self;
        let mut factory = ServiceStreamChannelFactory::new(&service.context.capabilities);
        let worker_lease = Arc::clone(&lease);
        let started = spawn_leased(worker_lease, move || {
            gigaam_transcription::MultiChannelSession::new_with_decoder(setup.0, &mut factory)
        })
        .await
        .map_err(|error| format!("stream session initialization task failed: {error}"))
        .and_then(core::convert::identity);
        let session = match started {
            Ok(session) => session,
            Err(message) => {
                return Err(LiveStartError {
                    message,
                    _lease: lease,
                });
            }
        };
        Ok((
            LiveSession {
                lease,
                state: Arc::new(std::sync::Mutex::new(Some(session))),
                metrics: service.metrics,
            },
            LiveDrain { phase },
        ))
    }
}
/// Drain subscription independent of the session operation handle.
pub struct LiveDrain {
    phase: tokio::sync::watch::Receiver<crate::admission::AdmissionPhase>,
}
impl LiveDrain {
    pub fn is_draining(&self) -> bool {
        *self.phase.borrow() == crate::admission::AdmissionPhase::Draining
    }
    pub async fn wait_for_drain(&mut self) -> Result<(), String> {
        while !self.is_draining() {
            self.phase.changed().await.map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
impl LiveSession {
    pub async fn push(
        &self,
        bytes: Vec<u8>,
    ) -> Result<gigaam_transcription::MultiChannelStep, String> {
        let state = Arc::clone(&self.state);
        let lease = Arc::clone(&self.lease);
        let result = spawn_leased(lease, move || {
            let mut state = state
                .lock()
                .map_err(|_| "stream session state is poisoned".to_owned())?;
            let session = state
                .as_mut()
                .ok_or_else(|| "stream session is already terminal".to_owned())?;
            Ok::<_, String>(session.push(&bytes))
        })
        .await
        .map_err(|e| format!("stream processing task failed: {e}"))??;
        self.project_result(result)
    }
    pub async fn finish(&self) -> Result<gigaam_transcription::MultiChannelStep, String> {
        let state = Arc::clone(&self.state);
        let lease = Arc::clone(&self.lease);
        let result = spawn_leased(lease, move || {
            let session = state
                .lock()
                .map_err(|_| "stream session state is poisoned".to_owned())?
                .take()
                .ok_or_else(|| "stream session is already terminal".to_owned())?;
            Ok::<_, String>(session.finish())
        })
        .await
        .map_err(|e| format!("stream finalization task failed: {e}"))??;
        self.project_result(result)
    }
    fn project_result(
        &self,
        result: Result<
            gigaam_transcription::MultiChannelStep,
            gigaam_transcription::MultiChannelFailure,
        >,
    ) -> Result<gigaam_transcription::MultiChannelStep, String> {
        match result {
            Ok(step) => {
                project_observations(&self.metrics, step.observations());
                Ok(step)
            }
            Err(failure) => {
                project_observations(&self.metrics, failure.observations());
                Err(failure.error().to_owned())
            }
        }
    }
}

#[cfg(test)]
mod lease_tests {
    use super::{LiveService, StreamInput, spawn_leased};
    use crate::admission::{AdmissionState, ServiceAdmission, WsPermitCount};
    use gigaam_transcription::SourceChannelCount;
    use std::{sync::Arc, time::Duration};

    #[tokio::test]
    async fn failed_open_keeps_capacity_until_terminal_delivery() -> Result<(), String> {
        use crate::{context::ApplicationContext, telemetry::PreparedTelemetry, tests};
        use gigaam_audio::SampleFormat;
        use gigaam_recognition::{SpeechProbabilityDetector, VadContext, VadProvider, VadRegistry};
        use gigaam_transcription::{EndpointSource, ObservationMode, StreamEmissionMode};

        struct FailingProvider;
        impl VadProvider for FailingProvider {
            fn id(&self) -> &'static str {
                "fixture"
            }

            fn preflight(&self, _: VadContext<'_>) -> Result<(), String> {
                Ok(())
            }

            fn create(
                &self,
                _: VadContext<'_>,
            ) -> Result<Box<dyn SpeechProbabilityDetector + Send>, String> {
                Err("controlled detector initialization failure".into())
            }
        }

        let fixture = tests::assembly_fixture()?;
        let mut providers = VadRegistry::default();
        providers.register(Arc::new(FailingProvider))?;
        let capabilities =
            tests::service_capabilities(&fixture)?.with_vad_provider(providers.select("fixture")?);
        let rate = fixture.frontend.sample_rate();
        let policy = tests::service_policy(rate, ObservationMode::disabled())?;
        let context = Arc::new(ApplicationContext::new(capabilities, policy)?);
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))?;
        let admission = Arc::new(AdmissionState::new(&settings)?);
        let metrics = Arc::new(crate::metrics::Metrics::new(1, 1));
        let producer = PreparedTelemetry::prepare(&settings, Arc::clone(&metrics))?.start()?;
        let service = LiveService::new(context, Arc::clone(&admission), metrics, producer);
        let setup = service.validate(StreamInput {
            rate,
            format: gigaam_format_raw::select(SampleFormat::Pcm16)?,
            channels: 1,
            emission: StreamEmissionMode::Words,
            config: service
                .defaults()
                .stream_base
                .with_endpoint_source(EndpointSource::Vad)?,
            turn_gap: 1.0,
            backchannel_max_sec: 0.0,
            dedup: false,
        })?;
        let reserved = service
            .reserve(setup)
            .map_err(|error| format!("{error:?}"))?;
        let Err(delivery) = reserved.start().await else {
            return Err("controlled provider must fail during construction".into());
        };
        assert!(
            delivery
                .message()
                .contains("controlled detector initialization failure")
        );
        assert_eq!(admission.available_ws(), 0);
        drop(delivery);
        assert_eq!(admission.available_ws(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn cancelled_await_keeps_capacity_until_worker_exits() {
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
            .expect("positive fixture limits and timeout");
        let admission =
            Arc::new(AdmissionState::new(&settings).expect("fixture admission is valid"));
        let permits = WsPermitCount::from_source_channels(
            SourceChannelCount::new(1).expect("mono is a valid source"),
        )
        .expect("mono uses one permit");
        let lease = Arc::new(
            admission
                .admit_ws(permits)
                .expect("the sole fixture slot is free"),
        );
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let task = tokio::spawn(async move {
            spawn_leased(lease, move || {
                let _ = entered_tx.send(());
                release_rx.recv().expect("the test releases the worker");
            })
            .await
            .expect("the fixture worker exits normally");
        });
        entered_rx.await.expect("the fixture worker entered");
        task.abort();
        let _ = task.await;
        assert_eq!(admission.available_ws(), 0);
        release_tx.send(()).expect("the fixture worker is waiting");
        tokio::time::timeout(Duration::from_secs(5), async {
            while admission.available_ws() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the fixture worker releases its permit before the deadline");
        assert!(admission.admit_ws(permits).is_ok());
    }
}
