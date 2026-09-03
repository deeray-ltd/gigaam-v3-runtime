// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! WebSocket `/v1/stream` adaptation, pre-upgrade admission, and terminal ownership.

use crate::admission::{
    AdmissionPhase, AdmissionRefusal, AdmissionState, WsAdmission, WsPermitCount,
};
use crate::context::ApplicationContext;
use crate::metrics::Metrics;
use crate::protocol::{QueryParameters, query_bool, query_finite_f32};
use crate::stream_response::{end_event, error_event, serialize_step};
use crate::telemetry::TelemetryProducer;
use axum::extract::ws::{CloseCode, CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{RawQuery, State};
use axum::http::StatusCode;
use axum::response::Response;
use gigaam_audio::{FrontendProcessor, RatePair, SampleFormat, SampleRate};
use gigaam_model_package::ModelPackage;
use gigaam_recognition::{ExecutionControl, ExecutionScheduler, ScheduledRecognizer, Vad};
use gigaam_transcription::{
    BackchannelDuration, BackchannelPolicy, CorrelationThreshold, EndpointDetector, EndpointSource,
    MultiChannelFailure, MultiChannelSession, MultiChannelStep, MultiChannelStreamOptions,
    MultiChannelStreamSetup, MultiChannelStreamSetupInput, OriginalChannel, SelectionWindowSamples,
    SourceChannelCount, StreamChannelFactory, StreamConfig, StreamEmissionMode, StreamLockPolicy,
    StreamSetup, StreamingChannelPolicy, TranscriptionObservation, TurnGap,
};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

type Session = MultiChannelSession<ScheduledRecognizer, Vad>;

const MAX_CHANNELS: usize = 8;

/// Pins both the per-message and per-frame upper bound as part of this protocol's contract,
/// instead of inheriting the underlying library defaults (which currently allow a 64 MiB
/// message assembled from 16 MiB frames). A message or frame above this bound refuses and ends
/// the session.
const WS_MESSAGE_LIMIT_BYTES: usize = 16 * 1024 * 1024;

/// Narrow state for the streaming protocol adapter.
#[derive(Clone)]
pub(crate) struct WsState {
    context: Arc<ApplicationContext>,
    admission: Arc<AdmissionState>,
    metrics: Arc<Metrics>,
    telemetry: TelemetryProducer,
}

impl WsState {
    pub(crate) fn new(
        context: Arc<ApplicationContext>,
        admission: Arc<AdmissionState>,
        metrics: Arc<Metrics>,
        telemetry: TelemetryProducer,
    ) -> Self {
        Self {
            context,
            admission,
            metrics,
            telemetry,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EmitMode {
    Words,
    Turns,
    Both,
}

impl EmitMode {
    fn stream_emission_mode(self) -> StreamEmissionMode {
        match self {
            Self::Words => StreamEmissionMode::Words,
            Self::Turns => StreamEmissionMode::Dialog,
            Self::Both => StreamEmissionMode::WordsAndDialog,
        }
    }
}

#[derive(Clone, Debug)]
struct StreamQuery {
    rate: SampleRate,
    format: SampleFormat,
    channels: usize,
    emit: EmitMode,
    config: StreamConfig,
    turn_gap: f32,
    backchannel_max_sec: f32,
    dedup: bool,
}

#[derive(Clone, Debug)]
struct StreamDefaults {
    stream_base: StreamConfig,
    dedup_default: bool,
    backchannel_max_sec: f32,
}

impl From<&ApplicationContext> for StreamDefaults {
    fn from(context: &ApplicationContext) -> Self {
        Self {
            stream_base: context.policy.stream_base.clone(),
            dedup_default: context.policy.dedup_default,
            backchannel_max_sec: context.policy.backchannel_max_seconds,
        }
    }
}

impl StreamQuery {
    fn parse(raw: Option<&str>, defaults: StreamDefaults) -> Result<Self, String> {
        let parameters = QueryParameters::parse(raw)?;
        parameters.reject_unknown(&[
            "model",
            "rate",
            "fmt",
            "channels",
            "emit",
            "endpoint",
            "horizon",
            "lock",
            "turn_gap",
            "dedup",
            "backchannel_max_ms",
        ])?;
        match parameters.get("model") {
            None | Some("ctc") => {}
            Some(value) => return Err(format!("stream supports only model=ctc, got {value:?}")),
        }
        let rate_text = parameters
            .get("rate")
            .ok_or_else(|| "rate parameter is required (8000..192000)".to_owned())?;
        let rate = rate_text
            .parse::<u32>()
            .map_err(|_| format!("rate must be an integer in 8000..192000, got {rate_text:?}"))?;
        if !(8000..=192000).contains(&rate) {
            return Err("rate must be in 8000..192000".into());
        }
        let rate = SampleRate::new(rate)?;
        RatePair::new(rate, defaults.stream_base.sample_rate())?;
        let format = match parameters.get("fmt") {
            None => SampleFormat::Pcm16,
            Some(value) => SampleFormat::parse(value)?,
        };
        let channels = match parameters.get("channels") {
            None => 1,
            Some(value) => value
                .parse::<usize>()
                .map_err(|_| format!("channels must be an integer in 1..{MAX_CHANNELS}"))?,
        };
        if !(1..=MAX_CHANNELS).contains(&channels) {
            return Err(format!("channels must be in 1..{MAX_CHANNELS}"));
        }
        let emit = match parameters.get("emit") {
            None | Some("turns") => EmitMode::Turns,
            Some("words") => EmitMode::Words,
            Some("both") => EmitMode::Both,
            Some(value) => return Err(format!("emit must be turns|words|both, got {value:?}")),
        };
        let config = StreamConfig::timing_changes()
            .with_horizon_sec(positive_f32(
                query_finite_f32(&parameters, "horizon", 5.0)?,
                "horizon",
            )?)?
            .apply(defaults.stream_base)?;
        let config = config.with_lock_policy(match query_bool(&parameters, "lock", false)? {
            true => StreamLockPolicy::CommitStable,
            false => StreamLockPolicy::Advisory,
        })?;
        let endpoint = match parameters.get("endpoint") {
            None | Some("blank") => EndpointSource::Blank,
            Some("vad") => EndpointSource::Vad,
            Some(value) => return Err(format!("endpoint must be blank or vad, got {value:?}")),
        };
        let config = config.with_endpoint_source(endpoint)?;
        let turn_gap = positive_f32(query_finite_f32(&parameters, "turn_gap", 0.8)?, "turn_gap")?;
        let backchannel_milliseconds = query_finite_f32(
            &parameters,
            "backchannel_max_ms",
            defaults.backchannel_max_sec * 1000.0,
        )?;
        if backchannel_milliseconds < 0.0 {
            return Err("backchannel_max_ms must be non-negative".into());
        }
        let dedup_requested = query_bool(&parameters, "dedup", defaults.dedup_default)?;
        if parameters.get("dedup").is_some() && channels == 1 {
            return Err("dedup is valid only when channels is greater than one".into());
        }
        let dedup = channels > 1 && dedup_requested;
        Ok(Self {
            rate,
            format,
            channels,
            emit,
            config,
            turn_gap,
            backchannel_max_sec: backchannel_milliseconds / 1000.0,
            dedup,
        })
    }
}

fn positive_f32(value: f32, key: &str) -> Result<f32, String> {
    if value <= 0.0 {
        return Err(format!("{key} must be greater than zero"));
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Control {
    End,
}

/// Inbound frames normalized before the generic stream state machine observes them.
#[derive(Clone, Debug, Eq, PartialEq)]
enum InboundFrame {
    Binary(Vec<u8>),
    Text(String),
    Close,
    Ping,
    Pong,
}

/// Distinguishes a capacity refusal, which the session must acknowledge with WebSocket close
/// code 1009, from every other receive failure, which keeps the existing silent-drop behavior.
#[derive(Clone, Debug, Eq, PartialEq)]
enum ReceiveError {
    MessageTooBig { size: usize, limit: usize },
    Transport(String),
}

/// WebSocket close code for a message or frame exceeding the configured capacity bound
/// (RFC 6455 Section 7.4.1, "Message Too Big").
const MESSAGE_TOO_BIG_CLOSE_CODE: CloseCode = 1009;

/// The distinct close outcomes a session can send: an ordinary end-of-session close, or a
/// capacity refusal the peer must be able to tell apart from every other close.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CloseReason {
    Normal,
    MessageTooBig,
}

impl CloseReason {
    /// The Close frame payload for this reason, or `None` for the ordinary bodiless close.
    fn frame(self) -> Option<CloseFrame> {
        match self {
            CloseReason::Normal => None,
            CloseReason::MessageTooBig => Some(CloseFrame {
                code: MESSAGE_TOO_BIG_CLOSE_CODE,
                reason: "message too big".into(),
            }),
        }
    }
}

/// Session operations which can move into blocking work without carrying a transport boundary.
trait StreamingSession: Send + 'static {
    fn push(&mut self, bytes: &[u8]) -> Result<MultiChannelStep, MultiChannelFailure>;
    fn finish(self) -> Result<MultiChannelStep, MultiChannelFailure>;
}

impl StreamingSession for Session {
    fn push(&mut self, bytes: &[u8]) -> Result<MultiChannelStep, MultiChannelFailure> {
        Self::push(self, bytes)
    }

    fn finish(self) -> Result<MultiChannelStep, MultiChannelFailure> {
        Self::finish(self)
    }
}

/// Effects owned by one transport while the shared stream terminal state machine remains generic.
trait SessionEffects: Send {
    fn receive(
        &mut self,
    ) -> impl std::future::Future<Output = Result<Option<InboundFrame>, ReceiveError>> + Send;
    fn send_text(
        &mut self,
        body: String,
    ) -> impl std::future::Future<Output = Result<(), String>> + Send;
    fn send_close(
        &mut self,
        reason: CloseReason,
    ) -> impl std::future::Future<Output = Result<(), String>> + Send;
    fn emit_access(&mut self, event: crate::log::AccessEvent);
}

/// Real Axum socket effects used by the production callback path.
struct AxumEffects {
    socket: WebSocket,
    telemetry: TelemetryProducer,
}

impl AxumEffects {
    fn new(socket: WebSocket, telemetry: TelemetryProducer) -> Self {
        Self { socket, telemetry }
    }
}

/// Serializes one WebSocket access event into the shared nonblocking telemetry lineage.
fn offer_access(telemetry: &TelemetryProducer, event: crate::log::AccessEvent) {
    telemetry.offer_access(crate::log::render(event));
}

/// Distinguishes a capacity refusal, surfaced by the underlying WebSocket library as a
/// `tungstenite` capacity error, from every other receive failure. The original text of every
/// other error is preserved for diagnostics.
fn classify_receive_error(error: axum::Error) -> ReceiveError {
    let message = error.to_string();
    match error.into_inner().downcast_ref::<tungstenite::Error>() {
        Some(tungstenite::Error::Capacity(tungstenite::error::CapacityError::MessageTooLong {
            size,
            max_size,
        })) => ReceiveError::MessageTooBig {
            size: *size,
            limit: *max_size,
        },
        _ => ReceiveError::Transport(message),
    }
}

impl SessionEffects for AxumEffects {
    async fn receive(&mut self) -> Result<Option<InboundFrame>, ReceiveError> {
        match self.socket.recv().await {
            Some(Ok(Message::Binary(bytes))) => Ok(Some(InboundFrame::Binary(bytes.to_vec()))),
            Some(Ok(Message::Text(text))) => Ok(Some(InboundFrame::Text(text.to_string()))),
            Some(Ok(Message::Close(_))) => Ok(Some(InboundFrame::Close)),
            Some(Ok(Message::Ping(_))) => Ok(Some(InboundFrame::Ping)),
            Some(Ok(Message::Pong(_))) => Ok(Some(InboundFrame::Pong)),
            Some(Err(error)) => Err(classify_receive_error(error)),
            None => Ok(None),
        }
    }

    async fn send_text(&mut self, body: String) -> Result<(), String> {
        self.socket
            .send(Message::Text(body.into()))
            .await
            .map_err(|error| format!("send WebSocket text frame: {error}"))
    }

    async fn send_close(&mut self, reason: CloseReason) -> Result<(), String> {
        self.socket
            .send(Message::Close(reason.frame()))
            .await
            .map_err(|error| format!("send WebSocket close frame: {error}"))
    }

    fn emit_access(&mut self, event: crate::log::AccessEvent) {
        offer_access(&self.telemetry, event);
    }
}

fn parse_control(text: &str) -> Result<Control, String> {
    let mut remaining = text;
    for token in ["{", "\"type\"", ":", "\"end\"", "}"] {
        remaining = remaining
            .trim_start()
            .strip_prefix(token)
            .ok_or_else(|| "control frame must be exactly {\"type\":\"end\"}".to_owned())?;
    }
    if remaining.trim().is_empty() {
        Ok(Control::End)
    } else {
        Err("control frame must be exactly {\"type\":\"end\"}".into())
    }
}

/// Service-owned capability factory; Transcription validates and warms every returned setup.
struct ServiceStreamChannelFactory {
    pack: Arc<ModelPackage>,
    frontend: Arc<FrontendProcessor>,
    scheduler: Arc<ExecutionScheduler>,
    intra_threads: Option<NonZeroUsize>,
}

impl ServiceStreamChannelFactory {
    fn new(
        pack: Arc<ModelPackage>,
        frontend: Arc<FrontendProcessor>,
        scheduler: Arc<ExecutionScheduler>,
        intra_threads: Option<NonZeroUsize>,
    ) -> Self {
        Self {
            pack,
            frontend,
            scheduler,
            intra_threads,
        }
    }
}

impl StreamChannelFactory for ServiceStreamChannelFactory {
    type Decoder = ScheduledRecognizer;
    type Detector = Vad;

    fn create_stream(
        &mut self,
        _channel: OriginalChannel,
        config: StreamConfig,
    ) -> Result<StreamSetup<Self::Decoder, Self::Detector>, String> {
        let control = ExecutionControl::without_deadline();
        let decoder = self.scheduler.tick_channel(control.clone());
        let detector = match config.endpoint_source() {
            EndpointSource::Blank => EndpointDetector::Blank,
            EndpointSource::Vad => {
                EndpointDetector::Vad(Vad::from_pack(&self.pack, self.intra_threads)?)
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

/// Validated setup data that is safe to compute before reserving an upgrade token.
struct PreparedStream {
    setup: MultiChannelStreamSetup,
}

impl PreparedStream {
    fn parse(raw: Option<&str>, context: &ApplicationContext) -> Result<Self, String> {
        let query = StreamQuery::parse(raw, StreamDefaults::from(context))?;
        let setup = multi_channel_setup(&query, context)?;
        Ok(Self { setup })
    }
}

fn multi_channel_setup(
    query: &StreamQuery,
    context: &ApplicationContext,
) -> Result<MultiChannelStreamSetup, String> {
    let source_channels = SourceChannelCount::new(query.channels)?;
    let channel_policy = match query.dedup {
        true => StreamingChannelPolicy::Deduplicate {
            threshold: CorrelationThreshold::new(f64::from(context.policy.dedup_threshold))?,
            analysis_window: SelectionWindowSamples::new(context.policy.dedup_window_samples)?,
        },
        false => StreamingChannelPolicy::AllChannels,
    };
    let backchannel_policy = if query.backchannel_max_sec == 0.0 {
        BackchannelPolicy::Disabled
    } else {
        BackchannelPolicy::MarkShorterThan(BackchannelDuration::new(query.backchannel_max_sec)?)
    };
    MultiChannelStreamSetup::new(MultiChannelStreamSetupInput {
        sample_format: query.format,
        source_sample_rate: query.rate,
        source_channels,
        stream_config: query.config.clone(),
        options: MultiChannelStreamOptions::new(
            channel_policy,
            query.emit.stream_emission_mode(),
            TurnGap::new(query.turn_gap)?,
            backchannel_policy,
        ),
    })
}

/// Records every committed domain observation before JSON construction or socket delivery.
fn project_observations(metrics: &Metrics, observations: &[TranscriptionObservation]) {
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

fn prepare_messages_for_delivery(
    metrics: &Metrics,
    observations: &[TranscriptionObservation],
    serialize: impl FnOnce() -> Vec<String>,
) -> Vec<String> {
    project_observations(metrics, observations);
    serialize()
}

fn prepare_step_for_delivery(
    metrics: &Metrics,
    source_channels: SourceChannelCount,
    step: &MultiChannelStep,
) -> Vec<String> {
    prepare_messages_for_delivery(metrics, step.observations(), || {
        serialize_step(step, source_channels)
    })
}

fn failure_message_after_projection(metrics: &Metrics, failure: MultiChannelFailure) -> String {
    project_observations(metrics, failure.observations());
    failure.error().to_owned()
}

fn locked<T>(session: &Arc<Mutex<Option<T>>>) -> Result<MutexGuard<'_, Option<T>>, String> {
    session
        .lock()
        .map_err(|_| "stream session state is poisoned".to_owned())
}

/// Transfers the sole terminal session owner out of shared callback state exactly once.
fn take_terminal_session<T>(state: &Arc<Mutex<Option<T>>>) -> Result<T, String> {
    let mut session = locked(state)?;
    session
        .take()
        .ok_or_else(|| "stream session is already terminal".to_owned())
}

/// Active terminal facts cannot be absent once a session has emitted its opening access event.
struct ActiveTerminalContext {
    request_id: String,
    opened_at: Instant,
    channels: SourceChannelCount,
    frames: u64,
}

impl ActiveTerminalContext {
    fn open<E: SessionEffects>(
        source_sample_rate: SampleRate,
        source_channels: SourceChannelCount,
        effects: &mut E,
    ) -> Self {
        let request_id = crate::log::new_req_id();
        effects.emit_access(crate::log::AccessEvent::WsOpen(crate::log::WsOpenAccess {
            request_id: request_id.clone(),
            channels: source_channels.get(),
            rate: source_sample_rate.hertz(),
        }));
        Self {
            request_id,
            opened_at: Instant::now(),
            channels: source_channels,
            frames: 0,
        }
    }

    fn record_frame(&mut self) {
        self.frames = self
            .frames
            .checked_add(1)
            .expect("a process-addressable WebSocket frame count cannot overflow u64");
    }

    /// Every active terminal has the complete data required for one close access event.
    fn close_event(self) -> crate::log::AccessEvent {
        crate::log::AccessEvent::WsClose(crate::log::WsCloseAccess {
            request_id: self.request_id,
            channels: self.channels.get(),
            frames: self.frames,
            milliseconds: self.opened_at.elapsed().as_millis(),
        })
    }
}

/// A terminal purpose owns the complete, mutually exclusive outcome of one accepted session.
enum TerminalPurpose {
    ShutdownBeforeOpen,
    ClientEnd(ActiveTerminalContext),
    PeerClosed(ActiveTerminalContext),
    ActiveShutdown(ActiveTerminalContext),
}

/// Peer delivery is a policy at a failure site, never an incidental property of its message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FailureDelivery {
    NotifyPeer,
    RecordOnly,
}

/// One outer owner records a failed session once with its protocol-independent details.
struct SessionFailure {
    phase: &'static str,
    message: String,
}

#[derive(Debug)]
enum PostWarmup<T> {
    Active(T),
    ShutdownBeforeOpen(T),
}

/// Cohesive facts retained by the production callback after construction and warmup succeed.
struct PostWarmupInput<S> {
    result: PostWarmup<S>,
    admission: WsAdmission,
    phase_rx: tokio::sync::watch::Receiver<AdmissionPhase>,
    source_sample_rate: SampleRate,
    source_channels: SourceChannelCount,
}

impl SessionFailure {
    fn new(phase: &'static str, message: impl Into<String>) -> Self {
        Self {
            phase,
            message: message.into(),
        }
    }
}

/// Owns one committed all-channel reservation until an accepted upgrade callback runs or drops.
struct CapturedUpgrade<C> {
    admission: WsAdmission,
    metrics: Arc<Metrics>,
    callback: C,
}

impl<C> CapturedUpgrade<C> {
    /// Callback execution starts the session metric before any callback side effect.
    fn execute<T>(self, execute: impl FnOnce(C, WsAdmission) -> T) -> T {
        self.metrics.ws_open();
        execute(self.callback, self.admission)
    }
}

/// Captures both the all-channel token and its lifecycle receiver before the callback can run.
fn capture_upgrade<C>(
    admission_state: &AdmissionState,
    metrics: Arc<Metrics>,
    permits: WsPermitCount,
    make_callback: impl FnOnce(tokio::sync::watch::Receiver<AdmissionPhase>) -> C,
) -> Result<CapturedUpgrade<C>, AdmissionRefusal> {
    let admission = admission_state.admit_ws(permits)?;
    let phase_rx = admission_state.subscribe();
    Ok(CapturedUpgrade {
        admission,
        metrics,
        callback: make_callback(phase_rx),
    })
}

struct UpgradeCallback {
    state: WsState,
    prepared: PreparedStream,
    phase_rx: tokio::sync::watch::Receiver<AdmissionPhase>,
}

/// Retains the exact captured admission token for every initialized session terminal path.
struct SessionLifetime<S> {
    _admission: WsAdmission,
    state: Arc<Mutex<Option<S>>>,
}

impl<S> SessionLifetime<S> {
    fn new(admission: WsAdmission, session: S) -> Self {
        Self {
            _admission: admission,
            state: Arc::new(Mutex::new(Some(session))),
        }
    }

    fn state(&self) -> Arc<Mutex<Option<S>>> {
        Arc::clone(&self.state)
    }
}

/// Completes one actual post-warmup result while retaining the captured token and receiver.
async fn run_post_warmup<S: StreamingSession, E: SessionEffects>(
    effects: &mut E,
    metrics: &Metrics,
    input: PostWarmupInput<S>,
) {
    let PostWarmupInput {
        result,
        admission,
        phase_rx,
        source_sample_rate,
        source_channels,
    } = input;
    match result {
        PostWarmup::ShutdownBeforeOpen(session) => {
            let lifetime = SessionLifetime::new(admission, session);
            let _ = phase_rx;
            resolve_terminal(
                effects,
                metrics,
                &lifetime,
                source_channels,
                TerminalPurpose::ShutdownBeforeOpen,
            )
            .await;
        }
        PostWarmup::Active(session) => {
            let lifetime = SessionLifetime::new(admission, session);
            let terminal_context =
                ActiveTerminalContext::open(source_sample_rate, source_channels, effects);
            let mut phase_rx = phase_rx;
            run_active(
                effects,
                metrics,
                &lifetime,
                source_channels,
                &mut phase_rx,
                terminal_context,
            )
            .await;
        }
    }
}

impl CapturedUpgrade<UpgradeCallback> {
    /// The sole production route which hands a captured admission token to Axum's upgrade future.
    fn into_response(self, upgrade: WebSocketUpgrade) -> Response {
        let upgrade = upgrade
            .max_message_size(WS_MESSAGE_LIMIT_BYTES)
            .max_frame_size(WS_MESSAGE_LIMIT_BYTES);
        upgrade.on_upgrade(move |socket| {
            self.execute(move |callback, admission| async move {
                handle(
                    socket,
                    callback.state,
                    callback.prepared,
                    callback.phase_rx,
                    admission,
                )
                .await;
            })
        })
    }
}

pub(crate) async fn stream(
    State(state): State<WsState>,
    RawQuery(query_text): RawQuery,
    upgrade: WebSocketUpgrade,
) -> Response {
    let prepared = match PreparedStream::parse(query_text.as_deref(), &state.context) {
        Ok(prepared) => prepared,
        Err(error) => {
            state.metrics.ws_reject();
            return crate::stream_response::pre_upgrade_refusal(StatusCode::BAD_REQUEST, error);
        }
    };
    let permits = match WsPermitCount::from_source_channels(prepared.setup.source_channels()) {
        Ok(permits) => permits,
        Err(error) => {
            state.metrics.ws_reject();
            return crate::stream_response::pre_upgrade_refusal(StatusCode::BAD_REQUEST, error);
        }
    };
    let callback_state = state.clone();
    match capture_upgrade(
        state.admission.as_ref(),
        Arc::clone(&state.metrics),
        permits,
        move |phase_rx| UpgradeCallback {
            state: callback_state,
            prepared,
            phase_rx,
        },
    ) {
        Ok(captured) => captured.into_response(upgrade),
        Err(refusal) => crate::stream_response::pre_upgrade_refusal(
            StatusCode::SERVICE_UNAVAILABLE,
            ws_refusal_message(&state.metrics, refusal),
        ),
    }
}

fn ws_refusal_message(metrics: &Metrics, refusal: AdmissionRefusal) -> &'static str {
    match refusal {
        AdmissionRefusal::Overloaded => {
            // An overload refusal records the established session and rejection history.
            metrics.ws_open();
            metrics.ws_reject();
            "too many active channels, retry later"
        }
        AdmissionRefusal::Draining => "draining",
    }
}

async fn send_text<E: SessionEffects>(effects: &mut E, body: String) -> Result<(), String> {
    effects.send_text(body).await
}

async fn send_close<E: SessionEffects>(effects: &mut E, reason: CloseReason) -> Result<(), String> {
    effects.send_close(reason).await
}

async fn send_error<E: SessionEffects>(effects: &mut E, message: &str) -> Result<(), String> {
    send_text(effects, error_event(message)).await?;
    send_close(effects, CloseReason::Normal).await
}

fn log_ws_error<E: SessionEffects>(effects: &mut E, phase: &'static str, message: &str) {
    effects.emit_access(crate::log::AccessEvent::WsError(
        crate::log::WsErrorAccess {
            phase,
            message: message.to_owned(),
        },
    ));
}

async fn report_session_failure<E: SessionEffects>(
    effects: &mut E,
    metrics: &Metrics,
    failure: SessionFailure,
    delivery: FailureDelivery,
) {
    record_primary_failure(effects, metrics, &failure);
    match delivery {
        FailureDelivery::NotifyPeer => {
            if let Err(error) = send_error(effects, &failure.message).await {
                // A peer-delivery problem is an access diagnostic, not a second failed session.
                log_ws_error(effects, "error_delivery", &error);
            }
        }
        FailureDelivery::RecordOnly => {}
    }
}

fn record_primary_failure<E: SessionEffects>(
    effects: &mut E,
    metrics: &Metrics,
    failure: &SessionFailure,
) {
    metrics.ws_error();
    log_ws_error(effects, failure.phase, &failure.message);
}

/// Resolves construction first, so ordinary failures have priority over a concurrent drain.
fn construct_then_classify<T, E>(
    construct: impl FnOnce() -> Result<T, E>,
    observe_channels: impl FnOnce(),
    phase: impl FnOnce() -> AdmissionPhase,
) -> Result<PostWarmup<T>, E> {
    let session = construct()?;
    observe_channels();
    match phase() {
        AdmissionPhase::Running => Ok(PostWarmup::Active(session)),
        AdmissionPhase::Draining => Ok(PostWarmup::ShutdownBeforeOpen(session)),
    }
}

async fn handle(
    socket: WebSocket,
    state: WsState,
    prepared: PreparedStream,
    phase_rx: tokio::sync::watch::Receiver<AdmissionPhase>,
    admission: WsAdmission,
) {
    let mut effects = AxumEffects::new(socket, state.telemetry.clone());
    let source_sample_rate = prepared.setup.source_sample_rate();
    let source_channels = prepared.setup.source_channels();
    let factory = ServiceStreamChannelFactory::new(
        Arc::clone(&state.context.capabilities.pack),
        Arc::clone(&state.context.capabilities.frontend),
        Arc::clone(&state.context.capabilities.ctc),
        state.context.capabilities.intra_threads,
    );
    let initialization = tokio::task::spawn_blocking(move || {
        let mut factory = factory;
        MultiChannelSession::new(prepared.setup, &mut factory)
    })
    .await;
    let constructed = match initialization {
        Ok(Ok(session)) => Ok(session),
        Ok(Err(error)) => Err(SessionFailure::new("stream", error)),
        Err(error) => Err(SessionFailure::new(
            "stream",
            format!("stream session initialization task failed: {error}"),
        )),
    };
    let post_warmup = match construct_then_classify(
        || constructed,
        || {
            // A successfully warmed session becomes observable before the final lifecycle read.
            state.metrics.ws_channels(source_channels.get());
        },
        || state.admission.phase(),
    ) {
        Ok(post_warmup) => post_warmup,
        Err(failure) => {
            report_session_failure(
                &mut effects,
                &state.metrics,
                failure,
                FailureDelivery::NotifyPeer,
            )
            .await;
            return;
        }
    };
    run_post_warmup(
        &mut effects,
        &state.metrics,
        PostWarmupInput {
            result: post_warmup,
            admission,
            phase_rx,
            source_sample_rate,
            source_channels,
        },
    )
    .await;
}

async fn run_active<S: StreamingSession, E: SessionEffects>(
    effects: &mut E,
    metrics: &Metrics,
    lifetime: &SessionLifetime<S>,
    source_channels: SourceChannelCount,
    phase_rx: &mut tokio::sync::watch::Receiver<AdmissionPhase>,
    mut terminal_context: ActiveTerminalContext,
) {
    let end = loop {
        enum Next {
            Shutdown(Result<(), tokio::sync::watch::error::RecvError>),
            Message(Result<Option<InboundFrame>, ReceiveError>),
        }
        let next = tokio::select! {
            biased;
            changed = phase_rx.changed() => Next::Shutdown(changed),
            message = effects.receive() => Next::Message(message),
        };
        match next {
            Next::Shutdown(Ok(())) => break TerminalPurpose::ActiveShutdown(terminal_context),
            Next::Shutdown(Err(error)) => {
                report_session_failure(
                    effects,
                    metrics,
                    SessionFailure::new("stream", format!("shutdown coordination failed: {error}")),
                    FailureDelivery::NotifyPeer,
                )
                .await;
                return;
            }
            Next::Message(Ok(Some(InboundFrame::Binary(bytes)))) => {
                terminal_context.record_frame();
                metrics.ws_frame();
                let session = lifetime.state();
                let result = tokio::task::spawn_blocking(move || {
                    let mut session = locked(&session)?;
                    let session = session
                        .as_mut()
                        .ok_or_else(|| "stream session is already terminal".to_owned())?;
                    Ok::<_, String>(session.push(&bytes))
                })
                .await;
                let step = match result {
                    Ok(Ok(Ok(step))) => step,
                    Ok(Ok(Err(failure))) => {
                        let error = failure_message_after_projection(metrics, failure);
                        report_session_failure(
                            effects,
                            metrics,
                            SessionFailure::new("stream", error),
                            FailureDelivery::NotifyPeer,
                        )
                        .await;
                        return;
                    }
                    Ok(Err(error)) => {
                        report_session_failure(
                            effects,
                            metrics,
                            SessionFailure::new("stream", error),
                            FailureDelivery::NotifyPeer,
                        )
                        .await;
                        return;
                    }
                    Err(error) => {
                        report_session_failure(
                            effects,
                            metrics,
                            SessionFailure::new(
                                "stream",
                                format!("stream processing task failed: {error}"),
                            ),
                            FailureDelivery::NotifyPeer,
                        )
                        .await;
                        return;
                    }
                };
                let messages = prepare_step_for_delivery(metrics, source_channels, &step);
                for message in messages {
                    if let Err(error) = send_text(effects, message).await {
                        report_session_failure(
                            effects,
                            metrics,
                            SessionFailure::new("event_delivery", error),
                            FailureDelivery::RecordOnly,
                        )
                        .await;
                        return;
                    }
                }
            }
            Next::Message(Ok(Some(InboundFrame::Text(text)))) => match parse_control(&text) {
                Ok(Control::End) => break TerminalPurpose::ClientEnd(terminal_context),
                Err(error) => {
                    report_session_failure(
                        effects,
                        metrics,
                        SessionFailure::new("stream", error),
                        FailureDelivery::NotifyPeer,
                    )
                    .await;
                    return;
                }
            },
            Next::Message(Ok(Some(InboundFrame::Close))) | Next::Message(Ok(None)) => {
                break TerminalPurpose::PeerClosed(terminal_context);
            }
            Next::Message(Ok(Some(InboundFrame::Ping)))
            | Next::Message(Ok(Some(InboundFrame::Pong))) => {}
            Next::Message(Err(ReceiveError::MessageTooBig { size, limit })) => {
                if let Err(error) = send_close(effects, CloseReason::MessageTooBig).await {
                    // A peer-delivery problem is an access diagnostic, not a second failure.
                    log_ws_error(effects, "error_delivery", &error);
                }
                report_session_failure(
                    effects,
                    metrics,
                    SessionFailure::new(
                        "receive",
                        format!("message too big: {size} bytes exceeds the {limit}-byte limit"),
                    ),
                    FailureDelivery::RecordOnly,
                )
                .await;
                return;
            }
            Next::Message(Err(ReceiveError::Transport(error))) => {
                report_session_failure(
                    effects,
                    metrics,
                    SessionFailure::new("receive", error),
                    FailureDelivery::RecordOnly,
                )
                .await;
                return;
            }
        }
    };
    resolve_terminal(effects, metrics, lifetime, source_channels, end).await;
}

async fn consume_finish<S: StreamingSession>(
    lifetime: &SessionLifetime<S>,
    metrics: &Metrics,
) -> Result<MultiChannelStep, SessionFailure> {
    let state = lifetime.state();
    let result = tokio::task::spawn_blocking(move || {
        let session = take_terminal_session(&state)?;
        Ok::<_, String>(session.finish())
    })
    .await;
    match result {
        Ok(Ok(Ok(step))) => Ok(step),
        Ok(Ok(Err(failure))) => Err(SessionFailure::new(
            "stream",
            failure_message_after_projection(metrics, failure),
        )),
        Ok(Err(error)) => Err(SessionFailure::new("stream", error)),
        Err(error) => Err(SessionFailure::new(
            "stream",
            format!("stream finalization task failed: {error}"),
        )),
    }
}

fn finish_failure_delivery(purpose: &TerminalPurpose) -> FailureDelivery {
    match purpose {
        TerminalPurpose::ShutdownBeforeOpen | TerminalPurpose::PeerClosed(_) => {
            FailureDelivery::RecordOnly
        }
        TerminalPurpose::ClientEnd(_) | TerminalPurpose::ActiveShutdown(_) => {
            FailureDelivery::NotifyPeer
        }
    }
}

async fn resolve_terminal<S: StreamingSession, E: SessionEffects>(
    effects: &mut E,
    metrics: &Metrics,
    lifetime: &SessionLifetime<S>,
    source_channels: SourceChannelCount,
    purpose: TerminalPurpose,
) {
    let delivery = finish_failure_delivery(&purpose);
    match purpose {
        TerminalPurpose::ShutdownBeforeOpen => {
            if let Err(error) = send_close(effects, CloseReason::Normal).await {
                report_session_failure(
                    effects,
                    metrics,
                    SessionFailure::new("shutdown_close", error),
                    FailureDelivery::RecordOnly,
                )
                .await;
            }
        }
        TerminalPurpose::PeerClosed(terminal_context) => {
            match consume_finish(lifetime, metrics).await {
                Ok(step) => {
                    project_observations(metrics, step.observations());
                    effects.emit_access(terminal_context.close_event());
                }
                Err(failure) => {
                    effects.emit_access(terminal_context.close_event());
                    report_session_failure(effects, metrics, failure, delivery).await;
                }
            }
        }
        TerminalPurpose::ClientEnd(terminal_context)
        | TerminalPurpose::ActiveShutdown(terminal_context) => {
            let step = match consume_finish(lifetime, metrics).await {
                Ok(step) => step,
                Err(failure) => {
                    report_session_failure(effects, metrics, failure, delivery).await;
                    return;
                }
            };
            let messages = prepare_step_for_delivery(metrics, source_channels, &step);
            for message in messages {
                if let Err(error) = send_text(effects, message).await {
                    report_session_failure(
                        effects,
                        metrics,
                        SessionFailure::new("final_event_delivery", error),
                        FailureDelivery::RecordOnly,
                    )
                    .await;
                    return;
                }
            }
            effects.emit_access(terminal_context.close_event());
            if let Err(error) = send_text(effects, end_event()).await {
                report_session_failure(
                    effects,
                    metrics,
                    SessionFailure::new("end_delivery", error),
                    FailureDelivery::RecordOnly,
                )
                .await;
                return;
            }
            if let Err(error) = send_close(effects, CloseReason::Normal).await {
                report_session_failure(
                    effects,
                    metrics,
                    SessionFailure::new("close_delivery", error),
                    FailureDelivery::RecordOnly,
                )
                .await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ActiveTerminalContext, AdmissionRefusal, CloseReason, Control, EmitMode, FailureDelivery,
        InboundFrame, PostWarmup, PostWarmupInput, ReceiveError, SessionEffects, SessionFailure,
        SessionLifetime, StreamDefaults, StreamQuery, StreamingSession, WS_MESSAGE_LIMIT_BYTES,
        WsState, capture_upgrade, construct_then_classify, offer_access, parse_control,
        prepare_messages_for_delivery, project_observations, record_primary_failure,
        report_session_failure, run_active, run_post_warmup, stream, take_terminal_session,
        ws_refusal_message,
    };
    use crate::admission::ServiceAdmission;
    use crate::admission::{AdmissionPhase, AdmissionState, WsPermitCount};
    use crate::context::ApplicationContext;
    use crate::metrics::Metrics;
    use crate::telemetry::{TelemetryProducer, test_support};
    use crate::{
        ServiceCapabilities, ServiceCapabilitiesParameters, ServicePolicy, ServicePolicyParameters,
    };
    use axum::Router;
    use axum::http::StatusCode;
    use axum::routing::get;
    use gigaam_audio::{
        ChannelAudioView, FeatureMatrixView, FrontendMode, FrontendProcessor, SampleFormat,
        SampleRate,
    };
    use gigaam_model_package::ModelPackage;
    use gigaam_recognition::{
        Decoded, Device, ExecutionControl, ExecutionScheduler, FrameRate,
        SpeechProbabilityDetector, WindowDecoder, Word,
    };
    use gigaam_transcription::{
        BackchannelPolicy, CorrelationThreshold, EndpointDetector, MultiChannelFailure,
        MultiChannelSession, MultiChannelStep, MultiChannelStreamOptions, MultiChannelStreamSetup,
        MultiChannelStreamSetupInput, ObservationMode, OriginalChannel, PairwiseChannelPolicy,
        SourceChannelCount, StreamChannelFactory, StreamConfig, StreamEmissionMode, StreamSetup,
        StreamingChannelPolicy, TranscriptionObservation, TurnGap, select_pairwise_channels,
    };
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    fn defaults() -> StreamDefaults {
        StreamDefaults {
            stream_base: StreamConfig::checked_default(
                SampleRate::new(16_000).expect("test sample rate is positive"),
            )
            .expect("test stream defaults are valid"),
            dedup_default: true,
            backchannel_max_sec: 0.0,
        }
    }

    /// A decoder that makes unexpected execution after a pre-upgrade refusal visible.
    struct RefusalPathDecoder;

    impl WindowDecoder for RefusalPathDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(25.0).expect("the controlled refusal decoder frame rate is positive")
        }

        fn decode(&mut self, _features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            Err("a pre-upgrade refusal must not initialize decoder work".into())
        }
    }

    /// Constructs the real state consumed by `stream` while retaining its telemetry lineage.
    fn refusal_path_state(
        fixture: &crate::tests::AssemblyFixture,
        admission: Arc<AdmissionState>,
        metrics: Arc<Metrics>,
        telemetry: TelemetryProducer,
    ) -> Result<WsState, String> {
        let capabilities = ServiceCapabilities::new(ServiceCapabilitiesParameters {
            pack: Arc::clone(&fixture.pack),
            frontend: Arc::clone(&fixture.frontend),
            ctc: Arc::new(ExecutionScheduler::spawn(RefusalPathDecoder)),
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
            observations: ObservationMode::disabled(),
            backchannel_max_seconds: 0.0,
        })?;
        let context = Arc::new(ApplicationContext::new(capabilities, policy)?);
        Ok(WsState::new(context, admission, metrics, telemetry))
    }

    /// A decoder whose warmup and every subsequent call succeed trivially, so a real post-upgrade
    /// session can be driven to `handle` without native ONNX Runtime work.
    struct WarmupSucceedsDecoder;

    impl WindowDecoder for WarmupSucceedsDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(25.0).expect("the controlled warmup decoder frame rate is positive")
        }

        fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            Decoded::new(
                Vec::new(),
                vec![false; features.frames()],
                features.frames(),
                0.0,
            )
        }
    }

    /// Constructs the real state consumed by `stream` for a session that reaches an active,
    /// post-upgrade socket instead of stopping at pre-upgrade admission.
    fn active_session_state(
        fixture: &crate::tests::AssemblyFixture,
        admission: Arc<AdmissionState>,
        metrics: Arc<Metrics>,
        telemetry: TelemetryProducer,
    ) -> Result<WsState, String> {
        let capabilities = ServiceCapabilities::new(ServiceCapabilitiesParameters {
            pack: Arc::clone(&fixture.pack),
            frontend: Arc::clone(&fixture.frontend),
            ctc: Arc::new(ExecutionScheduler::spawn(WarmupSucceedsDecoder)),
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
            observations: ObservationMode::disabled(),
            backchannel_max_seconds: 0.0,
        })?;
        let context = Arc::new(ApplicationContext::new(capabilities, policy)?);
        Ok(WsState::new(context, admission, metrics, telemetry))
    }

    /// Builds one complete client-to-server masked WebSocket frame carrying `payload_len` zero
    /// bytes, using the minimal length encoding the protocol requires (a 7-bit inline length, a
    /// 16-bit extended length, or a 64-bit extended length). A masked all-zero payload is
    /// exactly the mask key repeated, so the wire bytes are produced directly without a separate
    /// masking pass over a materialized zero buffer.
    fn masked_frame_of_zeros(fin: bool, opcode: u8, payload_len: usize) -> Vec<u8> {
        const MASK: [u8; 4] = [0x37, 0xfa, 0x21, 0x3d];
        let fin_bit = if fin { 0x80_u8 } else { 0x00_u8 };
        let mut frame = vec![fin_bit | opcode];
        if payload_len <= 125 {
            let length = u8::try_from(payload_len)
                .expect("a payload length of at most 125 bytes fits one byte");
            frame.push(0x80 | length);
        } else if payload_len <= 0xffff {
            let length = u16::try_from(payload_len)
                .expect("a payload length of at most 65535 bytes fits u16");
            frame.push(0x80 | 126);
            frame.extend_from_slice(&length.to_be_bytes());
        } else {
            let length = u64::try_from(payload_len)
                .expect("the test WebSocket frame payload length fits u64");
            frame.push(0x80 | 127);
            frame.extend_from_slice(&length.to_be_bytes());
        }
        frame.extend_from_slice(&MASK);
        frame.extend(MASK.iter().cycle().take(payload_len));
        frame
    }

    struct NoopEffects;

    impl SessionEffects for NoopEffects {
        async fn receive(&mut self) -> Result<Option<InboundFrame>, ReceiveError> {
            Ok(None)
        }

        async fn send_text(&mut self, _body: String) -> Result<(), String> {
            Ok(())
        }

        async fn send_close(&mut self, _reason: CloseReason) -> Result<(), String> {
            Ok(())
        }

        fn emit_access(&mut self, _event: crate::log::AccessEvent) {}
    }

    #[derive(Default)]
    struct RecorderEffects {
        inbound: VecDeque<Result<Option<InboundFrame>, ReceiveError>>,
        sent_text: Vec<String>,
        closes: usize,
        close_reasons: Vec<CloseReason>,
        access: Vec<crate::log::AccessEvent>,
        history: Vec<String>,
        fail_text: Option<String>,
    }

    impl RecorderEffects {
        fn with_inbound(
            inbound: impl IntoIterator<Item = Result<Option<InboundFrame>, ReceiveError>>,
        ) -> Self {
            Self {
                inbound: inbound.into_iter().collect(),
                ..Self::default()
            }
        }

        fn access_kinds(&self) -> Vec<&'static str> {
            self.access
                .iter()
                .map(|event| match event {
                    crate::log::AccessEvent::Transcribe(_) => "transcribe",
                    crate::log::AccessEvent::WsOpen(_) => "ws_open",
                    crate::log::AccessEvent::WsClose(_) => "ws_close",
                    crate::log::AccessEvent::WsError(_) => "ws_error",
                })
                .collect()
        }
    }

    impl SessionEffects for RecorderEffects {
        async fn receive(&mut self) -> Result<Option<InboundFrame>, ReceiveError> {
            match self.inbound.pop_front() {
                Some(next) => next,
                None => Ok(None),
            }
        }

        async fn send_text(&mut self, body: String) -> Result<(), String> {
            match self.fail_text.take() {
                Some(error) => Err(error),
                None => {
                    self.history.push(format!("text:{body}"));
                    self.sent_text.push(body);
                    Ok(())
                }
            }
        }

        async fn send_close(&mut self, reason: CloseReason) -> Result<(), String> {
            self.closes = self
                .closes
                .checked_add(1)
                .expect("controlled close count fits usize");
            self.close_reasons.push(reason);
            self.history.push("close".to_owned());
            Ok(())
        }

        fn emit_access(&mut self, event: crate::log::AccessEvent) {
            let kind = match &event {
                crate::log::AccessEvent::Transcribe(_) => "transcribe",
                crate::log::AccessEvent::WsOpen(_) => "ws_open",
                crate::log::AccessEvent::WsClose(_) => "ws_close",
                crate::log::AccessEvent::WsError(_) => "ws_error",
            };
            self.history.push(kind.to_owned());
            self.access.push(event);
        }
    }

    /// Test transport that preserves the generic terminal history while routing access through the
    /// exact production telemetry-offer helper.
    struct TelemetryRecorderEffects {
        recorder: RecorderEffects,
        telemetry: TelemetryProducer,
    }

    impl TelemetryRecorderEffects {
        fn with_inbound(
            telemetry: TelemetryProducer,
            inbound: impl IntoIterator<Item = Result<Option<InboundFrame>, ReceiveError>>,
        ) -> Self {
            Self {
                recorder: RecorderEffects::with_inbound(inbound),
                telemetry,
            }
        }

        fn into_recorder(self) -> RecorderEffects {
            self.recorder
        }
    }

    impl SessionEffects for TelemetryRecorderEffects {
        async fn receive(&mut self) -> Result<Option<InboundFrame>, ReceiveError> {
            self.recorder.receive().await
        }

        async fn send_text(&mut self, body: String) -> Result<(), String> {
            self.recorder.send_text(body).await
        }

        async fn send_close(&mut self, reason: CloseReason) -> Result<(), String> {
            self.recorder.send_close(reason).await
        }

        fn emit_access(&mut self, event: crate::log::AccessEvent) {
            offer_access(&self.telemetry, event.clone());
            self.recorder.emit_access(event);
        }
    }

    #[derive(Clone, Copy)]
    enum TerminalResult {
        Success,
        Failure,
    }

    struct TerminalSession {
        finishes: Arc<AtomicU64>,
        result: TerminalResult,
    }

    impl TerminalSession {
        fn new(finishes: Arc<AtomicU64>, result: TerminalResult) -> Self {
            Self { finishes, result }
        }
    }

    impl StreamingSession for TerminalSession {
        fn push(&mut self, _bytes: &[u8]) -> Result<MultiChannelStep, MultiChannelFailure> {
            Ok(MultiChannelStep::new(Vec::new(), Vec::new())
                .expect("an empty controlled stream step is valid"))
        }

        fn finish(self) -> Result<MultiChannelStep, MultiChannelFailure> {
            self.finishes.fetch_add(1, Ordering::Relaxed);
            match self.result {
                TerminalResult::Success => Ok(MultiChannelStep::new(Vec::new(), Vec::new())
                    .expect("an empty controlled stream step is valid")),
                TerminalResult::Failure => Err(MultiChannelFailure::new(
                    "controlled finish failure".to_owned(),
                    Vec::new(),
                )
                .expect("a nonempty controlled failure is valid")),
            }
        }
    }

    fn metric_counter(metrics: &Metrics, name: &str) -> u64 {
        let rendered = metrics.render(1, 1, 0, None);
        let prefix = format!("{name} ");
        let line = match rendered.lines().find(|line| line.starts_with(&prefix)) {
            Some(line) => line,
            None => panic!("metrics must include the {name} counter"),
        };
        let value = match line.strip_prefix(&prefix) {
            Some(value) => value,
            None => panic!("matched metric line must retain its counter value"),
        };
        value
            .parse::<u64>()
            .expect("the Prometheus counter projection is an unsigned integer")
    }

    async fn terminal_history_with_telemetry(
        telemetry: TelemetryProducer,
        result: TerminalResult,
    ) -> RecorderEffects {
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
            .expect("test admission settings are valid");
        let admission = AdmissionState::new(&settings).expect("test admission state is valid");
        let source_channels = SourceChannelCount::new(1).expect("one source channel is valid");
        let token = admission
            .admit_ws(
                WsPermitCount::from_source_channels(source_channels)
                    .expect("one source channel maps to a permit"),
            )
            .expect("the test session starts while running");
        let mut phase_rx = admission.subscribe();
        let metrics = Metrics::new(1, 1);
        let mut effects = TelemetryRecorderEffects::with_inbound(
            telemetry,
            [Ok(Some(InboundFrame::Text(
                "{\"type\":\"end\"}".to_owned(),
            )))],
        );
        let terminal = ActiveTerminalContext::open(
            SampleRate::new(16_000).expect("the controlled source sample rate is valid"),
            source_channels,
            &mut effects,
        );
        let finishes = Arc::new(AtomicU64::new(0));
        let lifetime =
            SessionLifetime::new(token, TerminalSession::new(Arc::clone(&finishes), result));
        run_active(
            &mut effects,
            &metrics,
            &lifetime,
            source_channels,
            &mut phase_rx,
            terminal,
        )
        .await;
        assert_eq!(finishes.load(Ordering::Relaxed), 1);
        drop(lifetime);
        assert_eq!(admission.available_ws(), 1);
        effects.into_recorder()
    }

    #[tokio::test]
    async fn telemetry_outcomes_preserve_websocket_terminal_histories_and_pre_upgrade_silence()
    -> Result<(), String> {
        let baseline = test_support::prepare(1, 1)?;
        let baseline_producer = baseline.producer();
        let expected_success =
            terminal_history_with_telemetry(baseline_producer.clone(), TerminalResult::Success)
                .await;
        let expected_failure =
            terminal_history_with_telemetry(baseline_producer, TerminalResult::Failure).await;
        assert_eq!(baseline.snapshot().entered, 4);

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
        let saturated_success =
            terminal_history_with_telemetry(saturated_producer.clone(), TerminalResult::Success)
                .await;
        let saturated_failure =
            terminal_history_with_telemetry(saturated_producer, TerminalResult::Failure).await;
        assert_eq!(saturated_success.history, expected_success.history);
        assert_eq!(
            saturated_success.access_kinds(),
            expected_success.access_kinds()
        );
        assert_eq!(saturated_failure.history, expected_failure.history);
        assert_eq!(
            saturated_failure.access_kinds(),
            expected_failure.access_kinds()
        );
        let saturated_snapshot = saturated.snapshot();
        assert_eq!(saturated_snapshot.entered, 8);
        assert_eq!(saturated_snapshot.queued, 4);
        assert_eq!(saturated_snapshot.undelivered, 4);
        assert_eq!(saturated_snapshot.queue_full, 4);

        let receiver_dropped = test_support::prepare(1, 1)?.into_receiver_dropped();
        let receiver_dropped_producer = receiver_dropped.producer();
        let receiver_dropped_success = terminal_history_with_telemetry(
            receiver_dropped_producer.clone(),
            TerminalResult::Success,
        )
        .await;
        let receiver_dropped_failure =
            terminal_history_with_telemetry(receiver_dropped_producer, TerminalResult::Failure)
                .await;
        assert_eq!(receiver_dropped_success.history, expected_success.history);
        assert_eq!(
            receiver_dropped_success.access_kinds(),
            expected_success.access_kinds()
        );
        assert_eq!(receiver_dropped_failure.history, expected_failure.history);
        assert_eq!(
            receiver_dropped_failure.access_kinds(),
            expected_failure.access_kinds()
        );
        let receiver_dropped_snapshot = receiver_dropped.snapshot();
        assert_eq!(receiver_dropped_snapshot.entered, 4);
        assert_eq!(receiver_dropped_snapshot.queued, 0);
        assert_eq!(receiver_dropped_snapshot.undelivered, 0);
        assert_eq!(receiver_dropped_snapshot.sink_closed, 4);

        let failing = test_support::prepare(1, 1)?.into_failing_stdout()?;
        let failing_producer = failing.producer();
        let failing_success =
            terminal_history_with_telemetry(failing_producer.clone(), TerminalResult::Success)
                .await;
        let failing_failure =
            terminal_history_with_telemetry(failing_producer.clone(), TerminalResult::Failure)
                .await;
        assert_eq!(failing_success.history, expected_success.history);
        assert_eq!(
            failing_success.access_kinds(),
            expected_success.access_kinds()
        );
        assert_eq!(failing_failure.history, expected_failure.history);
        assert_eq!(
            failing_failure.access_kinds(),
            expected_failure.access_kinds()
        );
        failing.wait_until_terminal();
        let failing_snapshot = failing.snapshot();
        assert_eq!(failing_snapshot.entered, 4);
        assert_eq!(failing_snapshot.queued, 0);
        assert_eq!(failing_snapshot.undelivered, 0);
        assert_eq!(failing_snapshot.sink_closed, 4);
        assert!(
            failing
                .metrics()
                .render(1, 1, 0, None)
                .contains("asr_telemetry_write_failures_total{destination=\"stdout\"} 1\n")
        );
        drop(failing_producer);
        failing.finish();

        let fixture = crate::tests::assembly_fixture()?;
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))?;
        let pre_upgrade = test_support::prepare(1, 1)?;
        let metrics = pre_upgrade.metrics_arc();
        let admission = Arc::new(AdmissionState::new(&settings)?);
        let state = refusal_path_state(
            &fixture,
            Arc::clone(&admission),
            Arc::clone(&metrics),
            pre_upgrade.producer(),
        )?;
        admission.begin_draining();
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|error| format!("bind WebSocket refusal listener: {error}"))?;
        let address = listener
            .local_addr()
            .map_err(|error| format!("read WebSocket refusal listener address: {error}"))?;
        let routes = Router::new()
            .route("/v1/stream", get(stream))
            .with_state(state);
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, routes)
                .with_graceful_shutdown(async {
                    shutdown_rx
                        .await
                        .expect("the WebSocket refusal test sends one listener shutdown signal");
                })
                .await
        });
        let mut client = TcpStream::connect(address)
            .await
            .map_err(|error| format!("connect WebSocket refusal client: {error}"))?;
        let request = format!(
            "GET /v1/stream?rate=16000 HTTP/1.1\r\nHost: {address}\r\nConnection: Upgrade, close\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
        );
        client
            .write_all(request.as_bytes())
            .await
            .map_err(|error| format!("write WebSocket refusal request: {error}"))?;
        let mut response = Vec::new();
        client
            .read_to_end(&mut response)
            .await
            .map_err(|error| format!("read WebSocket refusal response: {error}"))?;
        let response = String::from_utf8(response)
            .map_err(|error| format!("WebSocket refusal response must be UTF-8: {error}"))?;
        shutdown_tx
            .send(())
            .expect("the WebSocket refusal server remains live for its explicit shutdown");
        server
            .await
            .map_err(|error| format!("WebSocket refusal server task failed: {error}"))?
            .map_err(|error| format!("WebSocket refusal server stopped with error: {error}"))?;
        assert!(
            response.starts_with("HTTP/1.1 503"),
            "a draining WebSocket upgrade must preserve its refusal status: {response}"
        );
        assert!(
            response.contains("\r\n\r\n{\"error\":\"draining\"}"),
            "a draining WebSocket upgrade must preserve its refusal body: {response}"
        );
        assert_eq!(metric_counter(&metrics, "asr_ws_sessions_total"), 0);
        assert_eq!(metric_counter(&metrics, "asr_ws_rejected_total"), 0);
        assert_eq!(admission.available_ws(), 1);
        let pre_upgrade_snapshot = pre_upgrade.snapshot();
        assert_eq!(pre_upgrade_snapshot.entered, 0);
        assert_eq!(pre_upgrade_snapshot.remaining, 0);
        assert_eq!(pre_upgrade_snapshot.queued, 0);
        assert_eq!(pre_upgrade_snapshot.undelivered, 0);
        assert_eq!(pre_upgrade_snapshot.delivered, 0);
        assert_eq!(pre_upgrade_snapshot.queue_full, 0);
        assert_eq!(pre_upgrade_snapshot.sink_closed, 0);
        Ok(())
    }

    #[test]
    fn stream_query_percent_decodes_into_typed_configuration() {
        let query = StreamQuery::parse(
            Some(
                "rate=16000&fmt=f32&channels=2&emit=both&endpoint=vad&horizon=4%2E5&lock=true&turn_gap=0.8&dedup=false&backchannel_max_ms=250",
            ),
            defaults(),
        )
        .expect("a documented WebSocket query must parse");
        assert_eq!(
            query.rate,
            SampleRate::new(16_000).expect("test sample rate is positive")
        );
        assert_eq!(query.format, gigaam_audio::SampleFormat::F32);
        assert_eq!(query.channels, 2);
        assert_eq!(query.emit, EmitMode::Both);
        assert_eq!(
            query.config.endpoint_source(),
            gigaam_transcription::EndpointSource::Vad
        );
        assert_eq!(query.config.horizon_sec(), 4.5);
        assert_eq!(
            query.config.lock_policy(),
            gigaam_transcription::StreamLockPolicy::CommitStable
        );
        assert!(!query.dedup);
        assert_eq!(query.backchannel_max_sec, 0.25);
    }

    #[test]
    fn stream_query_refuses_unknown_duplicate_invalid_and_conflicting_values() {
        for raw in [
            "rate=16000&rate=8000",
            "rate=16000&unknown=value",
            "rate=16000&horizon=NaN",
            "rate=16000&channels=0",
            "rate=16000&dedup=true",
            "rate=16000&endpoint=other",
            "rate=16000&lock=on",
            "rate=16000&backchannel_max_ms=-1",
        ] {
            assert!(
                StreamQuery::parse(Some(raw), defaults()).is_err(),
                "query {raw:?} must be refused"
            );
        }
    }

    #[test]
    fn control_requires_the_exact_end_object() {
        assert_eq!(parse_control(" { \"type\" : \"end\" } "), Ok(Control::End));
        for text in [
            "{\"type\":\"send\"}",
            "{\"type\":\"e n d\"}",
            "{\"type\":\"end\",\"extra\":true}",
            "{\"type\":\"end\"} trailing",
        ] {
            assert!(
                parse_control(text).is_err(),
                "control {text:?} must be refused"
            );
        }
    }

    #[test]
    fn pre_upgrade_refusal_preserves_the_four_session_rejection_histories() {
        let metrics = Metrics::new(2, 2);
        assert_eq!(
            ws_refusal_message(&metrics, AdmissionRefusal::Overloaded),
            "too many active channels, retry later"
        );
        assert_eq!(metric_counter(&metrics, "asr_ws_sessions_total"), 1);
        assert_eq!(metric_counter(&metrics, "asr_ws_rejected_total"), 1);
        assert_eq!(
            ws_refusal_message(&metrics, AdmissionRefusal::Draining),
            "draining"
        );
        assert_eq!(metric_counter(&metrics, "asr_ws_sessions_total"), 1);
        assert_eq!(metric_counter(&metrics, "asr_ws_rejected_total"), 1);
    }

    #[tokio::test]
    async fn websocket_refusals_preserve_exact_shared_json_envelopes_and_metric_deltas() {
        let metrics = Metrics::new(1, 1);
        let overload = crate::stream_response::pre_upgrade_refusal(
            StatusCode::SERVICE_UNAVAILABLE,
            ws_refusal_message(&metrics, AdmissionRefusal::Overloaded),
        );
        assert_eq!(overload.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            overload.headers().get(axum::http::header::CONTENT_TYPE),
            Some(&axum::http::HeaderValue::from_static(
                "application/json; charset=utf-8",
            ))
        );
        assert_eq!(
            axum::body::to_bytes(overload.into_body(), usize::MAX)
                .await
                .expect("overload body is available"),
            "{\"error\":\"too many active channels, retry later\"}"
        );
        assert_eq!(metric_counter(&metrics, "asr_ws_sessions_total"), 1);
        assert_eq!(metric_counter(&metrics, "asr_ws_rejected_total"), 1);

        let draining = crate::stream_response::pre_upgrade_refusal(
            StatusCode::SERVICE_UNAVAILABLE,
            ws_refusal_message(&metrics, AdmissionRefusal::Draining),
        );
        assert_eq!(
            axum::body::to_bytes(draining.into_body(), usize::MAX)
                .await
                .expect("draining body is available"),
            "{\"error\":\"draining\"}"
        );
        assert_eq!(metric_counter(&metrics, "asr_ws_sessions_total"), 1);
        assert_eq!(metric_counter(&metrics, "asr_ws_rejected_total"), 1);
    }

    /// Walks unmasked server-to-client WebSocket frames in `bytes` (7-bit and 16-bit payload
    /// lengths) and returns the status code carried by the first Close frame (opcode `0x8`), if
    /// any. A close frame's own control payload is at most 125 bytes, so the 64-bit extended
    /// length case never applies to it; it is only skipped past correctly for any other frame.
    fn find_close_code(bytes: &[u8]) -> Option<u16> {
        let mut offset = 0;
        while offset + 2 <= bytes.len() {
            let opcode = bytes[offset] & 0x0f;
            let masked = bytes[offset + 1] & 0x80 != 0;
            let length_byte = bytes[offset + 1] & 0x7f;
            let (payload_len, header_len): (usize, usize) = match length_byte {
                126 => {
                    if offset + 4 > bytes.len() {
                        break;
                    }
                    let mut length_bytes = [0_u8; 2];
                    length_bytes.copy_from_slice(&bytes[offset + 2..offset + 4]);
                    (usize::from(u16::from_be_bytes(length_bytes)), 4)
                }
                127 => {
                    if offset + 10 > bytes.len() {
                        break;
                    }
                    let mut length_bytes = [0_u8; 8];
                    length_bytes.copy_from_slice(&bytes[offset + 2..offset + 10]);
                    let length = usize::try_from(u64::from_be_bytes(length_bytes))
                        .expect("a test trailing-frame payload length fits usize");
                    (length, 10)
                }
                direct => (usize::from(direct), 2),
            };
            let mask_len = if masked { 4 } else { 0 };
            let payload_start = offset + header_len + mask_len;
            let payload_end = payload_start
                .checked_add(payload_len)
                .expect("a test trailing-frame payload end fits usize");
            if payload_end > bytes.len() {
                break;
            }
            if opcode == 0x8 && payload_len >= 2 {
                let mut code_bytes = [0_u8; 2];
                code_bytes.copy_from_slice(&bytes[payload_start..payload_start + 2]);
                return Some(u16::from_be_bytes(code_bytes));
            }
            offset = payload_end;
        }
        None
    }

    #[tokio::test]
    async fn oversized_websocket_message_closes_with_code_1009_and_no_transcript()
    -> Result<(), String> {
        let fixture = crate::tests::assembly_fixture()?;
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(5))?;
        let admission = Arc::new(AdmissionState::new(&settings)?);
        let pre_upgrade = test_support::prepare(1, 1)?;
        let metrics = pre_upgrade.metrics_arc();
        let state = active_session_state(
            &fixture,
            Arc::clone(&admission),
            Arc::clone(&metrics),
            pre_upgrade.producer(),
        )?;
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|error| format!("bind oversized-message listener: {error}"))?;
        let address = listener
            .local_addr()
            .map_err(|error| format!("read oversized-message listener address: {error}"))?;
        let routes = Router::new()
            .route("/v1/stream", get(stream))
            .with_state(state);
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, routes)
                .with_graceful_shutdown(async {
                    shutdown_rx
                        .await
                        .expect("the oversized-message test sends one listener shutdown signal");
                })
                .await
        });

        let mut client = TcpStream::connect(address)
            .await
            .map_err(|error| format!("connect oversized-message client: {error}"))?;
        let request = format!(
            "GET /v1/stream?rate=16000 HTTP/1.1\r\nHost: {address}\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
        );
        client
            .write_all(request.as_bytes())
            .await
            .map_err(|error| format!("write oversized-message upgrade request: {error}"))?;

        // Read exactly through the end of the HTTP upgrade response headers, leaving the socket
        // positioned to observe only the bytes the server sends after the WebSocket upgrade.
        let mut response = Vec::new();
        let mut byte = [0_u8; 1];
        loop {
            let read = client
                .read(&mut byte)
                .await
                .map_err(|error| format!("read oversized-message upgrade response: {error}"))?;
            if read == 0 {
                return Err("oversized-message upgrade response ended before its headers".into());
            }
            response.push(byte[0]);
            if response.ends_with(b"\r\n\r\n") {
                break;
            }
        }
        let response = String::from_utf8(response).map_err(|error| {
            format!("oversized-message upgrade response must be UTF-8: {error}")
        })?;
        assert!(
            response.starts_with("HTTP/1.1 101"),
            "a valid query on running admission must upgrade: {response}"
        );

        // The per-frame limit equals the library default (16 MiB), so one oversized frame would
        // be rejected by the frame bound alone and would not exercise the tightened message
        // bound. Splitting the payload into two 16-MiB-or-smaller frames of a single fragmented
        // message keeps every frame within the frame limit while the assembled message exceeds
        // WS_MESSAGE_LIMIT_BYTES, so only the message-size contract this crate adds is under
        // test.
        let half = WS_MESSAGE_LIMIT_BYTES / 2;
        let first = masked_frame_of_zeros(false, 0x2, half);
        let second = masked_frame_of_zeros(true, 0x0, half + 4);
        // The server may close the connection once it reads enough of the fragmented message to
        // exceed the limit, before every payload byte is written; a write error is an expected
        // rejection signal, not a test failure, and the timeout guards against an unexpected
        // indefinite block.
        let _ = tokio::time::timeout(Duration::from_secs(10), client.write_all(&first)).await;
        let _ = tokio::time::timeout(Duration::from_secs(10), client.write_all(&second)).await;

        let mut trailing = Vec::new();
        tokio::time::timeout(Duration::from_secs(10), client.read_to_end(&mut trailing))
            .await
            .map_err(|_| {
                "the server never closed the session after the oversized message".to_owned()
            })?
            .map_err(|error| format!("read oversized-message trailing bytes: {error}"))?;
        let trailing_text = String::from_utf8_lossy(&trailing);
        assert!(
            !trailing_text.contains("\"type\":\"words\"")
                && !trailing_text.contains("\"type\":\"turns\"")
                && !trailing_text.contains("\"type\":\"final\"")
                && !trailing_text.contains("\"type\":\"end\""),
            "an oversized message must end the session without a transcript: {trailing_text:?}"
        );
        assert_eq!(
            find_close_code(&trailing),
            Some(1009),
            "an oversized message must close with WebSocket code 1009: {trailing:?}"
        );

        shutdown_tx.send(()).map_err(|_| {
            "the oversized-message server remains live for its explicit shutdown".to_owned()
        })?;
        server
            .await
            .map_err(|error| format!("oversized-message server task failed: {error}"))?
            .map_err(|error| format!("oversized-message server stopped with error: {error}"))?;
        Ok(())
    }

    #[test]
    fn refused_pre_upgrade_requests_never_register_a_callback_or_start_factory_work() {
        let settings = ServiceAdmission::new(1, 2, Duration::from_secs(1))
            .expect("test admission settings are valid");
        let admission = AdmissionState::new(&settings).expect("test admission state is valid");
        let metrics = Arc::new(Metrics::new(1, 2));
        let channels = SourceChannelCount::new(2).expect("two source channels are valid");
        let permits = WsPermitCount::from_source_channels(channels)
            .expect("the declared channel count fits admission");
        let held = admission
            .admit_ws(permits)
            .expect("the first running request occupies both channel permits");
        let callback_starts = Cell::new(0_u8);
        let overload = capture_upgrade(&admission, Arc::clone(&metrics), permits, |_| {
            callback_starts.set(
                callback_starts
                    .get()
                    .checked_add(1)
                    .expect("one controlled callback start fits u8"),
            );
        });
        assert!(matches!(overload, Err(AdmissionRefusal::Overloaded)));
        assert_eq!(
            callback_starts.get(),
            0,
            "overload must return before registering an upgrade callback or factory work"
        );

        drop(held);
        admission.begin_draining();
        let draining = capture_upgrade(&admission, Arc::clone(&metrics), permits, |_| {
            callback_starts.set(
                callback_starts
                    .get()
                    .checked_add(1)
                    .expect("one controlled callback start fits u8"),
            );
        });
        assert!(matches!(draining, Err(AdmissionRefusal::Draining)));
        assert_eq!(
            callback_starts.get(),
            0,
            "draining must return before registering an upgrade callback or factory work"
        );
    }

    #[test]
    fn abandoned_upgrade_callback_releases_every_token_without_counting_a_session() {
        let settings = ServiceAdmission::new(1, 2, Duration::from_secs(1))
            .expect("test admission settings are valid");
        let admission = AdmissionState::new(&settings).expect("test admission state is valid");
        let metrics = Metrics::new(1, 2);
        let channels = SourceChannelCount::new(2).expect("two source channels are valid");
        let token = admission
            .admit_ws(
                WsPermitCount::from_source_channels(channels)
                    .expect("the declared channel count fits semaphore admission"),
            )
            .expect("running state commits the callback-owned token");
        assert_eq!(admission.available_ws(), 0);
        drop(token);
        assert_eq!(admission.available_ws(), 2);
        assert_eq!(metric_counter(&metrics, "asr_ws_sessions_total"), 0);
    }

    #[test]
    fn production_capture_upgrade_holds_and_transfers_the_all_channel_token() {
        let settings = ServiceAdmission::new(1, 2, Duration::from_secs(1))
            .expect("test admission settings are valid");
        let admission = AdmissionState::new(&settings).expect("test admission state is valid");
        let metrics = Arc::new(Metrics::new(1, 2));
        let permits = WsPermitCount::from_source_channels(
            SourceChannelCount::new(2).expect("two source channels are valid"),
        )
        .expect("two permits are valid");

        let abandoned = capture_upgrade(&admission, Arc::clone(&metrics), permits, |phase_rx| {
            phase_rx
        })
        .expect("a running capture commits every channel permit");
        assert_eq!(admission.available_ws(), 0);
        assert_eq!(metric_counter(&metrics, "asr_ws_sessions_total"), 0);
        drop(abandoned);
        assert_eq!(admission.available_ws(), 2);
        assert_eq!(metric_counter(&metrics, "asr_ws_sessions_total"), 0);

        let captured = capture_upgrade(&admission, Arc::clone(&metrics), permits, |phase_rx| {
            phase_rx
        })
        .expect("a running capture commits every channel permit");
        let (phase_rx, token) = captured.execute(|phase_rx, token| (phase_rx, token));
        assert_eq!(metric_counter(&metrics, "asr_ws_sessions_total"), 1);
        assert_eq!(admission.available_ws(), 0);
        assert_eq!(*phase_rx.borrow(), AdmissionPhase::Running);
        drop(token);
        assert_eq!(admission.available_ws(), 2);
    }

    fn admitted_state() -> Arc<AdmissionState> {
        let settings = ServiceAdmission::new(1, 2, Duration::from_secs(1))
            .expect("test admission settings are valid");
        Arc::new(AdmissionState::new(&settings).expect("test admission state is valid"))
    }

    fn captured_callback(
        state: &AdmissionState,
        metrics: Arc<Metrics>,
        source_channels: SourceChannelCount,
    ) -> super::CapturedUpgrade<tokio::sync::watch::Receiver<AdmissionPhase>> {
        let permits = WsPermitCount::from_source_channels(source_channels)
            .expect("the controlled source channels fit all-channel admission");
        capture_upgrade(state, metrics, permits, |phase_rx| phase_rx)
            .expect("a running callback captures its admission token and receiver")
    }

    async fn resolve_pre_active_post_warmup(
        metrics: &Metrics,
        phase_rx: tokio::sync::watch::Receiver<AdmissionPhase>,
        admission: super::WsAdmission,
        source_channels: SourceChannelCount,
        result: PostWarmup<TerminalSession>,
    ) -> RecorderEffects {
        let mut effects = RecorderEffects::default();
        run_post_warmup(
            &mut effects,
            metrics,
            PostWarmupInput {
                result,
                admission,
                phase_rx,
                source_sample_rate: SampleRate::new(16000)
                    .expect("the controlled source sample rate is valid"),
                source_channels,
            },
        )
        .await;
        effects
    }

    fn assert_close_only_pre_active_effects(
        placement: &str,
        state: &AdmissionState,
        metrics: &Metrics,
        effects: &RecorderEffects,
        finishes: &AtomicU64,
        source_channels: SourceChannelCount,
    ) {
        assert_eq!(finishes.load(Ordering::Relaxed), 0, "{placement}");
        assert!(effects.sent_text.is_empty(), "{placement}");
        assert_eq!(effects.closes, 1, "{placement}");
        assert!(effects.access.is_empty(), "{placement}");
        assert_eq!(effects.history, vec!["close".to_owned()], "{placement}");
        assert_eq!(
            metric_counter(metrics, "asr_ws_sessions_total"),
            1,
            "{placement}"
        );
        assert_eq!(
            metric_counter(metrics, "asr_ws_rejected_total"),
            0,
            "{placement}"
        );
        let expected_source_channels = u64::try_from(source_channels.get())
            .expect("the controlled source channel count fits the Prometheus counter");
        assert_eq!(
            metric_counter(metrics, "asr_ws_channels_total"),
            expected_source_channels,
            "{placement}"
        );
        assert_eq!(
            metric_counter(metrics, "asr_ws_errors_total"),
            0,
            "{placement}"
        );
        assert_eq!(
            state.available_ws(),
            2,
            "{placement} must return the captured all-channel admission after Close-only completion"
        );
    }

    #[tokio::test]
    async fn construction_and_warmup_finish_before_each_pre_active_drain_placement() {
        let before_callback = admitted_state();
        let source_channels = SourceChannelCount::new(2).expect("two source channels are valid");
        let before_callback_metrics = Arc::new(Metrics::new(1, 2));
        let before_callback_captured = captured_callback(
            &before_callback,
            Arc::clone(&before_callback_metrics),
            source_channels,
        );
        before_callback.begin_draining();
        let (before_callback_rx, before_callback_token) =
            before_callback_captured.execute(|phase_rx, admission| (phase_rx, admission));
        let before_callback_trace = RefCell::new(Vec::new());
        let before_callback_finishes = Arc::new(AtomicU64::new(0));
        let before_callback_result = construct_then_classify(
            || {
                before_callback_trace.borrow_mut().push("warmup");
                Ok::<_, String>(TerminalSession::new(
                    Arc::clone(&before_callback_finishes),
                    TerminalResult::Success,
                ))
            },
            || {
                before_callback_trace.borrow_mut().push("channels");
                before_callback_metrics.ws_channels(source_channels.get());
            },
            || {
                before_callback_trace.borrow_mut().push("final phase");
                before_callback.phase()
            },
        )
        .expect("a committed callback completes warmup after a pre-callback drain");
        assert!(matches!(
            &before_callback_result,
            PostWarmup::ShutdownBeforeOpen(_)
        ));
        assert_eq!(
            before_callback_trace.into_inner(),
            vec!["warmup", "channels", "final phase"]
        );
        assert_eq!(
            metric_counter(&before_callback_metrics, "asr_ws_channels_total"),
            2
        );
        let before_callback_effects = resolve_pre_active_post_warmup(
            before_callback_metrics.as_ref(),
            before_callback_rx,
            before_callback_token,
            source_channels,
            before_callback_result,
        )
        .await;
        assert_close_only_pre_active_effects(
            "before callback",
            before_callback.as_ref(),
            before_callback_metrics.as_ref(),
            &before_callback_effects,
            before_callback_finishes.as_ref(),
            source_channels,
        );

        let during_warmup = admitted_state();
        let source_channels = SourceChannelCount::new(2).expect("two source channels are valid");
        let during_warmup_metrics = Arc::new(Metrics::new(1, 2));
        let during_warmup_captured = captured_callback(
            &during_warmup,
            Arc::clone(&during_warmup_metrics),
            source_channels,
        );
        let (during_warmup_rx, during_warmup_token) =
            during_warmup_captured.execute(|phase_rx, admission| (phase_rx, admission));
        let (warmup_started_tx, warmup_started_rx) = std::sync::mpsc::sync_channel(0);
        let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(0);
        let drain_state = Arc::clone(&during_warmup);
        let drainer = std::thread::spawn(move || {
            warmup_started_rx
                .recv()
                .expect("the controlled warmup must announce its blocked stage");
            let transition = drain_state.begin_draining();
            resume_tx
                .send(())
                .expect("the controlled warmup must still wait for drain transition");
            transition
        });
        let during_warmup_trace = RefCell::new(Vec::new());
        let during_warmup_finishes = Arc::new(AtomicU64::new(0));
        let during_warmup_result = construct_then_classify(
            || {
                during_warmup_trace.borrow_mut().push("warmup started");
                warmup_started_tx
                    .send(())
                    .expect("the drain coordinator must wait for the blocked warmup");
                resume_rx
                    .recv()
                    .expect("drain must complete while warmup remains blocked");
                during_warmup_trace.borrow_mut().push("warmup completed");
                Ok::<_, String>(TerminalSession::new(
                    Arc::clone(&during_warmup_finishes),
                    TerminalResult::Success,
                ))
            },
            || {
                during_warmup_trace.borrow_mut().push("channels");
                during_warmup_metrics.ws_channels(source_channels.get());
            },
            || {
                during_warmup_trace.borrow_mut().push("final phase");
                during_warmup.phase()
            },
        )
        .expect("a committed callback completes its blocked warmup after drain");
        assert_eq!(
            drainer
                .join()
                .expect("the drain coordinator must complete normally"),
            crate::admission::DrainTransition::Began
        );
        assert!(matches!(
            &during_warmup_result,
            PostWarmup::ShutdownBeforeOpen(_)
        ));
        assert_eq!(
            during_warmup_trace.into_inner(),
            vec![
                "warmup started",
                "warmup completed",
                "channels",
                "final phase"
            ]
        );
        assert_eq!(
            metric_counter(&during_warmup_metrics, "asr_ws_channels_total"),
            2
        );
        let during_warmup_effects = resolve_pre_active_post_warmup(
            during_warmup_metrics.as_ref(),
            during_warmup_rx,
            during_warmup_token,
            source_channels,
            during_warmup_result,
        )
        .await;
        assert_close_only_pre_active_effects(
            "during warmup",
            during_warmup.as_ref(),
            during_warmup_metrics.as_ref(),
            &during_warmup_effects,
            during_warmup_finishes.as_ref(),
            source_channels,
        );

        let after_warmup = admitted_state();
        let source_channels = SourceChannelCount::new(2).expect("two source channels are valid");
        let after_warmup_metrics = Arc::new(Metrics::new(1, 2));
        let after_warmup_captured = captured_callback(
            &after_warmup,
            Arc::clone(&after_warmup_metrics),
            source_channels,
        );
        let (after_warmup_rx, after_warmup_token) =
            after_warmup_captured.execute(|phase_rx, admission| (phase_rx, admission));
        let after_warmup_trace = RefCell::new(Vec::new());
        let after_warmup_finishes = Arc::new(AtomicU64::new(0));
        let after_warmup_result = construct_then_classify(
            || {
                after_warmup_trace.borrow_mut().push("warmup");
                Ok::<_, String>(TerminalSession::new(
                    Arc::clone(&after_warmup_finishes),
                    TerminalResult::Success,
                ))
            },
            || {
                after_warmup_trace.borrow_mut().push("channels");
                after_warmup_metrics.ws_channels(source_channels.get());
            },
            || {
                after_warmup_trace.borrow_mut().push("final phase");
                after_warmup.begin_draining();
                after_warmup.phase()
            },
        )
        .expect("a post-warmup drain classifies before Active entry");
        assert!(matches!(
            &after_warmup_result,
            PostWarmup::ShutdownBeforeOpen(_)
        ));
        assert_eq!(
            after_warmup_trace.into_inner(),
            vec!["warmup", "channels", "final phase"]
        );
        assert_eq!(
            metric_counter(&after_warmup_metrics, "asr_ws_channels_total"),
            2
        );
        let after_warmup_effects = resolve_pre_active_post_warmup(
            after_warmup_metrics.as_ref(),
            after_warmup_rx,
            after_warmup_token,
            source_channels,
            after_warmup_result,
        )
        .await;
        assert_close_only_pre_active_effects(
            "after warmup",
            after_warmup.as_ref(),
            after_warmup_metrics.as_ref(),
            &after_warmup_effects,
            after_warmup_finishes.as_ref(),
            source_channels,
        );

        let failure_state = admitted_state();
        let source_channels = SourceChannelCount::new(2).expect("two source channels are valid");
        let failure_metrics = Arc::new(Metrics::new(1, 2));
        let failure_captured = captured_callback(&failure_state, failure_metrics, source_channels);
        let (_failure_rx, failure_token) =
            failure_captured.execute(|phase_rx, admission| (phase_rx, admission));
        let channels_observed = RefCell::new(false);
        let phase_read = RefCell::new(false);
        let failure = construct_then_classify(
            || {
                failure_state.begin_draining();
                Err::<(), _>("warmup failed")
            },
            || *channels_observed.borrow_mut() = true,
            || {
                *phase_read.borrow_mut() = true;
                failure_state.phase()
            },
        );
        assert!(matches!(failure, Err("warmup failed")));
        assert!(
            !*channels_observed.borrow(),
            "failed construction must not observe successful channel capacity"
        );
        assert!(
            !*phase_read.borrow(),
            "a construction or warmup failure must retain priority over shutdown"
        );
        drop(failure_token);

        let active_state = admitted_state();
        let source_channels = SourceChannelCount::new(2).expect("two source channels are valid");
        let active_metrics = Arc::new(Metrics::new(1, 2));
        let active_captured =
            captured_callback(&active_state, Arc::clone(&active_metrics), source_channels);
        let (_active_rx, active_token) =
            active_captured.execute(|phase_rx, admission| (phase_rx, admission));
        let active = construct_then_classify(
            || Ok::<_, String>(()),
            || active_metrics.ws_channels(source_channels.get()),
            || active_state.phase(),
        )
        .expect("a running final read enters Active");
        assert!(matches!(active, PostWarmup::Active(())));
        assert_eq!(metric_counter(&active_metrics, "asr_ws_channels_total"), 2);
        drop(active_token);
    }

    #[test]
    fn primary_error_and_failed_error_delivery_count_one_failed_session() {
        let metrics = Metrics::new(1, 1);
        let failure = SessionFailure::new("stream", "primary failure");
        let mut effects = NoopEffects;
        record_primary_failure(&mut effects, &metrics, &failure);
        // Error-delivery diagnostics intentionally use logging only and do not recurse into metrics.
        super::log_ws_error(
            &mut effects,
            "error_delivery",
            "simulated client delivery failure",
        );
        assert_eq!(metric_counter(&metrics, "asr_ws_errors_total"), 1);
    }

    #[test]
    fn peer_close_and_eof_consume_one_real_session_finish_once() -> Result<(), String> {
        for peer_signal in ["close", "eof"] {
            let source_channels = SourceChannelCount::new(1)?;
            let (mut session, sample_rate) =
                serializer_session(source_channels, StreamEmissionMode::WordsAndDialog)?;
            session
                .push(&pcm16_interleaved_silence(
                    source_channels.get(),
                    sample_rate.as_usize()?,
                )?)
                .map_err(|failure| failure.error().to_owned())?;
            let state = Arc::new(std::sync::Mutex::new(Some(session)));
            let finished = take_terminal_session(&state).and_then(|session| {
                session
                    .finish()
                    .map_err(|failure| failure.error().to_owned())
            });
            if let Err(error) = finished {
                return Err(format!(
                    "peer {peer_signal} must consume its session finish successfully: {error}"
                ));
            }
            let consumed = match state.lock() {
                Ok(session) => session.is_none(),
                Err(_) => return Err("peer terminal test session state is poisoned".into()),
            };
            assert!(
                consumed,
                "peer {peer_signal} must consume the shared session exactly once"
            );
            let second_finish = take_terminal_session(&state);
            assert!(
                second_finish.is_err(),
                "peer {peer_signal} must not run a second terminal cleanup"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn generic_peer_close_and_eof_histories_close_once_without_delivery_on_success_or_failure()
     {
        for inbound in [Some(InboundFrame::Close), None] {
            for result in [TerminalResult::Success, TerminalResult::Failure] {
                let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
                    .expect("test admission settings are valid");
                let admission =
                    AdmissionState::new(&settings).expect("test admission state is valid");
                let source_channels =
                    SourceChannelCount::new(1).expect("one source channel is valid");
                let token = admission
                    .admit_ws(
                        WsPermitCount::from_source_channels(source_channels)
                            .expect("one channel maps to a permit"),
                    )
                    .expect("the test session starts while running");
                let mut phase_rx = admission.subscribe();
                let metrics = Metrics::new(1, 1);
                let mut effects = RecorderEffects::with_inbound([Ok(inbound.clone())]);
                let terminal = ActiveTerminalContext::open(
                    SampleRate::new(16000).expect("the controlled source sample rate is valid"),
                    source_channels,
                    &mut effects,
                );
                let finishes = Arc::new(AtomicU64::new(0));
                let lifetime = SessionLifetime::new(
                    token,
                    TerminalSession::new(Arc::clone(&finishes), result),
                );
                run_active(
                    &mut effects,
                    &metrics,
                    &lifetime,
                    source_channels,
                    &mut phase_rx,
                    terminal,
                )
                .await;
                assert_eq!(finishes.load(Ordering::Relaxed), 1);
                assert!(effects.sent_text.is_empty());
                assert_eq!(effects.closes, 0);
                match result {
                    TerminalResult::Success => {
                        assert_eq!(effects.access_kinds(), vec!["ws_open", "ws_close"]);
                        assert_eq!(metric_counter(&metrics, "asr_ws_errors_total"), 0);
                    }
                    TerminalResult::Failure => {
                        assert_eq!(
                            effects.access_kinds(),
                            vec!["ws_open", "ws_close", "ws_error"]
                        );
                        assert_eq!(metric_counter(&metrics, "asr_ws_errors_total"), 1);
                    }
                }
                drop(lifetime);
                assert_eq!(admission.available_ws(), 1);
            }
        }
    }

    /// A capacity refusal must acknowledge the peer with exactly one WebSocket-code-1009 close
    /// and no transcript text frame; every other receive failure keeps the existing silent-drop
    /// behavior and sends no close frame at all.
    #[tokio::test]
    async fn receive_error_close_frame_depends_on_the_error_kind() {
        let cases = [
            (
                ReceiveError::MessageTooBig {
                    size: 32,
                    limit: 16,
                },
                vec![CloseReason::MessageTooBig],
            ),
            (
                ReceiveError::Transport("controlled transport failure".to_owned()),
                Vec::new(),
            ),
        ];
        for (error, expected_close_reasons) in cases {
            let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
                .expect("test admission settings are valid");
            let admission = AdmissionState::new(&settings).expect("test admission state is valid");
            let source_channels = SourceChannelCount::new(1).expect("one source channel is valid");
            let token = admission
                .admit_ws(
                    WsPermitCount::from_source_channels(source_channels)
                        .expect("one channel maps to a permit"),
                )
                .expect("the test session starts while running");
            let mut phase_rx = admission.subscribe();
            let metrics = Metrics::new(1, 1);
            let mut effects = RecorderEffects::with_inbound([Err(error)]);
            let terminal = ActiveTerminalContext::open(
                SampleRate::new(16000).expect("the controlled source sample rate is valid"),
                source_channels,
                &mut effects,
            );
            let finishes = Arc::new(AtomicU64::new(0));
            let lifetime = SessionLifetime::new(
                token,
                TerminalSession::new(Arc::clone(&finishes), TerminalResult::Success),
            );
            run_active(
                &mut effects,
                &metrics,
                &lifetime,
                source_channels,
                &mut phase_rx,
                terminal,
            )
            .await;
            assert!(
                effects.sent_text.is_empty(),
                "a receive failure must never send a transcript or error text frame"
            );
            assert_eq!(effects.close_reasons, expected_close_reasons);
            assert_eq!(
                finishes.load(Ordering::Relaxed),
                0,
                "a receive failure ends the session without a terminal finish"
            );
            assert_eq!(metric_counter(&metrics, "asr_ws_errors_total"), 1);
            drop(lifetime);
            assert_eq!(admission.available_ws(), 1);
        }
    }

    #[tokio::test]
    async fn active_end_and_transition_window_retain_terminal_order_and_existing_receiver() {
        for drain_before_receive in [false, true] {
            let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
                .expect("test admission settings are valid");
            let admission = AdmissionState::new(&settings).expect("test admission state is valid");
            let source_channels = SourceChannelCount::new(1).expect("one source channel is valid");
            let token = admission
                .admit_ws(
                    WsPermitCount::from_source_channels(source_channels)
                        .expect("one channel maps to a permit"),
                )
                .expect("the test session starts while running");
            let mut phase_rx = admission.subscribe();
            assert_eq!(*phase_rx.borrow(), AdmissionPhase::Running);
            let metrics = Metrics::new(1, 1);
            let inbound = match drain_before_receive {
                true => Ok(None),
                false => Ok(Some(InboundFrame::Text("{\"type\":\"end\"}".to_owned()))),
            };
            let mut effects = RecorderEffects::with_inbound([inbound]);
            let terminal = ActiveTerminalContext::open(
                SampleRate::new(16000).expect("the controlled source sample rate is valid"),
                source_channels,
                &mut effects,
            );
            if drain_before_receive {
                admission.begin_draining();
            }
            let finishes = Arc::new(AtomicU64::new(0));
            let lifetime = SessionLifetime::new(
                token,
                TerminalSession::new(Arc::clone(&finishes), TerminalResult::Success),
            );
            run_active(
                &mut effects,
                &metrics,
                &lifetime,
                source_channels,
                &mut phase_rx,
                terminal,
            )
            .await;
            assert_eq!(finishes.load(Ordering::Relaxed), 1);
            assert_eq!(
                effects.history,
                vec![
                    "ws_open".to_owned(),
                    "ws_close".to_owned(),
                    "text:{\"type\":\"end\"}".to_owned(),
                    "close".to_owned(),
                ]
            );
            drop(lifetime);
            assert_eq!(admission.available_ws(), 1);
        }
    }

    #[tokio::test]
    async fn outer_error_delivery_records_one_primary_failure_without_recursion() {
        let metrics = Metrics::new(1, 1);
        let mut effects = RecorderEffects {
            fail_text: Some("controlled primary delivery failure".to_owned()),
            ..RecorderEffects::default()
        };
        report_session_failure(
            &mut effects,
            &metrics,
            SessionFailure::new("stream", "primary stream failure"),
            FailureDelivery::NotifyPeer,
        )
        .await;
        assert_eq!(metric_counter(&metrics, "asr_ws_errors_total"), 1);
        assert_eq!(effects.access_kinds(), vec!["ws_error", "ws_error"]);
        assert_eq!(
            effects.history,
            vec!["ws_error".to_owned(), "ws_error".to_owned()]
        );
        assert_eq!(effects.closes, 0);
    }

    #[test]
    fn committed_patch_observation_precedes_serialization_and_delivery_failure() {
        let metrics = Metrics::new(1, 1);
        let observation = TranscriptionObservation::dialog_patch_generated();
        let messages = prepare_messages_for_delivery(&metrics, &[observation], || {
            assert_eq!(
                metric_counter(&metrics, "asr_ws_turn_patches_total"),
                1,
                "metric projection must precede JSON serialization"
            );
            vec!["{\"type\":\"turns\"}".to_owned()]
        });
        assert_eq!(messages, vec!["{\"type\":\"turns\"}".to_owned()]);
        assert_eq!(metric_counter(&metrics, "asr_ws_turn_patches_total"), 1);
    }

    #[test]
    fn selection_observation_projects_the_exact_collapsed_source_count() -> Result<(), String> {
        let first = [-1.0, 0.0, 1.0];
        let second = [-1.0, 0.0, 1.0];
        let channels = [
            ChannelAudioView::new(&first)?,
            ChannelAudioView::new(&second)?,
        ];
        let selection = select_pairwise_channels(
            &channels,
            PairwiseChannelPolicy::Deduplicate(CorrelationThreshold::new(0.98)?),
        )?;
        let observation = TranscriptionObservation::selection_committed(
            SourceChannelCount::new(channels.len())?,
            selection,
        )?;
        let metrics = Metrics::new(1, 1);
        project_observations(&metrics, &[observation]);
        assert_eq!(metric_counter(&metrics, "asr_dedup_collapsed_total"), 1);
        Ok(())
    }

    static NEXT_SERIALIZER_FRONTEND: AtomicU64 = AtomicU64::new(0);

    /// Test-owned frontend inputs reuse the public package schema and are removed at scope exit.
    struct SerializerFrontendPackage {
        root: PathBuf,
    }

    impl SerializerFrontendPackage {
        fn new() -> Result<Self, String> {
            let sequence = NEXT_SERIALIZER_FRONTEND.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "gigaam-v3-runtime-ws-serializer-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&root)
                .map_err(|error| format!("create test frontend package: {error}"))?;
            let package = Self { root };
            fs::write(
                package.root.join("config.kv"),
                include_str!("../../../model/config.kv"),
            )
            .map_err(|error| format!("write test frontend configuration: {error}"))?;
            Ok(package)
        }

        fn path(&self) -> &Path {
            &self.root
        }

        fn write_f32_asset(
            &self,
            name: &str,
            dimensions: &[usize],
            values: &[f32],
        ) -> Result<(), String> {
            let expected = dimensions.iter().try_fold(1_usize, |total, dimension| {
                total
                    .checked_mul(*dimension)
                    .ok_or_else(|| "test frontend dimensions overflow usize".to_owned())
            })?;
            if values.len() != expected {
                return Err(format!(
                    "test frontend asset {name} has {} values, expected {expected}",
                    values.len()
                ));
            }
            let dimension_count = u32::try_from(dimensions.len())
                .map_err(|_| "test frontend dimension count exceeds u32".to_owned())?;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&dimension_count.to_le_bytes());
            for dimension in dimensions {
                let dimension = u32::try_from(*dimension)
                    .map_err(|_| "test frontend dimension exceeds u32".to_owned())?;
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            fs::write(self.root.join(name), bytes)
                .map_err(|error| format!("write test frontend asset {name}: {error}"))
        }
    }

    impl Drop for SerializerFrontendPackage {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.root) {
                panic!(
                    "test-owned serializer frontend cleanup must succeed for {}: {error}",
                    self.root.display()
                );
            }
        }
    }

    struct SerializerFrontend {
        processor: Arc<FrontendProcessor>,
        sample_rate: gigaam_audio::SampleRate,
    }

    fn test_weight_values(length: usize, asset: &str) -> Result<Vec<f32>, String> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|_| format!("reserve test frontend {asset} values"))?;
        values.resize(length, 1.0);
        Ok(values)
    }

    fn serializer_frontend() -> Result<SerializerFrontend, String> {
        let package = SerializerFrontendPackage::new()?;
        let model = ModelPackage::open(package.path())
            .map_err(|error| format!("open test frontend package: {error}"))?;
        let definition = model.frontend();
        let filterbank_bins = definition
            .n_fft()
            .checked_div(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "test frontend filterbank frequency bins overflow usize".to_owned())?;
        let filterbank_values = filterbank_bins
            .checked_mul(definition.n_mels())
            .ok_or_else(|| "test frontend filterbank value count overflows usize".to_owned())?;
        let window = test_weight_values(definition.n_fft(), "window")?;
        let filterbank = test_weight_values(filterbank_values, "filterbank")?;
        package.write_f32_asset("stft_window.f32", &[definition.n_fft()], &window)?;
        package.write_f32_asset(
            "mel_fbank.f32",
            &[filterbank_bins, definition.n_mels()],
            &filterbank,
        )?;
        let weights = model
            .frontend_weights()
            .map_err(|error| format!("load test frontend weights: {error}"))?;
        Ok(SerializerFrontend {
            processor: Arc::new(FrontendProcessor::new(
                definition,
                weights,
                FrontendMode::Scalar,
            )?),
            sample_rate: gigaam_audio::SampleRate::from_usize(
                definition.sample_rate(),
                "test serializer frontend",
            )?,
        })
    }

    struct SerializerDecoder {
        words: Vec<Word>,
    }

    impl WindowDecoder for SerializerDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(100.0).expect("the fixed serializer decoder frame rate is positive")
        }

        fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            Decoded::new(
                self.words.clone(),
                vec![false; features.frames()],
                features.frames(),
                0.0,
            )
        }
    }

    struct BlankDetector;

    impl SpeechProbabilityDetector for BlankDetector {
        fn probabilities(&mut self, _audio: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
            Err("blank-endpoint serializer sessions never invoke VAD".into())
        }
    }

    struct SerializerFactory {
        frontend: Arc<FrontendProcessor>,
        sample_rate: gigaam_audio::SampleRate,
        words: Vec<Vec<Word>>,
    }

    impl SerializerFactory {
        fn new() -> Result<Self, String> {
            let frontend = serializer_frontend()?;
            Ok(Self {
                frontend: frontend.processor,
                sample_rate: frontend.sample_rate,
                words: vec![
                    vec![Word::new("left".into(), 0.01, 0.10)?],
                    vec![Word::new("right".into(), 0.12, 0.20)?],
                ],
            })
        }
    }

    impl StreamChannelFactory for SerializerFactory {
        type Decoder = SerializerDecoder;
        type Detector = BlankDetector;

        fn create_stream(
            &mut self,
            channel: OriginalChannel,
            config: StreamConfig,
        ) -> Result<StreamSetup<Self::Decoder, Self::Detector>, String> {
            let words = self
                .words
                .get(channel.index())
                .cloned()
                .ok_or_else(|| "serializer factory lacks a channel word plan".to_owned())?;
            let control = ExecutionControl::without_deadline();
            Ok(StreamSetup {
                frontend: Arc::clone(&self.frontend),
                decoder: SerializerDecoder { words },
                config,
                detector: EndpointDetector::Blank,
                control,
            })
        }
    }

    fn serializer_stream_config(sample_rate: SampleRate) -> Result<StreamConfig, String> {
        StreamConfig::timing_changes()
            .with_window_sec(1.0)?
            .with_overlap_sec(0.0)?
            .with_step_sec(1.0)?
            .with_horizon_sec(0.001)?
            .with_seam_after_sec(0.0)?
            .with_keep_silence_sec(0.0)?
            .apply(StreamConfig::checked_default(sample_rate)?)
    }

    fn serializer_setup(
        source_channels: SourceChannelCount,
        sample_rate: gigaam_audio::SampleRate,
        emission_mode: StreamEmissionMode,
    ) -> Result<MultiChannelStreamSetup, String> {
        MultiChannelStreamSetup::new(MultiChannelStreamSetupInput {
            sample_format: SampleFormat::Pcm16,
            source_sample_rate: sample_rate,
            source_channels,
            stream_config: serializer_stream_config(sample_rate)?,
            options: MultiChannelStreamOptions::new(
                StreamingChannelPolicy::AllChannels,
                emission_mode,
                TurnGap::new(0.5)?,
                BackchannelPolicy::Disabled,
            ),
        })
    }

    fn pcm16_interleaved_silence(
        source_channels: usize,
        frames_per_channel: usize,
    ) -> Result<Vec<u8>, String> {
        let samples = frames_per_channel
            .checked_mul(source_channels)
            .ok_or_else(|| "serializer test sample count overflows usize".to_owned())?;
        let capacity = samples
            .checked_mul(2)
            .ok_or_else(|| "serializer test byte count overflows usize".to_owned())?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| "reserve serializer test PCM16 bytes".to_owned())?;
        for _ in 0..samples {
            bytes.extend_from_slice(&0_i16.to_le_bytes());
        }
        Ok(bytes)
    }

    fn serializer_session(
        source_channels: SourceChannelCount,
        emission_mode: StreamEmissionMode,
    ) -> Result<
        (
            MultiChannelSession<SerializerDecoder, BlankDetector>,
            gigaam_audio::SampleRate,
        ),
        String,
    > {
        let mut factory = SerializerFactory::new()?;
        let sample_rate = factory.sample_rate;
        let session = MultiChannelSession::new(
            serializer_setup(source_channels, sample_rate, emission_mode)?,
            &mut factory,
        )?;
        Ok((session, sample_rate))
    }

    fn serializer_step(
        source_channels: SourceChannelCount,
        emission_mode: StreamEmissionMode,
    ) -> Result<gigaam_transcription::MultiChannelStep, String> {
        let (mut session, sample_rate) = serializer_session(source_channels, emission_mode)?;
        let frames_per_channel = sample_rate.as_usize()?;
        session
            .push(&pcm16_interleaved_silence(
                source_channels.get(),
                frames_per_channel,
            )?)
            .map_err(|failure| failure.error().to_owned())
    }

    fn expected_stereo_word_messages() -> Vec<String> {
        vec![
            "{\"type\":\"words\",\"at\":1,\"revise_from\":0,\"words\":[{\"start\":0.01,\"end\":0.1,\"text\":\"left\",\"stable\":true}],\"channel\":0}".to_owned(),
            "{\"type\":\"stable\",\"at\":1,\"upto\":1,\"channel\":0}".to_owned(),
            "{\"type\":\"words\",\"at\":1,\"revise_from\":0,\"words\":[{\"start\":0.12,\"end\":0.2,\"text\":\"right\",\"stable\":true}],\"channel\":1}".to_owned(),
            "{\"type\":\"stable\",\"at\":1,\"upto\":1,\"channel\":1}".to_owned(),
        ]
    }

    fn expected_stereo_turns_message() -> String {
        "{\"type\":\"turns\",\"revise_from\":0,\"frontier\":-0.06,\"turns\":[{\"channel\":0,\"k\":0,\"start\":0.01,\"end\":0.1,\"text\":\"left\",\"stable\":true,\"final\":false,\"backchannel\":false},{\"channel\":1,\"k\":0,\"start\":0.12,\"end\":0.2,\"text\":\"right\",\"stable\":true,\"final\":false,\"backchannel\":false}]}".to_owned()
    }

    #[test]
    fn stream_emit_queries_select_the_exact_permitted_stereo_wire_groups() -> Result<(), String> {
        for (emit, expected_mode) in [
            ("words", StreamEmissionMode::Words),
            ("turns", StreamEmissionMode::Dialog),
            ("both", StreamEmissionMode::WordsAndDialog),
        ] {
            let raw_query = format!("rate=16000&channels=2&dedup=false&emit={emit}");
            let query = StreamQuery::parse(Some(&raw_query), defaults())?;
            let source_channels = SourceChannelCount::new(query.channels)?;
            let messages = crate::stream_response::serialize_step(
                &serializer_step(source_channels, query.emit.stream_emission_mode())?,
                source_channels,
            );
            let expected = match expected_mode {
                StreamEmissionMode::Words => expected_stereo_word_messages(),
                StreamEmissionMode::Dialog => vec![expected_stereo_turns_message()],
                StreamEmissionMode::WordsAndDialog => {
                    let mut messages = expected_stereo_word_messages();
                    messages.push(expected_stereo_turns_message());
                    messages
                }
            };
            assert_eq!(
                messages, expected,
                "emit={emit} must retain ordered wire groups"
            );
        }
        Ok(())
    }

    #[test]
    fn stream_serialization_preserves_stereo_order_and_mono_channel_omission() -> Result<(), String>
    {
        let stereo_sources = SourceChannelCount::new(2)?;
        let stereo = crate::stream_response::serialize_step(
            &serializer_step(stereo_sources, StreamEmissionMode::WordsAndDialog)?,
            stereo_sources,
        );
        let mut expected_stereo = expected_stereo_word_messages();
        expected_stereo.push(expected_stereo_turns_message());
        assert_eq!(stereo, expected_stereo);

        let mono_sources = SourceChannelCount::new(1)?;
        let mono = crate::stream_response::serialize_step(
            &serializer_step(mono_sources, StreamEmissionMode::WordsAndDialog)?,
            mono_sources,
        );
        assert_eq!(
            mono,
            vec![
                "{\"type\":\"words\",\"at\":1,\"revise_from\":0,\"words\":[{\"start\":0.01,\"end\":0.1,\"text\":\"left\",\"stable\":true}]}",
                "{\"type\":\"stable\",\"at\":1,\"upto\":1}",
                "{\"type\":\"turns\",\"revise_from\":0,\"frontier\":-0.06,\"turns\":[{\"channel\":0,\"k\":0,\"start\":0.01,\"end\":0.1,\"text\":\"left\",\"stable\":true,\"final\":false,\"backchannel\":false}]}",
            ]
        );
        Ok(())
    }
}
