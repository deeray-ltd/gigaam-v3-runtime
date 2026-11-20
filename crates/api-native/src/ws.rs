// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

// WebSocket `/v1/stream` adaptation, pre-upgrade admission, and terminal ownership.

use crate::stream_response::{applied_config_event, end_event, error_event, serialize_step};
use axum::extract::ws::{CloseCode, CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{RawQuery, State};
use axum::http::StatusCode;
use axum::response::Response;
use gigaam_audio::{RatePair, SampleFormat, SampleRate};
use gigaam_service_host::live::{LiveEvent, LiveTelemetry};
use gigaam_service_host::wire::{QueryParameters, query_bool, query_finite_f32};
use gigaam_transcription::{
    EndpointRules, EndpointSource, MultiChannelStep, SilenceSkipPolicy, SourceChannelCount,
    StreamConfig, StreamEmissionMode, StreamLockPolicy, VadProbability,
};
use std::time::Instant;

const MAX_CHANNELS: usize = 8;

/// Pins both the per-message and per-frame upper bound as part of this protocol's contract,
/// instead of inheriting the underlying library defaults (which currently allow a 64 MiB
/// message assembled from 16 MiB frames). A message or frame above this bound refuses and ends
/// the session.
const WS_MESSAGE_LIMIT_BYTES: usize = 16 * 1024 * 1024;

/// Narrow state for the streaming protocol adapter.
#[derive(Clone)]
pub(crate) struct WsState {
    live: gigaam_service_host::live::LiveService,
    telemetry: LiveTelemetry,
}

impl WsState {
    pub(crate) fn new(live: gigaam_service_host::live::LiveService) -> Self {
        let telemetry = live.telemetry();
        Self { live, telemetry }
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
    applied_config: bool,
}

#[derive(Clone, Debug)]
struct StreamDefaults {
    stream_base: StreamConfig,
    dedup_default: bool,
    backchannel_max_sec: f32,
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
            "vad_threshold",
            "silence_skip",
            "skip_vad_threshold",
            "silence_ms",
            "no_speech_ms",
            "window",
            "overlap",
            "step_ms",
            "horizon",
            "lock",
            "seam_after",
            "keep_silence",
            "applied_config",
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
        let base = defaults.stream_base;
        let rules = EndpointRules::new(
            query_milliseconds_as_seconds(
                &parameters,
                "silence_ms",
                base.rules().after_speech_seconds(),
            )?,
            query_milliseconds_as_seconds(
                &parameters,
                "no_speech_ms",
                base.rules().without_speech_seconds(),
            )?,
        )?;
        let config = StreamConfig::timing_changes()
            .with_window_sec(query_finite_f32(&parameters, "window", base.window_sec())?)?
            .with_overlap_sec(query_finite_f32(
                &parameters,
                "overlap",
                base.overlap_sec(),
            )?)?
            .with_step_sec(query_milliseconds_as_seconds(
                &parameters,
                "step_ms",
                base.step_sec(),
            )?)?
            .with_horizon_sec(positive_f32(
                query_finite_f32(&parameters, "horizon", 5.0)?,
                "horizon",
            )?)?
            .with_seam_after_sec(query_finite_f32(
                &parameters,
                "seam_after",
                base.seam_after_sec(),
            )?)?
            .with_keep_silence_sec(query_finite_f32(
                &parameters,
                "keep_silence",
                base.keep_silence_sec(),
            )?)?
            .apply(base)?
            .with_rules(rules)?;
        let config = config.with_lock_policy(match query_bool(&parameters, "lock", false)? {
            true => StreamLockPolicy::CommitStable,
            false => StreamLockPolicy::Advisory,
        })?;
        let endpoint = match parameters.get("endpoint") {
            None | Some("blank") => EndpointSource::Blank,
            Some("vad") => EndpointSource::Vad,
            Some(value) => return Err(format!("endpoint must be blank or vad, got {value:?}")),
        };
        let mut config = config.with_endpoint_source(endpoint)?;
        if parameters.get("vad_threshold").is_some() {
            if endpoint != EndpointSource::Vad {
                return Err("vad_threshold requires endpoint=vad".into());
            }
            let threshold = VadProbability::new(query_finite_f32(
                &parameters,
                "vad_threshold",
                config.vad_threshold().get(),
            )?)?;
            config = config.with_vad_threshold(threshold)?;
        }
        let skip_enabled = query_bool(&parameters, "silence_skip", false)?;
        if !skip_enabled && parameters.get("skip_vad_threshold").is_some() {
            return Err("skip_vad_threshold requires silence_skip=true".into());
        }
        if skip_enabled {
            let threshold = VadProbability::new(query_finite_f32(
                &parameters,
                "skip_vad_threshold",
                VadProbability::DEFAULT.get(),
            )?)?;
            config = config.with_silence_skip(SilenceSkipPolicy::AcousticWindow { threshold })?;
        }
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
            applied_config: query_bool(&parameters, "applied_config", false)?,
        })
    }
}

fn query_milliseconds_as_seconds(
    parameters: &QueryParameters,
    key: &str,
    default_seconds: f32,
) -> Result<f32, String> {
    match parameters.get(key) {
        Some(_) => {
            let milliseconds = query_finite_f32(parameters, key, 0.0)?;
            if milliseconds < 0.0 {
                return Err(format!("{key} must be nonnegative"));
            }
            Ok(milliseconds / 1_000.0)
        }
        None => Ok(default_seconds),
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

trait WsMetrics: Send + Sync {
    fn ws_open(&self);
    fn ws_reject(&self);
    fn ws_frame(&self);
    fn ws_error(&self);
}
impl WsMetrics for LiveTelemetry {
    fn ws_open(&self) {
        self.open();
    }
    fn ws_reject(&self) {
        self.reject();
    }
    fn ws_frame(&self) {
        self.frame();
    }
    fn ws_error(&self) {
        self.error();
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
    fn emit_access(&mut self, event: LiveEvent);
}

/// Real Axum socket effects used by the production callback path.
struct AxumEffects {
    socket: WebSocket,
    telemetry: LiveTelemetry,
}

impl AxumEffects {
    fn new(socket: WebSocket, telemetry: LiveTelemetry) -> Self {
        Self { socket, telemetry }
    }
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

    fn emit_access(&mut self, event: LiveEvent) {
        self.telemetry.emit(event);
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
        let request_id = LiveTelemetry::request_id();
        effects.emit_access(LiveEvent::Open {
            request_id: request_id.clone(),
            channels: source_channels.get(),
            rate: source_sample_rate.hertz(),
        });
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
    fn close_event(self) -> LiveEvent {
        LiveEvent::Close {
            request_id: self.request_id,
            channels: self.channels.get(),
            frames: self.frames,
            milliseconds: self.opened_at.elapsed().as_millis(),
        }
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

impl SessionFailure {
    fn new(phase: &'static str, message: impl Into<String>) -> Self {
        Self {
            phase,
            message: message.into(),
        }
    }
}

trait SessionOperations: Send + Sync {
    type Metrics: WsMetrics;
    async fn push_operation(
        &self,
        bytes: Vec<u8>,
        metrics: &Self::Metrics,
    ) -> Result<MultiChannelStep, SessionFailure>;
    async fn finish_operation(
        &self,
        metrics: &Self::Metrics,
    ) -> Result<MultiChannelStep, SessionFailure>;
    fn project_step(&self, metrics: &Self::Metrics, step: &MultiChannelStep);
}

impl SessionOperations for gigaam_service_host::live::LiveSession {
    type Metrics = LiveTelemetry;
    async fn push_operation(
        &self,
        bytes: Vec<u8>,
        _: &Self::Metrics,
    ) -> Result<MultiChannelStep, SessionFailure> {
        self.push(bytes)
            .await
            .map_err(|error| SessionFailure::new("stream", error))
    }
    async fn finish_operation(
        &self,
        _: &Self::Metrics,
    ) -> Result<MultiChannelStep, SessionFailure> {
        self.finish()
            .await
            .map_err(|error| SessionFailure::new("stream", error))
    }
    fn project_step(&self, _: &Self::Metrics, _: &MultiChannelStep) {}
}

trait DrainWatch: Send {
    async fn wait(&mut self) -> Result<(), String>;
}
impl DrainWatch for gigaam_service_host::live::LiveDrain {
    async fn wait(&mut self) -> Result<(), String> {
        self.wait_for_drain().await
    }
}

pub(crate) async fn stream(
    State(state): State<WsState>,
    RawQuery(query_text): RawQuery,
    upgrade: WebSocketUpgrade,
) -> Response {
    let service = &state.live;
    if !service.supports_stream() {
        state.telemetry.ws_reject();
        return crate::stream_response::pre_upgrade_refusal(
            StatusCode::BAD_REQUEST,
            "stream requires CTC",
        );
    }
    let defaults = service.defaults();
    let query = match StreamQuery::parse(
        query_text.as_deref(),
        StreamDefaults {
            stream_base: defaults.stream_base,
            dedup_default: defaults.dedup_default,
            backchannel_max_sec: defaults.backchannel_max_sec,
        },
    ) {
        Ok(query) => query,
        Err(error) => {
            state.telemetry.ws_reject();
            return crate::stream_response::pre_upgrade_refusal(StatusCode::BAD_REQUEST, error);
        }
    };
    let applied_config = query.applied_config;
    let setup = match service.validate(gigaam_service_host::live::StreamInput {
        rate: query.rate,
        format: query.format,
        channels: query.channels,
        emission: query.emit.stream_emission_mode(),
        config: query.config,
        turn_gap: query.turn_gap,
        backchannel_max_sec: query.backchannel_max_sec,
        dedup: query.dedup,
    }) {
        Ok(setup) => setup,
        Err(error) => {
            state.telemetry.ws_reject();
            return crate::stream_response::pre_upgrade_refusal(StatusCode::BAD_REQUEST, error);
        }
    };
    let source_sample_rate = setup.source_sample_rate();
    let source_channels = setup.source_channels();
    let applied_config = applied_config.then(|| setup.stream_config().clone());
    let reservation = match service.reserve(setup) {
        Ok(reservation) => reservation,
        Err(gigaam_service_host::live::LiveRefusal::Overloaded) => {
            return crate::stream_response::pre_upgrade_refusal(
                StatusCode::SERVICE_UNAVAILABLE,
                ws_refusal_message(
                    &state.telemetry,
                    gigaam_service_host::live::LiveRefusal::Overloaded,
                ),
            );
        }
        Err(gigaam_service_host::live::LiveRefusal::Draining) => {
            return crate::stream_response::pre_upgrade_refusal(
                StatusCode::SERVICE_UNAVAILABLE,
                ws_refusal_message(
                    &state.telemetry,
                    gigaam_service_host::live::LiveRefusal::Draining,
                ),
            );
        }
    };
    upgrade
        .max_message_size(WS_MESSAGE_LIMIT_BYTES)
        .max_frame_size(WS_MESSAGE_LIMIT_BYTES)
        .on_upgrade(move |socket| async move {
            state.telemetry.ws_open();
            let mut effects = AxumEffects::new(socket, state.telemetry.clone());
            let (session, mut drain) = match reservation.start().await {
                Ok(started) => started,
                Err(error) => {
                    report_session_failure(
                        &mut effects,
                        &state.telemetry,
                        SessionFailure::new("stream", error.message()),
                        FailureDelivery::NotifyPeer,
                    )
                    .await;
                    drop(error);
                    return;
                }
            };
            state.telemetry.channels(source_channels.get());
            if drain.is_draining() {
                resolve_terminal(
                    &mut effects,
                    &state.telemetry,
                    &session,
                    source_channels,
                    TerminalPurpose::ShutdownBeforeOpen,
                )
                .await;
            } else {
                let terminal =
                    ActiveTerminalContext::open(source_sample_rate, source_channels, &mut effects);
                run_active(
                    &mut effects,
                    &state.telemetry,
                    &session,
                    source_channels,
                    &mut drain,
                    terminal,
                    applied_config,
                )
                .await;
            }
        })
}

fn ws_refusal_message(
    metrics: &impl WsMetrics,
    refusal: gigaam_service_host::live::LiveRefusal,
) -> &'static str {
    match refusal {
        gigaam_service_host::live::LiveRefusal::Overloaded => {
            // An overload refusal records the established session and rejection history.
            metrics.ws_open();
            metrics.ws_reject();
            "too many active channels, retry later"
        }
        gigaam_service_host::live::LiveRefusal::Draining => "draining",
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
    effects.emit_access(LiveEvent::Error {
        phase,
        message: message.to_owned(),
    });
}

async fn report_session_failure<E: SessionEffects>(
    effects: &mut E,
    metrics: &impl WsMetrics,
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
    metrics: &impl WsMetrics,
    failure: &SessionFailure,
) {
    metrics.ws_error();
    log_ws_error(effects, failure.phase, &failure.message);
}

async fn run_active<S: SessionOperations, E: SessionEffects, D: DrainWatch>(
    effects: &mut E,
    metrics: &S::Metrics,
    lifetime: &S,
    source_channels: SourceChannelCount,
    phase_rx: &mut D,
    mut terminal_context: ActiveTerminalContext,
    mut applied_config: Option<StreamConfig>,
) {
    let end = loop {
        enum Next {
            Shutdown(Result<(), String>),
            AppliedConfig,
            Message(Result<Option<InboundFrame>, ReceiveError>),
        }
        let next = tokio::select! {
            biased;
            changed = phase_rx.wait() => Next::Shutdown(changed),
            () = std::future::ready(()), if applied_config.is_some() => Next::AppliedConfig,
            message = effects.receive() => Next::Message(message),
        };
        match next {
            Next::AppliedConfig => {
                let config = applied_config.take().expect(
                    "the applied-config branch is enabled only while a bound policy is pending",
                );
                if let Err(error) = send_text(effects, applied_config_event(&config)).await {
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
                let step = match lifetime.push_operation(bytes, metrics).await {
                    Ok(step) => step,
                    Err(failure) => {
                        report_session_failure(
                            effects,
                            metrics,
                            failure,
                            FailureDelivery::NotifyPeer,
                        )
                        .await;
                        return;
                    }
                };
                lifetime.project_step(metrics, &step);
                let messages = serialize_step(&step, source_channels);
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

async fn resolve_terminal<S: SessionOperations, E: SessionEffects>(
    effects: &mut E,
    metrics: &S::Metrics,
    lifetime: &S,
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
            match lifetime.finish_operation(metrics).await {
                Ok(step) => {
                    lifetime.project_step(metrics, &step);
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
            let step = match lifetime.finish_operation(metrics).await {
                Ok(step) => step,
                Err(failure) => {
                    report_session_failure(effects, metrics, failure, delivery).await;
                    return;
                }
            };
            lifetime.project_step(metrics, &step);
            let messages = serialize_step(&step, source_channels);
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
