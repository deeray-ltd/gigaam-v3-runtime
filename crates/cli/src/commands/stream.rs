// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Streaming-simulator command adapter with direct local Recognition composition.

use crate::composition;
use crate::configuration;
use crate::grammar::StreamInvocation;
use crate::numeric::{count_to_f32, stream_chunk};
use crate::projection;
use crate::projection::stream::{StreamProjection, StreamSummary};
use gigaam_audio::{ChannelAudio, load};
use gigaam_recognition::{ExecutionControl, RequiredEncoderRoles};
use gigaam_transcription::{
    BatchConfig, BatchSetup, BatchTranscriber, EndpointDetector, EndpointRules, PadPolicy,
    StreamConfig, StreamSession, StreamSetup,
};
use std::time::Instant;

/// Executes the direct stream simulator and preserves incremental client projection.
pub(crate) fn run(invocation: StreamInvocation) -> Result<(), String> {
    let runtime = configuration::runtime(&invocation.runtime, RequiredEncoderRoles::ctc())
        .map_err(|error| format!("runtime configuration: {error}"))?;
    let reference = match &invocation.reference {
        Some(path) => {
            Some(std::fs::read_to_string(path).map_err(|error| format!("--ref: {error}"))?)
        }
        None => None,
    };
    let trace_mode = configuration::trace_mode().map_err(|error| format!("ASR_TRACE: {error}"))?;
    let frontend_mode =
        configuration::frontend_mode().map_err(|error| format!("frontend: {error}"))?;
    let observations = projection::observations(trace_mode);

    let package = composition::open_package(&invocation.model)?;
    let sample_rate = composition::package_sample_rate(&package)?;
    let timing_changes = StreamConfig::timing_changes()
        .with_step_sec(invocation.step_seconds)
        .and_then(|changes| changes.with_horizon_sec(invocation.horizon_seconds))
        .and_then(|changes| changes.with_window_sec(invocation.window_seconds))
        .and_then(|changes| changes.with_overlap_sec(invocation.overlap_seconds))
        .map_err(|error| format!("stream configuration: {error}"))?;
    let base_config = StreamConfig::checked_default(sample_rate)
        .map_err(|error| format!("stream configuration: {error}"))?;
    let mut config = timing_changes
        .apply(base_config)
        .map_err(|error| format!("stream configuration: {error}"))?
        .with_lock_policy(invocation.lock_policy)
        .map_err(|error| format!("stream configuration: {error}"))?
        .with_endpoint_source(invocation.endpoint)
        .map_err(|error| format!("stream configuration: {error}"))?;
    if let Some(milliseconds) = invocation.silence_milliseconds {
        let rules = EndpointRules::new(
            milliseconds / 1_000.0,
            config.rules().without_speech_seconds(),
        )
        .map_err(|error| format!("--silence-ms: {error}"))?;
        config = config
            .with_rules(rules)
            .map_err(|error| format!("stream configuration: {error}"))?;
    }
    let batch_config = BatchConfig::new(
        config.sample_rate(),
        config.window_sec(),
        config.overlap_sec(),
        PadPolicy::PadToWindow,
    )
    .map_err(|error| format!("batch configuration: {error}"))?;
    let chunk = stream_chunk(invocation.chunk_milliseconds, config.sample_rate())?;

    composition::initialize_runtime(&runtime)?;
    let frontend = composition::frontend_for_package(&package, frontend_mode)
        .map_err(|error| format!("frontend: {error}"))?;
    let recognizer = composition::ctc_recognizer(&package, &runtime, invocation.precision)
        .map_err(|error| format!("decoder: {error}"))?;
    projection::emit_ctc_construction_notice(&recognizer);
    projection::emit_unverified_cuda_assignment(
        runtime.plan(),
        gigaam_recognition::EncoderRole::Ctc,
        recognizer.assignment_evidence(),
    );
    let loaded = load(&invocation.input).map_err(|error| format!("audio: {error}"))?;
    let audio = composition::resample_to(loaded, config.sample_rate())
        .map_err(|error| format!("resample audio: {error}"))?;
    let channel = first_channel(&audio)?;
    let model_rate = config
        .sample_rate()
        .as_usize()
        .map_err(|error| format!("stream sample rate: {error}"))?;
    let duration = count_to_f32(channel.len()) / count_to_f32(model_rate);
    let mut batch_transcriber = BatchTranscriber::new(BatchSetup {
        frontend: frontend.clone(),
        decoder: recognizer,
        config: batch_config,
        control: ExecutionControl::without_deadline(),
        observations,
    })
    .map_err(|error| format!("batch transcriber: {error}"))?;
    batch_transcriber
        .warmup()
        .map_err(|error| format!("batch warmup: {error}"))?;
    let batch = batch_transcriber
        .transcribe_channel(channel)
        .map_err(|error| format!("batch transcription: {error}"))?;
    let decoder = batch_transcriber.into_decoder();
    let detector = match config.endpoint_source() {
        gigaam_transcription::EndpointSource::Blank => EndpointDetector::Blank,
        gigaam_transcription::EndpointSource::Vad => EndpointDetector::Vad(
            composition::vad(&package, runtime.plan().intra_threads())
                .map_err(|error| format!("vad: {error}"))?,
        ),
    };
    let mut session = StreamSession::new(StreamSetup {
        frontend,
        decoder,
        config: config.clone(),
        detector,
        control: ExecutionControl::without_deadline(),
    })
    .map_err(|error| format!("stream session: {error}"))?;
    session
        .warmup()
        .map_err(|error| format!("stream warmup: {error}"))?;

    let mut client = Vec::new();
    let mut projection_state = StreamProjection::new(invocation.events);
    let started = Instant::now();
    let mut offset = 0usize;
    while offset < channel.len() {
        let end = offset
            .checked_add(chunk)
            .ok_or_else(|| "stream chunk endpoint overflows usize".to_owned())?
            .min(channel.len());
        let input = ChannelAudio::new(channel.samples()[offset..end].to_vec())
            .map_err(|error| format!("stream input: {error}"))?;
        let events = session
            .push(&input)
            .map_err(|error| format!("stream input: {error}"))?;
        projection_state.apply(events, &mut client)?;
        if client.as_slice() != session.transcript() {
            return Err("stream client transcript diverged from session".into());
        }
        offset = end;
    }
    let events = session
        .flush()
        .map_err(|error| format!("stream flush: {error}"))?;
    projection_state.apply(events, &mut client)?;
    if client.as_slice() != session.transcript() {
        return Err("stream client transcript diverged from session".into());
    }
    let wall_seconds = started.elapsed().as_secs_f32();
    projection_state.finish(StreamSummary {
        client: &client,
        batch_text: batch.text(),
        config: &config,
        chunk_milliseconds: invocation.chunk_milliseconds,
        precision: invocation.precision,
        decoder_seconds: session.decoder_seconds(),
        encoder_seconds: session.encoder_seconds(),
        decodes: session.decodes(),
        audio_seconds: duration,
        wall_seconds,
        reference: reference.as_deref(),
    })
}

fn first_channel(audio: &gigaam_audio::DecodedAudio) -> Result<&ChannelAudio, String> {
    audio
        .channels()
        .first()
        .ok_or_else(|| "stream audio must contain at least one channel".to_owned())
}
