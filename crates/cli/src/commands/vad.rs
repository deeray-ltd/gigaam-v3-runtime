// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Fixed-16-kHz VAD command adapter.

use crate::composition;
use crate::configuration;
use crate::grammar::VadInvocation;
use crate::numeric::{count_to_f32, usize_to_u32};
use crate::projection::{self, VadSummary};
use gigaam_audio::{SampleRate, load};
use gigaam_recognition::{RequiredEncoderRoles, vad::VAD_SR};
use gigaam_transcription::{
    VadDurations, VadPadding, VadPolicyConfig, VadProbability, VadThresholds, speech_segments,
};
use std::time::Instant;

/// Executes direct fixed-domain VAD with package binding before native construction.
pub(crate) fn run(invocation: VadInvocation) -> Result<(), String> {
    let runtime = configuration::runtime(&invocation.runtime, RequiredEncoderRoles::none())
        .map_err(|error| format!("runtime configuration: {error}"))?;
    let speech_threshold = VadProbability::new(invocation.speech_threshold)
        .map_err(|error| format!("VAD speech threshold: {error}"))?;
    let silence_threshold = VadProbability::new(invocation.silence_threshold)
        .map_err(|error| format!("VAD silence threshold: {error}"))?;
    let thresholds = VadThresholds::new(speech_threshold, silence_threshold)
        .map_err(|error| format!("VAD thresholds: {error}"))?;
    let durations = VadDurations::new(
        count_to_f32(invocation.minimum_speech_milliseconds),
        count_to_f32(invocation.minimum_silence_milliseconds),
    )
    .map_err(|error| format!("VAD durations: {error}"))?;
    let padding = VadPadding::new(count_to_f32(invocation.speech_padding_milliseconds))
        .map_err(|error| format!("VAD padding: {error}"))?;
    let config = VadPolicyConfig::new(thresholds, durations, padding);

    let package = composition::open_package(&invocation.model)?;
    composition::initialize_runtime(&runtime)?;
    let mut detector = composition::vad(&package, runtime.plan().intra_threads())
        .map_err(|error| format!("vad: {error}"))?;
    let loaded = load(&invocation.input)?;
    let target_rate = SampleRate::new(usize_to_u32(VAD_SR, "VAD sample rate")?)?;
    let audio = composition::resample_to(loaded, target_rate)
        .map_err(|error| format!("resample: {error}"))?;
    let channel = audio
        .channels()
        .first()
        .ok_or_else(|| "VAD audio must contain at least one channel".to_owned())?;
    let started = Instant::now();
    let segments = speech_segments(&mut detector, channel.view(), config)?;
    projection::vad(VadSummary {
        segments: &segments,
        sample_rate: VAD_SR,
        total_samples: channel.len(),
        elapsed: started.elapsed(),
    })
}
