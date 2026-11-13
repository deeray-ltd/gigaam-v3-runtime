// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Fixed-16-kHz VAD command adapter.

use crate::composition;
use crate::configuration;
use crate::grammar::{VadChannels, VadInvocation};
use crate::numeric::usize_to_u32;
use crate::projection::{self, VadSummary};
use gigaam_audio::{DecodedAudio, SampleRate, load};
use gigaam_recognition::{RequiredEncoderRoles, SpeechProbabilityDetector, vad::VAD_SR};
use gigaam_transcription::{SpeechSegment, VadPolicyConfig, speech_segments};
use std::time::{Duration, Instant};

/// Executes direct fixed-domain VAD with package binding before native construction.
pub(crate) fn run(invocation: VadInvocation) -> Result<(), String> {
    let runtime = configuration::runtime(&invocation.runtime, RequiredEncoderRoles::none())
        .map_err(|error| format!("runtime configuration: {error}"))?;
    let config = invocation.policy.bind()?;

    let package = composition::open_package(&invocation.model)?;
    let native = composition::initialize_runtime(&runtime)?;
    let mut detector = composition::vad(&package, &native, runtime.plan().intra_threads())
        .map_err(|error| format!("vad: {error}"))?;
    let loaded = load(&invocation.input)?;
    let target_rate = SampleRate::new(usize_to_u32(VAD_SR, "VAD sample rate")?)?;
    let audio = composition::resample_to(loaded, target_rate)
        .map_err(|error| format!("resample: {error}"))?;
    let results = detect_channels(&audio, &invocation.channels, &mut detector, config)?;
    if invocation.channels.is_explicit() {
        projection::vad_policy(config);
        projection::vad_selection(
            audio.channels().len(),
            results.iter().map(|result| result.index),
        );
    }
    for result in results {
        projection::vad(VadSummary {
            segments: &result.segments,
            sample_rate: VAD_SR,
            total_samples: audio.frames(),
            elapsed: result.elapsed,
            channel: invocation.channels.is_explicit().then_some(result.index),
        })?;
    }
    Ok(())
}

struct ChannelDetection {
    index: usize,
    segments: Vec<SpeechSegment>,
    elapsed: Duration,
}

/// Resolve the complete physical selection before any detector call; project only a complete result.
fn detect_channels<D: SpeechProbabilityDetector>(
    audio: &DecodedAudio,
    selection: &VadChannels,
    detector: &mut D,
    policy: VadPolicyConfig,
) -> Result<Vec<ChannelDetection>, String> {
    let selected = selection.resolve(audio.channels().len())?;
    let mut results = Vec::with_capacity(selected.len());
    for index in selected {
        let channel = audio
            .channels()
            .get(index)
            .ok_or_else(|| "validated VAD selection must belong to the source audio".to_owned())?;
        let started = Instant::now();
        let segments = speech_segments(detector, channel.view(), policy)?;
        results.push(ChannelDetection {
            index,
            segments,
            elapsed: started.elapsed(),
        });
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gigaam_audio::{ChannelAudio, ChannelAudioView};

    struct Detector {
        calls: Vec<f32>,
    }

    impl SpeechProbabilityDetector for Detector {
        fn probabilities(&mut self, audio: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
            let marker = audio.samples()[0];
            self.calls.push(marker);
            let probability = if marker == 0.0 { 0.0 } else { 1.0 };
            Ok(vec![probability; audio.len().div_ceil(512)])
        }
    }

    fn audio() -> DecodedAudio {
        DecodedAudio::new(
            SampleRate::new(16_000).expect("test VAD rate is positive"),
            [0.25, 0.0, 0.25]
                .into_iter()
                .map(|marker| {
                    ChannelAudio::new(vec![marker; 8192]).expect("finite test channel is valid")
                })
                .collect(),
        )
        .expect("three equal nonempty physical channels are valid")
    }

    #[test]
    fn explicit_all_retains_silent_and_duplicate_physical_channels() {
        let mut detector = Detector { calls: Vec::new() };
        let result = detect_channels(
            &audio(),
            &VadChannels::All,
            &mut detector,
            VadPolicyConfig::default(),
        )
        .expect("complete per-channel probabilities produce complete results");
        assert_eq!(detector.calls, vec![0.25, 0.0, 0.25]);
        assert_eq!(
            result
                .iter()
                .map(|channel| channel.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(result[0].segments.len(), 1);
        assert!(result[1].segments.is_empty());
        assert_eq!(result[0].segments, result[2].segments);
    }

    #[test]
    fn subset_is_physical_and_invalid_selection_precedes_every_detector_call() {
        let mut detector = Detector { calls: Vec::new() };
        assert!(
            detect_channels(
                &audio(),
                &VadChannels::Selected(vec![0, 3]),
                &mut detector,
                VadPolicyConfig::default()
            )
            .is_err()
        );
        assert!(detector.calls.is_empty());
        let result = detect_channels(
            &audio(),
            &VadChannels::Selected(vec![1, 2]),
            &mut detector,
            VadPolicyConfig::default(),
        )
        .expect("valid physical subset is preserved");
        assert_eq!(detector.calls, vec![0.0, 0.25]);
        assert_eq!(
            result
                .iter()
                .map(|channel| channel.index)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(result[0].segments.is_empty());
    }
}
