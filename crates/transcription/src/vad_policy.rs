// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Hysteresis, duration, and padding policy over injected speech probabilities.

use crate::contracts::{
    TemporalField, TemporalSampleCount, VadProbability, doubled_temporal_sample_count,
    temporal_sample_count,
};
use gigaam_audio::{ChannelAudioView, SampleRate};
use gigaam_recognition::{SpeechProbabilityDetector, vad::VAD_SR};

const HOP_SAMPLES: usize = 512;

/// The two probability thresholds used by hysteresis segmentation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadThresholds {
    speech: VadProbability,
    silence: VadProbability,
}

impl VadThresholds {
    /// Creates ordered finite probabilities in the closed unit interval.
    pub fn new(speech: VadProbability, silence: VadProbability) -> Result<Self, String> {
        if silence.get() > speech.get() {
            return Err("VAD silence threshold must not exceed the speech threshold".into());
        }
        Ok(Self { speech, silence })
    }

    pub const fn speech(self) -> VadProbability {
        self.speech
    }

    pub const fn silence(self) -> VadProbability {
        self.silence
    }
}

/// The minimum speech and silence durations used by VAD segmentation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadDurations {
    minimum_speech_milliseconds: f32,
    minimum_silence_milliseconds: f32,
    minimum_speech_samples: TemporalSampleCount,
    minimum_silence_samples: TemporalSampleCount,
}

impl VadDurations {
    /// Creates finite nonnegative duration constraints.
    pub fn new(
        minimum_speech_milliseconds: f32,
        minimum_silence_milliseconds: f32,
    ) -> Result<Self, String> {
        let sample_rate = fixed_vad_sample_rate()?;
        let minimum_speech_samples = temporal_sample_count(
            minimum_speech_milliseconds,
            sample_rate,
            TemporalField::VadMinimumSpeech,
        )?;
        let minimum_silence_samples = temporal_sample_count(
            minimum_silence_milliseconds,
            sample_rate,
            TemporalField::VadMinimumSilence,
        )?;
        Ok(Self {
            minimum_speech_milliseconds,
            minimum_silence_milliseconds,
            minimum_speech_samples,
            minimum_silence_samples,
        })
    }

    pub const fn minimum_speech_milliseconds(self) -> f32 {
        self.minimum_speech_milliseconds
    }

    pub const fn minimum_silence_milliseconds(self) -> f32 {
        self.minimum_silence_milliseconds
    }

    const fn minimum_speech_samples(self) -> TemporalSampleCount {
        self.minimum_speech_samples
    }

    const fn minimum_silence_samples(self) -> TemporalSampleCount {
        self.minimum_silence_samples
    }
}

/// The finite nonnegative temporal extension applied to accepted speech segments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadPadding {
    milliseconds: f32,
    one_sided_samples: TemporalSampleCount,
    doubled_samples: TemporalSampleCount,
}

impl VadPadding {
    /// Creates a finite nonnegative speech-segment padding duration.
    pub fn new(milliseconds: f32) -> Result<Self, String> {
        let sample_rate = fixed_vad_sample_rate()?;
        let one_sided_samples =
            temporal_sample_count(milliseconds, sample_rate, TemporalField::VadPadding)?;
        let doubled_samples = doubled_temporal_sample_count(
            one_sided_samples,
            sample_rate,
            TemporalField::VadDoubledPadding,
        )?;
        Ok(Self {
            milliseconds,
            one_sided_samples,
            doubled_samples,
        })
    }

    pub const fn milliseconds(self) -> f32 {
        self.milliseconds
    }

    const fn one_sided_samples(self) -> TemporalSampleCount {
        self.one_sided_samples
    }

    const fn doubled_samples(self) -> TemporalSampleCount {
        self.doubled_samples
    }
}

/// Constructor-validated policy for speech-probability segmentation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadPolicyConfig {
    thresholds: VadThresholds,
    durations: VadDurations,
    padding: VadPadding,
}

impl VadPolicyConfig {
    /// Combines independently validated threshold, duration, and padding values.
    pub const fn new(
        thresholds: VadThresholds,
        durations: VadDurations,
        padding: VadPadding,
    ) -> Self {
        Self {
            thresholds,
            durations,
            padding,
        }
    }

    pub const fn thresholds(self) -> VadThresholds {
        self.thresholds
    }

    pub const fn durations(self) -> VadDurations {
        self.durations
    }

    pub const fn padding(self) -> VadPadding {
        self.padding
    }
}

impl Default for VadPolicyConfig {
    fn default() -> Self {
        Self::new(
            VadThresholds {
                speech: VadProbability::DEFAULT,
                silence: VadProbability::new(0.35)
                    .expect("the fixed 0.35 VAD silence threshold is a valid probability"),
            },
            VadDurations::new(250.0, 100.0)
                .expect("250 ms minimum speech and 100 ms minimum silence are representable at the fixed 16 kHz VAD rate"),
            VadPadding::new(30.0)
                .expect("30 ms one-sided padding is representable at the fixed 16 kHz VAD rate in both one-sided and doubled sample counts"),
        )
    }
}

/// A half-open speech-sample range in the detector input waveform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeechSegment {
    start_sample: usize,
    end_sample: usize,
}

impl SpeechSegment {
    fn new(start_sample: usize, end_sample: usize, input_length: usize) -> Result<Self, String> {
        if start_sample >= end_sample {
            return Err("speech segment must have positive length".into());
        }
        if end_sample > input_length {
            return Err("speech segment exceeds the detector input length".into());
        }
        Ok(Self {
            start_sample,
            end_sample,
        })
    }

    pub const fn start_sample(self) -> usize {
        self.start_sample
    }

    pub const fn end_sample(self) -> usize {
        self.end_sample
    }
}

enum DetectionState {
    Idle,
    Speech {
        start_sample: usize,
        tentative_end_sample: Option<usize>,
    },
}

/// Obtains speech probabilities from the injected detector and applies the configured policy.
pub fn speech_segments<D: SpeechProbabilityDetector>(
    detector: &mut D,
    audio: ChannelAudioView<'_>,
    config: VadPolicyConfig,
) -> Result<Vec<SpeechSegment>, String> {
    let probabilities = detector.probabilities(audio)?;
    segments_from_probabilities(&probabilities, audio.len(), config)
}

fn segments_from_probabilities(
    probabilities: &[f32],
    audio_length: usize,
    config: VadPolicyConfig,
) -> Result<Vec<SpeechSegment>, String> {
    let minimum_speech = config.durations.minimum_speech_samples().get();
    let minimum_silence = config.durations.minimum_silence_samples().get();
    let padding = config.padding.one_sided_samples().get();
    let doubled_padding = config.padding.doubled_samples().get();

    let mut raw = Vec::new();
    let mut state = DetectionState::Idle;
    for (index, &probability) in probabilities.iter().enumerate() {
        validate_probability_value(probability)?;
        let current_sample = index
            .checked_mul(HOP_SAMPLES)
            .ok_or_else(|| "VAD probability frame offset overflows".to_owned())?;
        let mut closure = None;

        match &mut state {
            DetectionState::Idle if probability >= config.thresholds.speech.get() => {
                state = DetectionState::Speech {
                    start_sample: current_sample,
                    tentative_end_sample: None,
                };
            }
            DetectionState::Idle => {}
            DetectionState::Speech {
                tentative_end_sample,
                ..
            } if probability >= config.thresholds.speech.get() => {
                *tentative_end_sample = None;
            }
            DetectionState::Speech {
                start_sample,
                tentative_end_sample,
            } if probability < config.thresholds.silence.get() => {
                let end_sample = tentative_end_sample.get_or_insert(current_sample);
                let silence = current_sample
                    .checked_sub(*end_sample)
                    .ok_or_else(|| "VAD silence interval underflows".to_owned())?;
                if silence >= minimum_silence {
                    closure = Some((*start_sample, *end_sample));
                }
            }
            DetectionState::Speech { .. } => {}
        }

        if let Some((start_sample, end_sample)) = closure {
            let speech = end_sample
                .checked_sub(start_sample)
                .ok_or_else(|| "VAD speech interval underflows".to_owned())?;
            if speech > minimum_speech {
                raw.push(SpeechSegment::new(start_sample, end_sample, audio_length)?);
            }
            state = DetectionState::Idle;
        }
    }

    if let DetectionState::Speech { start_sample, .. } = state {
        let speech = audio_length
            .checked_sub(start_sample)
            .ok_or_else(|| "VAD terminal speech interval underflows".to_owned())?;
        if speech > minimum_speech {
            raw.push(SpeechSegment::new(
                start_sample,
                audio_length,
                audio_length,
            )?);
        }
    }

    apply_padding(raw, padding, doubled_padding, audio_length)
}

fn fixed_vad_sample_rate() -> Result<SampleRate, String> {
    SampleRate::from_usize(VAD_SR, "VAD fixed sample rate")
}

fn apply_padding(
    mut segments: Vec<SpeechSegment>,
    padding: usize,
    doubled_padding: usize,
    audio_length: usize,
) -> Result<Vec<SpeechSegment>, String> {
    for index in 0..segments.len() {
        if index == 0 {
            segments[index].start_sample =
                subtract_and_floor(segments[index].start_sample, padding);
        }

        if index + 1 < segments.len() {
            let silence = segments[index + 1]
                .start_sample
                .checked_sub(segments[index].end_sample)
                .ok_or_else(|| "VAD segments overlap before padding".to_owned())?;
            if silence < doubled_padding {
                let half_silence = silence / 2;
                segments[index].end_sample =
                    add_and_cap(segments[index].end_sample, half_silence, audio_length)?;
                segments[index + 1].start_sample =
                    subtract_and_floor(segments[index + 1].start_sample, half_silence);
            } else {
                segments[index].end_sample =
                    add_and_cap(segments[index].end_sample, padding, audio_length)?;
                segments[index + 1].start_sample =
                    subtract_and_floor(segments[index + 1].start_sample, padding);
            }
        } else {
            segments[index].end_sample =
                add_and_cap(segments[index].end_sample, padding, audio_length)?;
        }
    }

    segments
        .into_iter()
        .map(|segment| SpeechSegment::new(segment.start_sample, segment.end_sample, audio_length))
        .collect()
}

fn subtract_and_floor(value: usize, amount: usize) -> usize {
    value.saturating_sub(amount)
}

fn add_and_cap(value: usize, amount: usize, cap: usize) -> Result<usize, String> {
    if value > cap {
        return Err("speech segment end exceeds the detector input length".into());
    }
    let remaining = cap - value;
    if amount >= remaining {
        Ok(cap)
    } else {
        Ok(value + amount)
    }
}

fn validate_probability_value(value: f32) -> Result<(), String> {
    VadProbability::new(value)
        .map(|_| ())
        .map_err(|_| "speech detector returned a probability outside [0, 1]".into())
}

#[cfg(test)]
mod tests {
    use super::{
        SpeechSegment, TemporalField, VadDurations, VadPadding, VadPolicyConfig, VadThresholds,
        doubled_temporal_sample_count, fixed_vad_sample_rate, speech_segments,
        temporal_sample_count,
    };
    use crate::contracts::VadProbability;
    use gigaam_audio::{ChannelAudio, ChannelAudioView};
    use gigaam_primitives::usize_to_f32;
    use gigaam_recognition::SpeechProbabilityDetector;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FixedDetector {
        expected: ChannelAudio,
        probabilities: Vec<f32>,
    }

    struct CountingDetector {
        expected: ChannelAudio,
        probabilities: Vec<f32>,
        calls: Arc<AtomicUsize>,
    }

    impl SpeechProbabilityDetector for FixedDetector {
        fn probabilities(&mut self, audio: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
            if audio.samples() != self.expected.samples() {
                return Err("test detector received a different waveform".into());
            }
            Ok(self.probabilities.clone())
        }
    }

    impl SpeechProbabilityDetector for CountingDetector {
        fn probabilities(&mut self, audio: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
            if audio.samples() != self.expected.samples() {
                return Err("test detector received a different waveform".into());
            }
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.probabilities.clone())
        }
    }

    fn audio(length: usize) -> ChannelAudio {
        ChannelAudio::new(vec![0.0; length]).expect("test waveform contains only finite samples")
    }

    fn policy(
        speech_threshold: f32,
        silence_threshold: f32,
        minimum_speech_milliseconds: f32,
        minimum_silence_milliseconds: f32,
        padding_milliseconds: f32,
    ) -> VadPolicyConfig {
        VadPolicyConfig::new(
            VadThresholds::new(
                VadProbability::new(speech_threshold)
                    .expect("test speech threshold is a probability"),
                VadProbability::new(silence_threshold)
                    .expect("test silence threshold is a probability"),
            )
            .expect("test thresholds are ordered probabilities"),
            VadDurations::new(minimum_speech_milliseconds, minimum_silence_milliseconds)
                .expect("test durations are finite and nonnegative"),
            VadPadding::new(padding_milliseconds).expect("test padding is finite and nonnegative"),
        )
    }

    fn bind_policy_then<T>(
        minimum_speech_milliseconds: f32,
        padding_milliseconds: f32,
        continuation: impl FnOnce(VadPolicyConfig) -> Result<T, String>,
    ) -> Result<T, String> {
        let thresholds = VadThresholds::new(VadProbability::new(0.5)?, VadProbability::new(0.35)?)?;
        let durations = VadDurations::new(minimum_speech_milliseconds, 0.0)?;
        let padding = VadPadding::new(padding_milliseconds)?;
        continuation(VadPolicyConfig::new(thresholds, durations, padding))
    }

    fn assert_policy_refusal_has_no_detector_or_continuation_effect(
        minimum_speech_milliseconds: f32,
        padding_milliseconds: f32,
        expected_error: &str,
    ) -> Result<(), String> {
        let input = audio(1_536);
        let detector_calls = Arc::new(AtomicUsize::new(0));
        let continuation_calls = Arc::new(AtomicUsize::new(0));
        let mut detector = CountingDetector {
            expected: input.clone(),
            probabilities: vec![0.9, 0.1, 0.1],
            calls: Arc::clone(&detector_calls),
        };
        let result = bind_policy_then(
            minimum_speech_milliseconds,
            padding_milliseconds,
            |config| {
                continuation_calls.fetch_add(1, Ordering::SeqCst);
                speech_segments(&mut detector, input.view(), config).map(|_| ())
            },
        );
        let error = match result {
            Ok(()) => return Err("an overflowing VAD policy must refuse before inference".into()),
            Err(error) => error,
        };
        assert!(error.contains(expected_error));
        assert_eq!(continuation_calls.load(Ordering::SeqCst), 0);
        assert_eq!(detector_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[test]
    fn injected_probabilities_preserve_threshold_hysteresis_and_padding() {
        let input = audio(4_096);
        let mut detector = FixedDetector {
            expected: input.clone(),
            probabilities: vec![0.1, 0.5, 0.4, 0.2, 0.2],
        };
        let segments = speech_segments(
            &mut detector,
            input.view(),
            policy(0.5, 0.35, 0.0, 32.0, 32.0),
        )
        .expect("fixed detector returns probabilities over the declared waveform");
        assert_eq!(
            segments,
            vec![SpeechSegment {
                start_sample: 0,
                end_sample: 2_048,
            }]
        );
    }

    #[test]
    fn minimum_speech_duration_discards_short_detection() {
        let input = audio(4_096);
        let mut detector = FixedDetector {
            expected: input.clone(),
            probabilities: vec![0.1, 0.9, 0.1, 0.1],
        };
        let segments = speech_segments(
            &mut detector,
            input.view(),
            policy(0.5, 0.35, 64.0, 32.0, 0.0),
        )
        .expect("fixed detector returns probabilities over the declared waveform");
        assert!(segments.is_empty());
    }

    #[test]
    fn neighboring_segments_share_an_insufficient_gap_without_overlap() {
        let input = audio(8_192);
        let mut detector = FixedDetector {
            expected: input.clone(),
            probabilities: vec![0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
        };
        let segments = speech_segments(
            &mut detector,
            input.view(),
            policy(0.5, 0.35, 0.0, 32.0, 96.0),
        )
        .expect("fixed detector returns probabilities over the declared waveform");
        assert_eq!(
            segments,
            vec![
                SpeechSegment {
                    start_sample: 0,
                    end_sample: 1_024,
                },
                SpeechSegment {
                    start_sample: 1_024,
                    end_sample: 3_584,
                },
            ]
        );
    }

    #[test]
    fn invalid_policy_and_detector_values_refuse() {
        assert!(VadProbability::new(-0.1).is_err());
        assert!(VadProbability::new(1.1).is_err());
        assert!(
            VadThresholds::new(
                VadProbability::new(0.3).expect("test probability is valid"),
                VadProbability::new(0.4).expect("test probability is valid"),
            )
            .is_err()
        );
        assert!(VadDurations::new(-1.0, 0.0).is_err());
        assert!(VadPadding::new(f32::NAN).is_err());

        let input = audio(512);
        let mut detector = FixedDetector {
            expected: input.clone(),
            probabilities: vec![f32::NAN],
        };
        assert!(speech_segments(&mut detector, input.view(), VadPolicyConfig::default()).is_err());
    }

    #[test]
    fn fixed_vad_temporal_policy_refuses_finite_overflow_before_detector_execution() {
        let rate = fixed_vad_sample_rate().expect("Recognition's fixed VAD rate fits Audio");
        let minimum = VadDurations::new(f32::MAX, 0.0)
            .expect_err("a finite overflowing VAD duration must refuse during construction");
        assert!(minimum.contains("VAD minimum speech duration at 16000 Hz"));
        let one_sided = VadPadding::new(f32::MAX)
            .expect_err("a finite overflowing VAD padding must refuse during construction");
        assert!(one_sided.contains("VAD speech padding at 16000 Hz"));

        let first_count_above_half = (usize::MAX / 2)
            .checked_add(1)
            .expect("half of a nonzero platform capacity has a successor");
        let doubled_only_milliseconds = usize_to_f32(first_count_above_half) / 16.0;
        let one_sided_count =
            temporal_sample_count(doubled_only_milliseconds, rate, TemporalField::VadPadding)
                .expect("the capacity-derived one-sided padding count is representable");
        assert!(
            doubled_temporal_sample_count(one_sided_count, rate, TemporalField::VadDoubledPadding,)
                .is_err()
        );
        let doubled = VadPadding::new(doubled_only_milliseconds)
            .expect_err("doubled padding overflow must refuse before inference can begin");
        assert!(doubled.contains("VAD doubled speech padding at 16000 Hz"));

        assert!(VadDurations::new(0.0, 0.0).is_ok());
        assert!(VadPadding::new(0.0).is_ok());
    }

    #[test]
    fn vad_overflow_refusals_precede_injected_effects_and_one_segment_preserves_padding_branch_absence()
    -> Result<(), String> {
        assert_policy_refusal_has_no_detector_or_continuation_effect(
            f32::MAX,
            0.0,
            "VAD minimum speech duration at 16000 Hz",
        )?;
        assert_policy_refusal_has_no_detector_or_continuation_effect(
            0.0,
            f32::MAX,
            "VAD speech padding at 16000 Hz",
        )?;

        let rate = fixed_vad_sample_rate()?;
        let first_count_above_half = (usize::MAX / 2)
            .checked_add(1)
            .ok_or_else(|| "half of a nonzero platform capacity has no successor".to_owned())?;
        let doubled_only_milliseconds = usize_to_f32(first_count_above_half) / 16.0;
        let one_sided =
            temporal_sample_count(doubled_only_milliseconds, rate, TemporalField::VadPadding)?;
        assert!(
            doubled_temporal_sample_count(one_sided, rate, TemporalField::VadDoubledPadding)
                .is_err(),
            "the platform-derived witness must leave one-sided padding representable"
        );
        assert_policy_refusal_has_no_detector_or_continuation_effect(
            0.0,
            doubled_only_milliseconds,
            "VAD doubled speech padding at 16000 Hz",
        )?;

        let input = audio(1_536);
        let detector_calls = Arc::new(AtomicUsize::new(0));
        let mut detector = CountingDetector {
            expected: input.clone(),
            probabilities: vec![0.9, 0.1, 0.1],
            calls: Arc::clone(&detector_calls),
        };
        let segments = speech_segments(
            &mut detector,
            input.view(),
            policy(0.5, 0.35, 0.0, 32.0, 0.0),
        )?;
        assert_eq!(detector_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            segments,
            vec![SpeechSegment {
                start_sample: 0,
                end_sample: 512,
            }],
            "one accepted segment must retain the branch where no neighboring padding split exists"
        );
        Ok(())
    }
}
