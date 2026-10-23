// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Single-channel batch transcription state and window orchestration.

use crate::contracts::{
    BatchConfig, PadPolicy, StageTimings, Transcript, TranscriptWord, checkpoint,
};
use crate::observations::{
    ObservationMode, WindowEvidence, WindowEvidenceInput, WindowKind, WindowTiming,
};
use crate::padding::pad_noise;
use crate::stitch::{WindowWords, stitch_aligned, windows};
use crate::vad_policy::{VadPolicyConfig, segments_from_probabilities};
use gigaam_audio::{ChannelAudio, FeatureMatrix, FrontendProcessor};
use gigaam_primitives::{trunc_f32_to_usize, usize_to_f32, usize_to_u64_checked};
use gigaam_recognition::{
    ExecutionControl, FrameRate, SpeechProbabilityDetector, WindowDecoder, vad::VAD_SR,
};
use std::sync::Arc;

/// Per-window geometry computed once by the window loop and threaded into per-window decoding,
/// so the same values back both the transcript word cut and, when enabled, the window-evidence
/// record. `subsampling_factor` is `None` unless a window-evidence observer is installed, so its
/// derivation (and any refusal it could produce) never runs on the plain transcription path.
struct WindowGeometry {
    index: usize,
    kind: WindowKind,
    start_sec: f32,
    end_sec: f32,
    first_sample: usize,
    first_frame: usize,
    last_frame: usize,
    offset: f32,
    keep_until_sec: f32,
    subsampling_factor: Option<usize>,
    frontend_origin_sample: usize,
}

/// The waveform crop's location in the source channel and its first planned window index.
/// Feature and decoder coordinates remain local to this crop, even off the source hop grid.
#[derive(Clone, Copy, Default)]
struct SpanOrigin {
    sample: usize,
    window_index: usize,
}

/// Derives the encoder's frame-subsampling factor from a frontend's sample rate and hop length
/// and a decoder's output frame rate, instead of assuming a fixed value. `sample_rate /
/// hop_length` is the frontend's own feature-frame rate; dividing it by the encoder's output
/// frame rate must yield a whole number of frontend frames per encoder output frame. The batch
/// window loop and the CLI evidence-header writer both call this so the two stay consistent.
pub fn subsampling_factor(
    sample_rate: usize,
    hop_length: usize,
    frame_rate: FrameRate,
) -> Result<usize, String> {
    if hop_length == 0 {
        return Err("batch frontend hop length must be positive".into());
    }
    let feature_rate = usize_to_f32(sample_rate) / usize_to_f32(hop_length);
    let ratio = feature_rate / frame_rate.get();
    if !ratio.is_finite() || ratio < 1.0 {
        return Err(format!(
            "batch subsampling factor {ratio} must be finite and at least one"
        ));
    }
    let rounded = ratio.round();
    if (ratio - rounded).abs() > 1e-3 {
        return Err(format!(
            "batch subsampling factor {ratio} is not a whole number of encoder output frames"
        ));
    }
    trunc_f32_to_usize(rounded).map_err(|error| format!("batch subsampling factor: {error}"))
}

/// A mutable single-channel batch transcriber with one supplied recognition decoder.
pub struct BatchTranscriber<D: WindowDecoder> {
    config: BatchConfig,
    frontend: Arc<FrontendProcessor>,
    decoder: D,
    control: ExecutionControl,
    observations: ObservationMode,
    frame_rate: FrameRate,
    sample_rate: usize,
    timings: StageTimings,
}

/// Typed capabilities required to create one batch transcriber.
///
/// Every field is either an immutable capability or a constructor-validated value, so the grouped
/// input prevents positional ambiguity without exposing unchecked scalar configuration.
pub struct BatchSetup<D: WindowDecoder> {
    pub frontend: Arc<FrontendProcessor>,
    pub decoder: D,
    pub config: BatchConfig,
    pub control: ExecutionControl,
    pub observations: ObservationMode,
}

impl<D: WindowDecoder> BatchTranscriber<D> {
    pub fn new(setup: BatchSetup<D>) -> Result<Self, String> {
        if setup.config.sample_rate() != setup.frontend.sample_rate() {
            return Err(format!(
                "batch configuration sample rate {} Hz does not match frontend sample rate {} Hz",
                setup.config.sample_rate().hertz(),
                setup.frontend.sample_rate().hertz()
            ));
        }
        Ok(Self {
            config: setup.config,
            sample_rate: setup.frontend.sample_rate().as_usize()?,
            frame_rate: setup.decoder.frame_rate(),
            frontend: setup.frontend,
            decoder: setup.decoder,
            control: setup.control,
            observations: setup.observations,
            timings: StageTimings::default(),
        })
    }

    pub const fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    pub const fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    pub const fn timings(&self) -> StageTimings {
        self.timings
    }

    pub fn into_decoder(self) -> D {
        self.decoder
    }

    /// Warm up the full-window shape, the only shape used for files longer than a window.
    pub fn warmup(&mut self) -> Result<(), String> {
        self.warmup_lengths(&[self.config.full_window_samples()])
    }

    pub fn warmup_shapes(&mut self, shapes_sec: &[f32]) -> Result<(), String> {
        let mut lengths = Vec::with_capacity(shapes_sec.len());
        for &seconds in shapes_sec {
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err("batch warmup shape must be finite and positive".into());
            }
            let length = trunc_f32_to_usize(seconds * usize_to_f32(self.sample_rate))
                .map_err(|error| format!("warmup shape {seconds}: {error}"))?;
            lengths.push(length);
        }
        self.warmup_lengths(&lengths)
    }

    fn warmup_lengths(&mut self, lengths: &[usize]) -> Result<(), String> {
        for &length in lengths {
            checkpoint(&self.control)?;
            let mel = self.frontend.log_mel(&vec![0.0_f32; length])?;
            checkpoint(&self.control)?;
            self.decoder.decode(mel.view())?;
            checkpoint(&self.control)?;
        }
        // Warmup costs are deliberately excluded from application timings.
        self.timings = StageTimings::default();
        Ok(())
    }

    fn decode_frames(
        &mut self,
        all_features: &FeatureMatrix,
        geometry: WindowGeometry,
    ) -> Result<Vec<TranscriptWord>, String> {
        checkpoint(&self.control)?;
        let features = all_features.frame_range(geometry.first_frame, geometry.last_frame)?;
        let feature_frames = features.frames();
        checkpoint(&self.control)?;
        let started = std::time::Instant::now();
        let decoded = self.decoder.decode(features.view())?;
        checkpoint(&self.control)?;
        let total_seconds = started.elapsed().as_secs_f64();
        let encoder_seconds = decoded.encoder_seconds();
        let decode_seconds = total_seconds - encoder_seconds;
        if !total_seconds.is_finite()
            || !encoder_seconds.is_finite()
            || !decode_seconds.is_finite()
            || decode_seconds < 0.0
        {
            return Err("recognition decoder reported an invalid stage duration".into());
        }
        if let (Some(labels), Some(tokens), Some(word_spans), Some(subsampling_factor)) = (
            decoded.labels(),
            decoded.tokens(),
            decoded.word_spans(),
            geometry.subsampling_factor,
        ) {
            let source_offset =
                usize_to_f32(geometry.frontend_origin_sample) / usize_to_f32(self.sample_rate);
            let evidence = WindowEvidence::new(WindowEvidenceInput {
                index: geometry.index,
                kind: geometry.kind,
                start_sec: source_offset + geometry.start_sec,
                end_sec: source_offset + geometry.end_sec,
                first_sample: geometry
                    .frontend_origin_sample
                    .checked_add(geometry.first_sample)
                    .ok_or_else(|| "batch evidence source sample overflows".to_owned())?,
                frontend_origin_sample: geometry.frontend_origin_sample,
                first_feature_frame: geometry.first_frame,
                feature_frames,
                residue: geometry.first_frame % subsampling_factor,
                origin: geometry.first_frame / subsampling_factor,
                output_frames: decoded.output_frames(),
                keep_until_sec: source_offset + geometry.keep_until_sec,
                labels: labels.to_vec(),
                mask: decoded.silence().to_vec(),
                tokens: tokens.to_vec(),
                word_spans: word_spans.to_vec(),
            })?;
            self.observations.emit_evidence(evidence);
        }
        let words = decoded
            .into_words()
            .into_iter()
            .map(TranscriptWord::from_recognition)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|word| word.shifted(geometry.offset))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|word| word.start() < geometry.keep_until_sec)
            .collect();
        self.timings.add_encoder(encoder_seconds);
        self.timings.add_decode(decode_seconds);
        self.observations.emit(WindowTiming::new(
            usize_to_f32(geometry.frontend_origin_sample) / usize_to_f32(self.sample_rate)
                + geometry.offset,
            feature_frames,
            encoder_seconds,
        )?);
        Ok(words)
    }

    /// Transcribes one complete, validated Audio-owned channel.
    pub fn transcribe_channel(&mut self, input: &ChannelAudio) -> Result<Transcript, String> {
        self.transcribe_span(input.samples(), SpanOrigin::default())
    }

    /// Segments one complete channel at 16 kHz before recognition, borrowing an independently
    /// reset detector for this call. The caller retains detector ownership; no worker or native
    /// session is created here. Audio must already be at the configured model/frontend rate.
    ///
    /// Accepted spans use the existing padding/window policy and keep source-channel time.
    /// All-silence input performs no frontend or decoder work. With `PadPolicy::Exact`, spans
    /// no longer than the frontend FFT length retain the existing no-decode behavior.
    pub fn transcribe_channel_with_vad<V: SpeechProbabilityDetector>(
        &mut self,
        input: &ChannelAudio,
        detector: &mut V,
        policy: VadPolicyConfig,
    ) -> Result<Transcript, String> {
        checkpoint(&self.control)?;
        self.validate_vad_sample_rate()?;
        let started = std::time::Instant::now();
        let probabilities = detector.probabilities(input.view())?;
        checkpoint(&self.control)?;
        let segments = segments_from_probabilities(&probabilities, input.len(), policy)?;
        drop(probabilities);
        checkpoint(&self.control)?;
        self.timings.add_vad(started.elapsed().as_secs_f64());

        let mut words = Vec::new();
        let mut planned_windows = 0_usize;
        for segment in segments {
            checkpoint(&self.control)?;
            let samples = &input.samples()[segment.start_sample()..segment.end_sample()];
            let transcript = self.transcribe_span(
                samples,
                SpanOrigin {
                    sample: segment.start_sample(),
                    window_index: planned_windows,
                },
            )?;
            checkpoint(&self.control)?;
            planned_windows = planned_windows
                .checked_add(transcript.windows())
                .ok_or_else(|| "VAD batch planned window count overflows".to_owned())?;
            // Independent speech spans have no recognition overlap to stitch. In particular,
            // repeated words separated by silence must not be mistaken for duplicate context.
            words.extend(transcript.into_words());
        }
        checkpoint(&self.control)?;
        Ok(Transcript::new(words, planned_windows))
    }

    pub(crate) fn validate_vad_sample_rate(&self) -> Result<(), String> {
        if self.sample_rate != VAD_SR {
            return Err(format!(
                "batch VAD requires {VAD_SR} Hz model/frontend audio, got {} Hz",
                self.sample_rate
            ));
        }
        Ok(())
    }

    fn transcribe_span(
        &mut self,
        input_samples: &[f32],
        origin: SpanOrigin,
    ) -> Result<Transcript, String> {
        checkpoint(&self.control)?;
        let sample_rate_f32 = usize_to_f32(self.sample_rate);
        let full_length = self.config.full_window_samples();
        let real_seconds = usize_to_f32(input_samples.len()) / sample_rate_f32;
        let padded: Vec<f32>;
        let samples: &[f32] = if self.config.padding() == PadPolicy::PadToWindow
            && input_samples.len() < full_length
        {
            let mut value = input_samples.to_vec();
            let seed = usize_to_u64_checked(input_samples.len())
                .map_err(|error| format!("batch padding seed: {error}"))?;
            value.extend(pad_noise(full_length - input_samples.len(), seed));
            padded = value;
            &padded
        } else {
            input_samples
        };
        let total_seconds = usize_to_f32(samples.len()) / sample_rate_f32;
        // Keep a small punctuation tail but never include a word beginning in synthetic padding.
        let keep_until = real_seconds + 0.5;
        let frontend_started = std::time::Instant::now();
        let features = self.frontend.log_mel(samples)?;
        checkpoint(&self.control)?;
        self.timings
            .add_frontend(frontend_started.elapsed().as_secs_f64());
        let feature_count = features.frames();
        let hop = self.frontend.hop_length();
        let fft_length = self.frontend.fft_length();
        let windows = windows(
            total_seconds,
            self.config.window_sec(),
            self.config.overlap_sec(),
        )?;
        let subsampling_factor = if self.observations.evidence_enabled() {
            Some(subsampling_factor(
                self.sample_rate,
                self.frontend.hop_length(),
                self.frame_rate,
            )?)
        } else {
            None
        };
        let mut window_words = Vec::with_capacity(windows.len());
        for (window_index, window) in windows.iter().enumerate() {
            checkpoint(&self.control)?;
            let start = window.start();
            let end = window.end();
            let first_sample = trunc_f32_to_usize(start * sample_rate_f32)
                .map_err(|error| format!("batch window start {start}: {error}"))?;
            let unclamped_end = trunc_f32_to_usize(end * sample_rate_f32)
                .map_err(|error| format!("batch window end {end}: {error}"))?;
            let last_sample = unclamped_end.min(samples.len());
            let first_frame = first_sample.div_ceil(hop);
            let span = last_sample
                .checked_sub(first_sample)
                .ok_or_else(|| "batch window sample range is inverted".to_owned())?;
            let frame_count = if span > fft_length {
                1 + (span - fft_length) / hop
            } else {
                0
            };
            let last_frame = first_frame
                .checked_add(frame_count)
                .ok_or_else(|| "batch feature frame range overflows".to_owned())?
                .min(feature_count);
            if last_frame <= first_frame {
                continue;
            }
            let offset = usize_to_f32(first_frame) * usize_to_f32(hop) / sample_rate_f32;
            let kind = if window_index + 1 == windows.len() {
                WindowKind::Tail
            } else {
                WindowKind::Body
            };
            let words = self.decode_frames(
                &features,
                WindowGeometry {
                    index: origin
                        .window_index
                        .checked_add(window_index)
                        .ok_or_else(|| "batch window index overflows".to_owned())?,
                    kind,
                    start_sec: start,
                    end_sec: end,
                    first_sample,
                    first_frame,
                    last_frame,
                    offset,
                    keep_until_sec: end.min(keep_until),
                    subsampling_factor,
                    frontend_origin_sample: origin.sample,
                },
            )?;
            window_words.push(WindowWords::new(*window, words));
        }
        checkpoint(&self.control)?;
        let words = stitch_aligned(&window_words, 0.25)?
            .into_iter()
            .filter(|word| word.start() < real_seconds)
            .map(|word| {
                word.capped_end(real_seconds)?
                    .shifted(usize_to_f32(origin.sample) / sample_rate_f32)
            })
            .collect::<Result<Vec<_>, _>>()?;
        checkpoint(&self.control)?;
        Ok(Transcript::new(words, windows.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::{BatchSetup, BatchTranscriber, subsampling_factor};
    use crate::contracts::VadProbability;
    use crate::contracts::{BatchConfig, PadPolicy};
    use crate::observations::{
        ObservationMode, WindowEvidence, WindowEvidenceObserver, WindowKind, WindowTiming,
        WindowTimingObserver,
    };
    use crate::test_support;
    use crate::vad_policy::{VadDurations, VadPadding, VadPolicyConfig, VadThresholds};
    use gigaam_audio::{ChannelAudio, ChannelAudioView, SampleRate};
    use gigaam_recognition::{
        Decoded, ExecutionControl, ExecutionState, FrameRate, SpeechProbabilityDetector, Token,
        WindowDecoder, Word,
    };
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RecordingObserver {
        observations: Mutex<Vec<WindowTiming>>,
    }

    impl RecordingObserver {
        fn observations(&self) -> Vec<WindowTiming> {
            self.observations
                .lock()
                .expect("test observer lock must not be poisoned")
                .clone()
        }
    }

    impl WindowTimingObserver for RecordingObserver {
        fn observe(&self, observation: WindowTiming) {
            self.observations
                .lock()
                .expect("test observer lock must not be poisoned")
                .push(observation);
        }
    }

    struct DeterministicDecoder {
        calls: Arc<AtomicUsize>,
        cancel_after_decode: Option<ExecutionControl>,
    }

    struct FrameRateProbe {
        frame_rate_calls: Arc<AtomicUsize>,
    }

    impl WindowDecoder for FrameRateProbe {
        fn frame_rate(&self) -> FrameRate {
            self.frame_rate_calls.fetch_add(1, Ordering::SeqCst);
            FrameRate::new(8.0).expect("test decoder frame rate is positive")
        }

        fn decode(
            &mut self,
            _features: gigaam_audio::FeatureMatrixView<'_>,
        ) -> Result<Decoded, String> {
            Err("the mismatch test must refuse before decoder execution".into())
        }
    }

    impl DeterministicDecoder {
        fn ordinary(calls: Arc<AtomicUsize>) -> Self {
            Self {
                calls,
                cancel_after_decode: None,
            }
        }

        fn cancelling(calls: Arc<AtomicUsize>, control: ExecutionControl) -> Self {
            Self {
                calls,
                cancel_after_decode: Some(control),
            }
        }
    }

    impl WindowDecoder for DeterministicDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(8.0).expect("test decoder frame rate is positive")
        }

        fn decode(
            &mut self,
            features: gigaam_audio::FeatureMatrixView<'_>,
        ) -> Result<Decoded, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(control) = &self.cancel_after_decode {
                control.request_cancellation();
            }
            let word = Word::new("word".into(), 0.0, 0.1)
                .expect("test decoder word timestamps are coherent");
            Decoded::new(
                vec![word],
                vec![false; features.frames()],
                features.frames(),
                0.0,
            )
        }
    }

    /// A decoder that always exposes CTC-style labels and tokens, so evidence-observer tests do
    /// not depend on a real encoder.
    struct EvidenceDecoder {
        calls: Arc<AtomicUsize>,
    }

    impl WindowDecoder for EvidenceDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(8.0).expect("test decoder frame rate is positive")
        }

        fn decode(
            &mut self,
            features: gigaam_audio::FeatureMatrixView<'_>,
        ) -> Result<Decoded, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let frames = features.frames();
            let word = Word::new("word".into(), 0.0, 0.1)
                .expect("test decoder word timestamps are coherent");
            Decoded::new_with_labels(
                vec![word],
                vec![false; frames],
                frames,
                0.0,
                vec![1; frames],
                vec![Token::new(1, 0)],
                vec![(0, 0)],
            )
        }
    }

    struct EvidenceRecordingObserver {
        observations: Mutex<Vec<WindowEvidence>>,
    }

    impl EvidenceRecordingObserver {
        fn observations(&self) -> Vec<WindowEvidence> {
            self.observations
                .lock()
                .expect("test observer lock must not be poisoned")
                .clone()
        }
    }

    impl WindowEvidenceObserver for EvidenceRecordingObserver {
        fn observe(&self, observation: WindowEvidence) {
            self.observations
                .lock()
                .expect("test observer lock must not be poisoned")
                .push(observation);
        }
    }

    fn config() -> BatchConfig {
        BatchConfig::new(
            SampleRate::new(16).expect("test sample rate is positive"),
            1.0,
            0.25,
            PadPolicy::Exact,
        )
        .expect("test batch configuration must be valid")
    }

    fn input() -> ChannelAudio {
        ChannelAudio::new(vec![0.25; 48]).expect("test samples must be finite")
    }

    struct FixedVad {
        probabilities: Vec<f32>,
        calls: Arc<AtomicUsize>,
        cancel: Option<ExecutionControl>,
    }

    impl SpeechProbabilityDetector for FixedVad {
        fn probabilities(&mut self, _audio: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(control) = &self.cancel {
                control.request_cancellation();
            }
            Ok(self.probabilities.clone())
        }
    }

    fn vad(probabilities: Vec<f32>) -> FixedVad {
        FixedVad {
            probabilities,
            calls: Arc::new(AtomicUsize::new(0)),
            cancel: None,
        }
    }

    fn vad_policy(padding_milliseconds: f32) -> VadPolicyConfig {
        VadPolicyConfig::new(
            VadThresholds::new(
                VadProbability::DEFAULT,
                VadProbability::new(0.35)
                    .expect("the test silence threshold is in the unit interval"),
            )
            .expect("test thresholds are ordered"),
            VadDurations::new(0.0, 0.0).expect("zero minimum durations are valid"),
            VadPadding::new(padding_milliseconds).expect("test VAD padding is representable"),
        )
    }

    fn vad_config(padding: PadPolicy, window: f32) -> BatchConfig {
        BatchConfig::new(
            SampleRate::new(16_000).expect("test VAD rate is valid"),
            window,
            window / 4.0,
            padding,
        )
        .expect("test VAD batch geometry is valid")
    }

    fn vad_batch<D: WindowDecoder>(
        decoder: D,
        config: BatchConfig,
        observations: ObservationMode,
        control: ExecutionControl,
    ) -> BatchTranscriber<D> {
        BatchTranscriber::new(BatchSetup {
            frontend: test_support::frontend_at_sample_rate(16_000),
            decoder,
            config,
            control,
            observations,
        })
        .expect("test batch uses one consistent 16 kHz rate")
    }

    fn pcm(length: usize) -> ChannelAudio {
        ChannelAudio::new(vec![0.25; length]).expect("test PCM is finite")
    }

    #[test]
    fn all_speech_vad_preserves_complete_channel_words_and_observations() {
        let make = || {
            let evidence = Arc::new(EvidenceRecordingObserver {
                observations: Mutex::new(Vec::new()),
            });
            let timing = Arc::new(RecordingObserver {
                observations: Mutex::new(Vec::new()),
            });
            let transcriber = vad_batch(
                EvidenceDecoder {
                    calls: Arc::new(AtomicUsize::new(0)),
                },
                vad_config(PadPolicy::Exact, 0.128),
                ObservationMode::enabled(timing.clone()).with_evidence(evidence.clone()),
                ExecutionControl::without_deadline(),
            );
            (transcriber, evidence, timing)
        };
        let (mut plain, plain_evidence, plain_timing) = make();
        let (mut segmented, segmented_evidence, segmented_timing) = make();
        let input = pcm(6_144);
        let reference = plain
            .transcribe_channel(&input)
            .expect("plain batch succeeds");
        let result = segmented
            .transcribe_channel_with_vad(&input, &mut vad(vec![1.0; 12]), vad_policy(0.0))
            .expect("a full speech span uses the same batch geometry");
        assert_eq!(result, reference);
        assert_eq!(
            segmented_evidence.observations(),
            plain_evidence.observations()
        );
        assert_eq!(segmented_timing.observations(), plain_timing.observations());
        assert!(
            plain_evidence
                .observations()
                .iter()
                .all(|record| record.frontend_origin_sample() == 0)
        );
        assert_eq!(plain.timings().vad_seconds(), 0.0);
        assert!(segmented.timings().vad_seconds().is_finite());
    }

    #[test]
    fn all_silence_vad_skips_frontend_decoder_and_observations() {
        let calls = Arc::new(AtomicUsize::new(0));
        let evidence = Arc::new(EvidenceRecordingObserver {
            observations: Mutex::new(Vec::new()),
        });
        let mut batch = vad_batch(
            EvidenceDecoder {
                calls: calls.clone(),
            },
            vad_config(PadPolicy::PadToWindow, 0.128),
            ObservationMode::disabled().with_evidence(evidence.clone()),
            ExecutionControl::without_deadline(),
        );
        let result = batch
            .transcribe_channel_with_vad(&pcm(1_537), &mut vad(vec![0.0; 4]), vad_policy(0.0))
            .expect("complete silent coverage is a successful empty result");
        assert!(result.words().is_empty());
        assert_eq!(result.windows(), 0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(evidence.observations().is_empty());
        assert_eq!(batch.timings().frontend_seconds(), 0.0);
        assert_eq!(batch.timings().encoder_seconds(), 0.0);
        assert_eq!(batch.timings().decode_seconds(), 0.0);
    }

    #[test]
    fn disjoint_vad_spans_preserve_repeated_words_and_off_grid_source_coordinates() {
        let evidence = Arc::new(EvidenceRecordingObserver {
            observations: Mutex::new(Vec::new()),
        });
        let timing = Arc::new(RecordingObserver {
            observations: Mutex::new(Vec::new()),
        });
        let mut enabled = vad_batch(
            EvidenceDecoder {
                calls: Arc::new(AtomicUsize::new(0)),
            },
            vad_config(PadPolicy::Exact, 0.128),
            ObservationMode::enabled(timing.clone()).with_evidence(evidence.clone()),
            ExecutionControl::without_deadline(),
        );
        let mut disabled = vad_batch(
            EvidenceDecoder {
                calls: Arc::new(AtomicUsize::new(0)),
            },
            vad_config(PadPolicy::Exact, 0.128),
            ObservationMode::disabled(),
            ExecutionControl::without_deadline(),
        );
        let probabilities = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        // One sample of padding places both crops off this fixture's two-sample hop grid.
        let policy = vad_policy(0.0625);
        let input = pcm(4_096);
        let result = enabled
            .transcribe_channel_with_vad(&input, &mut vad(probabilities.clone()), policy)
            .expect("two separated padded speech spans transcribe");
        let without_observers = disabled
            .transcribe_channel_with_vad(&input, &mut vad(probabilities), policy)
            .expect("observation-free segmentation transcribes");
        assert_eq!(result, without_observers);
        assert_eq!(result.text(), "word word");
        assert_eq!(result.windows(), 2);
        let records = evidence.observations();
        let timings = timing.observations();
        for (index, (origin, end)) in [(511_usize, 1_025_usize), (2_559, 3_073)]
            .into_iter()
            .enumerate()
        {
            let start_sec = gigaam_primitives::usize_to_f32(origin) / 16_000.0;
            let end_sec = gigaam_primitives::usize_to_f32(end) / 16_000.0;
            let word = &result.words()[index];
            let record = &records[index];
            assert_eq!(word.start(), start_sec);
            assert!((word.end() - end_sec).abs() < 1e-6);
            assert_eq!(record.index(), index);
            assert_eq!(record.kind(), WindowKind::Tail);
            assert_eq!(record.frontend_origin_sample(), origin);
            assert_eq!(record.first_sample(), origin);
            assert_eq!(record.first_feature_frame(), 0);
            assert_eq!(record.origin(), 0);
            assert_eq!(record.residue(), 0);
            assert_eq!(record.start_sec(), start_sec);
            assert!((record.end_sec() - end_sec).abs() < 1e-6);
            assert!((record.keep_until_sec() - end_sec).abs() < 1e-6);
            assert_eq!(timings[index].offset_sec(), start_sec);
            assert_eq!(record.tokens(), &[Token::new(1, 0)]);
        }
    }

    #[test]
    fn vad_long_spans_keep_window_tail_geometry_and_channel_unique_indices() {
        let evidence = Arc::new(EvidenceRecordingObserver {
            observations: Mutex::new(Vec::new()),
        });
        let mut batch = vad_batch(
            EvidenceDecoder {
                calls: Arc::new(AtomicUsize::new(0)),
            },
            vad_config(PadPolicy::Exact, 0.128),
            ObservationMode::disabled().with_evidence(evidence.clone()),
            ExecutionControl::without_deadline(),
        );
        let probabilities = [vec![1.0; 8], vec![0.0; 4], vec![1.0; 8], vec![0.0; 4]].concat();
        let result = batch
            .transcribe_channel_with_vad(&pcm(12_288), &mut vad(probabilities), vad_policy(0.0))
            .expect("long speech spans use existing overlapping windows");
        let records = evidence.observations();
        assert_eq!(result.windows(), 6);
        assert_eq!(records.len(), 6);
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record.index(), index);
            assert_eq!(
                record.frontend_origin_sample(),
                if index < 3 { 0 } else { 6_144 }
            );
            assert_eq!(
                record.kind(),
                if index % 3 == 2 {
                    WindowKind::Tail
                } else {
                    WindowKind::Body
                }
            );
        }
        assert!(result.words().iter().any(|word| word.start() >= 0.384));
    }

    #[test]
    fn vad_short_spans_retain_exact_boundary_and_explicit_padding_semantics() {
        for length in [1_usize, 4, 5] {
            for padding in [PadPolicy::Exact, PadPolicy::PadToWindow] {
                let calls = Arc::new(AtomicUsize::new(0));
                let mut batch = vad_batch(
                    DeterministicDecoder::ordinary(calls.clone()),
                    vad_config(padding, 0.004),
                    ObservationMode::disabled(),
                    ExecutionControl::without_deadline(),
                );
                let result = batch
                    .transcribe_channel_with_vad(&pcm(length), &mut vad(vec![1.0]), vad_policy(0.0))
                    .expect("a short terminal span preserves the selected padding policy");
                assert_eq!(
                    result.windows(),
                    1,
                    "planned windows include no-feature exact windows"
                );
                let decoded = padding == PadPolicy::PadToWindow || length > 4;
                assert_eq!(calls.load(Ordering::SeqCst), usize::from(decoded));
                assert_eq!(result.words().len(), usize::from(decoded));
                for word in result.words() {
                    assert!(word.end() <= gigaam_primitives::usize_to_f32(length) / 16_000.0);
                }
            }
        }
    }

    #[test]
    fn vad_refuses_bad_coverage_or_domain_before_frontend_and_decoder() {
        for probabilities in [
            vec![],
            vec![0.0; 3],
            vec![0.0, f32::NAN],
            vec![f32::INFINITY, 0.0],
            vec![-0.1, 0.0],
            vec![0.0, 1.1],
        ] {
            let calls = Arc::new(AtomicUsize::new(0));
            let mut batch = vad_batch(
                DeterministicDecoder::ordinary(calls.clone()),
                vad_config(PadPolicy::Exact, 0.128),
                ObservationMode::disabled(),
                ExecutionControl::without_deadline(),
            );
            assert!(
                batch
                    .transcribe_channel_with_vad(
                        &pcm(513),
                        &mut vad(probabilities),
                        vad_policy(0.0)
                    )
                    .is_err()
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(batch.timings().frontend_seconds(), 0.0);
        }
    }

    #[test]
    fn enabled_vad_rate_refuses_before_detector_but_plain_batch_rate_remains_supported() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut batch = BatchTranscriber::new(BatchSetup {
            frontend: test_support::frontend(),
            decoder: DeterministicDecoder::ordinary(calls.clone()),
            config: config(),
            control: ExecutionControl::without_deadline(),
            observations: ObservationMode::disabled(),
        })
        .expect("legacy fixture rate remains valid for batch");
        let mut detector = vad(vec![1.0]);
        let error = batch
            .transcribe_channel_with_vad(&input(), &mut detector, vad_policy(0.0))
            .expect_err("VAD must refuse a non-16 kHz model rate");
        assert!(error.contains("requires 16000 Hz"));
        assert_eq!(detector.calls.load(Ordering::SeqCst), 0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(batch.transcribe_channel(&input()).is_ok());
        assert!(calls.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn vad_cancellation_prevents_detector_or_following_segments_at_each_execution_boundary() {
        for cancel_before in [true, false] {
            let control = ExecutionControl::without_deadline();
            if cancel_before {
                control.request_cancellation();
            }
            let calls = Arc::new(AtomicUsize::new(0));
            let mut batch = vad_batch(
                DeterministicDecoder::ordinary(calls.clone()),
                vad_config(PadPolicy::Exact, 0.128),
                ObservationMode::disabled(),
                control.clone(),
            );
            let mut detector = vad(vec![1.0; 4]);
            detector.cancel = Some(control);
            assert!(
                batch
                    .transcribe_channel_with_vad(&pcm(2_048), &mut detector, vad_policy(0.0))
                    .is_err()
            );
            assert_eq!(
                detector.calls.load(Ordering::SeqCst),
                usize::from(!cancel_before)
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(batch.timings().frontend_seconds(), 0.0);
        }
        let control = ExecutionControl::without_deadline();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut batch = vad_batch(
            DeterministicDecoder::cancelling(calls.clone(), control.clone()),
            vad_config(PadPolicy::Exact, 0.128),
            ObservationMode::disabled(),
            control,
        );
        assert!(
            batch
                .transcribe_channel_with_vad(
                    &pcm(2_048),
                    &mut vad(vec![1.0, 0.0, 1.0, 0.0]),
                    vad_policy(0.0)
                )
                .is_err()
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "cancellation after native work suppresses the next span"
        );
    }

    #[test]
    fn subsampling_factor_derives_whole_ratios_and_refuses_fractional_or_zero_input() {
        assert_eq!(
            subsampling_factor(
                16_000,
                160,
                FrameRate::new(25.0).expect("test frame rate is positive")
            ),
            Ok(4)
        );
        assert_eq!(
            subsampling_factor(
                16,
                2,
                FrameRate::new(8.0).expect("test frame rate is positive")
            ),
            Ok(1)
        );
        assert!(
            subsampling_factor(
                16_000,
                160,
                FrameRate::new(7.0).expect("test frame rate is positive")
            )
            .is_err(),
            "a non-whole-number ratio must refuse"
        );
        assert!(
            subsampling_factor(
                16_000,
                0,
                FrameRate::new(25.0).expect("test frame rate is positive")
            )
            .is_err(),
            "a zero hop length must refuse"
        );
    }

    #[test]
    fn bound_batch_rate_mismatch_refuses_before_decoder_capability_work() {
        let frame_rate_calls = Arc::new(AtomicUsize::new(0));
        let result = BatchTranscriber::new(BatchSetup {
            frontend: test_support::frontend(),
            decoder: FrameRateProbe {
                frame_rate_calls: Arc::clone(&frame_rate_calls),
            },
            config: BatchConfig::new(
                SampleRate::new(8).expect("test sample rate is positive"),
                1.0,
                0.0,
                PadPolicy::Exact,
            )
            .expect("the bound mismatched test configuration is otherwise valid"),
            control: ExecutionControl::without_deadline(),
            observations: ObservationMode::disabled(),
        });
        let error = match result {
            Ok(_) => panic!("a mismatched bound batch rate must refuse"),
            Err(error) => error,
        };
        assert!(error.contains("batch configuration sample rate 8 Hz"));
        assert_eq!(frame_rate_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn observation_mode_preserves_batch_outputs_and_enabled_mode_emits_exact_window_records() {
        let enabled_calls = Arc::new(AtomicUsize::new(0));
        let observer = Arc::new(RecordingObserver {
            observations: Mutex::new(Vec::new()),
        });
        let mut enabled = BatchTranscriber::new(BatchSetup {
            frontend: test_support::frontend(),
            decoder: DeterministicDecoder::ordinary(enabled_calls.clone()),
            config: config(),
            control: ExecutionControl::without_deadline(),
            observations: ObservationMode::enabled(observer.clone()),
        })
        .expect("test batch setup must be valid");
        let enabled_transcript = enabled
            .transcribe_channel(&input())
            .expect("ordinary decoder must transcribe every window");
        let observations = observer.observations();
        assert_eq!(observations.len(), enabled_calls.load(Ordering::SeqCst));
        assert_eq!(observations.len(), enabled_transcript.windows());
        let records: Vec<(f32, usize, f64)> = observations
            .iter()
            .map(|observation| {
                (
                    observation.offset_sec(),
                    observation.frames(),
                    observation.encoder_seconds(),
                )
            })
            .collect();
        assert_eq!(
            records,
            vec![(0.0, 7, 0.0), (0.75, 7, 0.0), (1.5, 7, 0.0), (2.0, 7, 0.0)],
            "enabled observation records must preserve successful window content and order"
        );

        let disabled_calls = Arc::new(AtomicUsize::new(0));
        let mut disabled = BatchTranscriber::new(BatchSetup {
            frontend: test_support::frontend(),
            decoder: DeterministicDecoder::ordinary(disabled_calls.clone()),
            config: config(),
            control: ExecutionControl::without_deadline(),
            observations: ObservationMode::disabled(),
        })
        .expect("test batch setup must be valid");
        let disabled_transcript = disabled
            .transcribe_channel(&input())
            .expect("ordinary decoder must transcribe in disabled observation mode");
        assert_eq!(
            disabled_calls.load(Ordering::SeqCst),
            enabled_calls.load(Ordering::SeqCst)
        );
        assert_eq!(disabled_transcript, enabled_transcript);
    }

    #[test]
    fn evidence_observer_emits_one_record_per_window_and_leaves_the_transcript_unchanged() {
        let enabled_calls = Arc::new(AtomicUsize::new(0));
        let observer = Arc::new(EvidenceRecordingObserver {
            observations: Mutex::new(Vec::new()),
        });
        let mut enabled = BatchTranscriber::new(BatchSetup {
            frontend: test_support::frontend(),
            decoder: EvidenceDecoder {
                calls: enabled_calls.clone(),
            },
            config: config(),
            control: ExecutionControl::without_deadline(),
            observations: ObservationMode::disabled().with_evidence(observer.clone()),
        })
        .expect("test batch setup must be valid");
        let enabled_transcript = enabled
            .transcribe_channel(&input())
            .expect("evidence decoder must transcribe every window");
        let records = observer.observations();
        assert_eq!(records.len(), enabled_calls.load(Ordering::SeqCst));
        assert_eq!(records.len(), enabled_transcript.windows());
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record.index(), index);
            let expected_kind = if index + 1 == records.len() {
                WindowKind::Tail
            } else {
                WindowKind::Body
            };
            assert_eq!(record.kind(), expected_kind);
            assert_eq!(record.output_frames(), record.labels().len());
            assert_eq!(record.output_frames(), record.mask().len());
            assert_eq!(record.tokens(), &[Token::new(1, 0)]);
            assert_eq!(record.word_spans(), &[(0, 0)]);
            for &(first, last) in record.word_spans() {
                assert!(first <= last);
                assert!(
                    last < record.tokens().len(),
                    "a word span must index within this record's own tokens list"
                );
            }
        }

        let disabled_calls = Arc::new(AtomicUsize::new(0));
        let mut disabled = BatchTranscriber::new(BatchSetup {
            frontend: test_support::frontend(),
            decoder: EvidenceDecoder {
                calls: disabled_calls.clone(),
            },
            config: config(),
            control: ExecutionControl::without_deadline(),
            observations: ObservationMode::disabled(),
        })
        .expect("test batch setup must be valid");
        let disabled_transcript = disabled
            .transcribe_channel(&input())
            .expect("evidence decoder must transcribe with the evidence channel disabled");
        assert_eq!(
            disabled_calls.load(Ordering::SeqCst),
            enabled_calls.load(Ordering::SeqCst)
        );
        assert_eq!(disabled_transcript, enabled_transcript);
    }

    #[test]
    fn evidence_geometry_matches_window_arithmetic_including_an_off_grid_tail_window() {
        // Frontend feature rate is sample_rate / hop_length = 64 / 2 = 32 Hz; the decoder's frame
        // rate is 8 Hz, so this fixture's subsampling factor is 32 / 8 = 4. A 100-sample (1.5625 s)
        // input with a 1.0 s window and 0.25 s overlap produces a body window starting on the
        // subsampling grid and a tail window widened backward to an off-grid start.
        let sample_rate = SampleRate::new(64).expect("test sample rate is positive");
        let observer = Arc::new(EvidenceRecordingObserver {
            observations: Mutex::new(Vec::new()),
        });
        let mut transcriber = BatchTranscriber::new(BatchSetup {
            frontend: test_support::frontend_at_sample_rate(64),
            decoder: EvidenceDecoder {
                calls: Arc::new(AtomicUsize::new(0)),
            },
            config: BatchConfig::new(sample_rate, 1.0, 0.25, PadPolicy::Exact)
                .expect("test batch configuration must be valid"),
            control: ExecutionControl::without_deadline(),
            observations: ObservationMode::disabled().with_evidence(observer.clone()),
        })
        .expect("test batch setup must be valid");
        let audio = ChannelAudio::new(vec![0.25; 100]).expect("test samples must be finite");
        transcriber
            .transcribe_channel(&audio)
            .expect("evidence decoder must transcribe every window");
        let records = observer.observations();
        assert_eq!(
            records.len(),
            2,
            "this fixture must produce exactly a body and a tail window"
        );
        for record in &records {
            assert_eq!(
                record.origin() * 4 + record.residue(),
                record.first_feature_frame(),
                "origin * subsampling_factor + residue must reconstruct the first feature frame"
            );
            assert!(
                record.residue() < 4,
                "residue must stay within one subsampling period"
            );
        }
        assert_eq!(
            (records[0].kind(), records[0].residue()),
            (WindowKind::Body, 0)
        );
        assert_eq!(
            (records[1].kind(), records[1].residue()),
            (WindowKind::Tail, 2)
        );
        assert_eq!(records[1].first_feature_frame(), 18);
        assert_eq!(records[1].origin(), 4);
    }

    #[test]
    fn cancellation_before_or_during_recognition_commits_no_transcript_and_starts_no_successor() {
        let before_calls = Arc::new(AtomicUsize::new(0));
        let before_control = ExecutionControl::without_deadline();
        before_control.request_cancellation();
        let mut before = BatchTranscriber::new(BatchSetup {
            frontend: test_support::frontend(),
            decoder: DeterministicDecoder::ordinary(before_calls.clone()),
            config: config(),
            control: before_control.clone(),
            observations: ObservationMode::disabled(),
        })
        .expect("test batch setup must be valid");
        assert!(before.transcribe_channel(&input()).is_err());
        assert_eq!(before_calls.load(Ordering::SeqCst), 0);
        assert_eq!(before_control.state(), ExecutionState::CancelRequested);

        let race_calls = Arc::new(AtomicUsize::new(0));
        let race_control = ExecutionControl::without_deadline();
        let mut race = BatchTranscriber::new(BatchSetup {
            frontend: test_support::frontend(),
            decoder: DeterministicDecoder::cancelling(race_calls.clone(), race_control.clone()),
            config: config(),
            control: race_control.clone(),
            observations: ObservationMode::disabled(),
        })
        .expect("test batch setup must be valid");
        assert!(race.transcribe_channel(&input()).is_err());
        assert_eq!(
            race_calls.load(Ordering::SeqCst),
            1,
            "a cancellation racing the first return prevents every successor window"
        );
        assert_eq!(race_control.state(), ExecutionState::CancelRequested);
    }
}
