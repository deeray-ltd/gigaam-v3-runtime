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
use crate::observations::{ObservationMode, WindowTiming};
use crate::padding::pad_noise;
use crate::stitch::{WindowWords, stitch_aligned, windows};
use gigaam_audio::{ChannelAudio, FeatureMatrix, FrontendProcessor};
use gigaam_primitives::{trunc_f32_to_usize, usize_to_f32, usize_to_u64_checked};
use gigaam_recognition::{ExecutionControl, FrameRate, WindowDecoder};
use std::sync::Arc;

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
        first_frame: usize,
        last_frame: usize,
        offset: f32,
        end_sec: f32,
    ) -> Result<Vec<TranscriptWord>, String> {
        checkpoint(&self.control)?;
        let features = all_features.frame_range(first_frame, last_frame)?;
        let frames = features.frames();
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
        let words = decoded
            .into_words()
            .into_iter()
            .map(TranscriptWord::from_recognition)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|word| word.shifted(offset))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|word| word.start() < end_sec)
            .collect();
        self.timings.add_encoder(encoder_seconds);
        self.timings.add_decode(decode_seconds);
        self.observations
            .emit(WindowTiming::new(offset, frames, encoder_seconds)?);
        Ok(words)
    }

    /// Transcribes one complete, validated Audio-owned channel.
    pub fn transcribe_channel(&mut self, input: &ChannelAudio) -> Result<Transcript, String> {
        checkpoint(&self.control)?;
        let input_samples = input.samples();
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
        let mut window_words = Vec::with_capacity(windows.len());
        for window in &windows {
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
            let words = self.decode_frames(
                &features,
                first_frame,
                last_frame,
                offset,
                end.min(keep_until),
            )?;
            window_words.push(WindowWords::new(*window, words));
        }
        checkpoint(&self.control)?;
        let words = stitch_aligned(&window_words, 0.25)?
            .into_iter()
            .filter(|word| word.start() < real_seconds)
            .map(|word| word.capped_end(real_seconds))
            .collect::<Result<Vec<_>, _>>()?;
        checkpoint(&self.control)?;
        Ok(Transcript::new(words, windows.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::{BatchSetup, BatchTranscriber};
    use crate::contracts::{BatchConfig, PadPolicy};
    use crate::observations::{ObservationMode, WindowTiming, WindowTimingObserver};
    use crate::test_support;
    use gigaam_audio::{ChannelAudio, SampleRate};
    use gigaam_recognition::{
        Decoded, ExecutionControl, ExecutionState, FrameRate, WindowDecoder, Word,
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
