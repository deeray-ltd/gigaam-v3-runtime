// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Single-channel incremental transcription state and revision events.

use crate::contracts::{
    ChannelSnapshot, EndpointSource, FinalEvent, FinalReason, FrontierEvent, SnapshotWord,
    StreamConfig, StreamEvent, StreamLockPolicy, StreamWord, TranscriptWord, WordFinality,
    WordStability, WordsEvent, checkpoint,
};
use crate::endpoint::{Endpoint, SpeechState, detect};
use crate::padding::pad_noise;
use crate::stitch::{Window, WindowWords, seam, stitch_aligned};
use crate::text::normalize_word;
use gigaam_audio::{
    ChannelAudio, ChannelAudioView, FeatureMatrixView, FrontendProcessor, FrontendScratch,
};
use gigaam_primitives::{trunc_f32_to_usize, usize_to_f32};
use gigaam_recognition::vad::{VAD_FPS, VAD_SR, mask};
use gigaam_recognition::{ExecutionControl, FrameRate, SpeechProbabilityDetector, WindowDecoder};
use std::sync::Arc;
use std::time::Instant;

/// The explicitly selected endpoint capability for a streaming session.
pub enum EndpointDetector<D> {
    Blank,
    Vad(D),
}

/// Typed capabilities required to create one streaming session.
pub struct StreamSetup<D: WindowDecoder, V: SpeechProbabilityDetector> {
    pub frontend: Arc<FrontendProcessor>,
    pub decoder: D,
    pub config: StreamConfig,
    pub detector: EndpointDetector<V>,
    pub control: ExecutionControl,
}

/// Mutable single-channel streaming transcription state.
pub struct StreamSession<D: WindowDecoder, V: SpeechProbabilityDetector> {
    config: StreamConfig,
    frontend: Arc<FrontendProcessor>,
    decoder: D,
    detector: EndpointDetector<V>,
    control: ExecutionControl,
    sample_rate: usize,
    frame_rate: FrameRate,
    full_samples: usize,
    full_frames: usize,
    noise: Vec<f32>,
    noise_mel: Vec<f32>,
    mel: Vec<f32>,
    /// Mel frames fully inside real audio that are already computed.
    real_full_frames: usize,
    scratch: FrontendScratch,
    column: Vec<f32>,
    frame: Vec<f32>,
    /// Absolute time of `buffer[0]`.
    start_time: f32,
    buffer: Vec<f32>,
    decoded_samples: usize,
    committed: Vec<StreamWord>,
    /// Exactly the word line a client reaches after applying every emitted patch.
    emitted: Vec<StreamWord>,
    final_sent: usize,
    stable_sent: usize,
    /// Current-window words at absolute times, before committed-frontier filtering.
    current: Vec<TranscriptWord>,
    /// Pre-cut words waiting for a seam.
    previous: Option<WindowWords>,
    /// Current-window words before this time are committed.
    cut_time: f32,
    decodes: usize,
    decode_seconds: f64,
    encoder_seconds: f64,
}

/// One decoded window before the session accepts it into its mutable application state.
struct DecodedCandidate {
    words: Vec<gigaam_recognition::Word>,
    silence: Vec<bool>,
    output_frames: usize,
    accounting: DecodedAccounting,
}

#[derive(Clone, Copy)]
struct DecodedAccounting {
    real_full_frames: usize,
    decoded_samples: usize,
    elapsed: f64,
    encoder_seconds: f64,
}

/// Selects whether a decode advances a live stream or finalizes its remaining input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StreamProgress {
    Advance,
    Flush,
}

impl<D: WindowDecoder, V: SpeechProbabilityDetector> StreamSession<D, V> {
    pub fn new(setup: StreamSetup<D, V>) -> Result<Self, String> {
        let StreamSetup {
            frontend,
            decoder,
            config,
            detector,
            control,
        } = setup;
        if config.sample_rate() != frontend.sample_rate() {
            return Err(format!(
                "stream configuration sample rate {} Hz does not match frontend sample rate {} Hz",
                config.sample_rate().hertz(),
                frontend.sample_rate().hertz()
            ));
        }
        if frontend.is_centered() {
            return Err("stream requires center=false in the frontend".into());
        }
        let sample_rate = config.sample_rate().as_usize()?;
        match (config.endpoint_source(), &detector) {
            (EndpointSource::Blank, EndpointDetector::Blank) => {}
            (EndpointSource::Vad, EndpointDetector::Vad(_)) => {
                if sample_rate != VAD_SR {
                    return Err(format!(
                        "vad endpoint requires 16 kHz, model has {sample_rate}"
                    ));
                }
            }
            (EndpointSource::Blank, EndpointDetector::Vad(_))
            | (EndpointSource::Vad, EndpointDetector::Blank) => {
                return Err(
                    "stream endpoint configuration does not match its injected detector".into(),
                );
            }
        }
        let full_samples = config.full_window_samples().get();
        let full_frames = frontend.log_mel(&vec![0.0_f32; full_samples])?.frames();
        let noise = pad_noise(full_samples, 0x5eed);
        let noise_features = frontend.log_mel(&noise)?;
        if noise_features.frames() != full_frames {
            return Err(
                "stream frontend frame count differs for equal validated window shapes".into(),
            );
        }
        let noise_mel = noise_features.into_values();
        let scratch = frontend.scratch()?;
        let mel_bins = frontend.mel_bins();
        let fft_length = frontend.fft_length();
        let frame_rate = decoder.frame_rate();
        Ok(Self {
            config,
            frontend,
            decoder,
            detector,
            control,
            sample_rate,
            frame_rate,
            full_samples,
            full_frames,
            noise,
            mel: noise_mel.clone(),
            noise_mel,
            real_full_frames: 0,
            scratch,
            column: vec![0.0; mel_bins],
            frame: vec![0.0; fft_length],
            start_time: 0.0,
            buffer: Vec::with_capacity(full_samples),
            decoded_samples: 0,
            committed: Vec::new(),
            emitted: Vec::new(),
            final_sent: 0,
            stable_sent: 0,
            current: Vec::new(),
            previous: None,
            cut_time: 0.0,
            decodes: 0,
            decode_seconds: 0.0,
            encoder_seconds: 0.0,
        })
    }

    /// Warm up the full window shape with deterministic padding.
    pub fn warmup(&mut self) -> Result<(), String> {
        checkpoint(&self.control)?;
        let features =
            FeatureMatrixView::new(self.frontend.mel_bins(), self.full_frames, &self.noise_mel)?;
        checkpoint(&self.control)?;
        self.decoder.decode(features)?;
        checkpoint(&self.control)
    }

    pub fn now(&self) -> f32 {
        self.start_time + usize_to_f32(self.buffer.len()) / usize_to_f32(self.sample_rate)
    }

    pub fn transcript(&self) -> &[StreamWord] {
        &self.emitted
    }

    pub const fn cut_time(&self) -> f32 {
        self.cut_time
    }

    pub fn committed(&self) -> &[StreamWord] {
        &self.committed
    }

    pub const fn decodes(&self) -> usize {
        self.decodes
    }

    pub const fn decoder_seconds(&self) -> f64 {
        self.decode_seconds
    }

    pub const fn encoder_seconds(&self) -> f64 {
        self.encoder_seconds
    }

    /// Produces the immutable dialogue input for one original channel.
    pub fn snapshot(&self, channel: usize) -> Result<ChannelSnapshot, String> {
        let words = self
            .emitted
            .iter()
            .enumerate()
            .map(|(index, word)| {
                SnapshotWord::new(
                    word.word().clone(),
                    if index < self.final_sent {
                        WordFinality::Final
                    } else {
                        WordFinality::Open
                    },
                    word.stability(),
                )
            })
            .collect();
        ChannelSnapshot::new(
            channel,
            words,
            self.cut_time,
            self.now() - self.config.horizon_sec(),
        )
    }

    /// Supplies validated model-rate samples and returns all resulting application events.
    pub fn push(&mut self, input: &ChannelAudio) -> Result<Vec<StreamEvent>, String> {
        let step = self.config.step_samples().get();
        let mut events = Vec::new();
        let mut remaining = input.samples();
        while !remaining.is_empty() {
            let room = self
                .full_samples
                .checked_sub(self.buffer.len())
                .ok_or_else(|| "stream buffer exceeds its validated window length".to_owned())?;
            let unprocessed = self
                .buffer
                .len()
                .checked_sub(self.decoded_samples)
                .ok_or_else(|| "stream decoded sample count exceeds its buffer".to_owned())?;
            let to_step = step.saturating_sub(unprocessed);
            let take = remaining.len().min(room).min(to_step);
            self.buffer.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            let grown = self
                .buffer
                .len()
                .checked_sub(self.decoded_samples)
                .ok_or_else(|| "stream decoded sample count exceeds its buffer".to_owned())?;
            if self.buffer.len() >= self.full_samples || grown >= step {
                events.extend(self.tick(StreamProgress::Advance)?);
            }
            if take == 0 && self.buffer.len() >= self.full_samples {
                return Err("stream buffer is full and did not shrink after a tick".into());
            }
        }
        Ok(events)
    }

    /// Ends the stream by decoding and committing its remaining tail.
    pub fn flush(&mut self) -> Result<Vec<StreamEvent>, String> {
        self.tick(StreamProgress::Flush)
    }

    fn decode(&mut self) -> Result<DecodedCandidate, String> {
        checkpoint(&self.control)?;
        let hop = self.frontend.hop_length();
        let fft_length = self.frontend.fft_length();
        let mel_bins = self.frontend.mel_bins();
        let length = self.buffer.len();
        let real_frames = if length >= fft_length {
            length
                .checked_sub(fft_length)
                .ok_or_else(|| "stream frame range underflows".to_owned())?
                / hop
                + 1
        } else {
            0
        };
        let padded_frames = length.div_ceil(hop).min(self.full_frames);
        for time in self.real_full_frames..padded_frames {
            let offset = time
                .checked_mul(hop)
                .ok_or_else(|| "stream feature offset overflows".to_owned())?;
            for sample in 0..fft_length {
                let index = offset
                    .checked_add(sample)
                    .ok_or_else(|| "stream feature sample index overflows".to_owned())?;
                self.frame[sample] = if index < length {
                    self.buffer[index]
                } else {
                    self.noise[index]
                };
            }
            self.frontend
                .frame_log_mel(&self.frame, &mut self.column, &mut self.scratch)?;
            for mel_bin in 0..mel_bins {
                let index = mel_bin
                    .checked_mul(self.full_frames)
                    .and_then(|value| value.checked_add(time))
                    .ok_or_else(|| "stream mel index overflows".to_owned())?;
                self.mel[index] = self.column[mel_bin];
            }
        }
        checkpoint(&self.control)?;
        let started = Instant::now();
        let features = FeatureMatrixView::new(mel_bins, self.full_frames, &self.mel)?;
        let decoded = self.decoder.decode(features)?;
        checkpoint(&self.control)?;
        let elapsed = started.elapsed().as_secs_f64();
        let (words, silence, output_frames, encoder_seconds) = decoded.into_parts();
        let decode_seconds = elapsed - encoder_seconds;
        if !elapsed.is_finite()
            || !encoder_seconds.is_finite()
            || !decode_seconds.is_finite()
            || decode_seconds < 0.0
        {
            return Err("recognition decoder reported an invalid stream duration".into());
        }
        Ok(DecodedCandidate {
            words,
            silence,
            output_frames,
            accounting: DecodedAccounting {
                real_full_frames: real_frames.min(self.full_frames),
                decoded_samples: length,
                elapsed,
                encoder_seconds,
            },
        })
    }

    fn commit_decode(&mut self, accounting: DecodedAccounting) -> Result<(), String> {
        let decode_seconds = self.decode_seconds + accounting.elapsed;
        let encoder_seconds = self.encoder_seconds + accounting.encoder_seconds;
        if !decode_seconds.is_finite() || !encoder_seconds.is_finite() {
            return Err("stream accumulated decoder duration is invalid".into());
        }
        let decodes = self
            .decodes
            .checked_add(1)
            .ok_or_else(|| "stream decode count overflows".to_owned())?;
        self.real_full_frames = accounting.real_full_frames;
        self.decode_seconds = decode_seconds;
        self.encoder_seconds = encoder_seconds;
        self.decodes = decodes;
        self.decoded_samples = accounting.decoded_samples;
        Ok(())
    }

    fn open_words(&self) -> Vec<TranscriptWord> {
        let mut words: Vec<TranscriptWord> = self
            .current
            .iter()
            .filter(|word| word.start() >= self.cut_time - 0.06)
            .cloned()
            .collect();
        while let Some(word) = words.first() {
            let duplicate = self.committed.iter().rev().take(12).any(|committed| {
                (committed.start() - word.start()).abs() <= 0.12
                    && normalize_word(committed.text()) == normalize_word(word.text())
            });
            if duplicate {
                words.remove(0);
            } else {
                break;
            }
        }
        words
    }

    fn open_line(&self) -> Result<Vec<TranscriptWord>, String> {
        let now = self.now();
        match &self.previous {
            Some(previous) => {
                let current =
                    WindowWords::new(Window::new(self.start_time, now)?, self.open_words());
                stitch_aligned(&[previous.clone(), current], 0.25)
            }
            None => Ok(self.open_words()),
        }
    }

    fn commit(&mut self, words: &[TranscriptWord]) {
        self.committed.extend(
            words
                .iter()
                .cloned()
                .map(|word| StreamWord::new(word, WordStability::Stable)),
        );
    }

    fn resolve_seam(&mut self, now: f32) -> Result<(), String> {
        let Some(previous) = self.previous.take() else {
            return Ok(());
        };
        let current = WindowWords::new(Window::new(self.start_time, now)?, self.open_words());
        let seam = seam(previous.words(), previous.end(), &current, 0.25)?;
        let committed = previous
            .words()
            .get(..seam.previous_count())
            .ok_or_else(|| "stream seam left index exceeds the previous window".to_owned())?
            .to_vec();
        self.commit(&committed);
        self.cut_time = current
            .words()
            .get(seam.current_skip())
            .map_or(now, TranscriptWord::start);
        Ok(())
    }

    fn reset_buffer(&mut self, retained_samples: usize) -> Result<(), String> {
        let retain = retained_samples.min(self.buffer.len());
        let length = self.buffer.len();
        let dropped = length
            .checked_sub(retain)
            .ok_or_else(|| "stream retained sample count exceeds its buffer".to_owned())?;
        self.start_time += usize_to_f32(dropped) / usize_to_f32(self.sample_rate);
        self.buffer.drain(..dropped);
        self.mel.copy_from_slice(&self.noise_mel);
        self.real_full_frames = 0;
        self.decoded_samples = self.buffer.len();
        self.current.clear();
        Ok(())
    }

    fn emit(
        &mut self,
        open: &[TranscriptWord],
        now: f32,
        reason: Option<FinalReason>,
        events: &mut Vec<StreamEvent>,
    ) -> Result<(), String> {
        let mut target = self.committed.clone();
        target.extend(open.iter().cloned().map(|word| {
            let stability = if word.start() < now - self.config.horizon_sec() {
                WordStability::Stable
            } else {
                WordStability::Revisable
            };
            StreamWord::new(word, stability)
        }));
        let shared = self.emitted.len().min(target.len());
        let revise_from = (0..shared)
            .find(|&index| !same(&self.emitted[index], &target[index]))
            .unwrap_or(shared);
        if revise_from < target.len() || revise_from < self.emitted.len() {
            events.push(StreamEvent::Words(WordsEvent::new(
                now,
                revise_from,
                target[revise_from..].to_vec(),
            )?));
            self.emitted.truncate(revise_from);
            self.emitted.extend_from_slice(&target[revise_from..]);
        }
        let stable = target
            .iter()
            .take_while(|word| word.stability() == WordStability::Stable)
            .count();
        if stable > self.stable_sent {
            self.stable_sent = stable;
            for word in &mut self.emitted[..stable] {
                word.set_stable();
            }
            events.push(StreamEvent::Stable(FrontierEvent::new(now, stable)?));
        }
        if self.committed.len() > self.final_sent {
            self.final_sent = self.committed.len();
            for word in &mut self.emitted[..self.final_sent] {
                word.set_stable();
            }
            let reason = reason.unwrap_or(FinalReason::ForcedOrLocked);
            events.push(StreamEvent::Final(FinalEvent::new(
                now,
                self.final_sent,
                reason,
            )?));
        }
        Ok(())
    }

    fn tick(&mut self, progress: StreamProgress) -> Result<Vec<StreamEvent>, String> {
        checkpoint(&self.control)?;
        let mut events = Vec::new();
        let length = self.buffer.len();
        if length == 0 {
            return Ok(events);
        }
        let length_seconds = usize_to_f32(length) / usize_to_f32(self.sample_rate);
        let now = self.start_time + length_seconds;
        let candidate = self.decode()?;
        checkpoint(&self.control)?;
        let DecodedCandidate {
            words: recognition_words,
            silence,
            output_frames,
            accounting,
        } = candidate;
        let words = recognition_words
            .into_iter()
            .map(TranscriptWord::from_recognition)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|word| word.start() < length_seconds)
            .map(|word| word.capped_end(length_seconds))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|word| word.shifted(self.start_time))
            .collect::<Result<Vec<_>, _>>()?;
        let upto = trunc_f32_to_usize(length_seconds * self.frame_rate.get())
            .map_err(|error| format!("stream endpoint frame count: {error}"))?
            .min(output_frames);
        let speech_state = if self.previous.is_some() || !words.is_empty() {
            SpeechState::Present
        } else {
            SpeechState::Absent
        };
        let endpoint = match &mut self.detector {
            EndpointDetector::Vad(detector) => {
                let audio = ChannelAudioView::new(&self.buffer)?;
                checkpoint(&self.control)?;
                let probabilities = detector.probabilities(audio)?;
                checkpoint(&self.control)?;
                let silence_mask = mask(&probabilities, self.config.vad_threshold().get())?;
                detect(
                    self.config.rules(),
                    &silence_mask,
                    silence_mask.len(),
                    VAD_FPS,
                    speech_state,
                )?
            }
            EndpointDetector::Blank => detect(
                self.config.rules(),
                &silence,
                upto,
                self.frame_rate.get(),
                speech_state,
            )?,
        };
        self.commit_decode(accounting)?;
        self.current = words;
        let full = length >= self.full_samples;
        let resolves_seam = match progress {
            StreamProgress::Advance => {
                length_seconds >= self.config.overlap_sec() + self.config.seam_after_sec()
                    || endpoint != Endpoint::None
                    || full
            }
            StreamProgress::Flush => true,
        };
        if self.previous.is_some() && resolves_seam {
            self.resolve_seam(now)?;
        }
        if self.config.lock_policy() == StreamLockPolicy::CommitStable && self.previous.is_none() {
            let lock_time = now - self.config.horizon_sec();
            let open = self.open_words();
            let committed = open
                .iter()
                .take_while(|word| word.start() < lock_time)
                .cloned()
                .collect::<Vec<_>>();
            if !committed.is_empty() {
                self.commit(&committed);
                self.cut_time = self.cut_time.max(lock_time);
            }
        }
        let open = self.open_line()?;
        match progress {
            StreamProgress::Flush => match speech_state {
                SpeechState::Present => {
                    self.commit(&open);
                    self.emit(&[], now, Some(FinalReason::Endpoint), &mut events)?;
                    self.reset_buffer(0)?;
                    self.cut_time = self.start_time;
                }
                SpeechState::Absent => {
                    self.emit(&open, now, None, &mut events)?;
                    self.reset_buffer(0)?;
                    self.cut_time = self.start_time;
                }
            },
            StreamProgress::Advance => match (endpoint, speech_state) {
                (Endpoint::AfterSpeech, SpeechState::Present) => {
                    self.commit(&open);
                    self.emit(&[], now, Some(FinalReason::Endpoint), &mut events)?;
                    self.reset_buffer(self.config.retained_silence_samples().get())?;
                    self.cut_time = self.start_time;
                }
                (Endpoint::NoSpeech, SpeechState::Absent)
                | (Endpoint::NoSpeech, SpeechState::Present) => {
                    self.emit(&open, now, None, &mut events)?;
                    self.reset_buffer(self.config.retained_silence_samples().get())?;
                    self.cut_time = self.start_time;
                }
                (Endpoint::AfterSpeech, SpeechState::Absent)
                | (Endpoint::None, SpeechState::Absent)
                | (Endpoint::None, SpeechState::Present) => {
                    if full {
                        self.emit(&open, now, None, &mut events)?;
                        self.previous =
                            Some(WindowWords::new(Window::new(self.start_time, now)?, open));
                        self.reset_buffer(self.config.overlap_samples().get())?;
                    } else {
                        self.emit(&open, now, None, &mut events)?;
                    }
                }
            },
        }
        Ok(events)
    }
}

fn same(left: &StreamWord, right: &StreamWord) -> bool {
    left.text() == right.text()
        && (left.start() - right.start()).abs() <= 0.1
        && (left.end() - right.end()).abs() <= 0.1
}

#[cfg(test)]
mod tests {
    use super::{EndpointDetector, StreamSession, StreamSetup};
    use crate::contracts::{EndpointSource, StreamConfig, StreamEvent, VadProbability};
    use gigaam_audio::{ChannelAudio, FeatureMatrixView, SampleRate};
    use gigaam_recognition::{
        Decoded, ExecutionControl, FrameRate, SpeechProbabilityDetector, WindowDecoder, Word,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CandidateDecoder {
        calls: Arc<AtomicUsize>,
    }

    struct FrameRateProbe {
        frame_rate_calls: Arc<AtomicUsize>,
    }

    impl WindowDecoder for FrameRateProbe {
        fn frame_rate(&self) -> FrameRate {
            self.frame_rate_calls.fetch_add(1, Ordering::SeqCst);
            FrameRate::new(8.0).expect("the fixed test frame rate is valid")
        }

        fn decode(&mut self, _features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            Err("the mismatch test must refuse before decoder execution".into())
        }
    }

    impl WindowDecoder for CandidateDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(8.0).expect("the fixed test frame rate is valid")
        }

        fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let word =
                Word::new("candidate".into(), 0.0, 0.001).expect("the fixed test word is coherent");
            Decoded::new(
                vec![word],
                vec![false; features.frames()],
                features.frames(),
                0.0,
            )
        }
    }

    struct CancellingDetector {
        control: ExecutionControl,
        calls: Arc<AtomicUsize>,
    }

    struct NonFiniteDetector {
        probability: f32,
        calls: Arc<AtomicUsize>,
    }

    impl SpeechProbabilityDetector for NonFiniteDetector {
        fn probabilities(
            &mut self,
            audio: gigaam_audio::ChannelAudioView<'_>,
        ) -> Result<Vec<f32>, String> {
            if audio.is_empty() {
                return Err("test detector requires a nonempty borrowed waveform".into());
            }
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![self.probability])
        }
    }

    impl SpeechProbabilityDetector for CancellingDetector {
        fn probabilities(
            &mut self,
            audio: gigaam_audio::ChannelAudioView<'_>,
        ) -> Result<Vec<f32>, String> {
            if audio.is_empty() {
                return Err("test detector requires a nonempty borrowed waveform".into());
            }
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.control.request_cancellation();
            Ok(vec![0.0])
        }
    }

    fn vad_config() -> StreamConfig {
        StreamConfig::timing_changes()
            .with_window_sec(0.001)
            .expect("test window duration is valid")
            .with_overlap_sec(0.0)
            .expect("test overlap duration is valid")
            .with_step_sec(0.001)
            .expect("test step duration is valid")
            .apply(
                StreamConfig::checked_default(
                    SampleRate::new(16_000).expect("the fixed VAD sample rate is positive"),
                )
                .expect("the default stream configuration is valid"),
            )
            .expect("test timing changes preserve a valid configuration")
            .with_endpoint_source(EndpointSource::Vad)
            .expect("the injected VAD endpoint source is valid")
            .with_vad_threshold(
                VadProbability::new(0.5).expect("the test VAD threshold is a probability"),
            )
            .expect("the typed VAD threshold preserves a valid configuration")
    }

    #[test]
    fn retained_silence_overflow_refuses_before_session_continuation_and_no_endpoint_keeps_it_unused()
    -> Result<(), String> {
        let rate = SampleRate::new(16)?;
        let continuation_calls = Arc::new(AtomicUsize::new(0));
        let configuration = StreamConfig::timing_changes()
            .with_keep_silence_sec(f32::MAX)?
            .apply(StreamConfig::checked_default(rate)?);
        let refused = configuration.and_then(|config| {
            continuation_calls.fetch_add(1, Ordering::SeqCst);
            StreamSession::new(StreamSetup {
                frontend: crate::test_support::frontend_at_sample_rate(rate.hertz()),
                decoder: CandidateDecoder {
                    calls: Arc::new(AtomicUsize::new(0)),
                },
                config,
                detector: EndpointDetector::<NonFiniteDetector>::Blank,
                control: ExecutionControl::without_deadline(),
            })
            .map(|_| ())
        });
        let error = match refused {
            Ok(()) => {
                return Err(
                    "overflowing retained silence must refuse before session continuation".into(),
                );
            }
            Err(error) => error,
        };
        assert!(error.contains("stream retained silence duration at 16 Hz"));
        assert_eq!(continuation_calls.load(Ordering::SeqCst), 0);

        let config = StreamConfig::timing_changes()
            .with_window_sec(1.0)?
            .with_overlap_sec(0.0)?
            .with_step_sec(0.5)?
            .with_keep_silence_sec(0.25)?
            .apply(StreamConfig::checked_default(rate)?)?;
        let decoder_calls = Arc::new(AtomicUsize::new(0));
        let mut session = StreamSession::new(StreamSetup {
            frontend: crate::test_support::frontend_at_sample_rate(rate.hertz()),
            decoder: CandidateDecoder {
                calls: Arc::clone(&decoder_calls),
            },
            config,
            detector: EndpointDetector::<NonFiniteDetector>::Blank,
            control: ExecutionControl::without_deadline(),
        })?;
        let input = ChannelAudio::new(vec![0.0; 8])?;
        let advancing = session.push(&input)?;
        assert!(
            advancing
                .iter()
                .all(|event| !matches!(event, StreamEvent::Final(_))),
            "a successful non-endpoint advance must not consume retained silence"
        );
        let terminal = session.flush()?;
        assert!(
            terminal
                .iter()
                .any(|event| matches!(event, StreamEvent::Final(_))),
            "the terminal flush must preserve the normal finalization behavior"
        );
        assert_eq!(decoder_calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn bound_stream_rate_mismatch_refuses_before_decoder_capability_work() {
        let frame_rate_calls = Arc::new(AtomicUsize::new(0));
        let result = StreamSession::new(StreamSetup {
            frontend: crate::test_support::frontend(),
            decoder: FrameRateProbe {
                frame_rate_calls: Arc::clone(&frame_rate_calls),
            },
            config: StreamConfig::checked_default(
                SampleRate::new(8).expect("test sample rate is positive"),
            )
            .expect("the bound mismatched stream configuration is otherwise valid"),
            detector: EndpointDetector::<NonFiniteDetector>::Blank,
            control: ExecutionControl::without_deadline(),
        });
        let error = match result {
            Ok(_) => panic!("a mismatched bound stream rate must refuse"),
            Err(error) => error,
        };
        assert!(error.contains("stream configuration sample rate 8 Hz"));
        assert_eq!(frame_rate_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn detector_cancellation_commits_no_stream_candidate_or_successor() {
        let control = ExecutionControl::for_request();
        let decoder_calls = Arc::new(AtomicUsize::new(0));
        let detector_calls = Arc::new(AtomicUsize::new(0));
        let mut session = StreamSession::new(StreamSetup {
            frontend: crate::test_support::frontend_at_sample_rate(16_000),
            decoder: CandidateDecoder {
                calls: Arc::clone(&decoder_calls),
            },
            config: vad_config(),
            detector: EndpointDetector::Vad(CancellingDetector {
                control: control.clone(),
                calls: Arc::clone(&detector_calls),
            }),
            control: control.clone(),
        })
        .expect("the test VAD session is valid");
        let input = ChannelAudio::new(vec![0.0; 16])
            .expect("the test input contains finite model-rate samples");

        assert!(session.push(&input).is_err());
        assert_eq!(decoder_calls.load(Ordering::SeqCst), 1);
        assert_eq!(detector_calls.load(Ordering::SeqCst), 1);
        assert_eq!(session.decodes(), 0);
        assert!(session.committed().is_empty());
        assert!(session.transcript().is_empty());
        assert!(
            session
                .snapshot(0)
                .expect("a cancelled candidate still yields a public empty snapshot")
                .words()
                .is_empty()
        );

        assert!(session.flush().is_err());
        assert_eq!(decoder_calls.load(Ordering::SeqCst), 1);
        assert_eq!(detector_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stream_refuses_nonfinite_vad_probabilities_before_committing_a_candidate() {
        for (name, probability) in [
            ("NaN", f32::NAN),
            ("positive infinity", f32::INFINITY),
            ("negative infinity", f32::NEG_INFINITY),
        ] {
            let control = ExecutionControl::for_request();
            let decoder_calls = Arc::new(AtomicUsize::new(0));
            let detector_calls = Arc::new(AtomicUsize::new(0));
            let mut session = StreamSession::new(StreamSetup {
                frontend: crate::test_support::frontend_at_sample_rate(16_000),
                decoder: CandidateDecoder {
                    calls: Arc::clone(&decoder_calls),
                },
                config: vad_config(),
                detector: EndpointDetector::Vad(NonFiniteDetector {
                    probability,
                    calls: Arc::clone(&detector_calls),
                }),
                control,
            })
            .expect("the test VAD session is valid");
            let input = ChannelAudio::new(vec![0.0; 16])
                .expect("the test input contains finite model-rate samples");

            assert!(
                session.push(&input).is_err(),
                "{name} VAD output must refuse the stream candidate"
            );
            assert_eq!(decoder_calls.load(Ordering::SeqCst), 1);
            assert_eq!(detector_calls.load(Ordering::SeqCst), 1);
            assert_eq!(session.decodes(), 0);
            assert!(session.committed().is_empty());
            assert!(session.transcript().is_empty());
        }
    }
}
