// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Validated application values shared by batch and streaming transcription.

use crate::channel_selection::{CorrelationThreshold, SelectionWindowSamples};
use crate::endpoint::EndpointRules;
use gigaam_audio::SampleRate;
use gigaam_primitives::{trunc_f32_to_usize, usize_to_f32};
use gigaam_recognition::{ExecutionControl, ExecutionState, Word as RecognitionWord};

fn finite_nonnegative(value: f32, name: &str) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be finite and nonnegative"));
    }
    Ok(())
}

/// A finite speech probability in the closed unit interval.
///
/// This is the single Transcription-owned value domain for threshold configuration and injected
/// detector output validation.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct VadProbability(f32);

impl VadProbability {
    pub const DEFAULT: Self = Self(0.5);

    pub fn new(value: f32) -> Result<Self, String> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err("VAD probability must be finite and within [0, 1]".into());
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

fn finite_positive(value: f32, name: &str) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{name} must be finite and positive"));
    }
    Ok(())
}

/// A checked configuration-derived sample count that cannot be confused with raw duration input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TemporalSampleCount(usize);

impl TemporalSampleCount {
    pub(crate) const fn get(self) -> usize {
        self.0
    }
}

/// The complete semantic field set for configuration-derived temporal sample counts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalField {
    BatchWindow,
    StreamWindow,
    StreamStep,
    StreamOverlapRetention,
    StreamRetainedSilence,
    VadMinimumSpeech,
    VadMinimumSilence,
    VadPadding,
    VadDoubledPadding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TemporalValueRule {
    Positive,
    Nonnegative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TemporalZeroRule {
    Refuse,
    Allow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TemporalUnit {
    Seconds,
    Milliseconds,
}

impl TemporalField {
    const fn name(self) -> &'static str {
        match self {
            Self::BatchWindow => "batch window duration",
            Self::StreamWindow => "stream window duration",
            Self::StreamStep => "stream step duration",
            Self::StreamOverlapRetention => "stream overlap duration",
            Self::StreamRetainedSilence => "stream retained silence duration",
            Self::VadMinimumSpeech => "VAD minimum speech duration",
            Self::VadMinimumSilence => "VAD minimum silence duration",
            Self::VadPadding => "VAD speech padding",
            Self::VadDoubledPadding => "VAD doubled speech padding",
        }
    }

    const fn value_rule(self) -> TemporalValueRule {
        match self {
            Self::BatchWindow | Self::StreamWindow | Self::StreamStep => {
                TemporalValueRule::Positive
            }
            Self::StreamOverlapRetention
            | Self::StreamRetainedSilence
            | Self::VadMinimumSpeech
            | Self::VadMinimumSilence
            | Self::VadPadding
            | Self::VadDoubledPadding => TemporalValueRule::Nonnegative,
        }
    }

    const fn zero_rule(self) -> TemporalZeroRule {
        match self {
            Self::BatchWindow | Self::StreamWindow | Self::StreamStep => TemporalZeroRule::Refuse,
            Self::StreamOverlapRetention
            | Self::StreamRetainedSilence
            | Self::VadMinimumSpeech
            | Self::VadMinimumSilence
            | Self::VadPadding
            | Self::VadDoubledPadding => TemporalZeroRule::Allow,
        }
    }

    const fn unit(self) -> TemporalUnit {
        match self {
            Self::BatchWindow
            | Self::StreamWindow
            | Self::StreamStep
            | Self::StreamOverlapRetention
            | Self::StreamRetainedSilence => TemporalUnit::Seconds,
            Self::VadMinimumSpeech
            | Self::VadMinimumSilence
            | Self::VadPadding
            | Self::VadDoubledPadding => TemporalUnit::Milliseconds,
        }
    }
}

fn temporal_error(field: TemporalField, rate: SampleRate, detail: &str) -> String {
    format!("{} at {} Hz {detail}", field.name(), rate.hertz())
}

/// Converts one semantic temporal value at its owner rate with the established truncation order.
pub(crate) fn temporal_sample_count(
    value: f32,
    rate: SampleRate,
    field: TemporalField,
) -> Result<TemporalSampleCount, String> {
    match field.value_rule() {
        TemporalValueRule::Positive if !value.is_finite() || value <= 0.0 => {
            return Err(temporal_error(field, rate, "must be finite and positive"));
        }
        TemporalValueRule::Nonnegative if !value.is_finite() || value < 0.0 => {
            return Err(temporal_error(
                field,
                rate,
                "must be finite and nonnegative",
            ));
        }
        TemporalValueRule::Positive | TemporalValueRule::Nonnegative => {}
    }
    let rate_samples = rate
        .as_usize()
        .map_err(|error| format!("{}: {error}", temporal_error(field, rate, "rate")))?;
    let scaled = match field.unit() {
        TemporalUnit::Seconds => value * usize_to_f32(rate_samples),
        TemporalUnit::Milliseconds => value / 1000.0 * usize_to_f32(rate_samples),
    };
    let samples = trunc_f32_to_usize(scaled)
        .map_err(|error| format!("{}: {error}", temporal_error(field, rate, "sample count")))?;
    match (field.zero_rule(), samples) {
        (TemporalZeroRule::Refuse, 0) => Err(temporal_error(
            field,
            rate,
            "must truncate to at least one sample",
        )),
        (TemporalZeroRule::Refuse | TemporalZeroRule::Allow, samples) => {
            Ok(TemporalSampleCount(samples))
        }
    }
}

/// Derives the fixed semantic doubled VAD padding before detector execution can begin.
pub(crate) fn doubled_temporal_sample_count(
    value: TemporalSampleCount,
    rate: SampleRate,
    field: TemporalField,
) -> Result<TemporalSampleCount, String> {
    let doubled = value
        .get()
        .checked_mul(2)
        .ok_or_else(|| temporal_error(field, rate, "sample count exceeds usize"))?;
    match (field.zero_rule(), doubled) {
        (TemporalZeroRule::Refuse, 0) => Err(temporal_error(
            field,
            rate,
            "must truncate to at least one sample",
        )),
        (TemporalZeroRule::Refuse | TemporalZeroRule::Allow, samples) => {
            Ok(TemporalSampleCount(samples))
        }
    }
}

/// A recognized word at the Transcription application boundary.
///
/// Recognition owns model-relative words. This value owns their validated application-time
/// projection and is the only word value exposed to application consumers.
#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptWord {
    text: String,
    start: f32,
    end: f32,
}

impl TranscriptWord {
    pub fn new(text: String, start: f32, end: f32) -> Result<Self, String> {
        if text.trim().is_empty() {
            return Err("transcript word text must not be empty".into());
        }
        finite_nonnegative(start, "transcript word start")?;
        finite_nonnegative(end, "transcript word end")?;
        if end < start {
            return Err("transcript word end must not precede its start".into());
        }
        Ok(Self { text, start, end })
    }

    pub fn from_recognition(word: RecognitionWord) -> Result<Self, String> {
        Self::new(word.text().to_owned(), word.start(), word.end())
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub const fn start(&self) -> f32 {
        self.start
    }

    pub const fn end(&self) -> f32 {
        self.end
    }

    pub fn shifted(self, offset: f32) -> Result<Self, String> {
        if !offset.is_finite() {
            return Err("transcript word offset must be finite".into());
        }
        Self::new(self.text, self.start + offset, self.end + offset)
    }

    pub fn capped_end(self, maximum: f32) -> Result<Self, String> {
        finite_nonnegative(maximum, "transcript word end boundary")?;
        if maximum < self.start {
            return Err("transcript word end boundary precedes the word start".into());
        }
        Self::new(self.text, self.start, self.end.min(maximum))
    }
}

/// How a batch clip shorter than its recognition window is presented to the decoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PadPolicy {
    Exact,
    PadToWindow,
}

/// Constructor-validated batch-window configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchConfig {
    sample_rate: SampleRate,
    window_sec: f32,
    overlap_sec: f32,
    padding: PadPolicy,
    full_window_samples: TemporalSampleCount,
}

impl BatchConfig {
    pub fn new(
        sample_rate: SampleRate,
        window_sec: f32,
        overlap_sec: f32,
        padding: PadPolicy,
    ) -> Result<Self, String> {
        let full_window_samples =
            temporal_sample_count(window_sec, sample_rate, TemporalField::BatchWindow)?;
        finite_nonnegative(overlap_sec, "batch overlap duration")?;
        if overlap_sec >= window_sec {
            return Err("batch overlap duration must be shorter than the window".into());
        }
        Ok(Self {
            sample_rate,
            window_sec,
            overlap_sec,
            padding,
            full_window_samples,
        })
    }

    /// Returns the model sample rate that bound every execution-derived batch count.
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    pub const fn window_sec(&self) -> f32 {
        self.window_sec
    }

    pub const fn overlap_sec(&self) -> f32 {
        self.overlap_sec
    }

    pub const fn padding(&self) -> PadPolicy {
        self.padding
    }

    /// Returns the validated full-window shape needed by warmup and batch execution.
    pub const fn full_window_samples(&self) -> usize {
        self.full_window_samples.get()
    }
}

/// Aggregated successful stage timings for one single-channel batch transcript.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageTimings {
    frontend_seconds: f64,
    encoder_seconds: f64,
    decode_seconds: f64,
}

impl StageTimings {
    pub const fn frontend_seconds(&self) -> f64 {
        self.frontend_seconds
    }

    pub const fn encoder_seconds(&self) -> f64 {
        self.encoder_seconds
    }

    pub const fn decode_seconds(&self) -> f64 {
        self.decode_seconds
    }

    pub(crate) fn add_frontend(&mut self, value: f64) {
        self.frontend_seconds += value;
    }

    pub(crate) fn add_encoder(&mut self, value: f64) {
        self.encoder_seconds += value;
    }

    pub(crate) fn add_decode(&mut self, value: f64) {
        self.decode_seconds += value;
    }

    pub fn combined(self, other: Self) -> Self {
        Self {
            frontend_seconds: self.frontend_seconds + other.frontend_seconds,
            encoder_seconds: self.encoder_seconds + other.encoder_seconds,
            decode_seconds: self.decode_seconds + other.decode_seconds,
        }
    }
}

/// The application result of transcribing one complete channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    text: String,
    words: Vec<TranscriptWord>,
    windows: usize,
}

impl Transcript {
    pub(crate) fn new(words: Vec<TranscriptWord>, windows: usize) -> Self {
        let text = words_to_text(&words);
        Self {
            text,
            words,
            windows,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn words(&self) -> &[TranscriptWord] {
        &self.words
    }

    pub fn into_words(self) -> Vec<TranscriptWord> {
        self.words
    }

    pub const fn windows(&self) -> usize {
        self.windows
    }
}

/// A client-visible streaming word with a stable-frontier observation.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamWord {
    word: TranscriptWord,
    stability: WordStability,
}

impl StreamWord {
    pub(crate) fn new(word: TranscriptWord, stability: WordStability) -> Self {
        Self { word, stability }
    }

    pub fn text(&self) -> &str {
        self.word.text()
    }

    pub const fn start(&self) -> f32 {
        self.word.start()
    }

    pub const fn end(&self) -> f32 {
        self.word.end()
    }

    pub const fn stability(&self) -> WordStability {
        self.stability
    }

    /// Returns this immutable client-visible word with a later typed stability observation.
    pub fn with_stability(mut self, stability: WordStability) -> Self {
        self.stability = stability;
        self
    }

    pub(crate) fn set_stable(&mut self) {
        self.stability = WordStability::Stable;
    }

    pub(crate) fn word(&self) -> &TranscriptWord {
        &self.word
    }
}

/// A replacement patch over the application-visible streaming word line.
#[derive(Clone, Debug, PartialEq)]
pub struct WordsEvent {
    at: f32,
    revise_from: usize,
    words: Vec<StreamWord>,
}

impl WordsEvent {
    pub(crate) fn new(at: f32, revise_from: usize, words: Vec<StreamWord>) -> Result<Self, String> {
        finite_nonnegative(at, "stream word event time")?;
        Ok(Self {
            at,
            revise_from,
            words,
        })
    }

    pub const fn at(&self) -> f32 {
        self.at
    }

    pub const fn revise_from(&self) -> usize {
        self.revise_from
    }

    pub fn words(&self) -> &[StreamWord] {
        &self.words
    }
}

/// A stable or final prefix frontier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierEvent {
    at: f32,
    upto: usize,
}

impl FrontierEvent {
    pub(crate) fn new(at: f32, upto: usize) -> Result<Self, String> {
        finite_nonnegative(at, "stream frontier event time")?;
        Ok(Self { at, upto })
    }

    pub const fn at(&self) -> f32 {
        self.at
    }

    pub const fn upto(&self) -> usize {
        self.upto
    }
}

/// A final prefix frontier and its utterance-boundary marker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinalEvent {
    frontier: FrontierEvent,
    reason: FinalReason,
}

impl FinalEvent {
    pub(crate) fn new(at: f32, upto: usize, reason: FinalReason) -> Result<Self, String> {
        Ok(Self {
            frontier: FrontierEvent::new(at, upto)?,
            reason,
        })
    }

    pub const fn at(&self) -> f32 {
        self.frontier.at()
    }

    pub const fn upto(&self) -> usize {
        self.frontier.upto()
    }

    pub const fn reason(&self) -> FinalReason {
        self.reason
    }
}

/// Why a streaming prefix became final.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinalReason {
    Endpoint,
    ForcedOrLocked,
}

/// One immutable streaming application event.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    Words(WordsEvent),
    Stable(FrontierEvent),
    Final(FinalEvent),
}

/// The endpoint source selected by a process adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EndpointSource {
    Blank,
    Vad,
}

/// How the stable frontier affects the committed streaming prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamLockPolicy {
    Advisory,
    CommitStable,
}

/// The exhaustive channel-activation policy for an incremental multi-channel stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StreamingChannelPolicy {
    /// Activate every original channel before the first input frame without correlation.
    AllChannels,
    /// Delay activation until the exact model-rate analysis window is available.
    Deduplicate {
        threshold: CorrelationThreshold,
        analysis_window: SelectionWindowSamples,
    },
}

/// The client-visible material selected from a multi-channel stream transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamEmissionMode {
    /// Emit only channel-labelled streaming word events.
    Words,
    /// Emit only revision-aware dialogue patches.
    Dialog,
    /// Emit channel-labelled word events followed by dialogue patches.
    WordsAndDialog,
}

/// Constructor-validated streaming configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamConfig {
    sample_rate: SampleRate,
    endpoint_source: EndpointSource,
    vad_threshold: VadProbability,
    window_sec: f32,
    overlap_sec: f32,
    step_sec: f32,
    horizon_sec: f32,
    lock_policy: StreamLockPolicy,
    seam_after_sec: f32,
    rules: EndpointRules,
    keep_silence_sec: f32,
    full_window_samples: TemporalSampleCount,
    step_samples: TemporalSampleCount,
    overlap_samples: TemporalSampleCount,
    retained_silence_samples: TemporalSampleCount,
}

/// Complete scalar input to one fresh stream configuration reconstruction.
#[derive(Clone, Copy)]
struct StreamConfigValues {
    endpoint_source: EndpointSource,
    vad_threshold: VadProbability,
    window_sec: f32,
    overlap_sec: f32,
    step_sec: f32,
    horizon_sec: f32,
    lock_policy: StreamLockPolicy,
    seam_after_sec: f32,
    rules: EndpointRules,
    keep_silence_sec: f32,
}

impl StreamConfigValues {
    fn from_config(config: &StreamConfig) -> Self {
        Self {
            endpoint_source: config.endpoint_source,
            vad_threshold: config.vad_threshold,
            window_sec: config.window_sec,
            overlap_sec: config.overlap_sec,
            step_sec: config.step_sec,
            horizon_sec: config.horizon_sec,
            lock_policy: config.lock_policy,
            seam_after_sec: config.seam_after_sec,
            rules: config.rules,
            keep_silence_sec: config.keep_silence_sec,
        }
    }
}

impl StreamConfig {
    fn new(sample_rate: SampleRate, values: StreamConfigValues) -> Result<Self, String> {
        let StreamConfigValues {
            endpoint_source,
            vad_threshold,
            window_sec,
            overlap_sec,
            step_sec,
            horizon_sec,
            lock_policy,
            seam_after_sec,
            rules,
            keep_silence_sec,
        } = values;
        finite_positive(window_sec, "stream window duration")?;
        finite_nonnegative(overlap_sec, "stream overlap duration")?;
        if overlap_sec >= window_sec {
            return Err("stream overlap duration must be shorter than the window".into());
        }
        finite_positive(horizon_sec, "stream horizon duration")?;
        finite_nonnegative(seam_after_sec, "stream seam delay")?;
        rules.validate()?;
        if endpoint_source == EndpointSource::Vad {
            let model_rate = sample_rate
                .as_usize()
                .map_err(|error| format!("vad endpoint model rate: {error}"))?;
            if model_rate != gigaam_recognition::vad::VAD_SR {
                return Err(format!(
                    "vad endpoint requires 16 kHz, model has {model_rate}"
                ));
            }
        }
        let full_window_samples =
            temporal_sample_count(window_sec, sample_rate, TemporalField::StreamWindow)?;
        let step_samples = temporal_sample_count(step_sec, sample_rate, TemporalField::StreamStep)?;
        let overlap_samples = temporal_sample_count(
            overlap_sec,
            sample_rate,
            TemporalField::StreamOverlapRetention,
        )?;
        let retained_silence_samples = temporal_sample_count(
            keep_silence_sec,
            sample_rate,
            TemporalField::StreamRetainedSilence,
        )?;
        Ok(Self {
            sample_rate,
            endpoint_source,
            vad_threshold,
            window_sec,
            overlap_sec,
            step_sec,
            horizon_sec,
            lock_policy,
            seam_after_sec,
            rules,
            keep_silence_sec,
            full_window_samples,
            step_samples,
            overlap_samples,
            retained_silence_samples,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        Self::new(self.sample_rate, StreamConfigValues::from_config(self)).map(|_| ())
    }

    pub fn checked_default(sample_rate: SampleRate) -> Result<Self, String> {
        Self::new(
            sample_rate,
            StreamConfigValues {
                endpoint_source: EndpointSource::Blank,
                vad_threshold: VadProbability::DEFAULT,
                window_sec: 30.0,
                overlap_sec: 6.0,
                step_sec: 0.5,
                horizon_sec: 5.0,
                lock_policy: StreamLockPolicy::Advisory,
                seam_after_sec: 2.0,
                rules: EndpointRules::default(),
                keep_silence_sec: 0.5,
            },
        )
    }

    pub fn with_endpoint_source(mut self, value: EndpointSource) -> Result<Self, String> {
        self.endpoint_source = value;
        self.validate()?;
        Ok(self)
    }

    pub fn with_horizon_sec(mut self, value: f32) -> Result<Self, String> {
        self.horizon_sec = value;
        self.validate()?;
        Ok(self)
    }

    pub fn with_lock_policy(mut self, value: StreamLockPolicy) -> Result<Self, String> {
        self.lock_policy = value;
        self.validate()?;
        Ok(self)
    }

    pub fn with_vad_threshold(mut self, value: VadProbability) -> Result<Self, String> {
        self.vad_threshold = value;
        self.validate()?;
        Ok(self)
    }

    pub fn with_rules(mut self, value: EndpointRules) -> Result<Self, String> {
        self.rules = value;
        self.validate()?;
        Ok(self)
    }

    /// Begins one all-or-nothing change to interdependent timing values.
    pub fn timing_changes() -> StreamTimingChanges {
        StreamTimingChanges::default()
    }

    /// Returns the model sample rate that bound every execution-derived stream count.
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    pub const fn endpoint_source(&self) -> EndpointSource {
        self.endpoint_source
    }

    pub const fn vad_threshold(&self) -> VadProbability {
        self.vad_threshold
    }

    pub const fn window_sec(&self) -> f32 {
        self.window_sec
    }

    pub const fn overlap_sec(&self) -> f32 {
        self.overlap_sec
    }

    pub const fn step_sec(&self) -> f32 {
        self.step_sec
    }

    pub const fn horizon_sec(&self) -> f32 {
        self.horizon_sec
    }

    pub const fn lock_policy(&self) -> StreamLockPolicy {
        self.lock_policy
    }

    pub const fn seam_after_sec(&self) -> f32 {
        self.seam_after_sec
    }

    pub const fn rules(&self) -> EndpointRules {
        self.rules
    }

    pub const fn keep_silence_sec(&self) -> f32 {
        self.keep_silence_sec
    }

    pub(crate) const fn full_window_samples(&self) -> TemporalSampleCount {
        self.full_window_samples
    }

    pub(crate) const fn step_samples(&self) -> TemporalSampleCount {
        self.step_samples
    }

    pub(crate) const fn overlap_samples(&self) -> TemporalSampleCount {
        self.overlap_samples
    }

    pub(crate) const fn retained_silence_samples(&self) -> TemporalSampleCount {
        self.retained_silence_samples
    }
}

/// A private-field builder that validates each scalar and applies all timing changes together.
///
/// It avoids exposing a partially invalid window/overlap relationship while an adapter changes
/// more than one setting.
#[derive(Default)]
pub struct StreamTimingChanges {
    window_sec: Option<f32>,
    overlap_sec: Option<f32>,
    step_sec: Option<f32>,
    horizon_sec: Option<f32>,
    seam_after_sec: Option<f32>,
    keep_silence_sec: Option<f32>,
}

impl StreamTimingChanges {
    pub fn with_window_sec(mut self, value: f32) -> Result<Self, String> {
        finite_positive(value, "stream window duration")?;
        self.window_sec = Some(value);
        Ok(self)
    }

    pub fn with_overlap_sec(mut self, value: f32) -> Result<Self, String> {
        finite_nonnegative(value, "stream overlap duration")?;
        self.overlap_sec = Some(value);
        Ok(self)
    }

    pub fn with_step_sec(mut self, value: f32) -> Result<Self, String> {
        finite_positive(value, "stream step duration")?;
        self.step_sec = Some(value);
        Ok(self)
    }

    pub fn with_horizon_sec(mut self, value: f32) -> Result<Self, String> {
        finite_positive(value, "stream horizon duration")?;
        self.horizon_sec = Some(value);
        Ok(self)
    }

    pub fn with_seam_after_sec(mut self, value: f32) -> Result<Self, String> {
        finite_nonnegative(value, "stream seam delay")?;
        self.seam_after_sec = Some(value);
        Ok(self)
    }

    pub fn with_keep_silence_sec(mut self, value: f32) -> Result<Self, String> {
        finite_nonnegative(value, "stream retained silence duration")?;
        self.keep_silence_sec = Some(value);
        Ok(self)
    }

    pub fn apply(self, base: StreamConfig) -> Result<StreamConfig, String> {
        let StreamConfig {
            sample_rate,
            endpoint_source,
            vad_threshold,
            window_sec,
            overlap_sec,
            step_sec,
            horizon_sec,
            lock_policy,
            seam_after_sec,
            rules,
            keep_silence_sec,
            ..
        } = base;
        StreamConfig::new(
            sample_rate,
            StreamConfigValues {
                endpoint_source,
                vad_threshold,
                window_sec: match self.window_sec {
                    Some(value) => value,
                    None => window_sec,
                },
                overlap_sec: match self.overlap_sec {
                    Some(value) => value,
                    None => overlap_sec,
                },
                step_sec: match self.step_sec {
                    Some(value) => value,
                    None => step_sec,
                },
                horizon_sec: match self.horizon_sec {
                    Some(value) => value,
                    None => horizon_sec,
                },
                lock_policy,
                seam_after_sec: match self.seam_after_sec {
                    Some(value) => value,
                    None => seam_after_sec,
                },
                rules,
                keep_silence_sec: match self.keep_silence_sec {
                    Some(value) => value,
                    None => keep_silence_sec,
                },
            },
        )
    }
}

/// One immutable channel word with the current final and stable observations.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotWord {
    word: TranscriptWord,
    finality: WordFinality,
    stability: WordStability,
}

impl SnapshotWord {
    pub(crate) fn new(
        word: TranscriptWord,
        finality: WordFinality,
        stability: WordStability,
    ) -> Self {
        Self {
            word,
            finality,
            stability,
        }
    }

    pub fn word(&self) -> &TranscriptWord {
        &self.word
    }

    pub const fn finality(&self) -> WordFinality {
        self.finality
    }

    pub const fn stability(&self) -> WordStability {
        self.stability
    }
}

/// Finality observation attached to a word in a channel snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WordFinality {
    Open,
    Final,
}

/// Stability observation attached to a word in a channel snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WordStability {
    Revisable,
    Stable,
}

/// Immutable streaming state supplied to dialogue reconstruction for one original channel.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelSnapshot {
    channel: usize,
    words: Vec<SnapshotWord>,
    cut_time: f32,
    stable_frontier: f32,
}

impl ChannelSnapshot {
    pub(crate) fn new(
        channel: usize,
        words: Vec<SnapshotWord>,
        cut_time: f32,
        stable_frontier: f32,
    ) -> Result<Self, String> {
        finite_nonnegative(cut_time, "channel committed frontier")?;
        if !stable_frontier.is_finite() {
            return Err("channel stable frontier must be finite".into());
        }
        Ok(Self {
            channel,
            words,
            cut_time,
            stable_frontier,
        })
    }

    pub const fn channel(&self) -> usize {
        self.channel
    }

    pub fn words(&self) -> &[SnapshotWord] {
        &self.words
    }

    pub const fn cut_time(&self) -> f32 {
        self.cut_time
    }

    pub const fn stable_frontier(&self) -> f32 {
        self.stable_frontier
    }
}

/// A channel-labelled application word in a final batch turn.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelWord {
    channel: usize,
    word: TranscriptWord,
}

impl ChannelWord {
    pub(crate) fn new(channel: usize, word: TranscriptWord) -> Self {
        Self { channel, word }
    }

    pub const fn channel(&self) -> usize {
        self.channel
    }

    pub fn word(&self) -> &TranscriptWord {
        &self.word
    }
}

/// One continuous channel-labelled batch turn.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    channel: usize,
    start: f32,
    end: f32,
    words: Vec<ChannelWord>,
}

impl Turn {
    pub(crate) fn new(
        channel: usize,
        start: f32,
        end: f32,
        words: Vec<ChannelWord>,
    ) -> Result<Self, String> {
        finite_nonnegative(start, "turn start")?;
        finite_nonnegative(end, "turn end")?;
        if end < start {
            return Err("turn end must not precede its start".into());
        }
        if words.is_empty() {
            return Err("turn must contain at least one word".into());
        }
        Ok(Self {
            channel,
            start,
            end,
            words,
        })
    }

    pub const fn channel(&self) -> usize {
        self.channel
    }

    pub const fn start(&self) -> f32 {
        self.start
    }

    pub const fn end(&self) -> f32 {
        self.end
    }

    pub fn words(&self) -> &[ChannelWord] {
        &self.words
    }

    pub fn text(&self) -> String {
        self.words
            .iter()
            .map(|word| word.word.text())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// One dialogue turn in the current revision line.
#[derive(Clone, Debug, PartialEq)]
pub struct DialogTurn {
    channel: usize,
    index: usize,
    start: f32,
    end: f32,
    text: String,
    stability: WordStability,
    finality: WordFinality,
    backchannel: BackchannelMark,
}

/// Internal grouped inputs for one validated dialogue turn.
pub(crate) struct DialogTurnData {
    pub channel: usize,
    pub index: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub stability: WordStability,
    pub finality: WordFinality,
    pub backchannel: BackchannelMark,
}

impl DialogTurn {
    pub(crate) fn new(input: DialogTurnData) -> Result<Self, String> {
        if input.text.trim().is_empty() {
            return Err("dialogue turn text must not be empty".into());
        }
        finite_nonnegative(input.start, "dialogue turn start")?;
        finite_nonnegative(input.end, "dialogue turn end")?;
        if input.end < input.start {
            return Err("dialogue turn end must not precede its start".into());
        }
        Ok(Self {
            channel: input.channel,
            index: input.index,
            start: input.start,
            end: input.end,
            text: input.text,
            stability: input.stability,
            finality: input.finality,
            backchannel: input.backchannel,
        })
    }

    pub const fn channel(&self) -> usize {
        self.channel
    }

    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn start(&self) -> f32 {
        self.start
    }

    pub const fn end(&self) -> f32 {
        self.end
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub const fn stability(&self) -> WordStability {
        self.stability
    }

    pub const fn finality(&self) -> WordFinality {
        self.finality
    }

    pub const fn backchannel(&self) -> BackchannelMark {
        self.backchannel
    }

    pub(crate) fn with_backchannel(mut self, value: BackchannelMark) -> Self {
        self.backchannel = value;
        self
    }
}

/// Whether dialogue reconstruction identified a short overlapping backchannel turn.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackchannelMark {
    No,
    Yes,
}

/// An immutable replacement patch for the dialogue line.
#[derive(Clone, Debug, PartialEq)]
pub struct TurnsPatch {
    revise_from: usize,
    turns: Vec<DialogTurn>,
    frontier: f32,
}

impl TurnsPatch {
    pub(crate) fn new(
        revise_from: usize,
        turns: Vec<DialogTurn>,
        frontier: f32,
    ) -> Result<Self, String> {
        if !frontier.is_finite() {
            return Err("dialogue frontier must be finite".into());
        }
        Ok(Self {
            revise_from,
            turns,
            frontier,
        })
    }

    pub const fn revise_from(&self) -> usize {
        self.revise_from
    }

    pub fn turns(&self) -> &[DialogTurn] {
        &self.turns
    }

    pub const fn frontier(&self) -> f32 {
        self.frontier
    }
}

pub(crate) fn words_to_text(words: &[TranscriptWord]) -> String {
    words
        .iter()
        .map(TranscriptWord::text)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Checks one caller-owned execution control without taking terminal ownership.
pub(crate) fn checkpoint(control: &ExecutionControl) -> Result<(), String> {
    match control.state() {
        ExecutionState::Ready | ExecutionState::Queued | ExecutionState::Running => Ok(()),
        ExecutionState::CancelRequested | ExecutionState::Cancelled => {
            Err("execution cancelled".into())
        }
        ExecutionState::Completed | ExecutionState::Failed => {
            Err("execution is already terminal".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BatchConfig, EndpointSource, PadPolicy, StreamConfig, StreamConfigValues, StreamEvent,
        StreamLockPolicy, TranscriptWord, VadProbability,
    };
    use crate::endpoint::EndpointRules;
    use crate::stream::{EndpointDetector, StreamSession, StreamSetup};
    use gigaam_audio::{ChannelAudio, FeatureMatrixView, SampleRate};
    use gigaam_recognition::{
        Decoded, ExecutionControl, FrameRate, SpeechProbabilityDetector, WindowDecoder, Word,
    };

    struct HistoryDecoder;

    struct BlankHistoryDetector;

    impl WindowDecoder for HistoryDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(8.0).expect("the fixed history decoder frame rate is positive")
        }

        fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            let word = Word::new("history".into(), 0.0, 0.001)
                .expect("the fixed history word is coherent");
            Decoded::new(
                vec![word],
                vec![false; features.frames()],
                features.frames(),
                0.0,
            )
        }
    }

    impl SpeechProbabilityDetector for BlankHistoryDetector {
        fn probabilities(
            &mut self,
            _audio: gigaam_audio::ChannelAudioView<'_>,
        ) -> Result<Vec<f32>, String> {
            Err("blank-endpoint history sessions never invoke a VAD detector".into())
        }
    }

    fn emitted_stream_history(config: StreamConfig) -> Result<Vec<StreamEvent>, String> {
        let mut session = StreamSession::new(StreamSetup {
            frontend: crate::test_support::frontend_at_sample_rate(config.sample_rate().hertz()),
            decoder: HistoryDecoder,
            config,
            detector: EndpointDetector::<BlankHistoryDetector>::Blank,
            control: ExecutionControl::without_deadline(),
        })?;
        let input = ChannelAudio::new(vec![0.0; 16])?;
        let mut events = session.push(&input)?;
        events.extend(session.flush()?);
        Ok(events)
    }

    #[test]
    fn transcript_words_refuse_invalid_application_values() {
        assert!(TranscriptWord::new(" ".into(), 0.0, 0.0).is_err());
        assert!(TranscriptWord::new("word".into(), 1.0, 0.0).is_err());
        assert!(TranscriptWord::new("word".into(), f32::NAN, 0.0).is_err());
    }

    #[test]
    fn batch_configuration_requires_a_real_overlap_interval() {
        let rate = SampleRate::new(16).expect("test sample rate is positive");
        assert!(BatchConfig::new(rate, 0.0, 0.0, PadPolicy::Exact).is_err());
        assert!(BatchConfig::new(rate, 1.0, 1.0, PadPolicy::Exact).is_err());
        assert!(BatchConfig::new(rate, 1.0, 0.5, PadPolicy::PadToWindow).is_ok());
    }

    #[test]
    fn stream_configuration_carries_a_typed_vad_probability() {
        let zero = VadProbability::new(0.0).expect("zero is a valid VAD probability");
        let config = StreamConfig::checked_default(
            SampleRate::new(16).expect("test sample rate is positive"),
        )
        .expect("the default stream configuration is valid")
        .with_vad_threshold(zero)
        .expect("a typed VAD probability preserves valid stream timing");
        assert_eq!(config.vad_threshold(), zero);
    }

    #[test]
    fn vad_probability_is_the_closed_finite_unit_interval() {
        assert_eq!(
            VadProbability::new(0.0)
                .expect("zero is a valid VAD probability")
                .get(),
            0.0
        );
        assert_eq!(
            VadProbability::new(1.0)
                .expect("one is a valid VAD probability")
                .get(),
            1.0
        );
        assert!(VadProbability::new(f32::NAN).is_err());
        assert!(VadProbability::new(f32::INFINITY).is_err());
    }

    #[test]
    fn bound_temporal_configuration_refuses_zero_progress_and_preserves_nonnegative_zero() {
        let rate_eight = SampleRate::new(8).expect("test sample rate is positive");
        let rate_sixteen = SampleRate::new(16).expect("test sample rate is positive");
        let below_one = BatchConfig::new(rate_eight, 0.124, 0.0, PadPolicy::Exact)
            .expect_err("a positive duration below one sample must refuse");
        assert!(below_one.contains("batch window duration at 8 Hz"));
        assert!(BatchConfig::new(rate_eight, 0.125, 0.0, PadPolicy::Exact).is_ok());
        assert!(BatchConfig::new(rate_sixteen, 0.0626, 0.0, PadPolicy::Exact).is_ok());

        let base = StreamConfig::checked_default(rate_sixteen)
            .expect("the documented stream configuration is valid");
        let tiny_step = StreamConfig::timing_changes()
            .with_step_sec(0.0624)
            .expect("a finite positive scalar is not yet an executable configuration")
            .apply(base.clone())
            .expect_err("a stream step below one sample must refuse without a fallback");
        assert!(tiny_step.contains("stream step duration at 16 Hz"));
        let tiny_window = StreamConfig::timing_changes()
            .with_window_sec(0.0624)
            .expect("a finite positive scalar is not yet an executable configuration")
            .with_overlap_sec(0.0)
            .expect("zero overlap keeps the tiny-window boundary independently executable")
            .apply(base.clone())
            .expect_err("a stream window below one sample must refuse without a fallback");
        assert!(tiny_window.contains("stream window duration at 16 Hz"));
        let zero_retention = StreamConfig::timing_changes()
            .with_overlap_sec(0.0)
            .expect("zero overlap remains a valid retention policy")
            .with_keep_silence_sec(0.0)
            .expect("zero retained silence remains a valid extension policy")
            .apply(base)
            .expect("zero retention fields must remain executable");
        assert_eq!(zero_retention.overlap_sec(), 0.0);
        assert_eq!(zero_retention.keep_silence_sec(), 0.0);
    }

    #[test]
    fn timing_apply_reconstructs_one_fresh_bound_configuration() -> Result<(), String> {
        let rate = SampleRate::new(16).expect("test sample rate is positive");
        let base = StreamConfig::checked_default(rate)
            .expect("the documented stream configuration is valid");
        let unchanged = base.clone();
        assert!(
            StreamConfig::timing_changes()
                .with_step_sec(0.001)
                .expect("a finite positive scalar is accepted before rate binding")
                .apply(base.clone())
                .is_err()
        );
        assert_eq!(
            base, unchanged,
            "failed timing application must not mutate its base"
        );

        let applied = StreamConfig::timing_changes()
            .with_window_sec(4.0)
            .expect("test window duration is valid")
            .with_overlap_sec(1.0)
            .expect("test overlap duration is valid")
            .with_step_sec(0.5)
            .expect("test step duration is valid")
            .with_keep_silence_sec(0.25)
            .expect("test retained silence duration is valid")
            .apply(base)
            .expect("complete timing application is executable");
        let fresh = StreamConfig::new(
            rate,
            StreamConfigValues {
                endpoint_source: EndpointSource::Blank,
                vad_threshold: VadProbability::DEFAULT,
                window_sec: 4.0,
                overlap_sec: 1.0,
                step_sec: 0.5,
                horizon_sec: 5.0,
                lock_policy: StreamLockPolicy::Advisory,
                seam_after_sec: 2.0,
                rules: EndpointRules::default(),
                keep_silence_sec: 0.25,
            },
        )
        .expect("the same final values construct a fresh executable configuration");
        assert_eq!(applied, fresh);
        let applied_events = emitted_stream_history(applied)?;
        let fresh_events = emitted_stream_history(fresh)?;
        assert!(
            !applied_events.is_empty(),
            "the configured stream history must expose application events"
        );
        assert_eq!(
            applied_events, fresh_events,
            "timing application must preserve the fresh configuration's emitted-event history"
        );
        Ok(())
    }

    #[test]
    fn finite_unrepresentable_temporal_configuration_refuses_at_its_bound_owner() {
        let rate = SampleRate::new(16).expect("test sample rate is positive");
        let batch = BatchConfig::new(rate, f32::MAX, 0.0, PadPolicy::Exact)
            .expect_err("a finite overflowing batch duration must refuse during binding");
        assert!(batch.contains("batch window duration at 16 Hz"));
        let stream = StreamConfig::timing_changes()
            .with_keep_silence_sec(f32::MAX)
            .expect("the scalar remains finite before its bound conversion")
            .apply(
                StreamConfig::checked_default(rate)
                    .expect("the documented stream configuration is valid"),
            )
            .expect_err("unrepresentable retained silence must refuse even before reset occurs");
        assert!(stream.contains("stream retained silence duration at 16 Hz"));
        let window = StreamConfig::timing_changes()
            .with_window_sec(f32::MAX)
            .expect("the scalar remains finite before its bound conversion")
            .apply(
                StreamConfig::checked_default(rate)
                    .expect("the documented stream configuration is valid"),
            )
            .expect_err("an overflowing stream window must refuse during rate binding");
        assert!(window.contains("stream window duration at 16 Hz"));
    }
}
