// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Online and offline multi-channel streaming and dialogue orchestration.

use crate::channel_selection::{
    ChannelSelection, CorrelationThreshold, OriginalChannel, PairwiseChannelPolicy,
    SelectionWindowSamples, SourceChannelCount, select_pairwise_channels,
};
use crate::contracts::{
    ChannelSnapshot, DialogTurn, StreamConfig, StreamEmissionMode, StreamEvent,
    StreamingChannelPolicy, TurnsPatch, WordFinality, WordStability,
};
use crate::dialog::{BackchannelPolicy, DialogMerger};
use crate::stream::{StreamSession, StreamSetup};
use crate::turns::TurnGap;
use gigaam_audio::{
    ChannelAudio, ChannelAudioView, ChannelCount, InterleavedFrameDecoder, RatePair, Resampler,
    ResamplerConfig, SampleFormat, SampleRate, StreamResampler,
};
use gigaam_recognition::{SpeechProbabilityDetector, WindowDecoder};
use std::cmp::Ordering;

/// Typed dialogue and activation choices for one multi-channel streaming session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiChannelStreamOptions {
    channel_policy: StreamingChannelPolicy,
    emission_mode: StreamEmissionMode,
    turn_gap: TurnGap,
    backchannel_policy: BackchannelPolicy,
}

impl MultiChannelStreamOptions {
    /// Groups independently validated activation, delivery, turn, and backchannel policies.
    pub const fn new(
        channel_policy: StreamingChannelPolicy,
        emission_mode: StreamEmissionMode,
        turn_gap: TurnGap,
        backchannel_policy: BackchannelPolicy,
    ) -> Self {
        Self {
            channel_policy,
            emission_mode,
            turn_gap,
            backchannel_policy,
        }
    }

    /// Returns the exhaustive channel-activation policy.
    pub const fn channel_policy(self) -> StreamingChannelPolicy {
        self.channel_policy
    }

    /// Returns the client-visible material selected from each successful transition.
    pub const fn emission_mode(self) -> StreamEmissionMode {
        self.emission_mode
    }

    /// Returns the validated within-channel pause that separates dialogue turns.
    pub const fn turn_gap(self) -> TurnGap {
        self.turn_gap
    }

    /// Returns the typed overlapping-backchannel policy.
    pub const fn backchannel_policy(self) -> BackchannelPolicy {
        self.backchannel_policy
    }
}

/// Named inputs for one validated multi-channel stream setup.
///
/// The input remains mutable only until [`MultiChannelStreamSetup::new`] validates and owns it.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiChannelStreamSetupInput {
    pub sample_format: SampleFormat,
    pub source_sample_rate: SampleRate,
    pub source_channels: SourceChannelCount,
    pub stream_config: StreamConfig,
    pub options: MultiChannelStreamOptions,
}

/// Complete immutable configuration shared by every channel of one incremental stream.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiChannelStreamSetup {
    sample_format: SampleFormat,
    source_channels: SourceChannelCount,
    rate_pair: RatePair,
    stream_config: StreamConfig,
    options: MultiChannelStreamOptions,
}

impl MultiChannelStreamSetup {
    /// Validates the audio shape, supported rate conversion, and single-channel stream contract.
    pub fn new(input: MultiChannelStreamSetupInput) -> Result<Self, String> {
        let MultiChannelStreamSetupInput {
            sample_format,
            source_sample_rate,
            source_channels,
            stream_config,
            options,
        } = input;
        let rate_pair = RatePair::new(source_sample_rate, stream_config.sample_rate())
            .map_err(|error| format!("multi-channel stream sample rates: {error}"))?;
        Ok(Self {
            sample_format,
            source_channels,
            rate_pair,
            stream_config,
            options,
        })
    }

    /// Returns the wire sample format accepted by this stream.
    pub const fn sample_format(&self) -> SampleFormat {
        self.sample_format
    }

    /// Returns the validated number of original interleaved source channels.
    pub const fn source_channels(&self) -> SourceChannelCount {
        self.source_channels
    }

    /// Returns the source sample rate before stream resampling.
    pub const fn source_sample_rate(&self) -> SampleRate {
        self.rate_pair.input()
    }

    /// Returns the model sample rate after stream resampling.
    pub const fn model_sample_rate(&self) -> SampleRate {
        self.stream_config.sample_rate()
    }

    /// Returns the validated single-channel stream contract supplied to each factory call.
    pub fn stream_config(&self) -> &StreamConfig {
        &self.stream_config
    }

    /// Returns the typed activation, emission, turn, and backchannel choices.
    pub const fn options(&self) -> MultiChannelStreamOptions {
        self.options
    }
}

/// The exhaustive offline duplicate-selection choice for complete model-rate audio.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OfflineChannelPolicy {
    /// Retain every original channel without evaluating correlation.
    Disabled,
    /// Apply the fixed strict pairwise dialogue-deduplication policy once.
    DialogDeduplication,
}

impl OfflineChannelPolicy {
    /// Returns the exact pairwise selection policy used by the offline workflow.
    pub const fn pairwise_policy(self) -> PairwiseChannelPolicy {
        match self {
            Self::Disabled => PairwiseChannelPolicy::Disabled,
            Self::DialogDeduplication => PairwiseChannelPolicy::dialog_deduplication(),
        }
    }
}

/// Named inputs for one validated complete-audio dialogue operation.
#[derive(Clone, Debug, PartialEq)]
pub struct OfflineDialogSetupInput {
    pub channel_policy: OfflineChannelPolicy,
    pub stream_config: StreamConfig,
    pub turn_gap: TurnGap,
    pub backchannel_policy: BackchannelPolicy,
}

/// Complete immutable configuration for the offline dialogue workflow.
#[derive(Clone, Debug, PartialEq)]
pub struct OfflineDialogSetup {
    channel_policy: OfflineChannelPolicy,
    stream_config: StreamConfig,
    turn_gap: TurnGap,
    backchannel_policy: BackchannelPolicy,
}

impl OfflineDialogSetup {
    /// Validates the single-channel stream contract used for every retained complete channel.
    pub fn new(input: OfflineDialogSetupInput) -> Result<Self, String> {
        let OfflineDialogSetupInput {
            channel_policy,
            stream_config,
            turn_gap,
            backchannel_policy,
        } = input;
        Ok(Self {
            channel_policy,
            stream_config,
            turn_gap,
            backchannel_policy,
        })
    }

    /// Returns the sample rate of the complete model-rate channel audio.
    pub const fn model_sample_rate(&self) -> SampleRate {
        self.stream_config.sample_rate()
    }

    /// Returns the exhaustive complete-audio duplicate-selection choice.
    pub const fn channel_policy(&self) -> OfflineChannelPolicy {
        self.channel_policy
    }

    /// Returns the validated single-channel stream contract used by each retained channel.
    pub fn stream_config(&self) -> &StreamConfig {
        &self.stream_config
    }

    /// Returns the validated within-channel pause that separates dialogue turns.
    pub const fn turn_gap(&self) -> TurnGap {
        self.turn_gap
    }

    /// Returns the typed overlapping-backchannel policy.
    pub const fn backchannel_policy(&self) -> BackchannelPolicy {
        self.backchannel_policy
    }
}

/// Creates one complete single-channel stream setup for a validated original identity.
///
/// The factory is the Transcription boundary for channel-local recognition capabilities. It does
/// not expose process, provider, scheduler, runtime, or transport state.
pub trait StreamChannelFactory {
    type Decoder: WindowDecoder;
    type Detector: SpeechProbabilityDetector;

    /// Returns channel-local capabilities carrying the supplied validated stream configuration.
    fn create_stream(
        &mut self,
        channel: OriginalChannel,
        stream_config: StreamConfig,
    ) -> Result<StreamSetup<Self::Decoder, Self::Detector>, String>;
}

/// One immutable streaming event paired with its original channel identity.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelStreamEvent {
    channel: OriginalChannel,
    event: StreamEvent,
}

impl ChannelStreamEvent {
    /// Pairs an immutable stream event with an original channel identity.
    pub const fn new(channel: OriginalChannel, event: StreamEvent) -> Self {
        Self { channel, event }
    }

    /// Returns the original source identity that produced this event.
    pub const fn channel(&self) -> OriginalChannel {
        self.channel
    }

    /// Returns the immutable single-channel stream event.
    pub fn event(&self) -> &StreamEvent {
        &self.event
    }
}

/// Ordered client-visible material from one complete multi-channel business transition.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiChannelEmissionGroup {
    channel_events: Vec<ChannelStreamEvent>,
    dialog_patch: Option<TurnsPatch>,
}

impl MultiChannelEmissionGroup {
    /// Validates ordered channel events and the optional dialogue patch that follows them.
    pub fn new(
        source_channels: SourceChannelCount,
        channel_events: Vec<ChannelStreamEvent>,
        dialog_patch: Option<TurnsPatch>,
    ) -> Result<Self, String> {
        if channel_events.is_empty() && dialog_patch.is_none() {
            return Err(
                "multi-channel emission group must contain an event or dialogue patch".into(),
            );
        }
        let mut previous = None;
        for event in &channel_events {
            event.channel.validate_against(source_channels)?;
            if let Some(previous_channel) = previous
                && event.channel.index() < previous_channel
            {
                return Err(
                    "multi-channel emission events must be ordered by ascending original channel"
                        .into(),
                );
            }
            previous = Some(event.channel.index());
        }
        Ok(Self {
            channel_events,
            dialog_patch,
        })
    }

    /// Returns channel-labelled stream events in ascending original-channel order.
    pub fn channel_events(&self) -> &[ChannelStreamEvent] {
        &self.channel_events
    }

    /// Returns the dialogue patch that follows every channel event in this group, when emitted.
    pub fn dialog_patch(&self) -> Option<&TurnsPatch> {
        self.dialog_patch.as_ref()
    }
}

/// Immutable evidence that one source-channel activation decision committed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelSelectionObservation {
    source_channels: SourceChannelCount,
    active_channels: ChannelSelection,
}

impl ChannelSelectionObservation {
    /// Validates one immutable source-to-active channel selection observation.
    pub fn new(
        source_channels: SourceChannelCount,
        active_channels: ChannelSelection,
    ) -> Result<Self, String> {
        active_channels.validate_against(source_channels)?;
        Ok(Self {
            source_channels,
            active_channels,
        })
    }

    /// Returns the complete source-channel count before selection.
    pub const fn source_channels(&self) -> SourceChannelCount {
        self.source_channels
    }

    /// Returns active original identities in ascending order.
    pub fn active_channels(&self) -> &[OriginalChannel] {
        self.active_channels.channels()
    }

    fn active_selection(&self) -> &ChannelSelection {
        &self.active_channels
    }
}

/// One immutable internal business transition that a transport can project independently.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TranscriptionObservation {
    /// A typed source-channel activation decision committed.
    ChannelSelectionCommitted(ChannelSelectionObservation),
    /// Dialogue reconstruction generated one replacement patch.
    DialogPatchGenerated,
}

impl TranscriptionObservation {
    /// Creates a committed channel-selection observation with validated source ownership.
    pub fn selection_committed(
        source_channels: SourceChannelCount,
        active_channels: ChannelSelection,
    ) -> Result<Self, String> {
        ChannelSelectionObservation::new(source_channels, active_channels)
            .map(Self::ChannelSelectionCommitted)
    }

    /// Creates the immutable observation for one generated dialogue patch.
    pub const fn dialog_patch_generated() -> Self {
        Self::DialogPatchGenerated
    }
}

/// Immutable result material from one successful incremental input transition.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiChannelStep {
    emission_groups: Vec<MultiChannelEmissionGroup>,
    observations: Vec<TranscriptionObservation>,
}

impl MultiChannelStep {
    /// Validates all committed observations returned with one successful transition.
    pub fn new(
        emission_groups: Vec<MultiChannelEmissionGroup>,
        observations: Vec<TranscriptionObservation>,
    ) -> Result<Self, String> {
        validate_transition_observations(&observations)?;
        Ok(Self {
            emission_groups,
            observations,
        })
    }

    /// Returns client-visible emission groups in their committed business order.
    pub fn emission_groups(&self) -> &[MultiChannelEmissionGroup] {
        &self.emission_groups
    }

    /// Returns immutable observations that committed before this step was returned.
    pub fn observations(&self) -> &[TranscriptionObservation] {
        &self.observations
    }
}

/// A failed transition together with business observations committed before the failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiChannelFailure {
    error: String,
    observations: Vec<TranscriptionObservation>,
}

impl MultiChannelFailure {
    /// Preserves a nonempty exact application error and prior committed observations.
    pub fn new(error: String, observations: Vec<TranscriptionObservation>) -> Result<Self, String> {
        if error.is_empty() {
            return Err("multi-channel failure error must not be empty".into());
        }
        validate_transition_observations(&observations)?;
        Ok(Self {
            error,
            observations,
        })
    }

    /// Returns the exact application error that prevented this transition from completing.
    pub fn error(&self) -> &str {
        &self.error
    }

    /// Returns only observations committed before the failure.
    pub fn observations(&self) -> &[TranscriptionObservation] {
        &self.observations
    }

    fn from_committed(error: String, observations: Vec<TranscriptionObservation>) -> Self {
        Self {
            error: nonempty_failure_error(error),
            observations,
        }
    }
}

fn nonempty_failure_error(error: String) -> String {
    if error.is_empty() {
        "multi-channel operation failed without an upstream error message".into()
    } else {
        error
    }
}

/// The exhaustive transport-neutral stage that stopped offline dialogue transcription.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OfflineDialogFailureOrigin {
    /// Input violated an offline workflow invariant outside pairwise selection.
    InputValidation,
    /// Pairwise channel selection could not produce an active source set.
    ChannelSelection,
    /// The caller could not provide channel-local recognition capabilities.
    Factory,
    /// A factory setup could not become a valid stream session.
    SessionConstruction,
    /// A newly constructed stream session could not warm its full window shape.
    SessionWarmup,
    /// A retained channel's complete audio could not be pushed into its session.
    SessionPush,
    /// A retained channel's terminal stream tail could not be flushed.
    SessionFlush,
    /// A completed retained session could not produce its dialogue snapshot.
    Snapshot,
    /// Completed channel snapshots could not be reconstructed into dialogue.
    Dialog,
    /// The completed dialogue could not satisfy the offline result contract.
    ResultValidation,
}

/// A failed offline dialogue operation with its exact stage, error, and committed observations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OfflineDialogFailure {
    origin: OfflineDialogFailureOrigin,
    error: String,
    observations: Vec<TranscriptionObservation>,
}

impl OfflineDialogFailure {
    /// Preserves a nonempty exact application error and the observations committed before it.
    pub fn new(
        origin: OfflineDialogFailureOrigin,
        error: String,
        observations: Vec<TranscriptionObservation>,
    ) -> Result<Self, String> {
        if error.is_empty() {
            return Err("offline dialogue failure error must not be empty".into());
        }
        validate_transition_observations(&observations)?;
        Ok(Self {
            origin,
            error,
            observations,
        })
    }

    /// Returns the stage whose failure prevents the offline dialogue result.
    pub const fn origin(&self) -> OfflineDialogFailureOrigin {
        self.origin
    }

    /// Returns the exact application error emitted by the failed stage.
    pub fn error(&self) -> &str {
        &self.error
    }

    /// Returns only observations committed before the failed stage.
    pub fn observations(&self) -> &[TranscriptionObservation] {
        &self.observations
    }

    fn from_committed(
        origin: OfflineDialogFailureOrigin,
        error: String,
        observations: Vec<TranscriptionObservation>,
    ) -> Self {
        Self {
            origin,
            error: nonempty_failure_error(error),
            observations,
        }
    }
}

/// Immutable terminal dialogue result from the offline multi-channel workflow.
#[derive(Clone, Debug, PartialEq)]
pub struct OfflineDialogResult {
    source_channels: SourceChannelCount,
    active_channels: ChannelSelection,
    dialogue: Vec<DialogTurn>,
    observations: Vec<TranscriptionObservation>,
}

impl OfflineDialogResult {
    /// Validates one terminal dialogue result against its immutable selection observation.
    pub fn new(
        source_channels: SourceChannelCount,
        active_channels: ChannelSelection,
        dialogue: Vec<DialogTurn>,
        observations: Vec<TranscriptionObservation>,
    ) -> Result<Self, String> {
        active_channels.validate_against(source_channels)?;
        validate_transition_observations(&observations)?;
        validate_offline_selection_observation(source_channels, &active_channels, &observations)?;
        validate_final_dialogue(&dialogue, &active_channels, source_channels)?;
        Ok(Self {
            source_channels,
            active_channels,
            dialogue,
            observations,
        })
    }

    /// Returns the original source-channel count before offline selection.
    pub const fn source_channels(&self) -> SourceChannelCount {
        self.source_channels
    }

    /// Returns retained original identities in ascending order.
    pub fn active_channels(&self) -> &[OriginalChannel] {
        self.active_channels.channels()
    }

    /// Returns final dialogue turns in dialogue order.
    pub fn dialogue(&self) -> &[DialogTurn] {
        &self.dialogue
    }

    /// Returns immutable business observations committed by the offline workflow.
    pub fn observations(&self) -> &[TranscriptionObservation] {
        &self.observations
    }
}

fn validate_transition_observations(
    observations: &[TranscriptionObservation],
) -> Result<(), String> {
    let selection_count = observations
        .iter()
        .filter(|observation| {
            matches!(
                observation,
                TranscriptionObservation::ChannelSelectionCommitted(_)
            )
        })
        .count();
    if selection_count > 1 {
        return Err(
            "one transcription transition cannot commit multiple channel selections".into(),
        );
    }
    Ok(())
}

fn validate_offline_selection_observation(
    source_channels: SourceChannelCount,
    active_channels: &ChannelSelection,
    observations: &[TranscriptionObservation],
) -> Result<(), String> {
    let mut selection_observation = None;
    for observation in observations {
        match observation {
            TranscriptionObservation::ChannelSelectionCommitted(selection) => {
                selection_observation = Some(selection);
            }
            TranscriptionObservation::DialogPatchGenerated => {
                return Err(
                    "offline dialogue result cannot contain an incremental patch observation"
                        .into(),
                );
            }
        }
    }
    let selection_observation = selection_observation.ok_or_else(|| {
        "offline dialogue result requires its committed channel selection".to_owned()
    })?;
    if selection_observation.source_channels() != source_channels
        || selection_observation.active_selection() != active_channels
    {
        return Err(
            "offline dialogue selection observation does not match the final result".into(),
        );
    }
    Ok(())
}

fn validate_final_dialogue(
    dialogue: &[DialogTurn],
    active_channels: &ChannelSelection,
    source_channels: SourceChannelCount,
) -> Result<(), String> {
    let mut next_turn_indices = vec![0_usize; source_channels.get()];
    for (position, turn) in dialogue.iter().enumerate() {
        if turn.finality() != WordFinality::Final || turn.stability() != WordStability::Stable {
            return Err("offline dialogue result requires stable final dialogue turns".into());
        }
        if !active_channels
            .channels()
            .iter()
            .any(|channel| channel.index() == turn.channel())
        {
            return Err("offline dialogue turn belongs to a non-active source channel".into());
        }
        let expected_index = next_turn_indices.get_mut(turn.channel()).ok_or_else(|| {
            "offline dialogue turn is outside the validated source channel count".to_owned()
        })?;
        if turn.index() != *expected_index {
            return Err(
                "offline dialogue turn indices must be consecutive per original channel".into(),
            );
        }
        *expected_index = expected_index
            .checked_add(1)
            .ok_or_else(|| "offline dialogue turn index overflows usize".to_owned())?;

        if let Some(previous) = position
            .checked_sub(1)
            .and_then(|index| dialogue.get(index))
            && dialogue_order(previous, turn) == Ordering::Greater
        {
            return Err("offline dialogue turns must remain in dialogue order".into());
        }
    }
    Ok(())
}

fn dialogue_order(left: &DialogTurn, right: &DialogTurn) -> Ordering {
    left.start()
        .total_cmp(&right.start())
        .then(left.channel().cmp(&right.channel()))
        .then(left.index().cmp(&right.index()))
}

/// Owns the complete online multi-channel transcription lifecycle from raw frames to dialogue.
pub struct MultiChannelSession<D: WindowDecoder, V: SpeechProbabilityDetector> {
    decoder: Option<InterleavedFrameDecoder>,
    source_channels: SourceChannelCount,
    candidates: Vec<OnlineCandidate<D, V>>,
    lifecycle: OnlineLifecycle,
    merger: DialogMerger,
    emission_mode: StreamEmissionMode,
}

enum OnlineLifecycle {
    Buffering {
        threshold: CorrelationThreshold,
        analysis_window: SelectionWindowSamples,
    },
    Active(ChannelSelection),
    Failed,
}

struct OnlineCandidate<D: WindowDecoder, V: SpeechProbabilityDetector> {
    channel: OriginalChannel,
    rate_adapter: Option<ChannelRateAdapter>,
    session: StreamSession<D, V>,
    pending: Vec<f32>,
}

enum ChannelRateAdapter {
    Identity,
    Resampled(StreamResampler),
}

impl ChannelRateAdapter {
    fn new(rate_pair: RatePair) -> Result<Self, String> {
        if rate_pair.is_identity() {
            return Ok(Self::Identity);
        }
        let resampler = Resampler::new(ResamplerConfig::new(rate_pair))?;
        Ok(Self::Resampled(StreamResampler::new(resampler)))
    }

    fn push(&mut self, input: ChannelAudio) -> Result<ChannelAudio, String> {
        match self {
            Self::Identity => Ok(input),
            Self::Resampled(resampler) => {
                resampler.push(input.samples()).and_then(ChannelAudio::new)
            }
        }
    }

    fn finish(self) -> Result<ChannelAudio, String> {
        match self {
            Self::Identity => ChannelAudio::new(Vec::new()),
            Self::Resampled(resampler) => resampler.finish().and_then(ChannelAudio::new),
        }
    }
}

impl<D: WindowDecoder, V: SpeechProbabilityDetector> MultiChannelSession<D, V> {
    /// Creates, validates, and warms one candidate session for every original source channel.
    pub fn new<F>(setup: MultiChannelStreamSetup, factory: &mut F) -> Result<Self, String>
    where
        F: StreamChannelFactory<Decoder = D, Detector = V>,
    {
        let decoder = InterleavedFrameDecoder::new(
            setup.sample_format,
            ChannelCount::new(setup.source_channels.get())?,
        )?;
        let lifecycle = match setup.options.channel_policy() {
            StreamingChannelPolicy::AllChannels => {
                OnlineLifecycle::Active(ChannelSelection::all(setup.source_channels)?)
            }
            StreamingChannelPolicy::Deduplicate {
                threshold,
                analysis_window,
            } => OnlineLifecycle::Buffering {
                threshold,
                analysis_window,
            },
        };
        let mut candidates = Vec::with_capacity(setup.source_channels.get());
        for index in 0..setup.source_channels.get() {
            let channel = OriginalChannel::new(index, setup.source_channels)?;
            let session = create_warmed_session(factory, channel, setup.stream_config.clone())
                .map_err(nonempty_failure_error)?;
            candidates.push(OnlineCandidate {
                channel,
                rate_adapter: Some(ChannelRateAdapter::new(setup.rate_pair)?),
                session,
                pending: Vec::new(),
            });
        }
        Ok(Self {
            decoder: Some(decoder),
            source_channels: setup.source_channels,
            candidates,
            lifecycle,
            merger: DialogMerger::new(setup.options.turn_gap())
                .with_backchannel_policy(setup.options.backchannel_policy()),
            emission_mode: setup.options.emission_mode(),
        })
    }

    /// Accepts one arbitrary raw-byte partition and returns only complete committed transitions.
    pub fn push(&mut self, bytes: &[u8]) -> Result<MultiChannelStep, MultiChannelFailure> {
        if matches!(self.lifecycle, OnlineLifecycle::Failed) {
            return Err(MultiChannelFailure::from_committed(
                "multi-channel session is already failed".into(),
                Vec::new(),
            ));
        }
        let mut observations = Vec::new();
        let decoded = match self.decoder.as_mut() {
            Some(decoder) => match decoder.push(bytes) {
                Ok(decoded) => decoded,
                Err(error) => return Err(self.fail(error, observations)),
            },
            None => {
                return Err(self.fail(
                    "multi-channel session cannot accept input after terminalization".into(),
                    observations,
                ));
            }
        };
        let Some(channels) = decoded else {
            return Ok(MultiChannelStep {
                emission_groups: Vec::new(),
                observations,
            });
        };
        let result = match self.buffering_parameters() {
            Some((threshold, analysis_window)) => {
                self.push_buffered(channels, threshold, analysis_window, &mut observations)
            }
            None => self.push_active(channels, &mut observations),
        };
        match result {
            Ok(emission_groups) => Ok(MultiChannelStep {
                emission_groups,
                observations,
            }),
            Err(error) => Err(self.fail(error, observations)),
        }
    }

    /// Consumes the stream, validates its terminal frame boundary, and flushes retained sessions.
    pub fn finish(mut self) -> Result<MultiChannelStep, MultiChannelFailure> {
        if matches!(self.lifecycle, OnlineLifecycle::Failed) {
            return Err(MultiChannelFailure::from_committed(
                "multi-channel session is already failed".into(),
                Vec::new(),
            ));
        }
        let decoder = match self.decoder.take() {
            Some(decoder) => decoder,
            None => {
                return Err(MultiChannelFailure::from_committed(
                    "multi-channel session terminal decoder is unavailable".into(),
                    Vec::new(),
                ));
            }
        };
        if let Err(error) = decoder.finish() {
            return Err(MultiChannelFailure::from_committed(error, Vec::new()));
        }
        let mut observations = Vec::new();
        let result = match self.buffering_parameters() {
            Some((threshold, analysis_window)) => {
                self.finish_buffered(threshold, analysis_window, &mut observations)
            }
            None => {
                let selection = match self.active_selection() {
                    Ok(selection) => selection.clone(),
                    Err(error) => {
                        return Err(MultiChannelFailure::from_committed(error, observations));
                    }
                };
                self.finish_active(selection, &mut observations)
            }
        };
        match result {
            Ok(emission_groups) => Ok(MultiChannelStep {
                emission_groups,
                observations,
            }),
            Err(error) => Err(MultiChannelFailure::from_committed(error, observations)),
        }
    }

    fn buffering_parameters(&self) -> Option<(CorrelationThreshold, SelectionWindowSamples)> {
        match &self.lifecycle {
            OnlineLifecycle::Buffering {
                threshold,
                analysis_window,
            } => Some((*threshold, *analysis_window)),
            OnlineLifecycle::Active(_) | OnlineLifecycle::Failed => None,
        }
    }

    fn active_selection(&self) -> Result<&ChannelSelection, String> {
        match &self.lifecycle {
            OnlineLifecycle::Active(selection) => Ok(selection),
            OnlineLifecycle::Buffering { .. } => {
                Err("multi-channel session has not committed a channel selection".into())
            }
            OnlineLifecycle::Failed => Err("multi-channel session is already failed".into()),
        }
    }

    fn fail(
        &mut self,
        error: String,
        observations: Vec<TranscriptionObservation>,
    ) -> MultiChannelFailure {
        self.lifecycle = OnlineLifecycle::Failed;
        MultiChannelFailure::from_committed(error, observations)
    }

    fn push_buffered(
        &mut self,
        channels: Vec<ChannelAudio>,
        threshold: CorrelationThreshold,
        analysis_window: SelectionWindowSamples,
        observations: &mut Vec<TranscriptionObservation>,
    ) -> Result<Vec<MultiChannelEmissionGroup>, String> {
        self.validate_decoded_channel_count(channels.len())?;
        for (candidate, channel) in self.candidates.iter_mut().zip(channels) {
            let adapter = candidate
                .rate_adapter
                .as_mut()
                .ok_or_else(|| "buffering channel rate adapter is already terminal".to_owned())?;
            let resampled = adapter.push(channel)?;
            candidate
                .pending
                .try_reserve(resampled.len())
                .map_err(|_| {
                    "multi-channel pending selection audio cannot reserve memory".to_owned()
                })?;
            candidate.pending.extend_from_slice(resampled.samples());
        }
        if self
            .candidates
            .iter()
            .any(|candidate| candidate.pending.len() < analysis_window.get())
        {
            return Ok(Vec::new());
        }
        let selection = self.select_pending(threshold, Some(analysis_window))?;
        let selection = self.commit_selection(selection, observations)?;
        let events = self.release_pending(&selection)?;
        self.incremental_groups(events, observations)
    }

    fn push_active(
        &mut self,
        channels: Vec<ChannelAudio>,
        observations: &mut Vec<TranscriptionObservation>,
    ) -> Result<Vec<MultiChannelEmissionGroup>, String> {
        self.validate_decoded_channel_count(channels.len())?;
        let selection = self.active_selection()?.clone();
        let mut events = Vec::new();
        for (candidate, channel) in self.candidates.iter_mut().zip(channels) {
            if !selection_contains(&selection, candidate.channel) {
                continue;
            }
            let adapter = candidate
                .rate_adapter
                .as_mut()
                .ok_or_else(|| "active channel rate adapter is already terminal".to_owned())?;
            let resampled = adapter.push(channel)?;
            append_channel_events(
                candidate.channel,
                candidate.session.push(&resampled)?,
                &mut events,
            );
        }
        self.incremental_groups(events, observations)
    }

    fn finish_buffered(
        &mut self,
        threshold: CorrelationThreshold,
        analysis_window: SelectionWindowSamples,
        observations: &mut Vec<TranscriptionObservation>,
    ) -> Result<Vec<MultiChannelEmissionGroup>, String> {
        self.finish_all_candidate_adapters()?;
        let exact_window = self
            .candidates
            .iter()
            .all(|candidate| candidate.pending.len() >= analysis_window.get());
        let selection = self.select_pending(
            threshold,
            if exact_window {
                Some(analysis_window)
            } else {
                None
            },
        )?;
        let selection = self.commit_selection(selection, observations)?;
        let pending_events = self.release_pending(&selection)?;
        let mut emission_groups = self.incremental_groups(pending_events, observations)?;
        let flush_events = self.flush_active_sessions(&selection)?;
        emission_groups.extend(self.terminal_groups(flush_events, observations)?);
        Ok(emission_groups)
    }

    fn finish_active(
        &mut self,
        selection: ChannelSelection,
        observations: &mut Vec<TranscriptionObservation>,
    ) -> Result<Vec<MultiChannelEmissionGroup>, String> {
        let events = self.finish_active_sessions(&selection)?;
        self.terminal_groups(events, observations)
    }

    fn validate_decoded_channel_count(&self, count: usize) -> Result<(), String> {
        if count != self.source_channels.get() {
            return Err(format!(
                "interleaved decoder produced {count} channels, expected {}",
                self.source_channels.get()
            ));
        }
        Ok(())
    }

    fn select_pending(
        &self,
        threshold: CorrelationThreshold,
        window: Option<SelectionWindowSamples>,
    ) -> Result<ChannelSelection, String> {
        let mut channels = Vec::with_capacity(self.candidates.len());
        for candidate in &self.candidates {
            let samples = match window {
                Some(window) => candidate.pending.get(..window.get()).ok_or_else(|| {
                    "multi-channel selection window exceeds pending model-rate audio".to_owned()
                })?,
                None => candidate.pending.as_slice(),
            };
            channels.push(ChannelAudioView::new(samples)?);
        }
        select_pairwise_channels(&channels, PairwiseChannelPolicy::Deduplicate(threshold))
    }

    fn commit_selection(
        &mut self,
        selection: ChannelSelection,
        observations: &mut Vec<TranscriptionObservation>,
    ) -> Result<ChannelSelection, String> {
        selection.validate_against(self.source_channels)?;
        let observation =
            TranscriptionObservation::selection_committed(self.source_channels, selection.clone())?;
        self.lifecycle = OnlineLifecycle::Active(selection.clone());
        observations.push(observation);
        Ok(selection)
    }

    fn release_pending(
        &mut self,
        selection: &ChannelSelection,
    ) -> Result<Vec<ChannelStreamEvent>, String> {
        let mut events = Vec::new();
        for candidate in &mut self.candidates {
            if !selection_contains(selection, candidate.channel) {
                candidate.pending.clear();
                continue;
            }
            let audio = ChannelAudio::new(std::mem::take(&mut candidate.pending))?;
            append_channel_events(
                candidate.channel,
                candidate.session.push(&audio)?,
                &mut events,
            );
        }
        Ok(events)
    }

    fn finish_all_candidate_adapters(&mut self) -> Result<(), String> {
        for candidate in &mut self.candidates {
            let adapter = candidate
                .rate_adapter
                .take()
                .ok_or_else(|| "candidate rate adapter is already terminal".to_owned())?;
            let tail = adapter.finish()?;
            candidate.pending.try_reserve(tail.len()).map_err(|_| {
                "multi-channel pending selection tail cannot reserve memory".to_owned()
            })?;
            candidate.pending.extend_from_slice(tail.samples());
        }
        Ok(())
    }

    fn finish_active_sessions(
        &mut self,
        selection: &ChannelSelection,
    ) -> Result<Vec<ChannelStreamEvent>, String> {
        let mut events = Vec::new();
        for channel in selection.channels() {
            let candidate = self.candidate_mut(*channel)?;
            let adapter = candidate
                .rate_adapter
                .take()
                .ok_or_else(|| "active channel rate adapter is already terminal".to_owned())?;
            let tail = adapter.finish()?;
            append_channel_events(
                candidate.channel,
                candidate.session.push(&tail)?,
                &mut events,
            );
            append_channel_events(candidate.channel, candidate.session.flush()?, &mut events);
        }
        Ok(events)
    }

    fn flush_active_sessions(
        &mut self,
        selection: &ChannelSelection,
    ) -> Result<Vec<ChannelStreamEvent>, String> {
        let mut events = Vec::new();
        for channel in selection.channels() {
            let candidate = self.candidate_mut(*channel)?;
            append_channel_events(candidate.channel, candidate.session.flush()?, &mut events);
        }
        Ok(events)
    }

    fn candidate_mut(
        &mut self,
        channel: OriginalChannel,
    ) -> Result<&mut OnlineCandidate<D, V>, String> {
        let candidate = self.candidates.get_mut(channel.index()).ok_or_else(|| {
            "selected channel is absent from the candidate session set".to_owned()
        })?;
        if candidate.channel != channel {
            return Err("candidate session identity does not match its selected channel".into());
        }
        Ok(candidate)
    }

    fn snapshots(&self) -> Result<Vec<ChannelSnapshot>, String> {
        let selection = self.active_selection()?;
        let mut snapshots = Vec::with_capacity(selection.channels().len());
        for channel in selection.channels() {
            let candidate = self.candidates.get(channel.index()).ok_or_else(|| {
                "selected channel is absent from the candidate session set".to_owned()
            })?;
            if candidate.channel != *channel {
                return Err(
                    "candidate session identity does not match its selected channel".into(),
                );
            }
            snapshots.push(candidate.session.snapshot(channel.index())?);
        }
        Ok(snapshots)
    }

    fn incremental_groups(
        &mut self,
        events: Vec<ChannelStreamEvent>,
        observations: &mut Vec<TranscriptionObservation>,
    ) -> Result<Vec<MultiChannelEmissionGroup>, String> {
        if self.emission_mode == StreamEmissionMode::Words {
            return self.groups_for(events, None, observations);
        }
        let snapshots = self.snapshots()?;
        let patch = self.merger.update(&snapshots)?;
        self.groups_for(events, patch, observations)
    }

    fn terminal_groups(
        &mut self,
        events: Vec<ChannelStreamEvent>,
        observations: &mut Vec<TranscriptionObservation>,
    ) -> Result<Vec<MultiChannelEmissionGroup>, String> {
        if self.emission_mode == StreamEmissionMode::Words {
            return self.groups_for(events, None, observations);
        }
        let snapshots = self.snapshots()?;
        let patch = self.merger.finalize(&snapshots)?;
        self.groups_for(events, patch, observations)
    }

    fn groups_for(
        &self,
        events: Vec<ChannelStreamEvent>,
        patch: Option<TurnsPatch>,
        observations: &mut Vec<TranscriptionObservation>,
    ) -> Result<Vec<MultiChannelEmissionGroup>, String> {
        if patch.is_some() {
            observations.push(TranscriptionObservation::dialog_patch_generated());
        }
        let channel_events = match self.emission_mode {
            StreamEmissionMode::Words | StreamEmissionMode::WordsAndDialog => events,
            StreamEmissionMode::Dialog => Vec::new(),
        };
        let dialog_patch = match self.emission_mode {
            StreamEmissionMode::Words => None,
            StreamEmissionMode::Dialog | StreamEmissionMode::WordsAndDialog => patch,
        };
        if channel_events.is_empty() && dialog_patch.is_none() {
            return Ok(Vec::new());
        }
        Ok(vec![MultiChannelEmissionGroup::new(
            self.source_channels,
            channel_events,
            dialog_patch,
        )?])
    }
}

/// Transcribes complete model-rate channels after one selection-first dialogue decision.
pub fn transcribe_offline_dialog<F>(
    setup: OfflineDialogSetup,
    factory: &mut F,
    channels: &[ChannelAudio],
) -> Result<OfflineDialogResult, OfflineDialogFailure>
where
    F: StreamChannelFactory,
{
    let source_channels = match SourceChannelCount::new(channels.len()) {
        Ok(source_channels) => source_channels,
        Err(error) => {
            return Err(OfflineDialogFailure::from_committed(
                OfflineDialogFailureOrigin::ChannelSelection,
                error,
                Vec::new(),
            ));
        }
    };
    let channel_views = channels.iter().map(ChannelAudio::view).collect::<Vec<_>>();
    let selection =
        match select_pairwise_channels(&channel_views, setup.channel_policy.pairwise_policy()) {
            Ok(selection) => selection,
            Err(error) => {
                return Err(OfflineDialogFailure::from_committed(
                    OfflineDialogFailureOrigin::ChannelSelection,
                    error,
                    Vec::new(),
                ));
            }
        };
    let mut observations = Vec::new();
    let selection_observation =
        match TranscriptionObservation::selection_committed(source_channels, selection.clone()) {
            Ok(observation) => observation,
            Err(error) => {
                return Err(OfflineDialogFailure::from_committed(
                    OfflineDialogFailureOrigin::ChannelSelection,
                    error,
                    observations,
                ));
            }
        };
    observations.push(selection_observation);
    let mut merger =
        DialogMerger::new(setup.turn_gap).with_backchannel_policy(setup.backchannel_policy);
    let mut snapshots = Vec::with_capacity(selection.channels().len());
    for channel in selection.channels() {
        let input = match channels.get(channel.index()) {
            Some(input) => input,
            None => {
                return Err(OfflineDialogFailure::from_committed(
                    OfflineDialogFailureOrigin::InputValidation,
                    "selected offline channel is absent from the validated input".into(),
                    observations,
                ));
            }
        };
        let mut session =
            match create_warmed_session_with_origin(factory, *channel, setup.stream_config.clone())
            {
                Ok(session) => session,
                Err(error) => {
                    return Err(OfflineDialogFailure::from_committed(
                        error.origin(),
                        error.into_error(),
                        observations,
                    ));
                }
            };
        if let Err(error) = session.push(input) {
            return Err(OfflineDialogFailure::from_committed(
                OfflineDialogFailureOrigin::SessionPush,
                error,
                observations,
            ));
        }
        if let Err(error) = session.flush() {
            return Err(OfflineDialogFailure::from_committed(
                OfflineDialogFailureOrigin::SessionFlush,
                error,
                observations,
            ));
        }
        match session.snapshot(channel.index()) {
            Ok(snapshot) => snapshots.push(snapshot),
            Err(error) => {
                return Err(OfflineDialogFailure::from_committed(
                    OfflineDialogFailureOrigin::Snapshot,
                    error,
                    observations,
                ));
            }
        }
    }
    if let Err(error) = merger.finalize(&snapshots) {
        return Err(OfflineDialogFailure::from_committed(
            OfflineDialogFailureOrigin::Dialog,
            error,
            observations,
        ));
    }
    let failure_observations = observations.clone();
    OfflineDialogResult::new(
        source_channels,
        selection,
        merger.dialog().to_vec(),
        observations,
    )
    .map_err(|error| {
        OfflineDialogFailure::from_committed(
            OfflineDialogFailureOrigin::ResultValidation,
            error,
            failure_observations,
        )
    })
}

fn create_warmed_session<D, V, F>(
    factory: &mut F,
    channel: OriginalChannel,
    stream_config: StreamConfig,
) -> Result<StreamSession<D, V>, String>
where
    D: WindowDecoder,
    V: SpeechProbabilityDetector,
    F: StreamChannelFactory<Decoder = D, Detector = V>,
{
    create_warmed_session_with_origin(factory, channel, stream_config)
        .map_err(WarmedSessionFailure::into_error)
}

enum WarmedSessionFailure {
    Factory(String),
    SessionConstruction(String),
    SessionWarmup(String),
}

impl WarmedSessionFailure {
    const fn origin(&self) -> OfflineDialogFailureOrigin {
        match self {
            Self::Factory(_) => OfflineDialogFailureOrigin::Factory,
            Self::SessionConstruction(_) => OfflineDialogFailureOrigin::SessionConstruction,
            Self::SessionWarmup(_) => OfflineDialogFailureOrigin::SessionWarmup,
        }
    }

    fn into_error(self) -> String {
        match self {
            Self::Factory(error)
            | Self::SessionConstruction(error)
            | Self::SessionWarmup(error) => error,
        }
    }
}

fn create_warmed_session_with_origin<D, V, F>(
    factory: &mut F,
    channel: OriginalChannel,
    stream_config: StreamConfig,
) -> Result<StreamSession<D, V>, WarmedSessionFailure>
where
    D: WindowDecoder,
    V: SpeechProbabilityDetector,
    F: StreamChannelFactory<Decoder = D, Detector = V>,
{
    let stream_setup = factory
        .create_stream(channel, stream_config.clone())
        .map_err(WarmedSessionFailure::Factory)?;
    if stream_setup.config != stream_config {
        return Err(WarmedSessionFailure::SessionConstruction(
            "stream channel factory changed the supplied stream configuration".into(),
        ));
    }
    if stream_setup.frontend.sample_rate() != stream_config.sample_rate() {
        return Err(WarmedSessionFailure::SessionConstruction(
            "stream channel factory frontend rate does not match the model sample rate".into(),
        ));
    }
    let mut session =
        StreamSession::new(stream_setup).map_err(WarmedSessionFailure::SessionConstruction)?;
    session
        .warmup()
        .map_err(WarmedSessionFailure::SessionWarmup)?;
    Ok(session)
}

fn selection_contains(selection: &ChannelSelection, channel: OriginalChannel) -> bool {
    selection.channels().contains(&channel)
}

fn append_channel_events(
    channel: OriginalChannel,
    events: Vec<StreamEvent>,
    target: &mut Vec<ChannelStreamEvent>,
) {
    target.extend(
        events
            .into_iter()
            .map(|event| ChannelStreamEvent::new(channel, event)),
    );
}

#[cfg(test)]
mod tests {
    use super::{
        ChannelStreamEvent, MultiChannelEmissionGroup, MultiChannelFailure, MultiChannelSession,
        MultiChannelStep, MultiChannelStreamOptions, MultiChannelStreamSetup,
        MultiChannelStreamSetupInput, OfflineChannelPolicy, OfflineDialogFailure,
        OfflineDialogFailureOrigin, OfflineDialogResult, OfflineDialogSetup,
        OfflineDialogSetupInput, StreamChannelFactory, TranscriptionObservation,
        transcribe_offline_dialog,
    };
    use crate::channel_selection::{
        ChannelSelection, CorrelationThreshold, OriginalChannel, SelectionWindowSamples,
        SourceChannelCount,
    };
    use crate::contracts::{
        BackchannelMark, DialogTurn, DialogTurnData, FrontierEvent, StreamConfig,
        StreamEmissionMode, StreamEvent, StreamingChannelPolicy, TurnsPatch, WordFinality,
        WordStability,
    };
    use crate::dialog::{BackchannelDuration, BackchannelPolicy};
    use crate::stream::{EndpointDetector, StreamSession, StreamSetup};
    use crate::turns::TurnGap;
    use gigaam_audio::{
        ChannelAudio, FeatureMatrixView, RatePair, Resampler, ResamplerConfig, SampleFormat,
        SampleRate,
    };
    use gigaam_recognition::{
        Decoded, ExecutionControl, FrameRate, SpeechProbabilityDetector, WindowDecoder, Word,
    };
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum DecoderBehavior {
        Succeeds,
        RefusesDuringWarmup,
        RefusesWithEmptyErrorDuringWarmup,
        RefusesAfterWarmup,
        RefusesWithEmptyErrorAfterWarmup,
        CancelsAfterWarmup,
    }

    /// Exact feature payload delivered to one decoder call in the stable Audio matrix layout.
    #[derive(Clone, Debug, Eq, PartialEq)]
    struct DecoderFeatureTrace {
        channel: usize,
        mel_bins: usize,
        frames: usize,
        value_bits: Vec<u32>,
    }

    #[derive(Clone, Debug, Default)]
    struct Trace {
        factory_channels: Vec<usize>,
        factory_configs: Vec<StreamConfig>,
        factory_rates: Vec<SampleRate>,
        decode_channels: Vec<usize>,
        decoder_features: Vec<DecoderFeatureTrace>,
    }

    struct RecordingFactory {
        frontend: Arc<gigaam_audio::FrontendProcessor>,
        trace: Arc<Mutex<Trace>>,
        behaviors: Vec<DecoderBehavior>,
        words: Vec<Vec<Word>>,
        config_override: Option<StreamConfig>,
    }

    /// A legal factory port that reports no diagnostic before any session can exist.
    struct EmptyDiagnosticFactory;

    struct RecordingDecoder {
        channel: usize,
        trace: Arc<Mutex<Trace>>,
        behavior: DecoderBehavior,
        control: ExecutionControl,
        words: Vec<Word>,
    }

    struct BlankDetector;

    impl SpeechProbabilityDetector for BlankDetector {
        fn probabilities(
            &mut self,
            _audio: gigaam_audio::ChannelAudioView<'_>,
        ) -> Result<Vec<f32>, String> {
            Err("blank-endpoint test sessions never invoke a VAD detector".into())
        }
    }

    impl WindowDecoder for RecordingDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(8.0).expect("the fixed test frame rate is positive")
        }

        fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            let call_number = {
                let mut trace = self
                    .trace
                    .lock()
                    .expect("test trace lock must not be poisoned");
                trace.decode_channels.push(self.channel);
                trace.decoder_features.push(DecoderFeatureTrace {
                    channel: self.channel,
                    mel_bins: features.mel_bins(),
                    frames: features.frames(),
                    value_bits: features
                        .values()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect(),
                });
                trace
                    .decode_channels
                    .iter()
                    .filter(|channel| **channel == self.channel)
                    .count()
            };
            match self.behavior {
                DecoderBehavior::Succeeds => {}
                DecoderBehavior::RefusesDuringWarmup => {
                    if call_number == 1 {
                        return Err("test decoder refused warmup".into());
                    }
                }
                DecoderBehavior::RefusesWithEmptyErrorDuringWarmup => {
                    if call_number == 1 {
                        return Err(String::new());
                    }
                }
                DecoderBehavior::RefusesAfterWarmup => {
                    if call_number > 1 {
                        return Err("test decoder refused its first live channel input".into());
                    }
                }
                DecoderBehavior::RefusesWithEmptyErrorAfterWarmup => {
                    if call_number > 1 {
                        return Err(String::new());
                    }
                }
                DecoderBehavior::CancelsAfterWarmup => {
                    if call_number > 1 {
                        self.control.request_cancellation();
                    }
                }
            }
            Decoded::new(
                self.words.clone(),
                vec![false; features.frames()],
                features.frames(),
                0.0,
            )
        }
    }

    impl RecordingFactory {
        fn new(
            model_sample_rate: SampleRate,
            words: Vec<Vec<Word>>,
            behaviors: Vec<DecoderBehavior>,
        ) -> Self {
            Self {
                frontend: crate::test_support::frontend_at_sample_rate(model_sample_rate.hertz()),
                trace: Arc::new(Mutex::new(Trace::default())),
                behaviors,
                words,
                config_override: None,
            }
        }

        fn trace(&self) -> Trace {
            self.trace
                .lock()
                .expect("test trace lock must not be poisoned")
                .clone()
        }
    }

    impl StreamChannelFactory for RecordingFactory {
        type Decoder = RecordingDecoder;
        type Detector = BlankDetector;

        fn create_stream(
            &mut self,
            channel: OriginalChannel,
            stream_config: StreamConfig,
        ) -> Result<StreamSetup<Self::Decoder, Self::Detector>, String> {
            let index = channel.index();
            {
                let mut trace = self
                    .trace
                    .lock()
                    .expect("test trace lock must not be poisoned");
                trace.factory_channels.push(index);
                trace.factory_rates.push(stream_config.sample_rate());
                trace.factory_configs.push(stream_config.clone());
            }
            let behavior = *self.behaviors.get(index).ok_or_else(|| {
                "test factory lacks a decoder behavior for its channel".to_owned()
            })?;
            let words = self
                .words
                .get(index)
                .cloned()
                .ok_or_else(|| "test factory lacks a word plan for its channel".to_owned())?;
            let control = ExecutionControl::without_deadline();
            Ok(StreamSetup {
                frontend: Arc::clone(&self.frontend),
                decoder: RecordingDecoder {
                    channel: index,
                    trace: Arc::clone(&self.trace),
                    behavior,
                    control: control.clone(),
                    words,
                },
                config: match &self.config_override {
                    Some(config) => config.clone(),
                    None => stream_config,
                },
                detector: EndpointDetector::Blank,
                control,
            })
        }
    }

    impl StreamChannelFactory for EmptyDiagnosticFactory {
        type Decoder = RecordingDecoder;
        type Detector = BlankDetector;

        fn create_stream(
            &mut self,
            _channel: OriginalChannel,
            _stream_config: StreamConfig,
        ) -> Result<StreamSetup<Self::Decoder, Self::Detector>, String> {
            Err(String::new())
        }
    }

    fn source_channels() -> Result<SourceChannelCount, String> {
        SourceChannelCount::new(2)
    }

    fn channel(index: usize) -> Result<OriginalChannel, String> {
        OriginalChannel::new(index, source_channels()?)
    }

    fn active_channels() -> Result<ChannelSelection, String> {
        let source_channels = source_channels()?;
        ChannelSelection::all(source_channels)
    }

    fn options() -> Result<MultiChannelStreamOptions, String> {
        Ok(MultiChannelStreamOptions::new(
            StreamingChannelPolicy::Deduplicate {
                threshold: CorrelationThreshold::new(0.98)?,
                analysis_window: SelectionWindowSamples::new(1_600)?,
            },
            StreamEmissionMode::WordsAndDialog,
            TurnGap::new(0.5)?,
            BackchannelPolicy::Disabled,
        ))
    }

    #[test]
    fn multi_channel_stream_options_preserve_each_typed_product_choice() -> Result<(), String> {
        let threshold = CorrelationThreshold::new(0.98)?;
        let analysis_window = SelectionWindowSamples::new(1_600)?;
        let turn_gap = TurnGap::new(0.5)?;
        let options = MultiChannelStreamOptions::new(
            StreamingChannelPolicy::Deduplicate {
                threshold,
                analysis_window,
            },
            StreamEmissionMode::WordsAndDialog,
            turn_gap,
            BackchannelPolicy::Disabled,
        );

        assert_eq!(
            options.channel_policy(),
            StreamingChannelPolicy::Deduplicate {
                threshold,
                analysis_window,
            }
        );
        assert_eq!(options.emission_mode(), StreamEmissionMode::WordsAndDialog);
        assert_eq!(options.turn_gap(), turn_gap);
        assert_eq!(options.backchannel_policy(), BackchannelPolicy::Disabled);
        Ok(())
    }

    fn model_rate() -> Result<SampleRate, String> {
        SampleRate::new(16)
    }

    fn fast_stream_config() -> Result<StreamConfig, String> {
        StreamConfig::timing_changes()
            .with_window_sec(1.0)?
            .with_overlap_sec(0.0)?
            .with_step_sec(0.25)?
            .with_horizon_sec(0.01)?
            .with_seam_after_sec(0.0)?
            .with_keep_silence_sec(0.0)?
            .apply(StreamConfig::checked_default(model_rate()?)?)
    }

    fn online_setup(
        sample_format: SampleFormat,
        source_sample_rate: SampleRate,
        channel_policy: StreamingChannelPolicy,
        emission_mode: StreamEmissionMode,
        backchannel_policy: BackchannelPolicy,
    ) -> Result<MultiChannelStreamSetup, String> {
        online_setup_for_channels(
            source_channels()?,
            sample_format,
            source_sample_rate,
            channel_policy,
            emission_mode,
            backchannel_policy,
        )
    }

    fn online_setup_for_channels(
        source_channels: SourceChannelCount,
        sample_format: SampleFormat,
        source_sample_rate: SampleRate,
        channel_policy: StreamingChannelPolicy,
        emission_mode: StreamEmissionMode,
        backchannel_policy: BackchannelPolicy,
    ) -> Result<MultiChannelStreamSetup, String> {
        MultiChannelStreamSetup::new(MultiChannelStreamSetupInput {
            sample_format,
            source_sample_rate,
            source_channels,
            stream_config: fast_stream_config()?,
            options: MultiChannelStreamOptions::new(
                channel_policy,
                emission_mode,
                TurnGap::new(0.3)?,
                backchannel_policy,
            ),
        })
    }

    fn success_behaviors() -> Vec<DecoderBehavior> {
        vec![DecoderBehavior::Succeeds, DecoderBehavior::Succeeds]
    }

    fn ordinary_words() -> Result<Vec<Vec<Word>>, String> {
        Ok(vec![
            vec![Word::new("left".into(), 0.01, 0.12)?],
            vec![Word::new("right".into(), 0.14, 0.22)?],
        ])
    }

    fn backchannel_words() -> Result<Vec<Vec<Word>>, String> {
        Ok(vec![
            vec![Word::new("long".into(), 0.01, 0.22)?],
            vec![Word::new("short".into(), 0.05, 0.10)?],
        ])
    }

    fn no_words() -> Vec<Vec<Word>> {
        vec![Vec::new(), Vec::new()]
    }

    fn three_channel_words() -> Result<Vec<Vec<Word>>, String> {
        Ok(vec![
            vec![Word::new("first".into(), 0.01, 0.12)?],
            Vec::new(),
            Vec::new(),
        ])
    }

    fn failure_error(failure: MultiChannelFailure) -> String {
        failure.error().to_owned()
    }

    fn offline_failure_error(failure: OfflineDialogFailure) -> String {
        failure.error().to_owned()
    }

    fn offline_setup() -> Result<OfflineDialogSetup, String> {
        OfflineDialogSetup::new(OfflineDialogSetupInput {
            channel_policy: OfflineChannelPolicy::Disabled,
            stream_config: fast_stream_config()?,
            turn_gap: TurnGap::new(0.3)?,
            backchannel_policy: BackchannelPolicy::Disabled,
        })
    }

    fn offline_channel(samples: Vec<f32>) -> Result<Vec<ChannelAudio>, String> {
        Ok(vec![ChannelAudio::new(samples)?])
    }

    fn required_offline_failure(
        result: Result<OfflineDialogResult, OfflineDialogFailure>,
    ) -> Result<OfflineDialogFailure, String> {
        match result {
            Ok(_) => Err("test offline dialogue operation must fail".into()),
            Err(failure) => Ok(failure),
        }
    }

    fn assert_committed_single_channel_selection(failure: &OfflineDialogFailure) {
        assert!(matches!(
            failure.observations(),
            [TranscriptionObservation::ChannelSelectionCommitted(selection)]
                if selection.active_channels().iter().map(|channel| channel.index()).collect::<Vec<_>>() == vec![0]
        ));
    }

    fn run_online(
        setup: MultiChannelStreamSetup,
        factory: &mut RecordingFactory,
        payload: &[u8],
        ends: &[usize],
    ) -> Result<Vec<MultiChannelStep>, String> {
        let mut session =
            MultiChannelSession::<RecordingDecoder, BlankDetector>::new(setup, factory)?;
        let mut steps = Vec::new();
        let mut start = 0_usize;
        for end in ends {
            if *end <= start || *end > payload.len() {
                return Err(
                    "test byte partitions must be strictly increasing payload boundaries".into(),
                );
            }
            steps.push(session.push(&payload[start..*end]).map_err(failure_error)?);
            start = *end;
        }
        if start != payload.len() {
            return Err("test byte partitions must end at the complete payload length".into());
        }
        steps.push(session.finish().map_err(failure_error)?);
        Ok(steps)
    }

    fn irregular_partition_ends(length: usize) -> Result<Vec<usize>, String> {
        if length == 0 {
            return Err("test payload must be nonempty".into());
        }
        let sizes = [1_usize, 3, 2, 5, 4];
        let mut ends = Vec::new();
        let mut offset = 0_usize;
        let mut index = 0_usize;
        while offset < length {
            let count = sizes[index % sizes.len()].min(length - offset);
            offset = offset
                .checked_add(count)
                .ok_or_else(|| "test byte partition offset overflows usize".to_owned())?;
            ends.push(offset);
            index = index
                .checked_add(1)
                .ok_or_else(|| "test byte partition index overflows usize".to_owned())?;
        }
        Ok(ends)
    }

    #[derive(Clone, Debug, PartialEq)]
    struct ObservedRun {
        events: Vec<ChannelStreamEvent>,
        patches: Vec<TurnsPatch>,
        observations: Vec<TranscriptionObservation>,
        dialogue: Vec<DialogTurn>,
    }

    fn observe_run(steps: &[MultiChannelStep]) -> Result<ObservedRun, String> {
        let mut observed = ObservedRun {
            events: Vec::new(),
            patches: Vec::new(),
            observations: Vec::new(),
            dialogue: Vec::new(),
        };
        for step in steps {
            observed.observations.extend_from_slice(step.observations());
            for group in step.emission_groups() {
                observed.events.extend_from_slice(group.channel_events());
                if let Some(patch) = group.dialog_patch() {
                    if patch.revise_from() > observed.dialogue.len() {
                        return Err(
                            "dialogue patch revises beyond the accumulated test line".into()
                        );
                    }
                    observed.dialogue.truncate(patch.revise_from());
                    observed.dialogue.extend_from_slice(patch.turns());
                    observed.patches.push(patch.clone());
                }
            }
        }
        Ok(observed)
    }

    fn committed_selections(steps: &[MultiChannelStep]) -> Vec<Vec<usize>> {
        steps
            .iter()
            .flat_map(MultiChannelStep::observations)
            .filter_map(|observation| match observation {
                TranscriptionObservation::ChannelSelectionCommitted(selection) => Some(
                    selection
                        .active_channels()
                        .iter()
                        .map(|channel| channel.index())
                        .collect(),
                ),
                TranscriptionObservation::DialogPatchGenerated => None,
            })
            .collect()
    }

    fn patch_observation_count(steps: &[MultiChannelStep]) -> usize {
        steps
            .iter()
            .flat_map(MultiChannelStep::observations)
            .filter(|observation| {
                matches!(observation, TranscriptionObservation::DialogPatchGenerated)
            })
            .count()
    }

    fn decoder_call_count(trace: &Trace, channel: usize) -> usize {
        trace
            .decode_channels
            .iter()
            .filter(|observed| **observed == channel)
            .count()
    }

    fn decoder_features_for_channel(trace: &Trace, channel: usize) -> Vec<DecoderFeatureTrace> {
        trace
            .decoder_features
            .iter()
            .filter(|observed| observed.channel == channel)
            .cloned()
            .collect()
    }

    /// Drives complete retained audio through a direct stream session, including construction warmup.
    fn complete_channel_reference_features(
        frontend: Arc<gigaam_audio::FrontendProcessor>,
        stream_config: StreamConfig,
        bytes: &[u8],
    ) -> Result<Vec<DecoderFeatureTrace>, String> {
        let trace = Arc::new(Mutex::new(Trace::default()));
        let control = ExecutionControl::without_deadline();
        let mut session = StreamSession::new(StreamSetup {
            frontend,
            decoder: RecordingDecoder {
                channel: 0,
                trace: Arc::clone(&trace),
                behavior: DecoderBehavior::Succeeds,
                control: control.clone(),
                words: Vec::new(),
            },
            config: stream_config,
            detector: EndpointDetector::<BlankDetector>::Blank,
            control,
        })?;
        session.warmup()?;
        let complete_audio = gigaam_audio::decode_samples(SampleFormat::Pcm16, bytes)?;
        session.push(&complete_audio)?;
        session.flush()?;
        Ok(trace
            .lock()
            .expect("test trace lock must not be poisoned")
            .decoder_features
            .clone())
    }

    fn raw_payload(format: SampleFormat) -> Vec<u8> {
        match format {
            SampleFormat::Pcm16 => {
                let frames = [
                    (1_000_i16, -1_000_i16),
                    (2_000_i16, -2_000_i16),
                    (3_000_i16, -3_000_i16),
                    (4_000_i16, -4_000_i16),
                    (5_000_i16, -5_000_i16),
                    (6_000_i16, -6_000_i16),
                ];
                let mut payload = Vec::new();
                for (left, right) in frames {
                    payload.extend_from_slice(&left.to_le_bytes());
                    payload.extend_from_slice(&right.to_le_bytes());
                }
                payload
            }
            SampleFormat::F32 => {
                let frames = [
                    (0.10_f32, -0.10_f32),
                    (0.20_f32, -0.20_f32),
                    (0.30_f32, -0.30_f32),
                    (0.40_f32, -0.40_f32),
                    (0.50_f32, -0.50_f32),
                    (0.60_f32, -0.60_f32),
                ];
                let mut payload = Vec::new();
                for (left, right) in frames {
                    payload.extend_from_slice(&left.to_le_bytes());
                    payload.extend_from_slice(&right.to_le_bytes());
                }
                payload
            }
            SampleFormat::Alaw => vec![
                0xd5, 0x55, 0xd6, 0x56, 0x95, 0x15, 0x96, 0x16, 0x85, 0x05, 0x86, 0x06,
            ],
            SampleFormat::Ulaw => vec![
                0xff, 0x7f, 0xfe, 0x7e, 0xef, 0x6f, 0xee, 0x6e, 0xdf, 0x5f, 0xde, 0x5e,
            ],
        }
    }

    fn f32_interleaved(left: &[f32], right: &[f32]) -> Result<Vec<u8>, String> {
        if left.len() != right.len() {
            return Err("test interleaved source channels must have equal lengths".into());
        }
        let mut payload = Vec::new();
        for (left_sample, right_sample) in left.iter().zip(right) {
            payload.extend_from_slice(&left_sample.to_le_bytes());
            payload.extend_from_slice(&right_sample.to_le_bytes());
        }
        Ok(payload)
    }

    fn f32_interleaved_three(
        first: &[f32],
        second: &[f32],
        third: &[f32],
    ) -> Result<Vec<u8>, String> {
        if first.len() != second.len() || second.len() != third.len() {
            return Err("test interleaved source channels must have equal lengths".into());
        }
        let mut payload = Vec::new();
        for ((first_sample, second_sample), third_sample) in first.iter().zip(second).zip(third) {
            payload.extend_from_slice(&first_sample.to_le_bytes());
            payload.extend_from_slice(&second_sample.to_le_bytes());
            payload.extend_from_slice(&third_sample.to_le_bytes());
        }
        Ok(payload)
    }

    fn three_channel_pcm16_payload() -> Vec<u8> {
        let frames = [
            (1_000_i16, 2_000_i16, 3_000_i16),
            (-1_000_i16, -2_000_i16, -3_000_i16),
            (4_000_i16, 5_000_i16, 6_000_i16),
            (-4_000_i16, -5_000_i16, -6_000_i16),
            (7_000_i16, 8_000_i16, 9_000_i16),
            (-7_000_i16, -8_000_i16, -9_000_i16),
        ];
        let mut payload = Vec::new();
        for (first, second, third) in frames {
            payload.extend_from_slice(&first.to_le_bytes());
            payload.extend_from_slice(&second.to_le_bytes());
            payload.extend_from_slice(&third.to_le_bytes());
        }
        payload
    }

    fn exact_window_overshoot_frames() -> [(i16, i16); 6] {
        [
            (1_000_i16, 1_000_i16),
            (-1_000_i16, -1_000_i16),
            (2_000_i16, 2_000_i16),
            (-2_000_i16, -2_000_i16),
            (30_000_i16, -30_000_i16),
            (-30_000_i16, 30_000_i16),
        ]
    }

    fn exact_window_overshoot_payload() -> Vec<u8> {
        let mut payload = Vec::new();
        for (left, right) in exact_window_overshoot_frames() {
            payload.extend_from_slice(&left.to_le_bytes());
            payload.extend_from_slice(&right.to_le_bytes());
        }
        payload
    }

    fn exact_window_retained_channel_payload() -> Vec<u8> {
        let mut payload = Vec::new();
        for (left, _) in exact_window_overshoot_frames() {
            payload.extend_from_slice(&left.to_le_bytes());
        }
        payload
    }

    fn stable_event(channel: OriginalChannel) -> Result<ChannelStreamEvent, String> {
        Ok(ChannelStreamEvent::new(
            channel,
            StreamEvent::Stable(FrontierEvent::new(0.0, 0)?),
        ))
    }

    fn final_turn(
        channel: usize,
        index: usize,
        finality: WordFinality,
        stability: WordStability,
    ) -> Result<DialogTurn, String> {
        DialogTurn::new(DialogTurnData {
            channel,
            index,
            start: 0.0,
            end: 0.1,
            text: "ready".into(),
            stability,
            finality,
            backchannel: BackchannelMark::No,
        })
    }

    #[test]
    fn stream_setup_validates_the_shared_rate_adapter_before_session_creation() -> Result<(), String>
    {
        let setup = MultiChannelStreamSetup::new(MultiChannelStreamSetupInput {
            sample_format: SampleFormat::Pcm16,
            source_sample_rate: SampleRate::new(48_000)?,
            source_channels: source_channels()?,
            stream_config: StreamConfig::checked_default(SampleRate::new(16_000)?)?,
            options: options()?,
        })?;
        assert_eq!(setup.sample_format(), SampleFormat::Pcm16);
        assert_eq!(setup.source_channels(), source_channels()?);
        assert_eq!(setup.source_sample_rate(), SampleRate::new(48_000)?);
        assert_eq!(setup.model_sample_rate(), SampleRate::new(16_000)?);
        let unsupported = MultiChannelStreamSetup::new(MultiChannelStreamSetupInput {
            sample_format: SampleFormat::Pcm16,
            source_sample_rate: SampleRate::new(1)?,
            source_channels: source_channels()?,
            stream_config: StreamConfig::checked_default(SampleRate::new(1_001)?)?,
            options: options()?,
        });
        assert!(unsupported.is_err());
        Ok(())
    }

    #[test]
    fn factories_observe_bound_setup_identity_and_pre_factory_refusals_have_no_effect()
    -> Result<(), String> {
        let rate = model_rate()?;
        let online_setup = online_setup(
            SampleFormat::Pcm16,
            rate,
            StreamingChannelPolicy::AllChannels,
            StreamEmissionMode::WordsAndDialog,
            BackchannelPolicy::Disabled,
        )?;
        let online_config = online_setup.stream_config().clone();
        let online_channels = online_setup.source_channels().get();
        let mut online_factory = RecordingFactory::new(rate, no_words(), success_behaviors());
        let _online = MultiChannelSession::<RecordingDecoder, BlankDetector>::new(
            online_setup,
            &mut online_factory,
        )?;
        let online_trace = online_factory.trace();
        assert_eq!(
            online_trace.factory_configs,
            vec![online_config.clone(); online_channels]
        );
        assert_eq!(online_trace.factory_rates, vec![rate; online_channels]);
        assert!(
            online_trace
                .factory_configs
                .iter()
                .all(|config| config.sample_rate() == rate),
            "every online factory call must receive the setup-bound model rate"
        );

        let offline_setup = offline_setup()?;
        let offline_config = offline_setup.stream_config().clone();
        let mut offline_factory = RecordingFactory::new(rate, no_words(), success_behaviors());
        let _offline = transcribe_offline_dialog(
            offline_setup,
            &mut offline_factory,
            &offline_channel(vec![0.1])?,
        )
        .map_err(offline_failure_error)?;
        let offline_trace = offline_factory.trace();
        assert_eq!(offline_trace.factory_configs, vec![offline_config]);
        assert_eq!(offline_trace.factory_rates, vec![rate]);

        let invalid_setup = MultiChannelStreamSetup::new(MultiChannelStreamSetupInput {
            sample_format: SampleFormat::Pcm16,
            source_sample_rate: SampleRate::new(1)?,
            source_channels: source_channels()?,
            stream_config: StreamConfig::checked_default(SampleRate::new(1_001)?)?,
            options: options()?,
        });
        let mut invalid_online_factory =
            RecordingFactory::new(rate, no_words(), success_behaviors());
        let invalid_online = invalid_setup.and_then(|setup| {
            MultiChannelSession::<RecordingDecoder, BlankDetector>::new(
                setup,
                &mut invalid_online_factory,
            )
        });
        assert!(
            invalid_online.is_err(),
            "an unsupported source-to-model rate pair must refuse before the online factory"
        );
        let invalid_online_trace = invalid_online_factory.trace();
        assert!(invalid_online_trace.factory_channels.is_empty());
        assert!(invalid_online_trace.factory_configs.is_empty());
        assert!(invalid_online_trace.factory_rates.is_empty());
        Ok(())
    }

    #[test]
    fn emission_groups_preserve_channel_order_and_dialogue_tail_position() -> Result<(), String> {
        let source_channels = source_channels()?;
        let patch = TurnsPatch::new(0, Vec::new(), 0.0)?;
        let group = MultiChannelEmissionGroup::new(
            source_channels,
            vec![
                stable_event(channel(0)?)?,
                stable_event(channel(0)?)?,
                stable_event(channel(1)?)?,
            ],
            Some(patch),
        )?;
        assert_eq!(group.channel_events().len(), 3);
        assert_eq!(group.channel_events()[2].channel().index(), 1);
        assert!(group.dialog_patch().is_some());

        let unordered = MultiChannelEmissionGroup::new(
            source_channels,
            vec![stable_event(channel(1)?)?, stable_event(channel(0)?)?],
            None,
        );
        assert!(unordered.is_err());
        assert!(MultiChannelEmissionGroup::new(source_channels, Vec::new(), None).is_err());
        Ok(())
    }

    #[test]
    fn step_and_failure_expose_only_valid_committed_observations() -> Result<(), String> {
        let source_channels = source_channels()?;
        let selection = active_channels()?;
        let observation =
            TranscriptionObservation::selection_committed(source_channels, selection.clone())?;
        let duplicate =
            MultiChannelStep::new(Vec::new(), vec![observation.clone(), observation.clone()]);
        assert!(duplicate.is_err());

        let failure = MultiChannelFailure::new("decoder refused input".into(), vec![observation])?;
        assert_eq!(failure.error(), "decoder refused input");
        assert_eq!(failure.observations().len(), 1);
        assert!(MultiChannelFailure::new(String::new(), Vec::new()).is_err());
        Ok(())
    }

    #[test]
    fn offline_failure_preserves_its_origin_raw_error_and_committed_observations()
    -> Result<(), String> {
        let source_channels = source_channels()?;
        let selection = active_channels()?;
        let observation =
            TranscriptionObservation::selection_committed(source_channels, selection)?;
        let failure = OfflineDialogFailure::new(
            OfflineDialogFailureOrigin::Snapshot,
            "snapshot serialization refused".into(),
            vec![observation.clone()],
        )?;

        assert_eq!(failure.origin(), OfflineDialogFailureOrigin::Snapshot);
        assert_eq!(failure.error(), "snapshot serialization refused");
        assert_eq!(failure.observations(), &[observation]);
        assert!(
            OfflineDialogFailure::new(
                OfflineDialogFailureOrigin::Factory,
                String::new(),
                Vec::new(),
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn offline_workflow_assigns_reachable_failure_origins_without_losing_raw_errors()
    -> Result<(), String> {
        let rate = model_rate()?;

        let mut selection_factory = RecordingFactory::new(rate, no_words(), success_behaviors());
        let selection_failure = required_offline_failure(transcribe_offline_dialog(
            offline_setup()?,
            &mut selection_factory,
            &[],
        ))?;
        assert_eq!(
            selection_failure.origin(),
            OfflineDialogFailureOrigin::ChannelSelection
        );
        assert_eq!(
            selection_failure.error(),
            "channel selection requires at least one source channel"
        );
        assert!(selection_failure.observations().is_empty());
        let selection_trace = selection_factory.trace();
        assert!(selection_trace.factory_channels.is_empty());
        assert!(selection_trace.factory_configs.is_empty());
        assert!(selection_trace.factory_rates.is_empty());

        let mut factory = RecordingFactory::new(rate, no_words(), Vec::new());
        let factory_failure = required_offline_failure(transcribe_offline_dialog(
            offline_setup()?,
            &mut factory,
            &offline_channel(vec![0.1])?,
        ))?;
        assert_eq!(
            factory_failure.origin(),
            OfflineDialogFailureOrigin::Factory
        );
        assert_eq!(
            factory_failure.error(),
            "test factory lacks a decoder behavior for its channel"
        );
        assert_committed_single_channel_selection(&factory_failure);

        let changed_config = StreamConfig::timing_changes()
            .with_step_sec(0.5)?
            .apply(fast_stream_config()?)?;
        let mut construction_factory =
            RecordingFactory::new(rate, no_words(), vec![DecoderBehavior::Succeeds]);
        construction_factory.config_override = Some(changed_config);
        let construction_failure = required_offline_failure(transcribe_offline_dialog(
            offline_setup()?,
            &mut construction_factory,
            &offline_channel(vec![0.1])?,
        ))?;
        assert_eq!(
            construction_failure.origin(),
            OfflineDialogFailureOrigin::SessionConstruction
        );
        assert_eq!(
            construction_failure.error(),
            "stream channel factory changed the supplied stream configuration"
        );
        assert_committed_single_channel_selection(&construction_failure);

        let mut warmup_factory =
            RecordingFactory::new(rate, no_words(), vec![DecoderBehavior::RefusesDuringWarmup]);
        let warmup_failure = required_offline_failure(transcribe_offline_dialog(
            offline_setup()?,
            &mut warmup_factory,
            &offline_channel(vec![0.1])?,
        ))?;
        assert_eq!(
            warmup_failure.origin(),
            OfflineDialogFailureOrigin::SessionWarmup
        );
        assert_eq!(warmup_failure.error(), "test decoder refused warmup");
        assert_committed_single_channel_selection(&warmup_failure);

        let mut push_factory =
            RecordingFactory::new(rate, no_words(), vec![DecoderBehavior::RefusesAfterWarmup]);
        let full_window_samples = rate.as_usize()?;
        let push_failure = required_offline_failure(transcribe_offline_dialog(
            offline_setup()?,
            &mut push_factory,
            &offline_channel(vec![0.1; full_window_samples])?,
        ))?;
        assert_eq!(
            push_failure.origin(),
            OfflineDialogFailureOrigin::SessionPush
        );
        assert_eq!(
            push_failure.error(),
            "test decoder refused its first live channel input"
        );
        assert_committed_single_channel_selection(&push_failure);

        let mut flush_factory =
            RecordingFactory::new(rate, no_words(), vec![DecoderBehavior::RefusesAfterWarmup]);
        let flush_failure = required_offline_failure(transcribe_offline_dialog(
            offline_setup()?,
            &mut flush_factory,
            &offline_channel(vec![0.1])?,
        ))?;
        assert_eq!(
            flush_failure.origin(),
            OfflineDialogFailureOrigin::SessionFlush
        );
        assert_eq!(
            flush_failure.error(),
            "test decoder refused its first live channel input"
        );
        assert_committed_single_channel_selection(&flush_failure);
        Ok(())
    }

    #[test]
    fn online_session_creation_keeps_shared_factory_construction_and_warmup_errors_exact()
    -> Result<(), String> {
        let rate = model_rate()?;
        let setup = || {
            online_setup(
                SampleFormat::Pcm16,
                rate,
                StreamingChannelPolicy::AllChannels,
                StreamEmissionMode::WordsAndDialog,
                BackchannelPolicy::Disabled,
            )
        };

        let mut factory = RecordingFactory::new(rate, no_words(), Vec::new());
        let factory_error = match MultiChannelSession::<RecordingDecoder, BlankDetector>::new(
            setup()?,
            &mut factory,
        ) {
            Ok(_) => return Err("online factory failure must refuse session construction".into()),
            Err(error) => error,
        };
        assert_eq!(
            factory_error,
            "test factory lacks a decoder behavior for its channel"
        );

        let changed_config = StreamConfig::timing_changes()
            .with_step_sec(0.5)?
            .apply(fast_stream_config()?)?;
        let mut construction_factory = RecordingFactory::new(rate, no_words(), success_behaviors());
        construction_factory.config_override = Some(changed_config);
        let construction_error = match MultiChannelSession::<RecordingDecoder, BlankDetector>::new(
            setup()?,
            &mut construction_factory,
        ) {
            Ok(_) => {
                return Err("online factory configuration mismatch must refuse the session".into());
            }
            Err(error) => error,
        };
        assert_eq!(
            construction_error,
            "stream channel factory changed the supplied stream configuration"
        );

        let mut warmup_factory =
            RecordingFactory::new(rate, no_words(), vec![DecoderBehavior::RefusesDuringWarmup]);
        let warmup_error = match MultiChannelSession::<RecordingDecoder, BlankDetector>::new(
            setup()?,
            &mut warmup_factory,
        ) {
            Ok(_) => return Err("online warmup failure must refuse session construction".into()),
            Err(error) => error,
        };
        assert_eq!(warmup_error, "test decoder refused warmup");
        Ok(())
    }

    #[test]
    fn online_session_creation_reports_a_nonempty_empty_factory_diagnostic() -> Result<(), String> {
        let rate = model_rate()?;
        let setup = online_setup(
            SampleFormat::Pcm16,
            rate,
            StreamingChannelPolicy::AllChannels,
            StreamEmissionMode::WordsAndDialog,
            BackchannelPolicy::Disabled,
        )?;
        let mut factory = EmptyDiagnosticFactory;

        let error = match MultiChannelSession::<RecordingDecoder, BlankDetector>::new(
            setup,
            &mut factory,
        ) {
            Ok(_) => {
                return Err("an empty factory diagnostic must refuse session construction".into());
            }
            Err(error) => error,
        };

        assert_eq!(
            error,
            "multi-channel operation failed without an upstream error message"
        );
        assert!(!error.is_empty());
        Ok(())
    }

    #[test]
    fn online_session_creation_reports_a_nonempty_empty_warmup_diagnostic() -> Result<(), String> {
        let rate = model_rate()?;
        let setup = online_setup(
            SampleFormat::Pcm16,
            rate,
            StreamingChannelPolicy::AllChannels,
            StreamEmissionMode::WordsAndDialog,
            BackchannelPolicy::Disabled,
        )?;
        let mut factory = RecordingFactory::new(
            rate,
            no_words(),
            vec![DecoderBehavior::RefusesWithEmptyErrorDuringWarmup],
        );

        let error = match MultiChannelSession::<RecordingDecoder, BlankDetector>::new(
            setup,
            &mut factory,
        ) {
            Ok(_) => {
                return Err("an empty warmup diagnostic must refuse session construction".into());
            }
            Err(error) => error,
        };

        assert_eq!(
            error,
            "multi-channel operation failed without an upstream error message"
        );
        assert!(!error.is_empty());
        assert_eq!(factory.trace().decode_channels, vec![0]);
        Ok(())
    }

    #[test]
    fn empty_upstream_error_becomes_an_explicit_typed_failure() -> Result<(), String> {
        let rate = model_rate()?;
        let setup = online_setup(
            SampleFormat::Pcm16,
            rate,
            StreamingChannelPolicy::AllChannels,
            StreamEmissionMode::WordsAndDialog,
            BackchannelPolicy::Disabled,
        )?;
        let mut factory = RecordingFactory::new(
            rate,
            no_words(),
            vec![
                DecoderBehavior::RefusesWithEmptyErrorAfterWarmup,
                DecoderBehavior::Succeeds,
            ],
        );
        let mut session =
            MultiChannelSession::<RecordingDecoder, BlankDetector>::new(setup, &mut factory)?;
        let failure = match session.push(&raw_payload(SampleFormat::Pcm16)) {
            Ok(_) => return Err("an empty upstream decoder error must fail the session".into()),
            Err(failure) => failure,
        };
        assert_eq!(
            failure.error(),
            "multi-channel operation failed without an upstream error message"
        );
        assert!(!failure.error().is_empty());
        assert_eq!(factory.trace().decode_channels, vec![0, 1, 0]);
        Ok(())
    }

    #[test]
    fn offline_empty_upstream_error_returns_a_typed_nonempty_failure() -> Result<(), String> {
        let rate = model_rate()?;
        let mut factory = RecordingFactory::new(
            rate,
            no_words(),
            vec![DecoderBehavior::RefusesWithEmptyErrorAfterWarmup],
        );
        let failure = required_offline_failure(transcribe_offline_dialog(
            offline_setup()?,
            &mut factory,
            &offline_channel(vec![0.1; rate.as_usize()?])?,
        ))?;

        assert_eq!(failure.origin(), OfflineDialogFailureOrigin::SessionPush);
        assert_eq!(
            failure.error(),
            "multi-channel operation failed without an upstream error message"
        );
        assert!(!failure.error().is_empty());
        assert_committed_single_channel_selection(&failure);
        Ok(())
    }

    #[test]
    fn offline_result_requires_matching_selection_and_stable_final_dialogue() -> Result<(), String>
    {
        let source_channels = source_channels()?;
        let active_channels = active_channels()?;
        let selection_observation = TranscriptionObservation::selection_committed(
            source_channels,
            active_channels.clone(),
        )?;
        let dialogue = vec![final_turn(
            0,
            0,
            WordFinality::Final,
            WordStability::Stable,
        )?];
        let result = OfflineDialogResult::new(
            source_channels,
            active_channels.clone(),
            dialogue,
            vec![selection_observation],
        )?;
        assert_eq!(result.active_channels(), &[channel(0)?, channel(1)?]);
        assert_eq!(result.dialogue().len(), 1);
        assert_eq!(result.observations().len(), 1);

        let unstable = OfflineDialogResult::new(
            source_channels,
            active_channels.clone(),
            vec![final_turn(
                0,
                0,
                WordFinality::Open,
                WordStability::Revisable,
            )?],
            vec![TranscriptionObservation::selection_committed(
                source_channels,
                active_channels,
            )?],
        );
        assert!(unstable.is_err());
        Ok(())
    }

    #[test]
    fn patch_observations_are_constructed_without_client_event_state() {
        assert!(matches!(
            TranscriptionObservation::dialog_patch_generated(),
            TranscriptionObservation::DialogPatchGenerated
        ));
    }

    #[test]
    fn raw_formats_and_arbitrary_partitions_preserve_incremental_semantics() -> Result<(), String> {
        let rate = model_rate()?;
        for format in [
            SampleFormat::Pcm16,
            SampleFormat::F32,
            SampleFormat::Alaw,
            SampleFormat::Ulaw,
        ] {
            let payload = raw_payload(format);
            let setup = online_setup(
                format,
                rate,
                StreamingChannelPolicy::AllChannels,
                StreamEmissionMode::WordsAndDialog,
                BackchannelPolicy::Disabled,
            )?;
            let mut one_shot_factory =
                RecordingFactory::new(rate, ordinary_words()?, success_behaviors());
            let one_shot = run_online(
                setup.clone(),
                &mut one_shot_factory,
                &payload,
                &[payload.len()],
            )?;
            let mut partitioned_factory =
                RecordingFactory::new(rate, ordinary_words()?, success_behaviors());
            let partitioned = run_online(
                setup,
                &mut partitioned_factory,
                &payload,
                &irregular_partition_ends(payload.len())?,
            )?;

            assert_eq!(
                observe_run(&one_shot)?,
                observe_run(&partitioned)?,
                "{format:?} must be insensitive to raw byte partitioning"
            );
            assert_eq!(one_shot_factory.trace().factory_channels, vec![0, 1]);
            assert_eq!(partitioned_factory.trace().factory_channels, vec![0, 1]);
        }
        Ok(())
    }

    #[test]
    fn deduplicate_uses_the_exact_window_and_releases_overshoot_once() -> Result<(), String> {
        let rate = model_rate()?;
        let payload = exact_window_overshoot_payload();
        let policy = StreamingChannelPolicy::Deduplicate {
            threshold: CorrelationThreshold::new(0.98)?,
            analysis_window: SelectionWindowSamples::new(4)?,
        };
        for cut_frames in [3_usize, 4, 5] {
            let setup = online_setup(
                SampleFormat::Pcm16,
                rate,
                policy,
                StreamEmissionMode::WordsAndDialog,
                BackchannelPolicy::Disabled,
            )?;
            let mut factory = RecordingFactory::new(rate, no_words(), success_behaviors());
            let reference_features = complete_channel_reference_features(
                Arc::clone(&factory.frontend),
                fast_stream_config()?,
                &exact_window_retained_channel_payload(),
            )?;
            let mut session =
                MultiChannelSession::<RecordingDecoder, BlankDetector>::new(setup, &mut factory)?;
            let cut_bytes = cut_frames
                .checked_mul(4)
                .ok_or_else(|| "test PCM16 cut offset overflows usize".to_owned())?;
            let first = session.push(&payload[..cut_bytes]).map_err(failure_error)?;
            if cut_frames == 3 {
                assert!(first.emission_groups().is_empty());
                assert!(first.observations().is_empty());
                assert_eq!(factory.trace().decode_channels, vec![0, 1]);
            }
            let second = session.push(&payload[cut_bytes..]).map_err(failure_error)?;
            let terminal = session.finish().map_err(failure_error)?;
            let steps = vec![first, second, terminal];

            assert_eq!(committed_selections(&steps), vec![vec![0]]);
            let trace = factory.trace();
            assert_eq!(trace.factory_channels, vec![0, 1]);
            assert_eq!(
                decoder_features_for_channel(&trace, 0),
                reference_features,
                "the retained channel must deliver the complete-audio decoder feature trace exactly once"
            );
            assert_eq!(
                decoder_call_count(&trace, 1),
                1,
                "the rejected channel must receive only construction warmup"
            );
        }
        Ok(())
    }

    #[test]
    fn terminal_partial_frame_refuses_before_tail_or_session_work() -> Result<(), String> {
        let model_rate = model_rate()?;
        let setup = online_setup(
            SampleFormat::Pcm16,
            SampleRate::new(8)?,
            StreamingChannelPolicy::AllChannels,
            StreamEmissionMode::WordsAndDialog,
            BackchannelPolicy::Disabled,
        )?;
        let mut factory = RecordingFactory::new(model_rate, no_words(), success_behaviors());
        let mut session =
            MultiChannelSession::<RecordingDecoder, BlankDetector>::new(setup, &mut factory)?;
        let before_terminal = session.push(&[0_u8, 0_u8, 0_u8]).map_err(failure_error)?;
        assert!(before_terminal.emission_groups().is_empty());
        assert!(before_terminal.observations().is_empty());
        let failure = match session.finish() {
            Ok(_) => return Err("a terminal partial interleaved frame must refuse".into()),
            Err(failure) => failure,
        };
        assert!(failure.error().contains("incomplete frame bytes"));
        assert!(failure.observations().is_empty());
        assert_eq!(factory.trace().decode_channels, vec![0, 1]);
        Ok(())
    }

    #[test]
    fn all_channels_activates_without_correlation_or_selection_observation() -> Result<(), String> {
        let rate = model_rate()?;
        let payload = exact_window_overshoot_payload();
        let setup = online_setup(
            SampleFormat::Pcm16,
            rate,
            StreamingChannelPolicy::AllChannels,
            StreamEmissionMode::WordsAndDialog,
            BackchannelPolicy::Disabled,
        )?;
        let mut factory = RecordingFactory::new(rate, no_words(), success_behaviors());
        let steps = run_online(setup, &mut factory, &payload, &[payload.len()])?;
        assert!(committed_selections(&steps).is_empty());
        let trace = factory.trace();
        assert_eq!(trace.factory_channels, vec![0, 1]);
        assert!(decoder_call_count(&trace, 0) > 1);
        assert!(decoder_call_count(&trace, 1) > 1);
        Ok(())
    }

    #[test]
    fn three_channel_stream_keeps_original_construction_and_drive_order() -> Result<(), String> {
        let rate = model_rate()?;
        let source_channels = SourceChannelCount::new(3)?;
        let setup = MultiChannelStreamSetup::new(MultiChannelStreamSetupInput {
            sample_format: SampleFormat::Pcm16,
            source_sample_rate: rate,
            source_channels,
            stream_config: fast_stream_config()?,
            options: MultiChannelStreamOptions::new(
                StreamingChannelPolicy::AllChannels,
                StreamEmissionMode::WordsAndDialog,
                TurnGap::new(0.3)?,
                BackchannelPolicy::Disabled,
            ),
        })?;
        let frames = [
            (1_000_i16, 2_000_i16, 3_000_i16),
            (-1_000_i16, -2_000_i16, -3_000_i16),
            (4_000_i16, 5_000_i16, 6_000_i16),
            (-4_000_i16, -5_000_i16, -6_000_i16),
            (7_000_i16, 8_000_i16, 9_000_i16),
            (-7_000_i16, -8_000_i16, -9_000_i16),
        ];
        let mut payload = Vec::new();
        for (first, second, third) in frames {
            payload.extend_from_slice(&first.to_le_bytes());
            payload.extend_from_slice(&second.to_le_bytes());
            payload.extend_from_slice(&third.to_le_bytes());
        }
        let mut factory = RecordingFactory::new(
            rate,
            vec![Vec::new(), Vec::new(), Vec::new()],
            vec![
                DecoderBehavior::Succeeds,
                DecoderBehavior::Succeeds,
                DecoderBehavior::Succeeds,
            ],
        );
        let steps = run_online(setup, &mut factory, &payload, &[payload.len()])?;
        assert!(committed_selections(&steps).is_empty());
        let trace = factory.trace();
        assert_eq!(trace.factory_channels, vec![0, 1, 2]);
        assert!(decoder_call_count(&trace, 0) > 1);
        assert!(decoder_call_count(&trace, 1) > 1);
        assert!(decoder_call_count(&trace, 2) > 1);
        Ok(())
    }

    #[test]
    fn first_live_error_or_cancellation_stops_later_active_channels() -> Result<(), String> {
        let rate = model_rate()?;
        let payload = raw_payload(SampleFormat::Pcm16);
        for behavior in [
            DecoderBehavior::RefusesAfterWarmup,
            DecoderBehavior::CancelsAfterWarmup,
        ] {
            let setup = online_setup(
                SampleFormat::Pcm16,
                rate,
                StreamingChannelPolicy::AllChannels,
                StreamEmissionMode::WordsAndDialog,
                BackchannelPolicy::Disabled,
            )?;
            let mut factory =
                RecordingFactory::new(rate, no_words(), vec![behavior, DecoderBehavior::Succeeds]);
            let mut session =
                MultiChannelSession::<RecordingDecoder, BlankDetector>::new(setup, &mut factory)?;
            let failure = match session.push(&payload) {
                Ok(_) => return Err("the first active decoder must stop this transition".into()),
                Err(failure) => failure,
            };
            assert!(failure.observations().is_empty());
            assert_eq!(factory.trace().decode_channels, vec![0, 1, 0]);
            let retry = match session.push(&payload) {
                Ok(_) => {
                    return Err("a failed multi-channel session must reject later input".into());
                }
                Err(failure) => failure,
            };
            assert!(retry.error().contains("already failed"));
            assert_eq!(factory.trace().decode_channels, vec![0, 1, 0]);
        }
        Ok(())
    }

    #[test]
    fn later_live_failure_or_cancellation_is_atomic_and_stops_the_third_channel()
    -> Result<(), String> {
        let rate = model_rate()?;
        let source_channels = SourceChannelCount::new(3)?;
        let payload = three_channel_pcm16_payload();
        for behavior in [
            DecoderBehavior::RefusesAfterWarmup,
            DecoderBehavior::CancelsAfterWarmup,
        ] {
            let setup = online_setup_for_channels(
                source_channels,
                SampleFormat::Pcm16,
                rate,
                StreamingChannelPolicy::AllChannels,
                StreamEmissionMode::WordsAndDialog,
                BackchannelPolicy::Disabled,
            )?;
            let mut factory = RecordingFactory::new(
                rate,
                three_channel_words()?,
                vec![
                    DecoderBehavior::Succeeds,
                    behavior,
                    DecoderBehavior::Succeeds,
                ],
            );
            let mut session =
                MultiChannelSession::<RecordingDecoder, BlankDetector>::new(setup, &mut factory)?;

            let failure = match session.push(&payload) {
                Ok(_) => {
                    return Err(
                        "a later active-channel failure must not publish a partial live step"
                            .into(),
                    );
                }
                Err(failure) => failure,
            };
            assert!(!failure.error().is_empty());
            assert!(failure.observations().is_empty());
            assert_eq!(factory.trace().decode_channels, vec![0, 1, 2, 0, 1]);

            let retry = match session.push(&payload) {
                Ok(_) => return Err("a failed multi-channel session must reject reuse".into()),
                Err(failure) => failure,
            };
            assert!(retry.error().contains("already failed"));
            assert!(retry.observations().is_empty());
            assert_eq!(factory.trace().decode_channels, vec![0, 1, 2, 0, 1]);
        }
        Ok(())
    }

    #[test]
    fn terminal_tail_error_stops_before_the_later_active_channel() -> Result<(), String> {
        let model_rate = model_rate()?;
        let setup = online_setup(
            SampleFormat::F32,
            SampleRate::new(8)?,
            StreamingChannelPolicy::AllChannels,
            StreamEmissionMode::WordsAndDialog,
            BackchannelPolicy::Disabled,
        )?;
        let payload = f32_interleaved(&[0.2_f32, -0.2_f32], &[0.2_f32, -0.2_f32])?;
        let mut factory = RecordingFactory::new(
            model_rate,
            no_words(),
            vec![
                DecoderBehavior::RefusesAfterWarmup,
                DecoderBehavior::Succeeds,
            ],
        );
        let mut session =
            MultiChannelSession::<RecordingDecoder, BlankDetector>::new(setup, &mut factory)?;
        let before_terminal = session.push(&payload).map_err(failure_error)?;
        assert!(before_terminal.emission_groups().is_empty());
        let failure = match session.finish() {
            Ok(_) => return Err("the first active terminal tail decode must refuse".into()),
            Err(failure) => failure,
        };
        assert!(failure.observations().is_empty());
        assert_eq!(factory.trace().decode_channels, vec![0, 1, 0]);
        Ok(())
    }

    #[test]
    fn later_terminal_tail_failure_or_cancellation_is_atomic_and_stops_the_third_channel()
    -> Result<(), String> {
        let source_rate = SampleRate::new(8)?;
        let model_rate = model_rate()?;
        let source_channels = SourceChannelCount::new(3)?;
        let payload = f32_interleaved_three(
            &[0.2_f32, -0.2_f32],
            &[0.3_f32, -0.3_f32],
            &[0.4_f32, -0.4_f32],
        )?;
        for behavior in [
            DecoderBehavior::RefusesAfterWarmup,
            DecoderBehavior::CancelsAfterWarmup,
        ] {
            let setup = online_setup_for_channels(
                source_channels,
                SampleFormat::F32,
                source_rate,
                StreamingChannelPolicy::AllChannels,
                StreamEmissionMode::WordsAndDialog,
                BackchannelPolicy::Disabled,
            )?;
            let mut factory = RecordingFactory::new(
                model_rate,
                three_channel_words()?,
                vec![
                    DecoderBehavior::Succeeds,
                    behavior,
                    DecoderBehavior::Succeeds,
                ],
            );
            let mut session =
                MultiChannelSession::<RecordingDecoder, BlankDetector>::new(setup, &mut factory)?;

            let before_terminal = session.push(&payload).map_err(failure_error)?;
            assert!(before_terminal.emission_groups().is_empty());
            assert!(before_terminal.observations().is_empty());
            assert_eq!(factory.trace().decode_channels, vec![0, 1, 2]);

            let failure = match session.finish() {
                Ok(_) => {
                    return Err(
                        "a later terminal-tail failure must not publish a partial terminal step"
                            .into(),
                    );
                }
                Err(failure) => failure,
            };
            assert!(!failure.error().is_empty());
            assert!(failure.observations().is_empty());
            assert_eq!(factory.trace().decode_channels, vec![0, 1, 2, 0, 0, 1]);
        }
        Ok(())
    }

    #[test]
    fn selection_observation_survives_a_failed_pending_release() -> Result<(), String> {
        let rate = model_rate()?;
        let setup = online_setup(
            SampleFormat::Pcm16,
            rate,
            StreamingChannelPolicy::Deduplicate {
                threshold: CorrelationThreshold::new(0.98)?,
                analysis_window: SelectionWindowSamples::new(4)?,
            },
            StreamEmissionMode::WordsAndDialog,
            BackchannelPolicy::Disabled,
        )?;
        let mut factory = RecordingFactory::new(
            rate,
            no_words(),
            vec![
                DecoderBehavior::RefusesAfterWarmup,
                DecoderBehavior::Succeeds,
            ],
        );
        let mut session =
            MultiChannelSession::<RecordingDecoder, BlankDetector>::new(setup, &mut factory)?;
        let failure = match session.push(&exact_window_overshoot_payload()) {
            Ok(_) => {
                return Err("the retained channel decoder must refuse its pending release".into());
            }
            Err(failure) => failure,
        };
        assert_eq!(failure.observations().len(), 1);
        assert!(matches!(
            failure.observations().first(),
            Some(TranscriptionObservation::ChannelSelectionCommitted(selection))
                if selection.active_channels().iter().map(|channel| channel.index()).collect::<Vec<_>>() == vec![0]
        ));
        assert_eq!(factory.trace().decode_channels, vec![0, 1, 0]);
        Ok(())
    }

    #[test]
    fn emission_modes_preserve_event_order_patch_observations_and_dialogue_state()
    -> Result<(), String> {
        let rate = model_rate()?;
        let payload = raw_payload(SampleFormat::Pcm16);
        let backchannel_policy =
            BackchannelPolicy::MarkShorterThan(BackchannelDuration::new(0.15)?);
        let mut expected_dialog_patch_observations = None;
        for mode in [
            StreamEmissionMode::Words,
            StreamEmissionMode::Dialog,
            StreamEmissionMode::WordsAndDialog,
        ] {
            let setup = online_setup(
                SampleFormat::Pcm16,
                rate,
                StreamingChannelPolicy::AllChannels,
                mode,
                backchannel_policy,
            )?;
            let mut factory =
                RecordingFactory::new(rate, backchannel_words()?, success_behaviors());
            let steps = run_online(setup, &mut factory, &payload, &[payload.len()])?;
            let observed = observe_run(&steps)?;
            let patch_observations = patch_observation_count(&steps);
            for group in steps.iter().flat_map(MultiChannelStep::emission_groups) {
                assert!(
                    group
                        .channel_events()
                        .windows(2)
                        .all(|pair| { pair[0].channel().index() <= pair[1].channel().index() })
                );
            }

            match mode {
                StreamEmissionMode::Words => {
                    assert!(!observed.events.is_empty());
                    assert!(observed.patches.is_empty());
                    assert_eq!(patch_observations, 0);
                }
                StreamEmissionMode::Dialog => {
                    assert!(observed.events.is_empty());
                    assert_eq!(observed.patches.len(), patch_observations);
                    assert!(patch_observations > 0);
                    expected_dialog_patch_observations = Some(patch_observations);
                }
                StreamEmissionMode::WordsAndDialog => {
                    assert!(!observed.events.is_empty());
                    assert_eq!(observed.patches.len(), patch_observations);
                    assert_eq!(expected_dialog_patch_observations, Some(patch_observations));
                    assert!(
                        observed
                            .patches
                            .windows(2)
                            .all(|pair| { pair[0].frontier() <= pair[1].frontier() })
                    );
                    assert!(observed.dialogue.iter().all(|turn| {
                        turn.finality() == WordFinality::Final
                            && turn.stability() == WordStability::Stable
                    }));
                    assert!(observed.dialogue.iter().any(|turn| {
                        turn.channel() == 1 && turn.backchannel() == BackchannelMark::Yes
                    }));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn nonidentity_terminal_tail_matches_offline_selection_and_dialogue() -> Result<(), String> {
        let source_rate = SampleRate::new(8)?;
        let model_rate = model_rate()?;
        let source_left = [0.2_f32, -0.2_f32];
        let source_right = [0.2_f32, -0.2_f32];
        let payload = f32_interleaved(&source_left, &source_right)?;
        let policy = StreamingChannelPolicy::Deduplicate {
            threshold: CorrelationThreshold::new(0.98)?,
            analysis_window: SelectionWindowSamples::new(4)?,
        };
        let setup = online_setup(
            SampleFormat::F32,
            source_rate,
            policy,
            StreamEmissionMode::WordsAndDialog,
            BackchannelPolicy::Disabled,
        )?;
        let mut online_factory =
            RecordingFactory::new(model_rate, ordinary_words()?, success_behaviors());
        let mut online = MultiChannelSession::<RecordingDecoder, BlankDetector>::new(
            setup,
            &mut online_factory,
        )?;
        let before_terminal = online.push(&payload).map_err(failure_error)?;
        assert!(before_terminal.emission_groups().is_empty());
        assert!(before_terminal.observations().is_empty());
        let terminal = online.finish().map_err(failure_error)?;
        let online_steps = vec![before_terminal, terminal];
        assert_eq!(committed_selections(&online_steps), vec![vec![0]]);
        let online_observed = observe_run(&online_steps)?;

        let resampler = Resampler::new(ResamplerConfig::new(RatePair::new(
            source_rate,
            model_rate,
        )?))?;
        let offline_channels = vec![
            ChannelAudio::new(resampler.process(&source_left)?)?,
            ChannelAudio::new(resampler.process(&source_right)?)?,
        ];
        assert_eq!(offline_channels[0].len(), 4);
        let offline_setup = OfflineDialogSetup::new(OfflineDialogSetupInput {
            channel_policy: OfflineChannelPolicy::DialogDeduplication,
            stream_config: fast_stream_config()?,
            turn_gap: TurnGap::new(0.3)?,
            backchannel_policy: BackchannelPolicy::Disabled,
        })?;
        let mut offline_factory =
            RecordingFactory::new(model_rate, ordinary_words()?, success_behaviors());
        let offline =
            transcribe_offline_dialog(offline_setup, &mut offline_factory, &offline_channels)
                .map_err(offline_failure_error)?;

        assert_eq!(
            offline
                .active_channels()
                .iter()
                .map(|channel| channel.index())
                .collect::<Vec<_>>(),
            vec![0]
        );
        assert_eq!(online_observed.dialogue, offline.dialogue());
        assert!(online_observed.dialogue.iter().all(|turn| {
            turn.finality() == WordFinality::Final && turn.stability() == WordStability::Stable
        }));
        assert_eq!(online_factory.trace().factory_channels, vec![0, 1]);
        assert_eq!(offline_factory.trace().factory_channels, vec![0]);
        assert_eq!(decoder_call_count(&online_factory.trace(), 0), 3);
        assert_eq!(decoder_call_count(&online_factory.trace(), 1), 1);
        Ok(())
    }
}
