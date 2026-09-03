// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Atomic multi-channel batch transcription over one mutable recognition decoder.

use crate::batch::{BatchSetup, BatchTranscriber};
use crate::channel_selection::{
    BatchChannelPolicy, DuplicateClassification, SourceChannelCount, select_batch_channels,
};
use crate::contracts::{StageTimings, Turn, checkpoint};
use crate::turns::{ChannelTranscript, TurnGap, channel_segments, turns};
use gigaam_audio::DecodedAudio;
use gigaam_recognition::{ExecutionControl, WindowDecoder};
use std::fmt::{Display, Formatter};

/// Typed options that determine multi-channel batch selection and final turn construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiChannelBatchOptions {
    channel_policy: BatchChannelPolicy,
    turn_gap: TurnGap,
}

impl MultiChannelBatchOptions {
    /// Combines independently validated channel-selection and turn-boundary policies.
    pub const fn new(channel_policy: BatchChannelPolicy, turn_gap: TurnGap) -> Self {
        Self {
            channel_policy,
            turn_gap,
        }
    }

    /// Returns the typed channel-selection policy.
    pub const fn channel_policy(self) -> BatchChannelPolicy {
        self.channel_policy
    }

    /// Returns the validated pause duration that separates final turns.
    pub const fn turn_gap(self) -> TurnGap {
        self.turn_gap
    }
}

/// Complete capabilities and typed options required for one multi-channel batch operation.
pub struct MultiChannelBatchSetup<D: WindowDecoder> {
    batch: BatchSetup<D>,
    options: MultiChannelBatchOptions,
}

impl<D: WindowDecoder> MultiChannelBatchSetup<D> {
    /// Groups one single-channel transcriber setup with its multi-channel application options.
    pub const fn new(batch: BatchSetup<D>, options: MultiChannelBatchOptions) -> Self {
        Self { batch, options }
    }
}

/// A refusal or failure from the all-or-nothing multi-channel batch operation.
#[derive(Debug, PartialEq, Eq)]
pub enum MultiChannelBatchError {
    /// Decoded audio cannot be represented at, or does not equal, the frontend sample rate.
    Input(String),
    /// Rate-compatible decoded audio cannot satisfy the typed batch channel-selection contract.
    Selection(String),
    /// Shared control, frontend or recognition work, or final projection prevented completion.
    Transcription(String),
}

impl Display for MultiChannelBatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(message) | Self::Selection(message) | Self::Transcription(message) => {
                formatter.write_str(message)
            }
        }
    }
}

/// The complete, immutable result of one multi-channel batch operation.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiChannelBatchResult {
    source_channels: SourceChannelCount,
    duplicate: DuplicateClassification,
    channels: Vec<ChannelTranscript>,
    segments: Vec<Turn>,
    turns: Vec<Turn>,
    timings: StageTimings,
}

/// Named inputs for constructing one validated immutable multi-channel batch result.
///
/// Segments and turns have the same collection type but different ordering contracts, so their
/// names are part of the construction boundary rather than interchangeable positional values.
struct MultiChannelBatchResultInput {
    source_channels: SourceChannelCount,
    duplicate: DuplicateClassification,
    channels: Vec<ChannelTranscript>,
    segments: Vec<Turn>,
    turns: Vec<Turn>,
    timings: StageTimings,
}

impl MultiChannelBatchResult {
    fn new(input: MultiChannelBatchResultInput) -> Result<Self, String> {
        validate_selected_channels(input.source_channels, &input.channels)?;
        validate_stage_timings(input.timings, "multi-channel batch result")?;
        Ok(Self {
            source_channels: input.source_channels,
            duplicate: input.duplicate,
            channels: input.channels,
            segments: input.segments,
            turns: input.turns,
            timings: input.timings,
        })
    }

    /// Returns the validated source-channel count before selection.
    pub const fn source_channels(&self) -> SourceChannelCount {
        self.source_channels
    }

    /// Returns the fixed-rule duplicate classification for the complete input.
    pub const fn duplicate(&self) -> DuplicateClassification {
        self.duplicate
    }

    /// Returns selected channel transcripts in ascending original-channel order.
    pub fn channels(&self) -> &[ChannelTranscript] {
        &self.channels
    }

    /// Returns final segments grouped by ascending original-channel order.
    pub fn segments(&self) -> &[Turn] {
        &self.segments
    }

    /// Returns final turns ordered by start time and then original-channel identity.
    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// Returns successful stage timings aggregated over every selected channel.
    pub const fn timings(&self) -> StageTimings {
        self.timings
    }
}

/// A one-shot multi-channel batch transcriber that reuses one mutable decoder sequentially.
pub struct MultiChannelBatchTranscriber<D: WindowDecoder> {
    batch: BatchTranscriber<D>,
    control: ExecutionControl,
    options: MultiChannelBatchOptions,
}

impl<D: WindowDecoder> MultiChannelBatchTranscriber<D> {
    /// Creates one multi-channel owner around a single mutable decoder and shared control state.
    pub fn new(setup: MultiChannelBatchSetup<D>) -> Result<Self, String> {
        // The extra handle observes the same caller-owned state before and after whole channels;
        // the embedded batch transcriber retains the original handle for window-level checks.
        let control = setup.batch.control.clone();
        let batch = BatchTranscriber::new(setup.batch)?;
        Ok(Self {
            batch,
            control,
            options: setup.options,
        })
    }

    /// Returns the sample rate required by the supplied single-channel batch transcriber.
    pub const fn sample_rate(&self) -> usize {
        self.batch.sample_rate()
    }

    /// Warms the one decoder before its single consuming batch operation.
    pub fn warmup(&mut self) -> Result<(), String> {
        self.batch.warmup()
    }

    /// Transcribes every selected source channel or returns no result.
    pub fn transcribe(
        mut self,
        input: &DecodedAudio,
    ) -> Result<MultiChannelBatchResult, MultiChannelBatchError> {
        checkpoint(&self.control).map_err(MultiChannelBatchError::Transcription)?;
        let input_sample_rate = input
            .sample_rate()
            .as_usize()
            .map_err(MultiChannelBatchError::Input)?;
        if input_sample_rate != self.batch.sample_rate() {
            return Err(MultiChannelBatchError::Input(format!(
                "batch input sample rate {input_sample_rate} Hz does not match frontend sample rate {} Hz",
                self.batch.sample_rate()
            )));
        }
        let channel_views = input
            .channels()
            .iter()
            .map(|channel| channel.view())
            .collect::<Vec<_>>();
        let selection = select_batch_channels(&channel_views, self.options.channel_policy())
            .map_err(MultiChannelBatchError::Selection)?;
        let source_channels = selection.source_channels();
        let duplicate = selection.duplicate();
        let selected_channels = selection.selection().channels();
        let mut channels = Vec::with_capacity(selected_channels.len());
        let mut timing_aggregate = ChannelTimingAggregate::new(self.batch.timings())
            .map_err(MultiChannelBatchError::Transcription)?;

        for selected in selected_channels {
            checkpoint(&self.control).map_err(MultiChannelBatchError::Transcription)?;
            let timings_before_channel = self.batch.timings();
            let channel = input
                .channels()
                .get(selected.index())
                .expect("validated channel selection must reference its source audio");
            let transcript = self
                .batch
                .transcribe_channel(channel)
                .map_err(MultiChannelBatchError::Transcription)?;
            checkpoint(&self.control).map_err(MultiChannelBatchError::Transcription)?;
            timing_aggregate
                .record(timings_before_channel, self.batch.timings())
                .map_err(MultiChannelBatchError::Transcription)?;
            let channel = ChannelTranscript::new(selected.index(), transcript.into_words())
                .map_err(MultiChannelBatchError::Transcription)?;
            channels.push(channel);
        }

        checkpoint(&self.control).map_err(MultiChannelBatchError::Transcription)?;
        let segments = channel_segments(&channels, self.options.turn_gap())
            .map_err(MultiChannelBatchError::Transcription)?;
        checkpoint(&self.control).map_err(MultiChannelBatchError::Transcription)?;
        let turns = turns(&channels, self.options.turn_gap())
            .map_err(MultiChannelBatchError::Transcription)?;
        checkpoint(&self.control).map_err(MultiChannelBatchError::Transcription)?;
        MultiChannelBatchResult::new(MultiChannelBatchResultInput {
            source_channels,
            duplicate,
            channels,
            segments,
            turns,
            timings: timing_aggregate.finish(),
        })
        .map_err(MultiChannelBatchError::Transcription)
    }
}

fn validate_selected_channels(
    source_channels: SourceChannelCount,
    channels: &[ChannelTranscript],
) -> Result<(), String> {
    let Some(first) = channels.first() else {
        return Err("multi-channel batch result requires at least one selected channel".into());
    };
    if first.channel() >= source_channels.get() {
        return Err("selected channel is outside the source channel count".into());
    }
    let mut previous = first.channel();
    for channel in &channels[1..] {
        if channel.channel() >= source_channels.get() {
            return Err("selected channel is outside the source channel count".into());
        }
        if channel.channel() <= previous {
            return Err("selected channel identities must be strictly ascending".into());
        }
        previous = channel.channel();
    }
    Ok(())
}

/// Aggregates timing deltas derived from consecutive cumulative batch snapshots.
///
/// A batch transcriber owns cumulative timings because one decoder processes consecutive windows.
/// This value derives each selected channel's contribution once, preserving that cumulative state
/// as the next required predecessor and refusing inconsistent timing observations.
#[derive(Clone, Copy, Debug)]
struct ChannelTimingAggregate {
    expected_before: StageTimings,
    total: StageTimings,
}

impl ChannelTimingAggregate {
    fn new(initial: StageTimings) -> Result<Self, String> {
        validate_stage_timings(initial, "initial cumulative batch")?;
        Ok(Self {
            expected_before: initial,
            total: StageTimings::default(),
        })
    }

    /// Records the contribution between consecutive cumulative timing snapshots.
    fn record(&mut self, before: StageTimings, after: StageTimings) -> Result<(), String> {
        validate_stage_timings(before, "previous cumulative batch")?;
        validate_stage_timings(after, "current cumulative batch")?;
        if before != self.expected_before {
            return Err("multi-channel batch timing snapshots are not consecutive".into());
        }
        let contribution = stage_timing_delta(before, after)?;
        let total = self.total.combined(contribution);
        validate_stage_timings(total, "multi-channel batch aggregate")?;
        self.expected_before = after;
        self.total = total;
        Ok(())
    }

    const fn finish(self) -> StageTimings {
        self.total
    }
}

fn stage_timing_delta(before: StageTimings, after: StageTimings) -> Result<StageTimings, String> {
    let frontend = stage_timing_delta_seconds(
        before.frontend_seconds(),
        after.frontend_seconds(),
        "frontend",
    )?;
    let encoder =
        stage_timing_delta_seconds(before.encoder_seconds(), after.encoder_seconds(), "encoder")?;
    let decode =
        stage_timing_delta_seconds(before.decode_seconds(), after.decode_seconds(), "decode")?;
    let mut contribution = StageTimings::default();
    contribution.add_frontend(frontend);
    contribution.add_encoder(encoder);
    contribution.add_decode(decode);
    Ok(contribution)
}

fn stage_timing_delta_seconds(before: f64, after: f64, stage: &str) -> Result<f64, String> {
    let delta = after - before;
    if !delta.is_finite() || delta < 0.0 {
        return Err(format!(
            "cumulative {stage} timing must be finite and nondecreasing"
        ));
    }
    Ok(delta)
}

fn validate_stage_timings(timings: StageTimings, context: &str) -> Result<(), String> {
    validate_stage_timing(timings.frontend_seconds(), context, "frontend")?;
    validate_stage_timing(timings.encoder_seconds(), context, "encoder")?;
    validate_stage_timing(timings.decode_seconds(), context, "decode")?;
    Ok(())
}

fn validate_stage_timing(value: f64, context: &str, stage: &str) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "{context} {stage} timing must be finite and nonnegative"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ChannelTimingAggregate, MultiChannelBatchError, MultiChannelBatchOptions,
        MultiChannelBatchSetup, MultiChannelBatchTranscriber,
    };
    use crate::batch::BatchSetup;
    use crate::channel_selection::{BatchChannelPolicy, DuplicateClassification};
    use crate::contracts::{BatchConfig, PadPolicy, StageTimings};
    use crate::observations::ObservationMode;
    use crate::turns::TurnGap;
    use gigaam_audio::{ChannelAudio, DecodedAudio, FeatureMatrixView, SampleRate};
    use gigaam_recognition::{Decoded, ExecutionControl, FrameRate, WindowDecoder, Word};
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Copy)]
    struct WordPlan {
        text: &'static str,
        start: f32,
        end: f32,
    }

    enum DecoderAction {
        Complete,
        CancelAfterFirst(ExecutionControl),
        FailOnCall(usize),
    }

    struct StatefulDecoder {
        calls: Arc<Mutex<Vec<usize>>>,
        next_call: usize,
        action: DecoderAction,
        words: Vec<WordPlan>,
    }

    impl StatefulDecoder {
        fn new(calls: Arc<Mutex<Vec<usize>>>, action: DecoderAction, words: Vec<WordPlan>) -> Self {
            Self {
                calls,
                next_call: 0,
                action,
                words,
            }
        }
    }

    impl WindowDecoder for StatefulDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(8.0).expect("test decoder frame rate is finite and positive")
        }

        fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            let call = self.next_call;
            self.next_call = self
                .next_call
                .checked_add(1)
                .ok_or_else(|| "test decoder call index overflows".to_owned())?;
            self.calls
                .lock()
                .expect("test call trace lock must not be poisoned")
                .push(call);
            match &self.action {
                DecoderAction::Complete => {}
                DecoderAction::CancelAfterFirst(control) => {
                    if call == 0 {
                        control.request_cancellation();
                    }
                }
                DecoderAction::FailOnCall(failing_call) => {
                    if call == *failing_call {
                        return Err(format!("test decoder fails on call {call}"));
                    }
                }
            }
            let plan = self
                .words
                .get(call)
                .ok_or_else(|| "test decoder received an unexpected call".to_owned())?;
            let word = Word::new(plan.text.into(), plan.start, plan.end)?;
            Decoded::new(
                vec![word],
                vec![false; features.frames()],
                features.frames(),
                0.0,
            )
        }
    }

    fn channel(samples: Vec<f32>) -> ChannelAudio {
        ChannelAudio::new(samples).expect("test channel samples must be finite")
    }

    fn audio_at_sample_rate(sample_rate: u32, channels: Vec<ChannelAudio>) -> DecodedAudio {
        DecodedAudio::new(
            SampleRate::new(sample_rate).expect("test sample rate is positive"),
            channels,
        )
        .expect("test channels have equal nonzero frame counts")
    }

    fn audio(channels: Vec<ChannelAudio>) -> DecodedAudio {
        audio_at_sample_rate(16, channels)
    }

    fn one_channel() -> DecodedAudio {
        audio(vec![channel(vec![0.25; 48])])
    }

    fn dual_mono() -> DecodedAudio {
        audio(vec![
            channel([0.0_f32, 0.5_f32, -0.5_f32, 0.25_f32].repeat(12)),
            channel([0.0_f32, 0.5_f32, -0.5_f32, 0.25_f32].repeat(12)),
        ])
    }

    fn three_channels() -> DecodedAudio {
        audio(vec![
            channel(vec![0.25; 48]),
            channel(vec![-0.25; 48]),
            channel([0.5_f32, -0.5_f32].repeat(24)),
        ])
    }

    fn plans(count: usize) -> Vec<WordPlan> {
        let candidates = [
            WordPlan {
                text: "late",
                start: 2.0,
                end: 2.2,
            },
            WordPlan {
                text: "early",
                start: 0.5,
                end: 0.7,
            },
            WordPlan {
                text: "middle",
                start: 1.0,
                end: 1.2,
            },
        ];
        candidates[..count].to_vec()
    }

    fn batch(
        decoder: StatefulDecoder,
        policy: BatchChannelPolicy,
        padding: PadPolicy,
        control: ExecutionControl,
    ) -> MultiChannelBatchTranscriber<StatefulDecoder> {
        let options = MultiChannelBatchOptions::new(
            policy,
            TurnGap::new(1.0).expect("test turn gap is valid"),
        );
        MultiChannelBatchTranscriber::new(MultiChannelBatchSetup::new(
            BatchSetup {
                frontend: crate::test_support::frontend(),
                decoder,
                config: BatchConfig::new(
                    SampleRate::new(16).expect("test sample rate is positive"),
                    4.0,
                    1.0,
                    padding,
                )
                .expect("test batch configuration is valid"),
                control,
                observations: ObservationMode::disabled(),
            },
            options,
        ))
        .expect("test multi-channel setup is valid")
    }

    fn channel_ids(result: &super::MultiChannelBatchResult) -> Vec<usize> {
        result
            .channels()
            .iter()
            .map(|channel| channel.channel())
            .collect()
    }

    fn calls(trace: &Arc<Mutex<Vec<usize>>>) -> Vec<usize> {
        trace
            .lock()
            .expect("test call trace lock must not be poisoned")
            .clone()
    }

    fn stage_timings(frontend: f64, encoder: f64, decode: f64) -> StageTimings {
        let mut timings = StageTimings::default();
        timings.add_frontend(frontend);
        timings.add_encoder(encoder);
        timings.add_decode(decode);
        timings
    }

    #[test]
    fn batch_policies_preserve_original_identity_for_one_two_and_three_channels() {
        let one_calls = Arc::new(Mutex::new(Vec::new()));
        let one = batch(
            StatefulDecoder::new(one_calls.clone(), DecoderAction::Complete, plans(1)),
            BatchChannelPolicy::separate_channels(),
            PadPolicy::Exact,
            ExecutionControl::without_deadline(),
        )
        .transcribe(&one_channel())
        .expect("one channel must transcribe");
        assert_eq!(one.source_channels().get(), 1);
        assert_eq!(one.duplicate(), DuplicateClassification::NotDualMono);
        assert_eq!(channel_ids(&one), vec![0]);

        let dual_calls = Arc::new(Mutex::new(Vec::new()));
        let dual = batch(
            StatefulDecoder::new(dual_calls.clone(), DecoderAction::Complete, plans(1)),
            BatchChannelPolicy::separate_channels(),
            PadPolicy::Exact,
            ExecutionControl::without_deadline(),
        )
        .transcribe(&dual_mono())
        .expect("dual mono must transcribe through original channel zero");
        assert_eq!(dual.source_channels().get(), 2);
        assert_eq!(dual.duplicate(), DuplicateClassification::DualMono);
        assert_eq!(channel_ids(&dual), vec![0]);

        let single_calls = Arc::new(Mutex::new(Vec::new()));
        let single = batch(
            StatefulDecoder::new(single_calls.clone(), DecoderAction::Complete, plans(1)),
            BatchChannelPolicy::single_output(),
            PadPolicy::Exact,
            ExecutionControl::without_deadline(),
        )
        .transcribe(&three_channels())
        .expect("single-output policy must transcribe original channel zero");
        assert_eq!(channel_ids(&single), vec![0]);

        let split_calls = Arc::new(Mutex::new(Vec::new()));
        let split = batch(
            StatefulDecoder::new(split_calls.clone(), DecoderAction::Complete, plans(3)),
            BatchChannelPolicy::separate_channels(),
            PadPolicy::Exact,
            ExecutionControl::without_deadline(),
        )
        .transcribe(&three_channels())
        .expect("separate-channel policy must preserve every three-channel identity");
        assert_eq!(split.source_channels().get(), 3);
        assert_eq!(split.duplicate(), DuplicateClassification::NotDualMono);
        assert_eq!(channel_ids(&split), vec![0, 1, 2]);
        assert_eq!(calls(&one_calls), vec![0]);
        assert_eq!(calls(&dual_calls), vec![0]);
        assert_eq!(calls(&single_calls), vec![0]);
        assert_eq!(calls(&split_calls), vec![0, 1, 2]);
    }

    #[test]
    fn one_mutable_decoder_preserves_channel_text_timing_segments_turns_and_aggregate_timings() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let result = batch(
            StatefulDecoder::new(trace.clone(), DecoderAction::Complete, plans(3)),
            BatchChannelPolicy::separate_channels(),
            PadPolicy::Exact,
            ExecutionControl::without_deadline(),
        )
        .transcribe(&three_channels())
        .expect("three independent channels must transcribe");

        assert_eq!(calls(&trace), vec![0, 1, 2]);
        assert_eq!(
            result
                .channels()
                .iter()
                .map(|channel| {
                    let word = channel
                        .words()
                        .first()
                        .expect("each one-window test channel yields one word");
                    (channel.channel(), word.text(), word.start(), word.end())
                })
                .collect::<Vec<_>>(),
            vec![
                (0, "late", 2.0, 2.2),
                (1, "early", 0.5, 0.7),
                (2, "middle", 1.0, 1.2),
            ]
        );
        assert_eq!(
            result
                .segments()
                .iter()
                .map(|segment| (segment.channel(), segment.text()))
                .collect::<Vec<_>>(),
            vec![
                (0, "late".to_owned()),
                (1, "early".to_owned()),
                (2, "middle".to_owned()),
            ]
        );
        assert_eq!(
            result
                .turns()
                .iter()
                .map(|turn| (turn.channel(), turn.text()))
                .collect::<Vec<_>>(),
            vec![
                (1, "early".to_owned()),
                (2, "middle".to_owned()),
                (0, "late".to_owned()),
            ]
        );
        assert!(result.timings().frontend_seconds().is_finite());
        assert!(result.timings().encoder_seconds().is_finite());
        assert!(result.timings().decode_seconds().is_finite());
        assert!(result.timings().frontend_seconds() >= 0.0);
        assert!(result.timings().encoder_seconds() >= 0.0);
        assert!(result.timings().decode_seconds() >= 0.0);
    }

    #[test]
    fn timing_aggregate_counts_each_selected_channel_once_across_every_stage() {
        let first_channel = stage_timings(1.0, 10.0, 100.0);
        let second_channel = stage_timings(2.0, 20.0, 200.0);
        let third_channel = stage_timings(3.0, 30.0, 300.0);
        let after_first = first_channel;
        let after_second = after_first.combined(second_channel);
        let after_third = after_second.combined(third_channel);

        let mut aggregate = ChannelTimingAggregate::new(StageTimings::default())
            .expect("zero initial timing is valid");
        aggregate
            .record(StageTimings::default(), after_first)
            .expect("the first cumulative channel timing is valid");
        aggregate
            .record(after_first, after_second)
            .expect("the second cumulative channel timing is valid");
        aggregate
            .record(after_second, after_third)
            .expect("the third cumulative channel timing is valid");

        let actual = aggregate.finish();
        let expected = first_channel
            .combined(second_channel)
            .combined(third_channel);
        assert_eq!(actual, expected);
        assert_eq!(
            (
                actual.frontend_seconds(),
                actual.encoder_seconds(),
                actual.decode_seconds(),
            ),
            (6.0, 60.0, 600.0)
        );
        assert_ne!(actual, third_channel);
        assert_ne!(actual, expected.combined(second_channel));
        assert_ne!(
            actual,
            stage_timings(0.0, 10.0, 100.0)
                .combined(stage_timings(0.0, 20.0, 200.0))
                .combined(stage_timings(0.0, 30.0, 300.0))
        );
        assert_ne!(
            actual,
            stage_timings(1.0, 0.0, 100.0)
                .combined(stage_timings(2.0, 0.0, 200.0))
                .combined(stage_timings(3.0, 0.0, 300.0))
        );
        assert_ne!(
            actual,
            stage_timings(1.0, 10.0, 0.0)
                .combined(stage_timings(2.0, 20.0, 0.0))
                .combined(stage_timings(3.0, 30.0, 0.0))
        );
    }

    #[test]
    fn timing_aggregate_refuses_nonconsecutive_nonmonotonic_and_nonfinite_snapshots() {
        let initial = stage_timings(1.0, 2.0, 3.0);
        let mut aggregate =
            ChannelTimingAggregate::new(initial).expect("finite initial timing is valid");
        assert!(
            aggregate
                .record(StageTimings::default(), stage_timings(2.0, 3.0, 4.0),)
                .is_err()
        );
        assert!(
            aggregate
                .record(initial, stage_timings(0.5, 3.0, 4.0))
                .is_err()
        );
        assert!(
            aggregate
                .record(initial, stage_timings(f64::INFINITY, 3.0, 4.0))
                .is_err()
        );
    }

    #[test]
    fn exact_and_padded_short_inputs_remain_distinct_batch_configurations() {
        let input = audio(vec![channel(vec![0.25; 4])]);
        let exact_trace = Arc::new(Mutex::new(Vec::new()));
        let exact = batch(
            StatefulDecoder::new(
                exact_trace.clone(),
                DecoderAction::Complete,
                vec![WordPlan {
                    text: "short",
                    start: 0.0,
                    end: 0.1,
                }],
            ),
            BatchChannelPolicy::single_output(),
            PadPolicy::Exact,
            ExecutionControl::without_deadline(),
        )
        .transcribe(&input)
        .expect("an exact short input remains a complete empty transcript");
        let padded_trace = Arc::new(Mutex::new(Vec::new()));
        let padded = batch(
            StatefulDecoder::new(
                padded_trace.clone(),
                DecoderAction::Complete,
                vec![WordPlan {
                    text: "short",
                    start: 0.0,
                    end: 0.1,
                }],
            ),
            BatchChannelPolicy::single_output(),
            PadPolicy::PadToWindow,
            ExecutionControl::without_deadline(),
        )
        .transcribe(&input)
        .expect("a padded short input must reach its complete decoder window");

        assert_eq!(calls(&exact_trace), Vec::<usize>::new());
        assert_eq!(calls(&padded_trace), vec![0]);
        assert!(exact.channels()[0].words().is_empty());
        assert_eq!(padded.channels()[0].words()[0].text(), "short");
    }

    #[test]
    fn cancellation_and_failure_return_no_result_or_successor_channel() {
        let before_trace = Arc::new(Mutex::new(Vec::new()));
        let before_control = ExecutionControl::without_deadline();
        before_control.request_cancellation();
        let before = batch(
            StatefulDecoder::new(before_trace.clone(), DecoderAction::Complete, plans(3)),
            BatchChannelPolicy::separate_channels(),
            PadPolicy::Exact,
            before_control,
        )
        .transcribe(&three_channels());
        assert!(before.is_err());
        assert_eq!(calls(&before_trace), Vec::<usize>::new());

        let cancellation_trace = Arc::new(Mutex::new(Vec::new()));
        let cancellation_control = ExecutionControl::without_deadline();
        let cancelled = batch(
            StatefulDecoder::new(
                cancellation_trace.clone(),
                DecoderAction::CancelAfterFirst(cancellation_control.clone()),
                plans(3),
            ),
            BatchChannelPolicy::separate_channels(),
            PadPolicy::Exact,
            cancellation_control,
        )
        .transcribe(&three_channels());
        assert!(cancelled.is_err());
        assert_eq!(calls(&cancellation_trace), vec![0]);

        let failure_trace = Arc::new(Mutex::new(Vec::new()));
        let failed = batch(
            StatefulDecoder::new(
                failure_trace.clone(),
                DecoderAction::FailOnCall(1),
                plans(3),
            ),
            BatchChannelPolicy::separate_channels(),
            PadPolicy::Exact,
            ExecutionControl::without_deadline(),
        )
        .transcribe(&three_channels());
        assert!(failed.is_err());
        assert_eq!(calls(&failure_trace), vec![0, 1]);
    }

    #[test]
    fn each_typed_channel_policy_preserves_the_same_application_result_for_same_audio() {
        let input = three_channels();
        for (policy, words) in [
            (BatchChannelPolicy::single_output(), plans(1)),
            (BatchChannelPolicy::separate_channels(), plans(3)),
        ] {
            let first_trace = Arc::new(Mutex::new(Vec::new()));
            let first = batch(
                StatefulDecoder::new(first_trace.clone(), DecoderAction::Complete, words.clone()),
                policy,
                PadPolicy::Exact,
                ExecutionControl::without_deadline(),
            )
            .transcribe(&input)
            .expect("the first deterministic policy execution must succeed");
            let second_trace = Arc::new(Mutex::new(Vec::new()));
            let second = batch(
                StatefulDecoder::new(second_trace.clone(), DecoderAction::Complete, words),
                policy,
                PadPolicy::Exact,
                ExecutionControl::without_deadline(),
            )
            .transcribe(&input)
            .expect("the second deterministic policy execution must succeed");

            assert_eq!(first.source_channels(), second.source_channels());
            assert_eq!(first.duplicate(), second.duplicate());
            assert_eq!(first.channels(), second.channels());
            assert_eq!(first.segments(), second.segments());
            assert_eq!(first.turns(), second.turns());
            assert_eq!(calls(&first_trace), calls(&second_trace));
        }
    }

    #[test]
    fn mismatched_input_rate_refuses_before_any_decoder_call() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let error = batch(
            StatefulDecoder::new(trace.clone(), DecoderAction::Complete, plans(2)),
            BatchChannelPolicy::separate_channels(),
            PadPolicy::Exact,
            ExecutionControl::without_deadline(),
        )
        .transcribe(&audio_at_sample_rate(
            8,
            vec![channel(vec![0.25; 48]), channel(vec![-0.25; 48])],
        ))
        .expect_err("a batch input at a rate different from the frontend must be refused");

        match error {
            MultiChannelBatchError::Input(message) => {
                assert_eq!(
                    message,
                    "batch input sample rate 8 Hz does not match frontend sample rate 16 Hz"
                );
            }
            MultiChannelBatchError::Selection(message) => {
                panic!("sample-rate refusal must precede channel selection: {message}");
            }
            MultiChannelBatchError::Transcription(message) => {
                panic!("sample-rate refusal must precede transcription: {message}");
            }
        }
        assert_eq!(calls(&trace), Vec::<usize>::new());
    }

    #[test]
    fn cancellation_precedes_a_simultaneous_input_rate_mismatch() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let control = ExecutionControl::without_deadline();
        control.request_cancellation();
        let error = batch(
            StatefulDecoder::new(trace.clone(), DecoderAction::Complete, plans(2)),
            BatchChannelPolicy::separate_channels(),
            PadPolicy::Exact,
            control,
        )
        .transcribe(&audio_at_sample_rate(
            8,
            vec![channel(vec![0.25; 48]), channel(vec![-0.25; 48])],
        ))
        .expect_err("an already-cancelled request must refuse before validating its input rate");

        match error {
            MultiChannelBatchError::Transcription(message) => {
                assert_eq!(message, "execution cancelled");
            }
            MultiChannelBatchError::Input(message) => {
                panic!("cancellation must precede input-rate validation: {message}");
            }
            MultiChannelBatchError::Selection(message) => {
                panic!("cancellation must precede channel selection: {message}");
            }
        }
        assert_eq!(calls(&trace), Vec::<usize>::new());
    }
}
