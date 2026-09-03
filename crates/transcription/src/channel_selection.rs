// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Typed channel-selection policies over validated borrowed audio.

use gigaam_audio::{ChannelAudioView, channel_correlation};
use std::num::NonZeroUsize;

const FIXED_DUPLICATE_THRESHOLD: CorrelationThreshold = CorrelationThreshold(0.98);

/// A validated positive count of source channels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceChannelCount(NonZeroUsize);

impl SourceChannelCount {
    /// Validates the number of original channels represented by a selection input.
    pub fn new(value: usize) -> Result<Self, String> {
        let count = NonZeroUsize::new(value)
            .ok_or_else(|| "channel selection requires at least one source channel".to_owned())?;
        Ok(Self(count))
    }

    /// Returns the validated number of original channels.
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

/// The stable identity of one source channel before any selection occurs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct OriginalChannel(usize);

impl OriginalChannel {
    /// Validates an original-channel identity against its source channel count.
    pub(crate) fn new(index: usize, source_channels: SourceChannelCount) -> Result<Self, String> {
        let channel = Self(index);
        channel.validate_against(source_channels)?;
        Ok(channel)
    }

    /// Refuses an identity that does not belong to this selection input.
    pub(crate) fn validate_against(
        self,
        source_channels: SourceChannelCount,
    ) -> Result<(), String> {
        if self.index() >= source_channels.get() {
            return Err(format!(
                "channel index {} is outside source channel count {}",
                self.index(),
                source_channels.get()
            ));
        }
        Ok(())
    }

    /// Returns the original, never-renumbered channel index.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// The output shape requested from batch transcription.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchProjection {
    /// Emit one transcript through the single-output batch projection.
    SingleOutput,
    /// Preserve separate source-channel transcripts for split or turn output.
    SeparateChannels,
}

/// A typed batch channel-selection policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchChannelPolicy {
    projection: BatchProjection,
}

impl BatchChannelPolicy {
    /// Selects the single-output batch projection.
    pub const fn single_output() -> Self {
        Self {
            projection: BatchProjection::SingleOutput,
        }
    }

    /// Selects the separate-channel batch projection used by split and turn output.
    pub const fn separate_channels() -> Self {
        Self {
            projection: BatchProjection::SeparateChannels,
        }
    }
}

/// The duplicate result reported by batch channel selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DuplicateClassification {
    /// Exactly two source channels exceeded the fixed duplicate threshold.
    DualMono,
    /// The source is not classified as dual mono.
    NotDualMono,
}

/// An immutable, validated correlation threshold in the closed unit interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrelationThreshold(f64);

impl CorrelationThreshold {
    /// Validates a finite duplicate threshold in the closed unit interval.
    pub fn new(value: f64) -> Result<Self, String> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err("channel correlation threshold must be finite and in [0, 1]".into());
        }
        Ok(Self(value))
    }
}

/// A validated nonzero model-rate sample count used as streaming selection evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectionWindowSamples(NonZeroUsize);

impl SelectionWindowSamples {
    /// Validates the exact number of post-resample samples admitted to correlation.
    pub fn new(value: usize) -> Result<Self, String> {
        let samples = NonZeroUsize::new(value).ok_or_else(|| {
            "streaming selection window must contain at least one sample".to_owned()
        })?;
        Ok(Self(samples))
    }

    /// Returns the exact model-rate sample count admitted to correlation.
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

/// A typed pairwise duplicate-selection policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PairwiseChannelPolicy {
    /// Retain every source channel without evaluating correlation.
    Disabled,
    /// Reject a channel only when its correlation exceeds this threshold against an earlier survivor.
    Deduplicate(CorrelationThreshold),
}

impl PairwiseChannelPolicy {
    /// Returns the fixed policy used by the offline dialogue command.
    pub const fn dialog_deduplication() -> Self {
        Self::Deduplicate(FIXED_DUPLICATE_THRESHOLD)
    }
}

/// A nonempty ordered subset of original source channels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelSelection {
    channels: Vec<OriginalChannel>,
}

impl ChannelSelection {
    fn first(source_channels: SourceChannelCount) -> Result<Self, String> {
        Self::new(
            source_channels,
            vec![OriginalChannel::new(0, source_channels)?],
        )
    }

    pub(crate) fn all(source_channels: SourceChannelCount) -> Result<Self, String> {
        let mut channels = Vec::with_capacity(source_channels.get());
        for index in 0..source_channels.get() {
            channels.push(OriginalChannel::new(index, source_channels)?);
        }
        Self::new(source_channels, channels)
    }

    fn new(
        source_channels: SourceChannelCount,
        channels: Vec<OriginalChannel>,
    ) -> Result<Self, String> {
        let selection = Self { channels };
        selection.validate_against(source_channels)?;
        Ok(selection)
    }

    /// Checks that this retained identity set belongs to the stated source and stays ordered.
    pub(crate) fn validate_against(
        &self,
        source_channels: SourceChannelCount,
    ) -> Result<(), String> {
        if self.channels.is_empty() {
            return Err("channel selection must retain at least one source channel".into());
        }
        let mut previous = None;
        for channel in &self.channels {
            channel.validate_against(source_channels).map_err(|_| {
                format!(
                    "selected channel {} is outside source channel count {}",
                    channel.index(),
                    source_channels.get()
                )
            })?;
            if let Some(previous_index) = previous
                && channel.index() <= previous_index
            {
                return Err("selected channel identities must be strictly ascending".into());
            }
            previous = Some(channel.index());
        }
        Ok(())
    }

    /// Returns retained source channels in ascending original identity order.
    pub fn channels(&self) -> &[OriginalChannel] {
        &self.channels
    }
}

/// Batch selection together with its typed dual-mono classification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchChannelSelection {
    source_channels: SourceChannelCount,
    selection: ChannelSelection,
    duplicate: DuplicateClassification,
}

impl BatchChannelSelection {
    /// Returns the validated count of original channels before batch selection.
    pub const fn source_channels(&self) -> SourceChannelCount {
        self.source_channels
    }

    /// Returns the retained original source channels.
    pub fn selection(&self) -> &ChannelSelection {
        &self.selection
    }

    /// Returns the fixed-rule duplicate classification for this batch source.
    pub const fn duplicate(&self) -> DuplicateClassification {
        self.duplicate
    }
}

/// Applies the exact batch channel-selection policy to validated borrowed audio.
pub fn select_batch_channels(
    channels: &[ChannelAudioView<'_>],
    policy: BatchChannelPolicy,
) -> Result<BatchChannelSelection, String> {
    let source_channels = SourceChannelCount::new(channels.len())?;
    select_batch_by(source_channels, policy, |first, second| {
        channel_correlation(channels[first.index()], channels[second.index()])
    })
}

fn select_batch_by(
    source_channels: SourceChannelCount,
    policy: BatchChannelPolicy,
    correlation: impl FnOnce(OriginalChannel, OriginalChannel) -> Result<f64, String>,
) -> Result<BatchChannelSelection, String> {
    let duplicate = match source_channels.get() {
        2 => {
            let first = OriginalChannel::new(0, source_channels)?;
            let second = OriginalChannel::new(1, source_channels)?;
            if correlation(first, second)? > FIXED_DUPLICATE_THRESHOLD.0 {
                DuplicateClassification::DualMono
            } else {
                DuplicateClassification::NotDualMono
            }
        }
        _ => DuplicateClassification::NotDualMono,
    };
    let selection = match (duplicate, policy.projection) {
        (DuplicateClassification::DualMono, BatchProjection::SingleOutput)
        | (DuplicateClassification::DualMono, BatchProjection::SeparateChannels)
        | (DuplicateClassification::NotDualMono, BatchProjection::SingleOutput) => {
            ChannelSelection::first(source_channels)?
        }
        (DuplicateClassification::NotDualMono, BatchProjection::SeparateChannels) => {
            ChannelSelection::all(source_channels)?
        }
    };
    Ok(BatchChannelSelection {
        source_channels,
        selection,
        duplicate,
    })
}

/// Applies the exact pairwise duplicate-selection policy to validated borrowed audio.
pub fn select_pairwise_channels(
    channels: &[ChannelAudioView<'_>],
    policy: PairwiseChannelPolicy,
) -> Result<ChannelSelection, String> {
    let source_channels = SourceChannelCount::new(channels.len())?;
    if channels.iter().copied().all(ChannelAudioView::is_empty) {
        // A stream can complete before its post-resample analysis window contains a sample.
        // Without correlation evidence, every original channel remains selected.
        return ChannelSelection::all(source_channels);
    }
    select_pairwise_by(source_channels, policy, |candidate, retained| {
        channel_correlation(channels[candidate.index()], channels[retained.index()])
    })
}

fn select_pairwise_by(
    source_channels: SourceChannelCount,
    policy: PairwiseChannelPolicy,
    mut correlation: impl FnMut(OriginalChannel, OriginalChannel) -> Result<f64, String>,
) -> Result<ChannelSelection, String> {
    let PairwiseChannelPolicy::Deduplicate(threshold) = policy else {
        return ChannelSelection::all(source_channels);
    };
    let mut retained = Vec::with_capacity(source_channels.get());
    for index in 0..source_channels.get() {
        let candidate = OriginalChannel::new(index, source_channels)?;
        let mut duplicate = false;
        for earlier in &retained {
            if correlation(candidate, *earlier)? > threshold.0 {
                duplicate = true;
                break;
            }
        }
        if !duplicate {
            retained.push(candidate);
        }
    }
    ChannelSelection::new(source_channels, retained)
}

#[cfg(test)]
mod tests {
    use super::{
        BatchChannelPolicy, CorrelationThreshold, DuplicateClassification, OriginalChannel,
        PairwiseChannelPolicy, SelectionWindowSamples, SourceChannelCount, select_batch_by,
        select_batch_channels, select_pairwise_by, select_pairwise_channels,
    };
    use gigaam_audio::ChannelAudio;

    fn indices(selection: &super::ChannelSelection) -> Vec<usize> {
        selection
            .channels()
            .iter()
            .map(|channel| channel.index())
            .collect()
    }

    #[test]
    fn selection_refuses_an_empty_source_channel_set() {
        assert!(select_batch_channels(&[], BatchChannelPolicy::single_output()).is_err());
        assert!(select_pairwise_channels(&[], PairwiseChannelPolicy::Disabled).is_err());
    }

    #[test]
    fn validated_threshold_and_channel_identity_refuse_invalid_values() -> Result<(), String> {
        for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
            assert!(CorrelationThreshold::new(threshold).is_err());
        }
        let source = SourceChannelCount::new(2)?;
        assert!(OriginalChannel::new(2, source).is_err());
        Ok(())
    }

    #[test]
    fn selection_window_requires_a_nonzero_model_rate_sample_count() -> Result<(), String> {
        assert!(SelectionWindowSamples::new(0).is_err());
        let window = SelectionWindowSamples::new(480)?;
        assert_eq!(window.get(), 480);
        Ok(())
    }

    #[test]
    fn pairwise_threshold_equality_retains_the_candidate() -> Result<(), String> {
        let source = SourceChannelCount::new(2)?;
        let selection = select_pairwise_by(
            source,
            PairwiseChannelPolicy::Deduplicate(CorrelationThreshold::new(0.98)?),
            |candidate, earlier| match (candidate.index(), earlier.index()) {
                (1, 0) => Ok(0.98),
                _ => Err("selection queried an unexpected channel pair".into()),
            },
        )?;
        assert_eq!(indices(&selection), vec![0, 1]);
        Ok(())
    }

    #[test]
    fn pairwise_strict_exceedance_rejects_the_candidate() -> Result<(), String> {
        let source = SourceChannelCount::new(2)?;
        let selection = select_pairwise_by(
            source,
            PairwiseChannelPolicy::Deduplicate(CorrelationThreshold::new(0.98)?),
            |candidate, earlier| match (candidate.index(), earlier.index()) {
                (1, 0) => Ok(0.980_000_000_1),
                _ => Err("selection queried an unexpected channel pair".into()),
            },
        )?;
        assert_eq!(indices(&selection), vec![0]);
        Ok(())
    }

    #[test]
    fn batch_exact_two_dual_mono_retains_original_channel_zero() -> Result<(), String> {
        let left = ChannelAudio::new(vec![0.0, 0.5, -0.5, 0.25])?;
        let right = ChannelAudio::new(vec![0.0, 0.5, -0.5, 0.25])?;
        let channels = [left.view(), right.view()];
        let selection = select_batch_channels(&channels, BatchChannelPolicy::separate_channels())?;
        assert_eq!(selection.duplicate(), DuplicateClassification::DualMono);
        assert_eq!(indices(selection.selection()), vec![0]);
        Ok(())
    }

    #[test]
    fn batch_fixed_threshold_retains_equality_and_suppresses_strict_exceedance()
    -> Result<(), String> {
        let source = SourceChannelCount::new(2)?;
        let retained = select_batch_by(source, BatchChannelPolicy::separate_channels(), |_, _| {
            Ok(0.98)
        })?;
        let suppressed =
            select_batch_by(source, BatchChannelPolicy::separate_channels(), |_, _| {
                Ok(0.980_000_000_1)
            })?;

        assert_eq!(retained.duplicate(), DuplicateClassification::NotDualMono);
        assert_eq!(indices(retained.selection()), vec![0, 1]);
        assert_eq!(suppressed.duplicate(), DuplicateClassification::DualMono);
        assert_eq!(indices(suppressed.selection()), vec![0]);
        Ok(())
    }

    #[test]
    fn batch_single_output_retains_original_channel_zero_without_dual_mono() -> Result<(), String> {
        let left = ChannelAudio::new(vec![0.0, 0.5, -0.5, 0.25])?;
        let right = ChannelAudio::new(vec![0.0, -0.5, 0.5, -0.25])?;
        let channels = [left.view(), right.view()];
        let selection = select_batch_channels(&channels, BatchChannelPolicy::single_output())?;
        assert_eq!(selection.duplicate(), DuplicateClassification::NotDualMono);
        assert_eq!(indices(selection.selection()), vec![0]);
        Ok(())
    }

    #[test]
    fn batch_more_than_two_channels_never_evaluates_pairwise_correlation() -> Result<(), String> {
        let first = ChannelAudio::new(vec![0.0])?;
        let second = ChannelAudio::new(vec![0.0, 0.5])?;
        let third = ChannelAudio::new(vec![0.0, 0.5, -0.5])?;
        let channels = [first.view(), second.view(), third.view()];
        let selection = select_batch_channels(&channels, BatchChannelPolicy::separate_channels())?;
        assert_eq!(selection.duplicate(), DuplicateClassification::NotDualMono);
        assert_eq!(indices(selection.selection()), vec![0, 1, 2]);
        Ok(())
    }

    #[test]
    fn pairwise_selection_is_greedy_and_preserves_the_lowest_original_identity()
    -> Result<(), String> {
        let source = SourceChannelCount::new(4)?;
        let selection = select_pairwise_by(
            source,
            PairwiseChannelPolicy::Deduplicate(CorrelationThreshold::new(0.98)?),
            |candidate, earlier| match (candidate.index(), earlier.index()) {
                (1, 0) => Ok(0.50),
                (2, 0) => Ok(0.50),
                (2, 1) => Ok(0.99),
                (3, 0) => Ok(0.99),
                _ => Err("selection queried an unexpected channel pair".into()),
            },
        )?;
        assert_eq!(indices(&selection), vec![0, 1]);
        Ok(())
    }

    #[test]
    fn disabled_pairwise_selection_preserves_every_identity_and_order() -> Result<(), String> {
        let first = ChannelAudio::new(vec![0.0])?;
        let second = ChannelAudio::new(vec![0.0, 0.5])?;
        let third = ChannelAudio::new(vec![0.0, 0.5, -0.5])?;
        let channels = [first.view(), second.view(), third.view()];
        let selection = select_pairwise_channels(&channels, PairwiseChannelPolicy::Disabled)?;
        assert_eq!(indices(&selection), vec![0, 1, 2]);
        Ok(())
    }

    #[test]
    fn pairwise_selection_without_analysis_samples_preserves_every_identity() -> Result<(), String>
    {
        let first = ChannelAudio::new(Vec::new())?;
        let second = ChannelAudio::new(Vec::new())?;
        let channels = [first.view(), second.view()];
        let policy = PairwiseChannelPolicy::Deduplicate(CorrelationThreshold::new(0.98)?);
        let selection = select_pairwise_channels(&channels, policy)?;
        assert_eq!(indices(&selection), vec![0, 1]);
        Ok(())
    }

    #[test]
    fn configured_pairwise_threshold_changes_the_observable_selection() -> Result<(), String> {
        let source = SourceChannelCount::new(2)?;
        let retained = select_pairwise_by(
            source,
            PairwiseChannelPolicy::Deduplicate(CorrelationThreshold::new(0.99)?),
            |candidate, earlier| match (candidate.index(), earlier.index()) {
                (1, 0) => Ok(0.985),
                _ => Err("selection queried an unexpected channel pair".into()),
            },
        )?;
        let rejected = select_pairwise_by(
            source,
            PairwiseChannelPolicy::Deduplicate(CorrelationThreshold::new(0.98)?),
            |candidate, earlier| match (candidate.index(), earlier.index()) {
                (1, 0) => Ok(0.985),
                _ => Err("selection queried an unexpected channel pair".into()),
            },
        )?;
        assert_eq!(indices(&retained), vec![0, 1]);
        assert_eq!(indices(&rejected), vec![0]);
        Ok(())
    }

    #[test]
    fn pairwise_correlation_domain_failures_are_explicit() -> Result<(), String> {
        let short = ChannelAudio::new(vec![0.0])?;
        let long = ChannelAudio::new(vec![0.0, 0.5])?;
        let channels = [short.view(), long.view()];
        let policy = PairwiseChannelPolicy::Deduplicate(CorrelationThreshold::new(0.98)?);
        assert!(select_pairwise_channels(&channels, policy).is_err());
        Ok(())
    }
}
