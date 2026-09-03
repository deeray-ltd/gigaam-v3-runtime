// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Batch and streaming transcription application contracts.

mod batch;
mod channel_selection;
mod contracts;
mod dialog;
mod endpoint;
mod multi_batch;
mod multi_stream;
mod observations;
mod padding;
mod stitch;
mod stream;
mod text;
mod turns;
mod vad_policy;

#[cfg(test)]
mod test_support;

pub use batch::{BatchSetup, BatchTranscriber};
pub use channel_selection::{
    BatchChannelPolicy, BatchChannelSelection, ChannelSelection, CorrelationThreshold,
    DuplicateClassification, OriginalChannel, PairwiseChannelPolicy, SelectionWindowSamples,
    SourceChannelCount, select_batch_channels, select_pairwise_channels,
};
pub use contracts::{
    BackchannelMark, BatchConfig, ChannelSnapshot, ChannelWord, DialogTurn, EndpointSource,
    FinalReason, PadPolicy, SnapshotWord, StageTimings, StreamConfig, StreamEmissionMode,
    StreamEvent, StreamLockPolicy, StreamTimingChanges, StreamWord, StreamingChannelPolicy,
    Transcript, TranscriptWord, Turn, TurnsPatch, VadProbability, WordFinality, WordStability,
};
pub use dialog::{BackchannelDuration, BackchannelPolicy, DialogMerger};
pub use endpoint::{Endpoint, EndpointRules, SpeechState, detect, segments, trailing_silence};
pub use multi_batch::{
    MultiChannelBatchError, MultiChannelBatchOptions, MultiChannelBatchResult,
    MultiChannelBatchSetup, MultiChannelBatchTranscriber,
};
pub use multi_stream::{
    ChannelSelectionObservation, ChannelStreamEvent, MultiChannelEmissionGroup,
    MultiChannelFailure, MultiChannelSession, MultiChannelStep, MultiChannelStreamOptions,
    MultiChannelStreamSetup, MultiChannelStreamSetupInput, OfflineChannelPolicy,
    OfflineDialogFailure, OfflineDialogFailureOrigin, OfflineDialogResult, OfflineDialogSetup,
    OfflineDialogSetupInput, StreamChannelFactory, TranscriptionObservation,
    transcribe_offline_dialog,
};
pub use observations::{ObservationMode, WindowTiming, WindowTimingObserver};
pub use stitch::{
    Seam, Window, WindowWords, seam, stitch_aligned, window_shapes, windows, windows_bucketed,
    words_to_text,
};
pub use stream::{EndpointDetector, StreamSession, StreamSetup};
pub use text::{normalize_word, word_count, word_edits};
pub use turns::{ChannelTranscript, TurnGap, dialog_text, merge, turns};
pub use vad_policy::{
    SpeechSegment, VadDurations, VadPadding, VadPolicyConfig, VadThresholds, speech_segments,
};
