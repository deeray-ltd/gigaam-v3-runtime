// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Validated audio ingestion, resampling, channel analysis, and log-mel frontend processing.
//!
//! This crate owns audio representations from untrusted encoded bytes through model-ready feature
//! matrices. It deliberately does not read process configuration, resolve model-package paths, or
//! depend on recognition, transcription, service, CLI, provider, or ONNX Runtime types.

mod channel_analysis;
mod contracts;
mod decoded_input;
mod fft;
mod formats;
mod frontend;
mod g711;
mod live_input;
mod replay;
mod resampling;

pub use channel_analysis::channel_correlation;
pub use contracts::{
    ChannelAudio, ChannelAudioView, ChannelCount, DecodedAudio, EncodedAudio, FeatureMatrix,
    FeatureMatrixView, FrontendMode, SampleFormat, SampleRate,
};
pub use decoded_input::{
    AudioFile, AudioMetadata, DecodedChunk, FileChunkReader, FileDecoder, collect_audio,
};
pub use formats::{FileDecoderProvider, FileDecoders, FileFormatMatch};
pub use frontend::{FrontendProcessor, FrontendScratch};
pub use g711::{alaw_to_i16, ulaw_to_i16};
pub use live_input::{
    LiveChunkReader, LiveDecoder, LiveDecoderProvider, LiveDecoders, SelectedLiveDecoder,
};
pub use replay::{ReplayAudio, ReplayError};
pub use resampling::{RatePair, Resampler, ResamplerConfig, StreamResampler, resample_audio};

#[cfg(test)]
mod tests;
