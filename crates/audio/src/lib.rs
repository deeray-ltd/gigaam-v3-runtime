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
mod codecs;
mod contracts;
mod dispatch;
mod fft;
mod frontend;
mod g711;
mod interleaved;
mod resampling;
mod wav;

pub use channel_analysis::channel_correlation;
pub use codecs::decode_bytes;
pub use contracts::{
    ChannelAudio, ChannelAudioView, ChannelCount, DecodedAudio, EncodedAudio, FeatureMatrix,
    FeatureMatrixView, FrontendMode, SampleFormat, SampleRate,
};
pub use dispatch::{load, load_bytes, read_wav};
pub use frontend::{FrontendProcessor, FrontendScratch};
pub use interleaved::{InterleavedFrameDecoder, decode_samples};
pub use resampling::{RatePair, Resampler, ResamplerConfig, StreamResampler, resample_audio};
pub use wav::parse_wav;

#[cfg(test)]
mod tests;
