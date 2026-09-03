// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

use gigaam_primitives::{u32_to_f32, u32_to_usize, usize_to_f32};

/// A validated nonzero sample rate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct SampleRate(u32);

impl SampleRate {
    pub fn new(hertz: u32) -> Result<Self, String> {
        if hertz == 0 {
            return Err("sample rate must be nonzero".into());
        }
        Ok(Self(hertz))
    }

    pub fn from_usize(value: usize, context: &str) -> Result<Self, String> {
        let hertz =
            u32::try_from(value).map_err(|_| format!("{context} sample rate exceeds u32"))?;
        Self::new(hertz).map_err(|error| format!("{context}: {error}"))
    }

    pub const fn hertz(self) -> u32 {
        self.0
    }

    pub fn as_usize(self) -> Result<usize, String> {
        u32_to_usize(self.0).map_err(|error| format!("sample rate: {error}"))
    }
}

/// A validated positive number of interleaved audio channels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelCount(usize);

impl ChannelCount {
    pub fn new(value: usize) -> Result<Self, String> {
        if value == 0 {
            return Err("channel count must be nonzero".into());
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

/// One channel of finite internal samples. The sample vector remains private so callers cannot
/// bypass validation by mutating audio after construction.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelAudio {
    samples: Vec<f32>,
}

impl ChannelAudio {
    /// Constructs an internal waveform. Resampling output is allowed to overshoot normalized
    /// external bounds because finite FIR ringing is valid, but non-finite values always refuse.
    pub fn new(samples: Vec<f32>) -> Result<Self, String> {
        validate_internal_samples(&samples)?;
        Ok(Self { samples })
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Borrows this validated waveform without exposing a mutable sample vector.
    pub fn view(&self) -> ChannelAudioView<'_> {
        // `ChannelAudio::new` validates the private buffer and no public operation can mutate it.
        ChannelAudioView {
            samples: &self.samples,
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }
}

/// A validated immutable borrowed waveform for a recognition operation.
///
/// The view does not own or retain audio. Callers that construct it from a raw slice validate
/// finite internal samples at the Audio boundary before passing it to another capability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelAudioView<'a> {
    samples: &'a [f32],
}

impl<'a> ChannelAudioView<'a> {
    pub fn new(samples: &'a [f32]) -> Result<Self, String> {
        validate_internal_samples(samples)?;
        Ok(Self { samples })
    }

    pub const fn samples(self) -> &'a [f32] {
        self.samples
    }

    pub const fn len(self) -> usize {
        self.samples.len()
    }

    pub const fn is_empty(self) -> bool {
        self.samples.is_empty()
    }
}

fn validate_internal_samples(samples: &[f32]) -> Result<(), String> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("audio samples must be finite".into());
    }
    Ok(())
}

/// A complete decoded multichannel input with one rate and equal-length nonempty channels.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedAudio {
    sample_rate: SampleRate,
    channels: Vec<ChannelAudio>,
}

/// Validated encoded input accepted by the codec-dispatch boundary.
///
/// A format hint is optional because container signatures can identify many inputs. When present,
/// it is a normalized nonempty ASCII extension rather than an arbitrary free-form selector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedAudio {
    bytes: Vec<u8>,
    format_hint: Option<String>,
}

impl EncodedAudio {
    pub fn new(bytes: Vec<u8>, format_hint: Option<String>) -> Result<Self, String> {
        if bytes.is_empty() {
            return Err("encoded audio bytes must not be empty".into());
        }
        let format_hint = match format_hint {
            Some(hint) => {
                if hint.is_empty() {
                    return Err("encoded audio format hint must not be empty when present".into());
                }
                if !hint.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
                    return Err(
                        "encoded audio format hint must contain only ASCII letters and digits"
                            .into(),
                    );
                }
                Some(hint.to_ascii_lowercase())
            }
            None => None,
        };
        Ok(Self { bytes, format_hint })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn format_hint(&self) -> Option<&str> {
        self.format_hint.as_deref()
    }

    pub fn into_parts(self) -> (Vec<u8>, Option<String>) {
        (self.bytes, self.format_hint)
    }
}

impl DecodedAudio {
    pub fn new(sample_rate: SampleRate, channels: Vec<ChannelAudio>) -> Result<Self, String> {
        ChannelCount::new(channels.len())?;
        let frames = channels
            .first()
            .map(ChannelAudio::len)
            .ok_or_else(|| "decoded audio must contain channels".to_owned())?;
        if frames == 0 {
            return Err("decoded audio channels must not be empty".into());
        }
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("decoded audio channels must have equal frame counts".into());
        }
        Ok(Self {
            sample_rate,
            channels,
        })
    }

    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    pub fn channels(&self) -> &[ChannelAudio] {
        &self.channels
    }

    pub fn channel_count(&self) -> ChannelCount {
        // The constructor validates nonempty channels and this type cannot be mutated externally.
        ChannelCount(self.channels.len())
    }

    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, ChannelAudio::len)
    }

    pub fn duration_seconds(&self) -> f32 {
        usize_to_f32(self.frames()) / u32_to_f32(self.sample_rate.hertz())
    }

    pub fn into_parts(self) -> (SampleRate, Vec<ChannelAudio>) {
        (self.sample_rate, self.channels)
    }
}

/// Exact wire sample representation accepted by the raw interleaved decoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleFormat {
    Pcm16,
    F32,
    Alaw,
    Ulaw,
}

impl SampleFormat {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "pcm16" => Ok(Self::Pcm16),
            "f32" => Ok(Self::F32),
            "alaw" => Ok(Self::Alaw),
            "ulaw" => Ok(Self::Ulaw),
            _ => Err(format!(
                "sample format must be pcm16|f32|alaw|ulaw, got {value:?}"
            )),
        }
    }

    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Pcm16 => 2,
            Self::F32 => 4,
            Self::Alaw | Self::Ulaw => 1,
        }
    }
}

/// A validated `[mel][time]` model feature matrix. Zero frames are valid for clips shorter than an
/// uncentered FFT window; mel dimensions and values remain validated.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureMatrix {
    mel_bins: usize,
    frames: usize,
    values: Vec<f32>,
}

impl FeatureMatrix {
    pub(crate) fn new(mel_bins: usize, frames: usize, values: Vec<f32>) -> Result<Self, String> {
        validate_feature_matrix(mel_bins, frames, &values)?;
        Ok(Self {
            mel_bins,
            frames,
            values,
        })
    }

    /// Constructs a validated model feature matrix from an explicit `[mel][time]` value buffer.
    ///
    /// This is the narrow cross-capability constructor used when a caller derives one validated
    /// frame range from an existing Audio feature matrix. It preserves Audio's dimensional and
    /// finite-value boundary instead of requiring Recognition to accept unrelated scalar shapes.
    pub fn from_values(mel_bins: usize, frames: usize, values: Vec<f32>) -> Result<Self, String> {
        Self::new(mel_bins, frames, values)
    }

    pub fn mel_bins(&self) -> usize {
        self.mel_bins
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Borrows this validated `[mel][time]` matrix for one recognition operation.
    pub fn view(&self) -> FeatureMatrixView<'_> {
        // `FeatureMatrix::new` validates the private dimensions and buffer, which cannot be
        // mutated through the public API.
        FeatureMatrixView {
            mel_bins: self.mel_bins,
            frames: self.frames,
            values: &self.values,
        }
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }

    /// Returns an owned contiguous `[mel][time]` range. The bounds are frame indices and must
    /// describe one non-reversed range inside this matrix.
    pub fn frame_range(&self, start: usize, end: usize) -> Result<Self, String> {
        if start > end || end > self.frames {
            return Err(format!(
                "feature matrix frame range {start}..{end} is outside 0..{}",
                self.frames
            ));
        }
        let frames = end
            .checked_sub(start)
            .ok_or_else(|| "feature matrix frame range underflows".to_owned())?;
        let mut values = Vec::new();
        let capacity = self
            .mel_bins
            .checked_mul(frames)
            .ok_or_else(|| "feature matrix frame range dimensions overflow usize".to_owned())?;
        values
            .try_reserve_exact(capacity)
            .map_err(|_| "feature matrix frame range cannot reserve memory".to_owned())?;
        for mel_bin in 0..self.mel_bins {
            let row_start = mel_bin
                .checked_mul(self.frames)
                .and_then(|offset| offset.checked_add(start))
                .ok_or_else(|| "feature matrix frame range start overflows usize".to_owned())?;
            let row_end = row_start
                .checked_add(frames)
                .ok_or_else(|| "feature matrix frame range end overflows usize".to_owned())?;
            let row = self.values.get(row_start..row_end).ok_or_else(|| {
                "feature matrix values do not match their validated dimensions".to_owned()
            })?;
            values.extend_from_slice(row);
        }
        Self::new(self.mel_bins, frames, values)
    }
}

/// A validated immutable borrowed `[mel][time]` matrix for a recognition operation.
///
/// The view carries dimensions with its slice so a downstream port cannot accept an untyped
/// sample buffer. It intentionally has no mutable or retained access to the source matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureMatrixView<'a> {
    mel_bins: usize,
    frames: usize,
    values: &'a [f32],
}

impl<'a> FeatureMatrixView<'a> {
    pub fn new(mel_bins: usize, frames: usize, values: &'a [f32]) -> Result<Self, String> {
        validate_feature_matrix(mel_bins, frames, values)?;
        Ok(Self {
            mel_bins,
            frames,
            values,
        })
    }

    pub const fn mel_bins(self) -> usize {
        self.mel_bins
    }

    pub const fn frames(self) -> usize {
        self.frames
    }

    pub const fn values(self) -> &'a [f32] {
        self.values
    }

    /// Materializes an independently owned matrix for a queue whose work outlives this borrow.
    pub fn to_owned(self) -> FeatureMatrix {
        // Both constructors validate the dimensions and values, and the view fields remain
        // private. Cloning the validated slice therefore preserves the FeatureMatrix invariant.
        FeatureMatrix {
            mel_bins: self.mel_bins,
            frames: self.frames,
            values: self.values.to_vec(),
        }
    }
}

fn validate_feature_matrix(mel_bins: usize, frames: usize, values: &[f32]) -> Result<(), String> {
    if mel_bins == 0 {
        return Err("feature matrix mel dimension must be nonzero".into());
    }
    let expected = mel_bins
        .checked_mul(frames)
        .ok_or_else(|| "feature matrix dimensions overflow usize".to_owned())?;
    if values.len() != expected {
        return Err(format!(
            "feature matrix has {} values, expected {expected}",
            values.len()
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("feature matrix values must be finite".into());
    }
    Ok(())
}

/// Exact frontend execution mode chosen by a process/configuration boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontendMode {
    Scalar,
    Batched,
}

impl FrontendMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "scalar" => Ok(Self::Scalar),
            "batched" => Ok(Self::Batched),
            _ => Err(format!(
                "frontend mode must be scalar|batched, got {value:?}"
            )),
        }
    }
}
