// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

use crate::assets::{RelativeAsset, ValidatedArtifact, read_vocabulary};
use crate::error::PackageError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchemaVersion(u8);

impl SchemaVersion {
    pub(crate) const V1: Self = Self(1);

    pub fn value(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderPrecision {
    Fp32,
    Fp16Io32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputLayout {
    TimeThenDimension,
    DimensionThenTime,
}

#[derive(Clone, Debug)]
pub struct EncoderArtifact {
    artifact: ValidatedArtifact,
    tensor_contract: EncoderTensorContract,
    precision: EncoderPrecision,
    output_dimension: usize,
    output_layout: OutputLayout,
}

impl EncoderArtifact {
    pub(crate) fn new(
        artifact: ValidatedArtifact,
        tensor_contract: EncoderTensorContract,
        precision: EncoderPrecision,
        output_dimension: usize,
        output_layout: OutputLayout,
    ) -> Self {
        Self {
            artifact,
            tensor_contract,
            precision,
            output_dimension,
            output_layout,
        }
    }

    pub fn artifact(&self) -> &ValidatedArtifact {
        &self.artifact
    }

    pub fn tensor_contract(&self) -> &EncoderTensorContract {
        &self.tensor_contract
    }

    pub fn precision(&self) -> EncoderPrecision {
        self.precision
    }

    pub fn output_dimension(&self) -> usize {
        self.output_dimension
    }

    pub fn output_layout(&self) -> OutputLayout {
        self.output_layout
    }
}

/// Exact ONNX encoder tensor roles for V1 graph artifacts.
///
/// The schema admits exactly one data input, one length input, one data output, and one
/// length output. Callers use named roles rather than positional configuration vectors.
#[derive(Clone, Debug)]
pub struct EncoderTensorContract {
    data_input: String,
    length_input: String,
    data_output: String,
    length_output: String,
}

impl EncoderTensorContract {
    pub(crate) fn new(inputs: [String; 2], outputs: [String; 2]) -> Self {
        let [data_input, length_input] = inputs;
        let [data_output, length_output] = outputs;
        Self {
            data_input,
            length_input,
            data_output,
            length_output,
        }
    }

    pub fn data_input(&self) -> &str {
        &self.data_input
    }

    pub fn length_input(&self) -> &str {
        &self.length_input
    }

    pub fn data_output(&self) -> &str {
        &self.data_output
    }

    pub fn length_output(&self) -> &str {
        &self.length_output
    }
}

#[derive(Clone, Debug)]
pub struct FrontendWeights {
    window_dimensions: Vec<usize>,
    window_values: Vec<f32>,
    filterbank_dimensions: Vec<usize>,
    filterbank_values: Vec<f32>,
}

impl FrontendWeights {
    pub(crate) fn new(
        window_dimensions: Vec<usize>,
        window_values: Vec<f32>,
        filterbank_dimensions: Vec<usize>,
        filterbank_values: Vec<f32>,
    ) -> Self {
        Self {
            window_dimensions,
            window_values,
            filterbank_dimensions,
            filterbank_values,
        }
    }

    pub fn window_dimensions(&self) -> &[usize] {
        &self.window_dimensions
    }

    pub fn window_values(&self) -> &[f32] {
        &self.window_values
    }

    pub fn filterbank_dimensions(&self) -> &[usize] {
        &self.filterbank_dimensions
    }

    pub fn filterbank_values(&self) -> &[f32] {
        &self.filterbank_values
    }
}

#[derive(Clone, Debug)]
pub struct RnntAssets {
    decoder: ValidatedArtifact,
    joint: ValidatedArtifact,
    vocabulary: ValidatedArtifact,
}

impl RnntAssets {
    pub(crate) fn new(
        decoder: ValidatedArtifact,
        joint: ValidatedArtifact,
        vocabulary: ValidatedArtifact,
    ) -> Self {
        Self {
            decoder,
            joint,
            vocabulary,
        }
    }

    pub fn decoder(&self) -> &ValidatedArtifact {
        &self.decoder
    }

    pub fn joint(&self) -> &ValidatedArtifact {
        &self.joint
    }

    pub fn load_vocabulary(&self) -> Result<Vec<String>, PackageError> {
        read_vocabulary(&self.vocabulary)
    }
}

#[derive(Clone, Debug)]
pub struct FrontendDefinition {
    pub(crate) sample_rate: usize,
    pub(crate) n_mels: usize,
    pub(crate) hop_length: usize,
    pub(crate) n_fft: usize,
    pub(crate) center: bool,
    pub(crate) log_clamp_min: f64,
    pub(crate) log_clamp_max: f64,
    pub(crate) frames_per_second: f64,
    pub(crate) window: RelativeAsset,
    pub(crate) filterbank: RelativeAsset,
}

impl FrontendDefinition {
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    pub fn center(&self) -> bool {
        self.center
    }

    pub fn log_clamp_min(&self) -> f64 {
        self.log_clamp_min
    }

    pub fn log_clamp_max(&self) -> f64 {
        self.log_clamp_max
    }

    pub fn frames_per_second(&self) -> f64 {
        self.frames_per_second
    }

    pub(crate) fn window_asset(&self) -> &RelativeAsset {
        &self.window
    }

    pub(crate) fn filterbank_asset(&self) -> &RelativeAsset {
        &self.filterbank
    }
}

#[derive(Clone, Debug)]
pub struct CtcDefinition {
    pub(crate) vocabulary: RelativeAsset,
    pub(crate) blank_id: usize,
    pub(crate) encoder_fp16io32: RelativeAsset,
    pub(crate) encoder_fp32: RelativeAsset,
    pub(crate) tensor_contract: EncoderTensorContract,
    pub(crate) output_dimension: usize,
    pub(crate) output_layout: OutputLayout,
}

impl CtcDefinition {
    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    pub(crate) fn vocabulary_asset(&self) -> &RelativeAsset {
        &self.vocabulary
    }

    pub(crate) fn encoder_asset(&self, precision: EncoderPrecision) -> &RelativeAsset {
        match precision {
            EncoderPrecision::Fp32 => &self.encoder_fp32,
            EncoderPrecision::Fp16Io32 => &self.encoder_fp16io32,
        }
    }

    pub(crate) fn tensor_contract(&self) -> &EncoderTensorContract {
        &self.tensor_contract
    }

    pub fn output_dimension(&self) -> usize {
        self.output_dimension
    }

    pub(crate) fn output_layout(&self) -> OutputLayout {
        self.output_layout
    }
}

#[derive(Clone, Debug)]
pub struct RnntDefinition {
    pub(crate) vocabulary: RelativeAsset,
    pub(crate) blank_id: usize,
    pub(crate) prediction_hidden: usize,
    pub(crate) max_symbols_per_step: usize,
    pub(crate) encoder_fp16io32: RelativeAsset,
    pub(crate) decoder_fp16: RelativeAsset,
    pub(crate) joint_fp16: RelativeAsset,
    pub(crate) encoder_fp32: RelativeAsset,
    pub(crate) decoder_fp32: RelativeAsset,
    pub(crate) joint_fp32: RelativeAsset,
    pub(crate) output_dimension: usize,
    pub(crate) output_layout: OutputLayout,
    pub(crate) encoder_tensor_contract: EncoderTensorContract,
    pub(crate) decoder_input_names: [String; 3],
    pub(crate) decoder_output_names: [String; 3],
    pub(crate) joint_input_names: [String; 2],
    pub(crate) joint_output_name: String,
}

impl RnntDefinition {
    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    pub fn prediction_hidden(&self) -> usize {
        self.prediction_hidden
    }

    pub fn max_symbols_per_step(&self) -> usize {
        self.max_symbols_per_step
    }

    pub fn decoder_input_names(&self) -> &[String; 3] {
        &self.decoder_input_names
    }

    pub fn decoder_output_names(&self) -> &[String; 3] {
        &self.decoder_output_names
    }

    pub fn joint_input_names(&self) -> &[String; 2] {
        &self.joint_input_names
    }

    pub fn joint_output_name(&self) -> &str {
        &self.joint_output_name
    }

    pub(crate) fn vocabulary_asset(&self) -> &RelativeAsset {
        &self.vocabulary
    }

    pub(crate) fn encoder_asset(&self, precision: EncoderPrecision) -> &RelativeAsset {
        match precision {
            EncoderPrecision::Fp32 => &self.encoder_fp32,
            EncoderPrecision::Fp16Io32 => &self.encoder_fp16io32,
        }
    }

    pub(crate) fn decoder_asset(&self, precision: EncoderPrecision) -> &RelativeAsset {
        match precision {
            EncoderPrecision::Fp32 => &self.decoder_fp32,
            EncoderPrecision::Fp16Io32 => &self.decoder_fp16,
        }
    }

    pub(crate) fn joint_asset(&self, precision: EncoderPrecision) -> &RelativeAsset {
        match precision {
            EncoderPrecision::Fp32 => &self.joint_fp32,
            EncoderPrecision::Fp16Io32 => &self.joint_fp16,
        }
    }

    pub(crate) fn encoder_tensor_contract(&self) -> &EncoderTensorContract {
        &self.encoder_tensor_contract
    }

    pub fn output_dimension(&self) -> usize {
        self.output_dimension
    }

    pub(crate) fn output_layout(&self) -> OutputLayout {
        self.output_layout
    }
}

#[derive(Clone, Debug)]
pub struct VadDefinition {
    pub(crate) model: RelativeAsset,
}

impl VadDefinition {
    pub(crate) fn model_asset(&self) -> &RelativeAsset {
        &self.model
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RetainedMetadata {
    pub(crate) win_length: Option<String>,
    pub(crate) subsampling_factor: Option<String>,
    pub(crate) pos_emb_max_len: Option<String>,
    pub(crate) ctc_encoder_fp16: Option<String>,
    pub(crate) rnnt_pred_layers: Option<String>,
    pub(crate) rnnt_encoder_fp16: Option<String>,
    pub(crate) source: Option<String>,
    pub(crate) exported: Option<String>,
}

impl RetainedMetadata {
    /// Retained V1 fields are opaque. This only preserves the parser invariant that a
    /// supplied metadata value is non-empty; it assigns no artifact or runtime meaning.
    pub(crate) fn values_are_nonempty(&self) -> bool {
        [
            self.win_length.as_deref(),
            self.subsampling_factor.as_deref(),
            self.pos_emb_max_len.as_deref(),
            self.ctc_encoder_fp16.as_deref(),
            self.rnnt_pred_layers.as_deref(),
            self.rnnt_encoder_fp16.as_deref(),
            self.source.as_deref(),
            self.exported.as_deref(),
        ]
        .into_iter()
        .flatten()
        .all(|value| !value.is_empty())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PackageDefinition {
    schema_version: SchemaVersion,
    frontend: FrontendDefinition,
    ctc: CtcDefinition,
    rnnt: RnntDefinition,
    vad: VadDefinition,
    retained: RetainedMetadata,
}

impl PackageDefinition {
    pub(crate) fn new(
        schema_version: SchemaVersion,
        frontend: FrontendDefinition,
        ctc: CtcDefinition,
        rnnt: RnntDefinition,
        vad: VadDefinition,
        retained: RetainedMetadata,
    ) -> Self {
        Self {
            schema_version,
            frontend,
            ctc,
            rnnt,
            vad,
            retained,
        }
    }

    pub(crate) fn schema_version(&self) -> SchemaVersion {
        self.schema_version
    }

    pub(crate) fn frontend(&self) -> &FrontendDefinition {
        &self.frontend
    }

    pub(crate) fn ctc(&self) -> &CtcDefinition {
        &self.ctc
    }

    pub(crate) fn rnnt(&self) -> &RnntDefinition {
        &self.rnnt
    }

    pub(crate) fn vad(&self) -> &VadDefinition {
        &self.vad
    }

    pub(crate) fn retained(&self) -> &RetainedMetadata {
        &self.retained
    }
}
