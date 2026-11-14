// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Admission of one explicitly identified V3 variant. This does not load a runtime.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::assets::{self, RelativeAsset, ValidatedArtifact, validate_regular_asset};
use crate::definition::{
    ArtifactFormat, EncoderArtifact, EncoderPrecision, EncoderTensorContract, FrontendDefinition,
    FrontendWeights, OutputLayout, RnntAssets, VadDefinition,
};
use crate::error::PackageError;
use crate::schema::{
    self, parse_asset, parse_bool, parse_encoder_tensor_contract, parse_f64, parse_layout,
    parse_names_array, parse_single_name, parse_usize, take,
};
use gigaam_primitives::{f64_to_f32, usize_to_f64};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFamily {
    Ctc,
    Rnnt,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerKind {
    Characters,
    SentencePiece,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariantId {
    V3Ctc,
    V3Rnnt,
    V3E2eCtc,
    V3E2eRnnt,
}
impl VariantId {
    pub fn family(self) -> ModelFamily {
        match self {
            Self::V3Ctc | Self::V3E2eCtc => ModelFamily::Ctc,
            Self::V3Rnnt | Self::V3E2eRnnt => ModelFamily::Rnnt,
        }
    }
    pub fn tokenizer_kind(self) -> TokenizerKind {
        match self {
            Self::V3Ctc | Self::V3Rnnt => TokenizerKind::Characters,
            Self::V3E2eCtc | Self::V3E2eRnnt => TokenizerKind::SentencePiece,
        }
    }
}
impl std::str::FromStr for VariantId {
    type Err = PackageError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "v3_ctc" => Ok(Self::V3Ctc),
            "v3_rnnt" => Ok(Self::V3Rnnt),
            "v3_e2e_ctc" => Ok(Self::V3E2eCtc),
            "v3_e2e_rnnt" => Ok(Self::V3E2eRnnt),
            _ => Err(PackageError::InvalidValue {
                key: "variant",
                value: value.to_string(),
                reason: "expected an explicit V3 variant",
            }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TokenizerDefinition {
    kind: TokenizerKind,
    vocabulary: RelativeAsset,
    vocabulary_size: usize,
    blank_id: usize,
    model: Option<RelativeAsset>,
}
impl TokenizerDefinition {
    pub fn kind(&self) -> TokenizerKind {
        self.kind
    }
    pub fn vocabulary_path(&self) -> &Path {
        self.vocabulary.path()
    }
    pub fn model_path(&self) -> Option<&Path> {
        self.model.as_ref().map(RelativeAsset::path)
    }
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }
    pub fn blank_id(&self) -> usize {
        self.blank_id
    }
}
#[derive(Clone, Debug)]
struct EncoderDefinition {
    fp32: RelativeAsset,
    fp16io32: Option<RelativeAsset>,
    fp16: Option<RelativeAsset>,
    contract: EncoderTensorContract,
    output_dimension: usize,
    output_layout: OutputLayout,
}
impl EncoderDefinition {
    fn asset(&self, precision: EncoderPrecision) -> Result<&RelativeAsset, PackageError> {
        let selected = match precision {
            EncoderPrecision::Fp32 => Some(&self.fp32),
            EncoderPrecision::Fp16Io32 => self.fp16io32.as_ref(),
            EncoderPrecision::Fp16 => self.fp16.as_ref(),
        };
        selected.ok_or(PackageError::Compatibility {
            field: "encoder precision",
            reason: "selected encoder precision is not declared",
        })
    }
}
#[derive(Clone, Debug)]
pub struct VariantCtcDefinition {
    encoder: EncoderDefinition,
}
impl VariantCtcDefinition {
    pub fn tensor_contract(&self) -> &EncoderTensorContract {
        &self.encoder.contract
    }
    pub fn output_dimension(&self) -> usize {
        self.encoder.output_dimension
    }
    pub fn output_layout(&self) -> OutputLayout {
        self.encoder.output_layout
    }
}
#[derive(Clone, Debug)]
pub struct VariantRnntDefinition {
    encoder: EncoderDefinition,
    decoder_fp32: RelativeAsset,
    joint_fp32: RelativeAsset,
    decoder_fp16: Option<RelativeAsset>,
    joint_fp16: Option<RelativeAsset>,
    prediction_hidden: usize,
    prediction_layers: usize,
    max_symbols_per_step: usize,
    decoder_inputs: [String; 3],
    decoder_outputs: [String; 3],
    joint_inputs: [String; 2],
    joint_output: String,
}
impl VariantRnntDefinition {
    pub fn encoder_tensor_contract(&self) -> &EncoderTensorContract {
        &self.encoder.contract
    }
    pub fn output_dimension(&self) -> usize {
        self.encoder.output_dimension
    }
    pub fn output_layout(&self) -> OutputLayout {
        self.encoder.output_layout
    }
    pub fn prediction_hidden(&self) -> usize {
        self.prediction_hidden
    }
    pub fn prediction_layers(&self) -> usize {
        self.prediction_layers
    }
    pub fn max_symbols_per_step(&self) -> usize {
        self.max_symbols_per_step
    }
    pub fn decoder_input_names(&self) -> &[String; 3] {
        &self.decoder_inputs
    }
    pub fn decoder_output_names(&self) -> &[String; 3] {
        &self.decoder_outputs
    }
    pub fn joint_input_names(&self) -> &[String; 2] {
        &self.joint_inputs
    }
    pub fn joint_output_name(&self) -> &str {
        &self.joint_output
    }
}
#[derive(Clone, Debug)]
pub enum VariantArtifacts {
    Ctc(Box<VariantCtcDefinition>),
    Rnnt(Box<VariantRnntDefinition>),
}

#[derive(Clone, Debug)]
pub struct VariantPackage {
    root: PathBuf,
    variant: VariantId,
    artifact_format: ArtifactFormat,
    frontend: FrontendDefinition,
    win_length: usize,
    subsampling_factor: usize,
    tokenizer: TokenizerDefinition,
    artifacts: VariantArtifacts,
    vad: Option<VadDefinition>,
}
impl VariantPackage {
    pub fn open(root: &Path) -> Result<Self, PackageError> {
        let root = fs::canonicalize(root)
            .map_err(|e| PackageError::io("open model package", root.to_path_buf(), e))?;
        if !fs::metadata(&root)
            .map_err(|e| PackageError::io("inspect model package", root.clone(), e))?
            .is_dir()
        {
            return Err(incompatible("model package", "expected a directory"));
        }
        let path = root.join("config.kv");
        let text =
            fs::read_to_string(&path).map_err(|e| PackageError::io("read config.kv", path, e))?;
        Self::parse(root, &text)
    }
    pub fn variant(&self) -> VariantId {
        self.variant
    }
    pub fn artifact_format(&self) -> ArtifactFormat {
        self.artifact_format
    }
    pub fn frontend(&self) -> &FrontendDefinition {
        &self.frontend
    }
    pub fn win_length(&self) -> usize {
        self.win_length
    }
    pub fn subsampling_factor(&self) -> usize {
        self.subsampling_factor
    }
    pub fn tokenizer(&self) -> &TokenizerDefinition {
        &self.tokenizer
    }
    pub fn artifacts(&self) -> &VariantArtifacts {
        &self.artifacts
    }
    pub fn vad(&self) -> Option<&VadDefinition> {
        self.vad.as_ref()
    }
    pub fn vad_model(&self) -> Result<Option<ValidatedArtifact>, PackageError> {
        self.vad
            .as_ref()
            .map(|vad| validate_regular_asset(&self.root, vad.model_asset()))
            .transpose()
    }
    pub fn frontend_weights(&self) -> Result<FrontendWeights, PackageError> {
        let window = validate_regular_asset(&self.root, self.frontend.window_asset())?;
        let filterbank = validate_regular_asset(&self.root, self.frontend.filterbank_asset())?;
        let (wd, wv) = assets::read_f32(&window)?;
        let (fd, fv) = assets::read_f32(&filterbank)?;
        Ok(FrontendWeights::new(wd, wv, fd, fv))
    }
    pub fn encoder(&self, precision: EncoderPrecision) -> Result<EncoderArtifact, PackageError> {
        let encoder = match &self.artifacts {
            VariantArtifacts::Ctc(v) => &v.encoder,
            VariantArtifacts::Rnnt(v) => &v.encoder,
        };
        let artifact = validate_regular_asset(&self.root, encoder.asset(precision)?)?;
        Ok(EncoderArtifact::new(
            artifact,
            encoder.contract.clone(),
            precision,
            encoder.output_dimension,
            encoder.output_layout,
        ))
    }
    pub fn vocabulary(&self) -> Result<Vec<String>, PackageError> {
        let artifact = validate_regular_asset(&self.root, &self.tokenizer.vocabulary)?;
        let rows = assets::read_vocabulary(&artifact)?;
        if rows.len() != self.tokenizer.vocabulary_size || rows.iter().any(String::is_empty) {
            return Err(PackageError::InvalidAssetData {
                key: "tokenizer.vocab",
                path: artifact.path().to_path_buf(),
                reason: "expected exactly the declared nonblank vocabulary rows".to_string(),
            });
        }
        Ok(rows)
    }
    /// Loads the selected tokenizer and validates its indexed vocabulary.
    pub fn load_tokenizer(&self) -> Result<crate::TokenizerMetadata, PackageError> {
        let vocabulary = self.vocabulary()?;
        match self.tokenizer_model()? {
            None => Ok(crate::TokenizerMetadata::characters(vocabulary)),
            Some(artifact) => {
                let path = artifact.path().to_path_buf();
                let bytes = fs::read(&path).map_err(|error| {
                    PackageError::io("read tokenizer model", path.clone(), error)
                })?;
                crate::tokenizer::read_sentencepiece(&bytes, vocabulary).map_err(|reason| {
                    PackageError::InvalidAssetData {
                        key: "tokenizer.model",
                        path,
                        reason,
                    }
                })
            }
        }
    }
    pub fn tokenizer_model(&self) -> Result<Option<ValidatedArtifact>, PackageError> {
        self.tokenizer
            .model
            .as_ref()
            .map(|v| validate_regular_asset(&self.root, v))
            .transpose()
    }
    pub fn rnnt_assets(&self, precision: EncoderPrecision) -> Result<RnntAssets, PackageError> {
        let rnnt = match &self.artifacts {
            VariantArtifacts::Rnnt(v) => v,
            VariantArtifacts::Ctc(_) => {
                return Err(incompatible("rnnt", "package contains a CTC variant"));
            }
        };
        if precision == EncoderPrecision::Fp16 {
            let _ = rnnt.encoder.asset(precision)?;
        }
        let (decoder, joint) = match precision {
            EncoderPrecision::Fp32 => (&rnnt.decoder_fp32, &rnnt.joint_fp32),
            EncoderPrecision::Fp16Io32 | EncoderPrecision::Fp16 => (
                rnnt.decoder_fp16.as_ref().ok_or(incompatible(
                    "rnnt precision",
                    "FP16 heads are not declared",
                ))?,
                rnnt.joint_fp16.as_ref().ok_or(incompatible(
                    "rnnt precision",
                    "FP16 heads are not declared",
                ))?,
            ),
        };
        let _ = self.vocabulary()?;
        Ok(RnntAssets::new(
            validate_regular_asset(&self.root, decoder)?,
            validate_regular_asset(&self.root, joint)?,
            validate_regular_asset(&self.root, &self.tokenizer.vocabulary)?,
        ))
    }
    fn parse(root: PathBuf, text: &str) -> Result<Self, PackageError> {
        let (mut values, lines) = schema::parse_lines(text)?;
        let version = take(&mut values, "format_version")?;
        if version != "3" {
            return Err(PackageError::UnsupportedFormatVersion { value: version });
        }
        let variant: VariantId = take(&mut values, "variant")?.parse()?;
        let artifact_format = match take(&mut values, "artifact_format")?.as_str() {
            "onnx" => ArtifactFormat::Onnx,
            "trt_ep_context" => ArtifactFormat::TrtEpContext,
            value => {
                return Err(PackageError::InvalidValue {
                    key: "artifact_format",
                    value: value.to_string(),
                    reason: "expected onnx or trt_ep_context",
                });
            }
        };
        let frontend = FrontendDefinition {
            sample_rate: parse_usize(&mut values, "sample_rate")?,
            n_mels: parse_usize(&mut values, "n_mels")?,
            hop_length: parse_usize(&mut values, "hop_length")?,
            n_fft: parse_usize(&mut values, "n_fft")?,
            center: parse_bool(&mut values, "center")?,
            log_clamp_min: parse_f64(&mut values, "log_clamp_min")?,
            log_clamp_max: parse_f64(&mut values, "log_clamp_max")?,
            frames_per_second: parse_f64(&mut values, "frames_per_sec")?,
            window: parse_asset(&mut values, "frontend.window")?,
            filterbank: parse_asset(&mut values, "frontend.filterbank")?,
        };
        let win_length = parse_usize(&mut values, "win_length")?;
        let subsampling_factor = parse_usize(&mut values, "subsampling_factor")?;
        let kind = match take(&mut values, "tokenizer.kind")?.as_str() {
            "characters" => TokenizerKind::Characters,
            "sentencepiece" => TokenizerKind::SentencePiece,
            value => {
                return Err(PackageError::InvalidValue {
                    key: "tokenizer.kind",
                    value: value.to_string(),
                    reason: "expected characters or sentencepiece",
                });
            }
        };
        if kind != variant.tokenizer_kind() {
            return Err(incompatible("tokenizer.kind", "does not match variant"));
        }
        let tokenizer = TokenizerDefinition {
            kind,
            vocabulary: parse_asset(&mut values, "tokenizer.vocab")?,
            vocabulary_size: parse_usize(&mut values, "tokenizer.vocab_size")?,
            blank_id: parse_usize(&mut values, "tokenizer.blank_id")?,
            model: if kind == TokenizerKind::SentencePiece {
                Some(parse_asset(&mut values, "tokenizer.model")?)
            } else {
                None
            },
        };
        let artifacts = match variant.family() {
            ModelFamily::Ctc => VariantArtifacts::Ctc(Box::new(VariantCtcDefinition {
                encoder: parse_encoder(&mut values, ModelFamily::Ctc)?,
            })),
            ModelFamily::Rnnt => {
                let encoder = parse_encoder(&mut values, ModelFamily::Rnnt)?;
                let decoder_fp32 = parse_asset(&mut values, "rnnt.decoder_fp32")?;
                let joint_fp32 = parse_asset(&mut values, "rnnt.joint_fp32")?;
                let decoder_fp16 = optional_asset(&mut values, "rnnt.decoder_fp16")?;
                let joint_fp16 = optional_asset(&mut values, "rnnt.joint_fp16")?;
                if decoder_fp16.is_some() != joint_fp16.is_some() {
                    return Err(incompatible(
                        "rnnt.decoder_fp16/rnnt.joint_fp16",
                        "FP16 heads must be declared together",
                    ));
                }
                VariantArtifacts::Rnnt(Box::new(VariantRnntDefinition {
                    encoder,
                    decoder_fp32,
                    joint_fp32,
                    decoder_fp16,
                    joint_fp16,
                    prediction_hidden: parse_usize(&mut values, "rnnt.pred_hidden")?,
                    prediction_layers: parse_usize(&mut values, "rnnt.pred_layers")?,
                    max_symbols_per_step: parse_usize(&mut values, "rnnt.max_symbols_per_step")?,
                    decoder_inputs: parse_names_array(&mut values, "rnnt.decoder_inputs")?,
                    decoder_outputs: parse_names_array(&mut values, "rnnt.decoder_outputs")?,
                    joint_inputs: parse_names_array(&mut values, "rnnt.joint_inputs")?,
                    joint_output: parse_single_name(&mut values, "rnnt.joint_outputs")?,
                }))
            }
        };
        let vad = optional_asset(&mut values, "vad.model")?.map(|model| VadDefinition { model });
        let _ = values.remove("source");
        let _ = values.remove("exported");
        if let Some(key) = values.keys().next() {
            return Err(PackageError::UnknownKey {
                line: lines
                    .get(key)
                    .copied()
                    .ok_or(incompatible("config.kv", "parsed key has no source line"))?,
                key: key.clone(),
            });
        }
        let package = Self {
            root,
            variant,
            artifact_format,
            frontend,
            win_length,
            subsampling_factor,
            tokenizer,
            artifacts,
            vad,
        };
        package.validate()?;
        Ok(package)
    }
    fn validate(&self) -> Result<(), PackageError> {
        let f = &self.frontend;
        if [
            f.sample_rate,
            f.n_mels,
            f.hop_length,
            f.n_fft,
            self.win_length,
            self.subsampling_factor,
        ]
        .contains(&0)
            || self.win_length > f.n_fft
        {
            return Err(incompatible(
                "frontend",
                "dimensions must be positive and window must fit FFT",
            ));
        }
        if f.log_clamp_min <= 0.0 || f.log_clamp_max < f.log_clamp_min {
            return Err(incompatible("log clamps", "must be positive and ordered"));
        }
        let denominator = f
            .hop_length
            .checked_mul(self.subsampling_factor)
            .ok_or(incompatible("frames_per_sec", "rate denominator overflows"))?;
        let declared = f64_to_f32(f.frames_per_second);
        let derived = f64_to_f32(usize_to_f64(f.sample_rate) / usize_to_f64(denominator));
        if !declared.is_finite()
            || declared <= 0.0
            || !derived.is_finite()
            || derived <= 0.0
            || declared != derived
        {
            return Err(incompatible(
                "frames_per_sec",
                "must match the derived rate at runtime precision",
            ));
        }
        let t = &self.tokenizer;
        if t.vocabulary_size == 0 || t.blank_id != t.vocabulary_size {
            return Err(incompatible(
                "tokenizer.blank_id",
                "must equal the positive nonblank vocabulary size",
            ));
        }
        let classes = t.vocabulary_size.checked_add(1).ok_or(incompatible(
            "tokenizer.vocab_size",
            "class count overflows",
        ))?;
        match &self.artifacts {
            VariantArtifacts::Ctc(v) if v.encoder.output_dimension != classes => {
                return Err(incompatible(
                    "ctc.out_dim",
                    "must equal vocabulary size plus blank",
                ));
            }
            VariantArtifacts::Rnnt(v)
                if [
                    v.encoder.output_dimension,
                    v.prediction_hidden,
                    v.prediction_layers,
                    v.max_symbols_per_step,
                ]
                .contains(&0) =>
            {
                return Err(incompatible(
                    "rnnt",
                    "dimensions and symbol limit must be positive",
                ));
            }
            _ => {}
        }
        Ok(())
    }
}
fn incompatible(field: &'static str, reason: &'static str) -> PackageError {
    PackageError::Compatibility { field, reason }
}
fn optional_asset(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> Result<Option<RelativeAsset>, PackageError> {
    values
        .remove(key)
        .map(|v| RelativeAsset::parse(key, v))
        .transpose()
}
fn parse_encoder(
    values: &mut BTreeMap<String, String>,
    family: ModelFamily,
) -> Result<EncoderDefinition, PackageError> {
    let (fp32, fp16io32, fp16, inputs, outputs, dim, layout) = match family {
        ModelFamily::Ctc => (
            "ctc.encoder_fp32",
            "ctc.encoder_fp16io32",
            "ctc.encoder_fp16",
            "ctc.input_names",
            "ctc.output_names",
            "ctc.out_dim",
            "ctc.out_layout",
        ),
        ModelFamily::Rnnt => (
            "rnnt.encoder_fp32",
            "rnnt.encoder_fp16io32",
            "rnnt.encoder_fp16",
            "rnnt.input_names",
            "rnnt.output_names",
            "rnnt.out_dim",
            "rnnt.out_layout",
        ),
    };
    Ok(EncoderDefinition {
        fp32: parse_asset(values, fp32)?,
        fp16io32: optional_asset(values, fp16io32)?,
        fp16: optional_asset(values, fp16)?,
        contract: parse_encoder_tensor_contract(values, inputs, outputs)?,
        output_dimension: parse_usize(values, dim)?,
        output_layout: parse_layout(values, layout)?,
    })
}
