// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! ONNX Runtime RNN-T prediction and joint adapters with the current greedy decoder contract.

use crate::contracts::{Decoded, Encoder, FrameRate, WindowDecoder};
use crate::ctc;
use crate::native_output::{
    ExpectedCardinality, NativeOutputRole, validate_f16_output, validate_f32_output,
};
use crate::ort_assignment::CudaAssignmentEvidence;
use crate::ort_encoder::{EncodedRnntWindow, OrtEncoder};
use crate::provider::{Device, ProviderPlan};
use crate::rnnt::{self, RnntFrameSource, RnntTransition};
use crate::rnnt_beam::{self, RnntBeamFrameValue, RnntBeamOptions, RnntBeamTransition};
use gigaam_audio::FeatureMatrixView;
use gigaam_model_package::{
    EncoderPrecision, ModelPackage, RnntAssets, VariantArtifacts, VariantPackage,
};
use gigaam_primitives::{f64_to_f32, usize_to_i64};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::time::Instant;

struct PredictionState<T> {
    decoder: Vec<T>,
    hidden: Vec<T>,
    cell: Vec<T>,
}

struct OrtFrameSource<'a, T> {
    values: &'a [T],
    output_frames: usize,
    encoder_dimension: usize,
}

impl<T> RnntFrameSource<T> for OrtFrameSource<'_, T> {
    fn output_frames(&self) -> usize {
        self.output_frames
    }

    fn frame(&self, index: usize) -> &[T] {
        let start = index * self.encoder_dimension;
        &self.values[start..start + self.encoder_dimension]
    }
}

struct OrtRnntTransition<T> {
    prediction: Session,
    joint: Session,
    prediction_inputs: [String; 3],
    prediction_outputs: [String; 3],
    joint_inputs: [String; 2],
    joint_output: String,
    prediction_hidden: usize,
    encoder_dimension: usize,
    blank: usize,
    prepared_frame: Vec<T>,
}

impl<T: SessionFloat> OrtRnntTransition<T> {
    fn prediction_step(
        &mut self,
        label: i64,
        hidden: &[T],
        cell: &[T],
    ) -> Result<PredictionState<T>, String> {
        let hidden_size = self.prediction_hidden;
        let input =
            Tensor::from_array(([1_usize, 1], vec![label])).map_err(|error| error.to_string())?;
        let hidden_input = Tensor::from_array(([1_usize, 1, hidden_size], hidden.to_vec()))
            .map_err(|error| error.to_string())?;
        let cell_input = Tensor::from_array(([1_usize, 1, hidden_size], cell.to_vec()))
            .map_err(|error| error.to_string())?;
        let output = self
            .prediction
            .run(ort::inputs![
                self.prediction_inputs[0].as_str() => input,
                self.prediction_inputs[1].as_str() => hidden_input,
                self.prediction_inputs[2].as_str() => cell_input,
            ])
            .map_err(|error| error.to_string())?;
        let (decoder_shape, decoder) = output[self.prediction_outputs[0].as_str()]
            .try_extract_tensor::<T>()
            .map_err(|error| error.to_string())?;
        let (hidden_shape, hidden) = output[self.prediction_outputs[1].as_str()]
            .try_extract_tensor::<T>()
            .map_err(|error| error.to_string())?;
        let (cell_shape, cell) = output[self.prediction_outputs[2].as_str()]
            .try_extract_tensor::<T>()
            .map_err(|error| error.to_string())?;
        T::validate(
            NativeOutputRole::RnntPredictionDecoder,
            decoder_shape,
            decoder,
            ExpectedCardinality::Exact(self.prediction_hidden),
        )?;
        T::validate(
            NativeOutputRole::RnntPredictionHidden,
            hidden_shape,
            hidden,
            ExpectedCardinality::Exact(self.prediction_hidden),
        )?;
        T::validate(
            NativeOutputRole::RnntPredictionCell,
            cell_shape,
            cell,
            ExpectedCardinality::Exact(self.prediction_hidden),
        )?;
        validate_vector_axes(
            "prediction decoder",
            decoder_shape,
            self.prediction_hidden,
            3,
        )?;
        validate_vector_axes("prediction hidden", hidden_shape, self.prediction_hidden, 3)?;
        validate_vector_axes("prediction cell", cell_shape, self.prediction_hidden, 3)?;
        Ok(PredictionState {
            decoder: decoder.to_vec(),
            hidden: hidden.to_vec(),
            cell: cell.to_vec(),
        })
    }

    fn joint_row(&mut self, frame: &[T], decoder: &[T]) -> Result<Vec<f32>, String> {
        let encoder = Tensor::from_array(([1_usize, self.encoder_dimension, 1], frame.to_vec()))
            .map_err(|error| error.to_string())?;
        let prediction =
            Tensor::from_array(([1_usize, self.prediction_hidden, 1], decoder.to_vec()))
                .map_err(|error| error.to_string())?;
        let output = self
            .joint
            .run(ort::inputs![
                self.joint_inputs[0].as_str() => encoder,
                self.joint_inputs[1].as_str() => prediction,
            ])
            .map_err(|error| error.to_string())?;
        let (shape, logits) = output[self.joint_output.as_str()]
            .try_extract_tensor::<T>()
            .map_err(|error| error.to_string())?;
        let expected = self
            .blank
            .checked_add(1)
            .ok_or("RNN-T vocabulary overflow")?;
        T::validate(
            NativeOutputRole::RnntJoint,
            shape,
            logits,
            ExpectedCardinality::Exact(expected),
        )?;
        validate_vector_axes("joint", shape, expected, 4)?;
        let mut row = Vec::new();
        row.try_reserve_exact(expected)
            .map_err(|_| "RNN-T joint row allocation failed")?;
        row.extend(logits.iter().map(|value| value.to_f32()));
        Ok(row)
    }

    fn joint_argmax(
        &mut self,
        frame: &[T],
        decoder: &[T],
        scored: bool,
    ) -> Result<(usize, Option<f64>), String> {
        let encoder = Tensor::from_array(([1_usize, self.encoder_dimension, 1], frame.to_vec()))
            .map_err(|error| error.to_string())?;
        let prediction =
            Tensor::from_array(([1_usize, self.prediction_hidden, 1], decoder.to_vec()))
                .map_err(|error| error.to_string())?;
        let output = self
            .joint
            .run(ort::inputs![
                self.joint_inputs[0].as_str() => encoder,
                self.joint_inputs[1].as_str() => prediction,
            ])
            .map_err(|error| error.to_string())?;
        let (shape, logits) = output[self.joint_output.as_str()]
            .try_extract_tensor::<T>()
            .map_err(|error| error.to_string())?;
        let expected = self
            .blank
            .checked_add(1)
            .ok_or_else(|| "RNNT blank identifier overflows joint vocabulary size".to_owned())?;
        T::validate(
            NativeOutputRole::RnntJoint,
            shape,
            logits,
            ExpectedCardinality::Exact(expected),
        )?;
        validate_vector_axes("joint", shape, expected, 4)?;
        let mut best = 0_usize;
        for (index, &value) in logits.iter().enumerate() {
            if value > logits[best] {
                best = index;
            }
        }
        let score = if scored {
            Some(crate::contracts::selected_log_probability(
                &logits
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
                best,
            )?)
        } else {
            None
        };
        Ok((best, score))
    }
}

fn validate_vector_axes(
    role: &str,
    shape: &[i64],
    width: usize,
    rank: usize,
) -> Result<(), String> {
    let width = i64::try_from(width).map_err(|_| format!("{role} width exceeds i64"))?;
    if shape.len() != rank
        || shape.last() != Some(&width)
        || shape[..shape.len().saturating_sub(1)]
            .iter()
            .any(|&axis| axis != 1)
    {
        return Err(format!("unexpected RNN-T {role} output shape {shape:?}"));
    }
    Ok(())
}

trait SessionFloat:
    ort::value::PrimitiveTensorElementType + std::fmt::Debug + PartialOrd + Copy + Send + Sync + 'static
{
    fn zero() -> Self;
    fn to_f32(self) -> f32;
    fn validate(
        role: NativeOutputRole,
        shape: &[i64],
        values: &[Self],
        expected: ExpectedCardinality,
    ) -> Result<(), String>;
}
impl SessionFloat for f32 {
    fn zero() -> Self {
        0.0
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn validate(
        role: NativeOutputRole,
        shape: &[i64],
        values: &[Self],
        expected: ExpectedCardinality,
    ) -> Result<(), String> {
        validate_f32_output(role, shape, values, expected)
    }
}
impl SessionFloat for half::f16 {
    fn zero() -> Self {
        half::f16::ZERO
    }
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
    fn validate(
        role: NativeOutputRole,
        shape: &[i64],
        values: &[Self],
        expected: ExpectedCardinality,
    ) -> Result<(), String> {
        validate_f16_output(role, shape, values, expected)
    }
}
impl<T: SessionFloat> OrtRnntTransition<T> {
    fn start_state(&mut self, blank: usize) -> Result<PredictionState<T>, String> {
        if blank != self.blank {
            return Err(
                "RNNT transition blank identifier does not match its native session".into(),
            );
        }
        let zeros = vec![T::zero(); self.prediction_hidden];
        let label =
            usize_to_i64(blank).map_err(|error| format!("RNNT blank token identifier: {error}"))?;
        self.prediction_step(label, &zeros, &zeros)
    }
    fn advance_state(
        &mut self,
        token: usize,
        state: &mut PredictionState<T>,
    ) -> Result<(), String> {
        let label =
            usize_to_i64(token).map_err(|error| format!("RNNT token identifier: {error}"))?;
        *state = self.prediction_step(label, &state.hidden, &state.cell)?;
        Ok(())
    }
}
macro_rules! native_transition {
    ($ty:ty) => {
        impl RnntTransition<$ty> for OrtRnntTransition<$ty> {
            type State = PredictionState<$ty>;
            fn start(&mut self, blank: usize) -> Result<Self::State, String> {
                self.start_state(blank)
            }
            fn select(&mut self, frame: &[$ty], state: &Self::State) -> Result<usize, String> {
                self.joint_argmax(frame, &state.decoder, false)
                    .map(|result| result.0)
            }
            fn select_scored(
                &mut self,
                frame: &[$ty],
                state: &Self::State,
            ) -> Result<(usize, f64), String> {
                let (token, score) = self.joint_argmax(frame, &state.decoder, true)?;
                Ok((
                    token,
                    score.ok_or_else(|| "missing RNN-T joint score".to_owned())?,
                ))
            }
            fn advance(&mut self, token: usize, state: &mut Self::State) -> Result<(), String> {
                self.advance_state(token, state)
            }
        }
    };
}
native_transition!(f32);
native_transition!(half::f16);
impl RnntTransition<f32> for OrtRnntTransition<half::f16> {
    type State = PredictionState<half::f16>;
    fn start(&mut self, blank: usize) -> Result<Self::State, String> {
        self.start_state(blank)
    }
    fn prepare_frame(&mut self, frame: &[f32]) -> Result<(), String> {
        self.prepared_frame = frame.iter().copied().map(half::f16::from_f32).collect();
        if self.prepared_frame.iter().any(|v| !v.is_finite()) {
            return Err("RNN-T frame cannot be represented as finite FP16".into());
        }
        Ok(())
    }
    fn select(&mut self, _: &[f32], state: &Self::State) -> Result<usize, String> {
        let frame = self.prepared_frame.clone();
        self.joint_argmax(&frame, &state.decoder, false)
            .map(|result| result.0)
    }
    fn select_scored(&mut self, _: &[f32], state: &Self::State) -> Result<(usize, f64), String> {
        let frame = self.prepared_frame.clone();
        let (token, score) = self.joint_argmax(&frame, &state.decoder, true)?;
        Ok((
            token,
            score.ok_or_else(|| "missing RNN-T joint score".to_owned())?,
        ))
    }
    fn advance(&mut self, token: usize, state: &mut Self::State) -> Result<(), String> {
        self.advance_state(token, state)
    }
}

macro_rules! native_beam_transition {
    ($ty:ty) => {
        impl RnntBeamTransition<$ty> for OrtRnntTransition<$ty> {
            type State = PredictionState<$ty>;
            fn start(&mut self, blank: usize) -> Result<Self::State, String> {
                self.start_state(blank)
            }
            fn prepare_frame(&mut self, _: &[$ty]) -> Result<(), String> {
                Ok(())
            }
            fn distribution(
                &mut self,
                frame: &[$ty],
                state: &Self::State,
            ) -> Result<Vec<f32>, String> {
                self.joint_row(frame, &state.decoder)
            }
            fn advance(
                &mut self,
                token: usize,
                state: &Self::State,
            ) -> Result<Self::State, String> {
                let label = usize_to_i64(token)
                    .map_err(|error| format!("RNN-T token identifier: {error}"))?;
                self.prediction_step(label, &state.hidden, &state.cell)
            }
        }
    };
}
native_beam_transition!(f32);
native_beam_transition!(half::f16);
impl RnntBeamTransition<f32> for OrtRnntTransition<half::f16> {
    type State = PredictionState<half::f16>;
    fn start(&mut self, blank: usize) -> Result<Self::State, String> {
        self.start_state(blank)
    }
    fn prepare_frame(&mut self, frame: &[f32]) -> Result<(), String> {
        <Self as RnntTransition<f32>>::prepare_frame(self, frame)
    }
    fn distribution(&mut self, _: &[f32], state: &Self::State) -> Result<Vec<f32>, String> {
        let frame = self.prepared_frame.clone();
        self.joint_row(&frame, &state.decoder)
    }
    fn advance(&mut self, token: usize, state: &Self::State) -> Result<Self::State, String> {
        let label =
            usize_to_i64(token).map_err(|error| format!("RNN-T token identifier: {error}"))?;
        self.prediction_step(label, &state.hidden, &state.cell)
    }
}

enum Transition {
    F32(OrtRnntTransition<f32>),
    F16(OrtRnntTransition<half::f16>),
}

/// A native RNN-T decoder using the selected provider plan for each neural session.
pub(crate) struct RnntDecoder {
    encoder: OrtEncoder,
    transition: Transition,
    vocabulary: Vec<String>,
    projector: Option<crate::TokenTextProjector>,
    blank: usize,
    encoder_dimension: usize,
    max_symbols_per_step: usize,
    frame_rate: FrameRate,
    beam_width: Option<usize>,
}

struct RnntParts {
    encoder: OrtEncoder,
    assets: RnntAssets,
    vocabulary: Vec<String>,
    projector: Option<crate::TokenTextProjector>,
    blank: usize,
    encoder_dimension: usize,
    max_symbols_per_step: usize,
    frame_rate: FrameRate,
    prediction_hidden: usize,
    prediction_inputs: [String; 3],
    prediction_outputs: [String; 3],
    joint_inputs: [String; 2],
    joint_output: String,
}

fn head_session(
    path: &Path,
    runtime: &crate::NativeRuntime,
    plan: &ProviderPlan,
) -> Result<Session, String> {
    let mut builder = Session::builder(runtime.environment()).map_err(|error| error.to_string())?;
    if let Some(threads) = plan.config.intra_threads {
        builder = builder
            .with_intra_threads(threads.get())
            .map_err(|error| error.to_string())?;
    }
    match plan.device() {
        Device::Cpu => {}
        Device::Cuda => {
            #[cfg(feature = "cuda")]
            {
                builder = builder
                    .with_config_entry("session.record_ep_graph_assignment_info", "1")
                    .map_err(|error| error.to_string())?
                    .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                    .map_err(|error| error.to_string())?
                    .with_disable_cpu_fallback()
                    .map_err(|error| error.to_string())?;
            }
            #[cfg(not(feature = "cuda"))]
            return Err(
                "cuda EP is unavailable in this build; rebuild with --features cuda".into(),
            );
        }
        Device::Tensorrt => {
            #[cfg(feature = "tensorrt")]
            {
                builder = builder
                    .with_execution_providers([ort::ep::TensorRT::default()
                        .build()
                        .error_on_failure()])
                    .map_err(|error| error.to_string())?
                    .with_disable_cpu_fallback()
                    .map_err(|error| error.to_string())?;
            }
            #[cfg(not(feature = "tensorrt"))]
            return Err("tensorrt EP requires building with --features tensorrt".into());
        }
    }
    let session = builder
        .commit_from_file(path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if plan.device() == Device::Cuda {
        let evidence = crate::ort_assignment::inspect(&session)?;
        eprintln!(
            "RNN-T head {} CUDA assignment: {} CUDA nodes, {} CPU nodes",
            path.display(),
            evidence.cuda_assignments(),
            evidence.cpu_assignments()
        );
        eprintln!(
            "RNN-T head {} CPU operator assignments: {:?}",
            path.display(),
            evidence.cpu_operator_types()
        );
    }
    Ok(session)
}

fn validate_precision(
    precision: EncoderPrecision,
    full_fp16: bool,
    device: Device,
) -> Result<(), String> {
    if full_fp16 && precision == EncoderPrecision::Fp32 {
        return Err("FP16 RNN-T heads require an FP16 encoder selection".into());
    }
    if !full_fp16 && precision == EncoderPrecision::Fp16 {
        return Err("native FP16 encoder requires FP16 RNN-T heads".into());
    }
    if full_fp16 && device == Device::Cpu {
        return Err("full RNN-T FP16 requires CUDA or TensorRT".into());
    }
    Ok(())
}

fn preflight_variant(
    pack: &VariantPackage,
    precision: EncoderPrecision,
    full_fp16: bool,
    device: Device,
) -> Result<crate::TokenTextProjector, String> {
    let definition = match pack.artifacts() {
        VariantArtifacts::Rnnt(definition) => definition,
        VariantArtifacts::Ctc(_) => return Err("selected variant is not RNN-T".into()),
    };
    if definition.prediction_layers() != 1 {
        return Err("RNN-T adapter requires one prediction layer".into());
    }
    validate_precision(precision, full_fp16, device)?;
    crate::TokenTextProjector::new(pack.load_tokenizer().map_err(|error| error.to_string())?)
}

impl RnntDecoder {
    /// Creates an RNN-T decoder; `precision` applies to the encoder only.
    pub(crate) fn from_pack(
        pack: &ModelPackage,
        context: (&crate::NativeRuntime, &ProviderPlan),
        precision: EncoderPrecision,
        full_fp16: bool,
    ) -> Result<Self, String> {
        let (runtime, plan) = context;
        validate_precision(precision, full_fp16, plan.device())?;
        let encoder = OrtEncoder::rnnt(pack, runtime, plan, precision)?;
        let assets = pack
            .rnnt_assets(if full_fp16 {
                precision
            } else {
                EncoderPrecision::Fp32
            })
            .map_err(|error| error.to_string())?;
        let vocabulary = assets
            .load_vocabulary()
            .map_err(|error| error.to_string())?;
        let definition = pack.rnnt();
        let parts = RnntParts {
            encoder,
            assets,
            vocabulary,
            projector: None,
            blank: definition.blank_id(),
            encoder_dimension: definition.output_dimension(),
            max_symbols_per_step: definition.max_symbols_per_step(),
            frame_rate: FrameRate::new(f64_to_f32(pack.frontend().frames_per_second()))?,
            prediction_hidden: definition.prediction_hidden(),
            prediction_inputs: definition.decoder_input_names().clone(),
            prediction_outputs: definition.decoder_output_names().clone(),
            joint_inputs: definition.joint_input_names().clone(),
            joint_output: definition.joint_output_name().to_owned(),
        };
        Self::from_parts(parts, runtime, plan, full_fp16)
    }

    pub(crate) fn from_variant(
        pack: &VariantPackage,
        context: (&crate::NativeRuntime, &ProviderPlan),
        precision: EncoderPrecision,
        full_fp16: bool,
    ) -> Result<Self, String> {
        let (runtime, plan) = context;
        let projector = preflight_variant(pack, precision, full_fp16, plan.device())?;
        let definition = match pack.artifacts() {
            VariantArtifacts::Rnnt(definition) => definition,
            VariantArtifacts::Ctc(_) => return Err("selected variant is not RNN-T".into()),
        };
        let vocabulary = pack.vocabulary().map_err(|error| error.to_string())?;
        let assets = pack
            .rnnt_assets(if full_fp16 {
                precision
            } else {
                EncoderPrecision::Fp32
            })
            .map_err(|error| error.to_string())?;
        let encoder = OrtEncoder::variant(pack, runtime, plan, precision)?;
        let parts = RnntParts {
            encoder,
            assets,
            vocabulary,
            projector: Some(projector),
            blank: pack.tokenizer().blank_id(),
            encoder_dimension: definition.output_dimension(),
            max_symbols_per_step: definition.max_symbols_per_step(),
            frame_rate: FrameRate::new(f64_to_f32(pack.frontend().frames_per_second()))?,
            prediction_hidden: definition.prediction_hidden(),
            prediction_inputs: definition.decoder_input_names().clone(),
            prediction_outputs: definition.decoder_output_names().clone(),
            joint_inputs: definition.joint_input_names().clone(),
            joint_output: definition.joint_output_name().to_owned(),
        };
        Self::from_parts(parts, runtime, plan, full_fp16)
    }

    fn from_parts(
        parts: RnntParts,
        runtime: &crate::NativeRuntime,
        plan: &ProviderPlan,
        full_fp16: bool,
    ) -> Result<Self, String> {
        let RnntParts {
            encoder,
            assets,
            vocabulary,
            projector,
            blank,
            encoder_dimension,
            max_symbols_per_step,
            frame_rate,
            prediction_hidden,
            prediction_inputs,
            prediction_outputs,
            joint_inputs,
            joint_output,
        } = parts;
        if encoder.out_dim() != encoder_dimension {
            return Err(format!(
                "RNNT: encoder out_dim {} ≠ rnnt.out_dim {encoder_dimension}",
                encoder.out_dim()
            ));
        }
        if vocabulary.len() < blank {
            return Err(format!(
                "RNNT: vocabulary {} is shorter than blank {blank}",
                vocabulary.len()
            ));
        }
        let prediction = head_session(assets.decoder().path(), runtime, plan)?;
        let joint = head_session(assets.joint().path(), runtime, plan)?;
        macro_rules! make_transition {
            () => {
                OrtRnntTransition {
                    prediction,
                    joint,
                    prediction_inputs,
                    prediction_outputs,
                    joint_inputs,
                    joint_output,
                    prediction_hidden,
                    encoder_dimension,
                    blank,
                    prepared_frame: Vec::new(),
                }
            };
        }
        let transition = if full_fp16 {
            Transition::F16(make_transition!())
        } else {
            Transition::F32(make_transition!())
        };
        Ok(Self {
            encoder,
            transition,
            vocabulary,
            projector,
            blank,
            encoder_dimension,
            max_symbols_per_step,
            frame_rate,
            beam_width: None,
        })
    }

    pub(crate) fn from_variant_beam(
        pack: &VariantPackage,
        context: (&crate::NativeRuntime, &ProviderPlan),
        selection: crate::RnntBeamSelection,
    ) -> Result<Self, String> {
        let width = selection.width;
        let precision = selection.precision.encoder;
        let full_fp16 = selection.precision.rnnt_heads_fp16;
        if width == 0 {
            return Err("RNN-T beam width must be positive".into());
        }
        let definition = match pack.artifacts() {
            VariantArtifacts::Rnnt(definition) => definition,
            VariantArtifacts::Ctc(_) => return Err("selected variant is not RNN-T".into()),
        };
        let vocabulary = pack
            .tokenizer()
            .blank_id()
            .checked_add(1)
            .ok_or("RNN-T vocabulary overflow")?;
        width
            .checked_mul(vocabulary)
            .ok_or("RNN-T candidate count overflow")?;
        definition
            .max_symbols_per_step()
            .checked_add(1)
            .and_then(|n| n.checked_mul(width))
            .ok_or("RNN-T completed count overflow")?;
        let mut decoder = Self::from_variant(pack, context, precision, full_fp16)?;
        decoder.beam_width = Some(width);
        Ok(decoder)
    }

    /// The selected encoder precision. Prediction and joint assets remain fp32 CPU assets.
    pub(crate) const fn encoder_precision(&self) -> EncoderPrecision {
        self.encoder.precision()
    }

    pub(crate) const fn assignment_evidence(&self) -> Option<&CudaAssignmentEvidence> {
        self.encoder.assignment_evidence()
    }
}

struct DecodeSettings<'a> {
    blank: usize,
    cap: usize,
    detail: crate::ScoreDetail,
    beam_width: Option<usize>,
    control: &'a crate::ExecutionControl,
}
fn decode_selected<T, R, F>(
    transition: &mut R,
    frames: &F,
    settings: &DecodeSettings<'_>,
) -> Result<(Vec<crate::Token>, Vec<bool>), String>
where
    T: rnnt::FiniteFrameValue + RnntBeamFrameValue,
    R: RnntTransition<T> + RnntBeamTransition<T>,
    F: RnntFrameSource<T>,
{
    if let Some(width) = settings.beam_width {
        let vocabulary = settings
            .blank
            .checked_add(1)
            .ok_or("RNN-T vocabulary overflow")?;
        let output = rnnt_beam::decode(
            transition,
            frames,
            RnntBeamOptions {
                vocabulary,
                blank: settings.blank,
                width,
                max_symbols_per_step: settings.cap,
                detail: settings.detail,
            },
            settings.control,
        )?;
        Ok(output.into_parts())
    } else {
        let output = rnnt::greedy_typed_in_context(
            transition,
            frames,
            settings.blank,
            settings.cap,
            settings.detail == crate::ScoreDetail::TokenLogProbabilities,
            settings.control,
        )?;
        Ok((output.tokens().to_vec(), output.silence().to_vec()))
    }
}

impl WindowDecoder for RnntDecoder {
    fn silence_capability(&self) -> crate::SilenceCapability {
        crate::SilenceCapability::TokenAbsence
    }

    fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
        self.decode_with_detail(features, crate::contracts::ScoreDetail::None)
    }

    fn decode_with_detail(
        &mut self,
        features: FeatureMatrixView<'_>,
        detail: crate::contracts::ScoreDetail,
    ) -> Result<Decoded, String> {
        let control = crate::ExecutionControl::without_deadline();
        self.decode_in_context(features, detail, &control)
    }

    fn decode_in_context(
        &mut self,
        features: FeatureMatrixView<'_>,
        detail: crate::contracts::ScoreDetail,
        control: &crate::ExecutionControl,
    ) -> Result<Decoded, String> {
        control.checkpoint()?;
        let started = Instant::now();
        let encoded = self.encoder.forward_rnnt(features)?;
        control.checkpoint()?;
        let encoder_seconds = started.elapsed().as_secs_f64();
        let settings = DecodeSettings {
            blank: self.blank,
            cap: self.max_symbols_per_step,
            detail,
            beam_width: self.beam_width,
            control,
        };
        let (tokens, silence, output_frames) = match (&mut self.transition, encoded) {
            (Transition::F32(transition), EncodedRnntWindow::F32(values, count)) => {
                let frames = OrtFrameSource {
                    values: &values,
                    output_frames: count,
                    encoder_dimension: self.encoder_dimension,
                };
                let (tokens, silence) = decode_selected(transition, &frames, &settings)?;
                (tokens, silence, count)
            }
            (Transition::F16(transition), EncodedRnntWindow::F32(values, count)) => {
                let frames = OrtFrameSource {
                    values: &values,
                    output_frames: count,
                    encoder_dimension: self.encoder_dimension,
                };
                let (tokens, silence) = decode_selected(transition, &frames, &settings)?;
                (tokens, silence, count)
            }
            (Transition::F16(transition), EncodedRnntWindow::F16(values, count)) => {
                let frames = OrtFrameSource {
                    values: &values,
                    output_frames: count,
                    encoder_dimension: self.encoder_dimension,
                };
                let (tokens, silence) = decode_selected(transition, &frames, &settings)?;
                (tokens, silence, count)
            }
            (Transition::F32(_), EncodedRnntWindow::F16(_, _)) => {
                return Err("native FP16 encoder requires FP16 RNN-T heads".into());
            }
        };
        control.checkpoint()?;
        if self.beam_width.is_some() && self.projector.is_none() {
            return Err("RNN-T beam requires selected tokenizer projection".into());
        }
        let ctc::ProjectedWords { words, spans } = ctc::project_words(
            &tokens,
            &self.vocabulary,
            self.projector.as_ref(),
            self.frame_rate,
        )?;
        control.checkpoint()?;
        Decoded::new_with_tokens(crate::contracts::DecodedTokensInput {
            words,
            silence,
            output_frames,
            encoder_seconds,
            tokens,
            word_spans: spans,
        })
    }
}

#[cfg(test)]
mod shape_tests {
    use super::validate_vector_axes;

    #[test]
    fn rnnt_outputs_require_width_on_the_final_axis() {
        assert!(validate_vector_axes("prediction", &[1, 1, 8], 8, 3).is_ok());
        assert!(validate_vector_axes("joint", &[1, 1, 1, 5], 5, 4).is_ok());
        assert!(validate_vector_axes("prediction", &[1, 1, 1], 1, 3).is_ok());
        assert!(validate_vector_axes("joint", &[1, 1, 1, 1], 1, 4).is_ok());
        assert!(validate_vector_axes("prediction", &[8, 1, 1], 8, 3).is_err());
        assert!(validate_vector_axes("prediction", &[1, 8, 1], 8, 3).is_err());
        assert!(validate_vector_axes("joint", &[5, 1, 1, 1], 5, 4).is_err());
        assert!(validate_vector_axes("joint", &[1, 1, 5, 1], 5, 4).is_err());
        assert!(validate_vector_axes("joint", &[1, 1, 5], 5, 4).is_err());
    }
}

#[cfg(test)]
mod variant_precision_tests {
    use super::*;
    #[test]
    fn variant_preflight_refuses_precision_before_missing_assets() {
        let root =
            std::env::temp_dir().join(format!("gigaam-rnnt-precision-{}", std::process::id()));
        std::fs::create_dir(&root).expect("create isolated package");
        let config = "format_version=3\nvariant=v3_rnnt\nartifact_format=onnx\nsample_rate=16000\nn_mels=64\nhop_length=160\nwin_length=320\nn_fft=320\ncenter=false\nlog_clamp_min=1e-09\nlog_clamp_max=1000000000\nsubsampling_factor=4\nframes_per_sec=25\nfrontend.window=window\nfrontend.filterbank=filter\ntokenizer.kind=characters\ntokenizer.vocab=missing-vocab\ntokenizer.vocab_size=2\ntokenizer.blank_id=2\nrnnt.encoder_fp32=missing-encoder\nrnnt.input_names=x,length\nrnnt.output_names=y,length\nrnnt.out_dim=7\nrnnt.out_layout=t_d\nrnnt.decoder_fp32=missing-decoder\nrnnt.joint_fp32=missing-joint\nrnnt.pred_hidden=5\nrnnt.pred_layers=1\nrnnt.max_symbols_per_step=3\nrnnt.decoder_inputs=x,h,c\nrnnt.decoder_outputs=y,h,c\nrnnt.joint_inputs=e,d\nrnnt.joint_outputs=joint\n";
        std::fs::write(root.join("config.kv"), config).expect("write configuration");
        let pack = VariantPackage::open(&root).expect("metadata admits missing unselected assets");
        let error = preflight_variant(&pack, EncoderPrecision::Fp32, true, Device::Cuda)
            .expect_err("precision must refuse");
        assert_eq!(error, "FP16 RNN-T heads require an FP16 encoder selection");
        std::fs::remove_dir_all(root).expect("remove test package");
    }

    #[test]
    fn fp32_encoder_with_fp16_heads_refuses_before_assets_or_sessions() {
        let error = validate_precision(EncoderPrecision::Fp32, true, Device::Cuda)
            .expect_err("mixed selection must refuse");
        assert_eq!(error, "FP16 RNN-T heads require an FP16 encoder selection");
        assert!(validate_precision(EncoderPrecision::Fp16Io32, true, Device::Cuda).is_ok());
        assert!(validate_precision(EncoderPrecision::Fp16, true, Device::Cuda).is_ok());
    }
}

#[cfg(test)]
mod beam_dispatch_tests {
    use super::*;
    struct Frames<T>(Vec<T>);
    impl<T> RnntFrameSource<T> for Frames<T> {
        fn output_frames(&self) -> usize {
            1
        }
        fn frame(&self, _: usize) -> &[T] {
            &self.0
        }
    }
    #[derive(Default)]
    struct Fake {
        greedy_calls: usize,
        beam_calls: usize,
        advances: usize,
    }
    impl<T> RnntTransition<T> for Fake {
        type State = Vec<usize>;
        fn start(&mut self, _: usize) -> Result<Self::State, String> {
            Ok(vec![])
        }
        fn select(&mut self, _: &[T], _: &Self::State) -> Result<usize, String> {
            self.greedy_calls += 1;
            Ok(2)
        }
        fn advance(&mut self, _: usize, _: &mut Self::State) -> Result<(), String> {
            Err("greedy blank must not advance".into())
        }
    }
    impl<T> RnntBeamTransition<T> for Fake {
        type State = Vec<usize>;
        fn start(&mut self, _: usize) -> Result<Self::State, String> {
            Ok(vec![])
        }
        fn prepare_frame(&mut self, _: &[T]) -> Result<(), String> {
            Ok(())
        }
        fn distribution(&mut self, _: &[T], _: &Self::State) -> Result<Vec<f32>, String> {
            self.beam_calls += 1;
            Ok(vec![0.0, 0.0, 1.0])
        }
        fn advance(&mut self, token: usize, state: &Self::State) -> Result<Self::State, String> {
            self.advances += 1;
            let mut next = state.clone();
            next.push(token);
            Ok(next)
        }
    }
    fn verify_dispatch<T: rnnt::FiniteFrameValue + RnntBeamFrameValue>(
        frames: Frames<T>,
    ) -> Result<(), String> {
        let control = crate::ExecutionControl::without_deadline();
        let mut fake = Fake::default();
        let mut settings = DecodeSettings {
            blank: 2,
            cap: 1,
            detail: crate::ScoreDetail::None,
            beam_width: None,
            control: &control,
        };
        let (tokens, silence) = decode_selected(&mut fake, &frames, &settings)?;
        assert!(tokens.is_empty());
        assert_eq!(silence, vec![true]);
        assert_eq!(fake.greedy_calls, 1);
        assert_eq!(fake.beam_calls, 0);
        settings.beam_width = Some(2);
        let _ = decode_selected(&mut fake, &frames, &settings)?;
        assert_eq!(fake.greedy_calls, 1);
        assert!(fake.beam_calls > 0);
        assert!(fake.advances > 0);
        Ok(())
    }
    #[test]
    fn selected_mode_dispatches_both_frame_precisions() -> Result<(), String> {
        verify_dispatch(Frames(vec![1.0_f32]))?;
        verify_dispatch(Frames(vec![half::f16::from_f32(1.0)]))
    }
}
