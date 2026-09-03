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
use crate::native_output::{ExpectedCardinality, NativeOutputRole, validate_f32_output};
use crate::ort_assignment::CudaAssignmentEvidence;
use crate::ort_encoder::OrtEncoder;
use crate::provider::ProviderPlan;
use crate::rnnt::{self, RnntFrameSource, RnntTransition};
use gigaam_audio::FeatureMatrixView;
use gigaam_model_package::{EncoderPrecision, ModelPackage};
use gigaam_primitives::{f64_to_f32, usize_to_i64};
use ort::session::Session;
use ort::value::Tensor;
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Instant;

struct PredictionState {
    decoder: Vec<f32>,
    hidden: Vec<f32>,
    cell: Vec<f32>,
}

struct OrtFrameSource<'a> {
    values: &'a [f32],
    output_frames: usize,
    encoder_dimension: usize,
}

impl RnntFrameSource for OrtFrameSource<'_> {
    fn output_frames(&self) -> usize {
        self.output_frames
    }

    fn frame(&self, index: usize) -> &[f32] {
        let start = index * self.encoder_dimension;
        &self.values[start..start + self.encoder_dimension]
    }
}

struct OrtRnntTransition {
    prediction: Session,
    joint: Session,
    prediction_inputs: [String; 3],
    prediction_outputs: [String; 3],
    joint_inputs: [String; 2],
    joint_output: String,
    prediction_hidden: usize,
    encoder_dimension: usize,
    blank: usize,
}

impl OrtRnntTransition {
    fn prediction_step(
        &mut self,
        label: i64,
        hidden: &[f32],
        cell: &[f32],
    ) -> Result<PredictionState, String> {
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
            .try_extract_tensor::<f32>()
            .map_err(|error| error.to_string())?;
        let (hidden_shape, hidden) = output[self.prediction_outputs[1].as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|error| error.to_string())?;
        let (cell_shape, cell) = output[self.prediction_outputs[2].as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|error| error.to_string())?;
        validate_f32_output(
            NativeOutputRole::RnntPredictionDecoder,
            decoder_shape,
            decoder,
            ExpectedCardinality::Exact(self.prediction_hidden),
        )?;
        validate_f32_output(
            NativeOutputRole::RnntPredictionHidden,
            hidden_shape,
            hidden,
            ExpectedCardinality::Exact(self.prediction_hidden),
        )?;
        validate_f32_output(
            NativeOutputRole::RnntPredictionCell,
            cell_shape,
            cell,
            ExpectedCardinality::Exact(self.prediction_hidden),
        )?;
        Ok(PredictionState {
            decoder: decoder.to_vec(),
            hidden: hidden.to_vec(),
            cell: cell.to_vec(),
        })
    }

    fn joint_argmax(&mut self, frame: &[f32], decoder: &[f32]) -> Result<usize, String> {
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
            .try_extract_tensor::<f32>()
            .map_err(|error| error.to_string())?;
        let expected = self
            .blank
            .checked_add(1)
            .ok_or_else(|| "RNNT blank identifier overflows joint vocabulary size".to_owned())?;
        validate_f32_output(
            NativeOutputRole::RnntJoint,
            shape,
            logits,
            ExpectedCardinality::Exact(expected),
        )?;
        let mut best = 0_usize;
        for (index, &value) in logits.iter().enumerate() {
            if value > logits[best] {
                best = index;
            }
        }
        Ok(best)
    }
}

impl RnntTransition for OrtRnntTransition {
    type State = PredictionState;

    fn start(&mut self, blank: usize) -> Result<Self::State, String> {
        if blank != self.blank {
            return Err(
                "RNNT transition blank identifier does not match its native session".into(),
            );
        }
        let hidden = vec![0.0_f32; self.prediction_hidden];
        let cell = vec![0.0_f32; self.prediction_hidden];
        let label =
            usize_to_i64(blank).map_err(|error| format!("RNNT blank token identifier: {error}"))?;
        self.prediction_step(label, &hidden, &cell)
    }

    fn select(&mut self, frame: &[f32], state: &Self::State) -> Result<usize, String> {
        self.joint_argmax(frame, &state.decoder)
    }

    fn advance(&mut self, token: usize, state: &mut Self::State) -> Result<(), String> {
        let label =
            usize_to_i64(token).map_err(|error| format!("RNNT token identifier: {error}"))?;
        *state = self.prediction_step(label, &state.hidden, &state.cell)?;
        Ok(())
    }
}

/// A native RNN-T decoder. Its encoder follows the selected provider plan; prediction and joint
/// are intentional CPU-only session roles and use the fp32 package assets.
pub(crate) struct RnntDecoder {
    encoder: OrtEncoder,
    transition: OrtRnntTransition,
    vocabulary: Vec<String>,
    blank: usize,
    encoder_dimension: usize,
    max_symbols_per_step: usize,
    frame_rate: FrameRate,
}

fn cpu_session(path: &Path, intra_threads: Option<NonZeroUsize>) -> Result<Session, String> {
    let mut builder = Session::builder().map_err(|error| error.to_string())?;
    if let Some(threads) = intra_threads {
        builder = builder
            .with_intra_threads(threads.get())
            .map_err(|error| error.to_string())?;
    }
    builder
        .commit_from_file(path)
        .map_err(|error| format!("{}: {error}", path.display()))
}

impl RnntDecoder {
    /// Creates an RNN-T decoder with one selected accelerator or CPU encoder and CPU prediction
    /// and joint sessions. `precision` applies to the encoder only.
    pub(crate) fn from_pack(
        pack: &ModelPackage,
        plan: &ProviderPlan,
        precision: EncoderPrecision,
    ) -> Result<Self, String> {
        let encoder = OrtEncoder::rnnt(pack, plan, precision)?;
        let assets = pack
            .rnnt_assets(EncoderPrecision::Fp32)
            .map_err(|error| error.to_string())?;
        let prediction = cpu_session(assets.decoder().path(), plan.config.intra_threads)?;
        let joint = cpu_session(assets.joint().path(), plan.config.intra_threads)?;
        let vocabulary = assets
            .load_vocabulary()
            .map_err(|error| error.to_string())?;
        let definition = pack.rnnt();
        let blank = definition.blank_id();
        let encoder_dimension = definition.output_dimension();
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
        let transition = OrtRnntTransition {
            prediction,
            joint,
            prediction_inputs: definition.decoder_input_names().clone(),
            prediction_outputs: definition.decoder_output_names().clone(),
            joint_inputs: definition.joint_input_names().clone(),
            joint_output: definition.joint_output_name().to_owned(),
            prediction_hidden: definition.prediction_hidden(),
            encoder_dimension,
            blank,
        };
        Ok(Self {
            encoder,
            transition,
            vocabulary,
            blank,
            encoder_dimension,
            max_symbols_per_step: definition.max_symbols_per_step(),
            frame_rate: FrameRate::new(f64_to_f32(pack.frontend().frames_per_second()))?,
        })
    }

    /// The selected encoder precision. Prediction and joint assets remain fp32 CPU assets.
    pub(crate) const fn encoder_precision(&self) -> EncoderPrecision {
        self.encoder.precision()
    }

    pub(crate) const fn assignment_evidence(&self) -> Option<&CudaAssignmentEvidence> {
        self.encoder.assignment_evidence()
    }
}

impl WindowDecoder for RnntDecoder {
    fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
        let started = Instant::now();
        let (encoded, output_frames) = self.encoder.forward(features)?;
        let encoder_seconds = started.elapsed().as_secs_f64();
        let frames = OrtFrameSource {
            values: &encoded,
            output_frames,
            encoder_dimension: self.encoder_dimension,
        };
        let output = rnnt::greedy(
            &mut self.transition,
            &frames,
            self.blank,
            self.max_symbols_per_step,
        )?;
        let words = ctc::tokens_to_words(output.tokens(), &self.vocabulary, self.frame_rate)?;
        Decoded::new(
            words,
            output.silence().to_vec(),
            output_frames,
            encoder_seconds,
        )
    }
}
