// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! CPU-only ONNX Runtime speech-probability execution for the selected VAD artifact.

use crate::NativeRuntime;
use crate::contracts::SpeechProbabilityDetector;
use crate::native_output::{ExpectedCardinality, NativeOutputRole, validate_f32_output};
use gigaam_audio::ChannelAudioView;
use gigaam_model_package::{ModelPackage, VariantPackage};
use ort::session::{Session, SessionOutputs};
use ort::value::{Tensor, TensorRef};
use std::num::NonZeroUsize;

const HOP_SAMPLES: usize = 512;
const CONTEXT_SAMPLES: usize = 64;
const INPUT_FRAME_SAMPLES: usize = HOP_SAMPLES + CONTEXT_SAMPLES;
const RECURRENT_STATE_SIZE: usize = 128;
const BLOCK_FRAMES: usize = 32;
const BLOCK_SAMPLES: usize = BLOCK_FRAMES * HOP_SAMPLES;
const RECURRENT_SHAPE: [usize; 3] = [1, 1, RECURRENT_STATE_SIZE];

struct RecurrentState {
    hidden: [f32; RECURRENT_STATE_SIZE],
    cell: [f32; RECURRENT_STATE_SIZE],
}

impl RecurrentState {
    fn initial() -> Self {
        Self {
            hidden: [0.0; RECURRENT_STATE_SIZE],
            cell: [0.0; RECURRENT_STATE_SIZE],
        }
    }
}

/// Native speech-probability detector. This small VAD graph is an intentional CPU-only role.
pub struct Vad {
    session: Session,
    input: String,
    hidden: String,
    cell: String,
    output: String,
}

impl Vad {
    /// Opens the selected VAD artifact with ONNX Runtime's built-in CPU provider.
    pub fn from_pack(
        pack: &ModelPackage,
        runtime: &NativeRuntime,
        intra_threads: Option<NonZeroUsize>,
    ) -> Result<Self, String> {
        let artifact = pack.vad_model().map_err(|error| error.to_string())?;
        Self::open(artifact.path(), runtime, intra_threads)
    }

    /// Opens the optional VAD selected by a V3 package.
    pub fn from_variant(
        pack: &VariantPackage,
        runtime: &NativeRuntime,
        intra_threads: Option<NonZeroUsize>,
    ) -> Result<Self, String> {
        let artifact = pack
            .vad_model()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "selected package does not declare VAD".to_string())?;
        Self::open(artifact.path(), runtime, intra_threads)
    }

    fn open(
        path: &std::path::Path,
        runtime: &NativeRuntime,
        intra_threads: Option<NonZeroUsize>,
    ) -> Result<Self, String> {
        let mut builder =
            Session::builder(runtime.environment()).map_err(|error| error.to_string())?;
        if let Some(threads) = intra_threads {
            builder = builder
                .with_intra_threads(threads.get())
                .map_err(|error| error.to_string())?;
        }
        let session = builder
            .commit_from_file(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(Self {
            session,
            input: "input".into(),
            hidden: "h".into(),
            cell: "c".into(),
            output: "speech_probs".into(),
        })
    }

    fn run_frames(
        &mut self,
        input: &[f32],
        frames: usize,
        state: Option<&mut RecurrentState>,
    ) -> Result<Vec<f32>, String> {
        let initial = RecurrentState::initial();
        let input_state = state.as_deref().unwrap_or(&initial);
        let input_tensor = TensorRef::from_array_view(([frames, INPUT_FRAME_SAMPLES], input))
            .map_err(|error| error.to_string())?;
        let hidden = Tensor::from_array((RECURRENT_SHAPE, input_state.hidden.to_vec()))
            .map_err(|error| error.to_string())?;
        let cell = Tensor::from_array((RECURRENT_SHAPE, input_state.cell.to_vec()))
            .map_err(|error| error.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs![
                self.input.as_str() => input_tensor,
                self.hidden.as_str() => hidden,
                self.cell.as_str() => cell,
            ])
            .map_err(|error| error.to_string())?;
        let (shape, probabilities) = outputs
            .get(self.output.as_str())
            .ok_or_else(|| format!("missing VAD output: {}", self.output))?
            .try_extract_tensor::<f32>()
            .map_err(|error| error.to_string())?;
        validate_vad_output(shape, probabilities, frames)?;
        if let Some(state) = state {
            let hidden = recurrent_output(&outputs, "hn")?;
            let cell = recurrent_output(&outputs, "cn")?;
            *state = RecurrentState { hidden, cell };
        }
        Ok(probabilities.to_vec())
    }
}

impl SpeechProbabilityDetector for Vad {
    fn probabilities(&mut self, audio: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }
        let samples = audio.samples();
        let frames = samples.len().div_ceil(HOP_SAMPLES);
        let input_size = frames
            .checked_mul(INPUT_FRAME_SAMPLES)
            .ok_or_else(|| "VAD input dimensions overflow usize".to_owned())?;
        let mut input = vec![0.0_f32; input_size];
        fill_input(samples, &mut input, &mut [0.0; CONTEXT_SAMPLES]);
        self.run_frames(&input, frames, None)
    }

    fn probabilities_bounded(
        &mut self,
        audio_length: usize,
        read: &mut dyn FnMut(&mut [f32]) -> Result<(), String>,
        emit: &mut dyn FnMut(&[f32]) -> Result<(), String>,
    ) -> Result<(), String> {
        if audio_length == 0 {
            return Ok(());
        }
        let mut state = RecurrentState::initial();
        let mut context = [0.0; CONTEXT_SAMPLES];
        let mut samples = vec![0.0; BLOCK_SAMPLES];
        let mut input = vec![0.0; BLOCK_FRAMES * INPUT_FRAME_SAMPLES];
        let mut remaining = audio_length;
        while remaining != 0 {
            let valid_samples = remaining.min(BLOCK_SAMPLES);
            let block = &mut samples[..valid_samples];
            read(block)?;
            if block.iter().any(|value| !value.is_finite()) {
                return Err("VAD input samples must be finite".into());
            }
            let frames = valid_samples.div_ceil(HOP_SAMPLES);
            let rows = &mut input[..frames * INPUT_FRAME_SAMPLES];
            fill_input(block, rows, &mut context);
            emit(&self.run_frames(rows, frames, Some(&mut state))?)?;
            remaining -= valid_samples;
        }
        Ok(())
    }
}

// Each row receives real predecessor samples from the previous full row,
// including across native runs. Only the first context and final partial row are padded.
fn fill_input(samples: &[f32], input: &mut [f32], context: &mut [f32; CONTEXT_SAMPLES]) {
    let (rows, remainder) = input.as_chunks_mut::<INPUT_FRAME_SAMPLES>();
    debug_assert!(remainder.is_empty(), "each VAD row has a fixed width");
    for (row, hop) in rows.iter_mut().zip(samples.chunks(HOP_SAMPLES)) {
        row[..CONTEXT_SAMPLES].copy_from_slice(context);
        let hop_end = CONTEXT_SAMPLES + hop.len();
        row[CONTEXT_SAMPLES..hop_end].copy_from_slice(hop);
        if hop.len() == HOP_SAMPLES {
            context.copy_from_slice(&hop[HOP_SAMPLES - CONTEXT_SAMPLES..]);
        } else {
            row[hop_end..].fill(0.0);
        }
    }
}

fn recurrent_output(
    outputs: &SessionOutputs<'_>,
    name: &str,
) -> Result<[f32; RECURRENT_STATE_SIZE], String> {
    let (shape, state) = outputs
        .get(name)
        .ok_or_else(|| format!("missing VAD recurrent output: {name}"))?
        .try_extract_tensor::<f32>()
        .map_err(|error| format!("VAD {name}: {error}"))?;
    validate_recurrent_output(shape, state).map_err(|error| format!("VAD {name}: {error}"))
}

fn validate_recurrent_output(
    shape: &[i64],
    state: &[f32],
) -> Result<[f32; RECURRENT_STATE_SIZE], String> {
    const STATE_WIDTH: i64 = 128;
    if shape != [1, 1, STATE_WIDTH] {
        return Err(format!("invalid VAD recurrent output shape: {shape:?}"));
    }
    validate_f32_output(
        NativeOutputRole::Vad,
        shape,
        state,
        ExpectedCardinality::Exact(RECURRENT_STATE_SIZE),
    )?;
    state
        .try_into()
        .map_err(|_| "invalid VAD recurrent output cardinality".to_owned())
}

fn validate_vad_output(shape: &[i64], probabilities: &[f32], frames: usize) -> Result<(), String> {
    validate_f32_output(
        NativeOutputRole::Vad,
        shape,
        probabilities,
        ExpectedCardinality::Exact(frames),
    )?;
    let frames = i64::try_from(frames).map_err(|_| "VAD frame count exceeds i64")?;
    if shape != [frames] {
        return Err(format!("invalid VAD output shape: {shape:?}"));
    }
    if probabilities
        .iter()
        .any(|value| !(0.0..=1.0).contains(value))
    {
        return Err("VAD output probabilities must be in 0..=1".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CONTEXT_SAMPLES, HOP_SAMPLES, INPUT_FRAME_SAMPLES, RECURRENT_STATE_SIZE, fill_input,
        validate_recurrent_output, validate_vad_output,
    };

    #[test]
    fn real_context_crosses_run_boundaries_and_only_the_tail_is_zero_padded() {
        let mut first = [1.0; HOP_SAMPLES];
        first[HOP_SAMPLES - CONTEXT_SAMPLES..].fill(2.0);
        let mut row = [5.0; INPUT_FRAME_SAMPLES];
        let mut context = [0.0; CONTEXT_SAMPLES];
        fill_input(&first, &mut row, &mut context);
        assert_eq!(&row[..CONTEXT_SAMPLES], &[0.0; CONTEXT_SAMPLES]);
        assert_eq!(&row[CONTEXT_SAMPLES..], &first);

        fill_input(&[3.0; 5], &mut row, &mut context);
        assert_eq!(&row[..CONTEXT_SAMPLES], &[2.0; CONTEXT_SAMPLES]);
        assert_eq!(&row[CONTEXT_SAMPLES..CONTEXT_SAMPLES + 5], &[3.0; 5]);
        assert!(row[CONTEXT_SAMPLES + 5..].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn recurrent_outputs_require_the_complete_finite_state() {
        let mut state = [0.0; RECURRENT_STATE_SIZE];
        assert!(validate_recurrent_output(&[1, 1, 128], &state).is_ok());
        assert!(validate_recurrent_output(&[1, 128], &state).is_err());
        assert!(validate_recurrent_output(&[1, 1, 128], &state[..127]).is_err());
        state[127] = f32::INFINITY;
        assert!(validate_recurrent_output(&[1, 1, 128], &state).is_err());
    }

    #[test]
    fn native_probability_layout_and_domain_are_checked() {
        assert!(validate_vad_output(&[2], &[0.0, 1.0], 2).is_ok());
        assert!(validate_vad_output(&[1, 2], &[0.0, 1.0], 2).is_err());
        assert!(validate_vad_output(&[2], &[0.0], 2).is_err());
        assert!(validate_vad_output(&[2], &[0.0, f32::NAN], 2).is_err());
        assert!(validate_vad_output(&[2], &[0.0, 1.1], 2).is_err());
    }
}
