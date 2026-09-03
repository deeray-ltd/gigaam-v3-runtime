// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! CPU-only ONNX Runtime speech-probability execution for the selected VAD artifact.

use crate::contracts::SpeechProbabilityDetector;
use crate::native_output::{ExpectedCardinality, NativeOutputRole, validate_f32_output};
use gigaam_audio::ChannelAudioView;
use gigaam_model_package::ModelPackage;
use gigaam_primitives::{isize_to_usize, usize_to_isize};
use ort::session::Session;
use ort::value::Tensor;
use std::num::NonZeroUsize;

const HOP_SAMPLES: usize = 512;
const CONTEXT_SAMPLES: usize = 64;
const INPUT_FRAME_SAMPLES: usize = HOP_SAMPLES + CONTEXT_SAMPLES;
const RECURRENT_STATE_SIZE: usize = 128;

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
        intra_threads: Option<NonZeroUsize>,
    ) -> Result<Self, String> {
        let artifact = pack.vad_model().map_err(|error| error.to_string())?;
        let path = artifact.path();
        let mut builder = Session::builder().map_err(|error| error.to_string())?;
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
        for frame_index in 0..frames {
            let sample_offset = frame_index
                .checked_mul(HOP_SAMPLES)
                .ok_or_else(|| "VAD input frame offset overflows".to_owned())?;
            let base = usize_to_isize(sample_offset)
                .map_err(|error| format!("VAD input frame offset: {error}"))?
                - usize_to_isize(CONTEXT_SAMPLES)
                    .map_err(|error| format!("VAD context length: {error}"))?;
            let row_offset = frame_index
                .checked_mul(INPUT_FRAME_SAMPLES)
                .ok_or_else(|| "VAD input row offset overflows".to_owned())?;
            for sample_index in 0..INPUT_FRAME_SAMPLES {
                let index = base
                    + usize_to_isize(sample_index)
                        .map_err(|error| format!("VAD frame sample offset: {error}"))?;
                if let Ok(index) = isize_to_usize(index)
                    && index < samples.len()
                {
                    input[row_offset + sample_index] = samples[index];
                }
            }
        }
        let input_tensor = Tensor::from_array(([frames, INPUT_FRAME_SAMPLES], input))
            .map_err(|error| error.to_string())?;
        let hidden = Tensor::from_array((
            [1_usize, 1, RECURRENT_STATE_SIZE],
            vec![0.0_f32; RECURRENT_STATE_SIZE],
        ))
        .map_err(|error| error.to_string())?;
        let cell = Tensor::from_array((
            [1_usize, 1, RECURRENT_STATE_SIZE],
            vec![0.0_f32; RECURRENT_STATE_SIZE],
        ))
        .map_err(|error| error.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs![
                self.input.as_str() => input_tensor,
                self.hidden.as_str() => hidden,
                self.cell.as_str() => cell,
            ])
            .map_err(|error| error.to_string())?;
        let (shape, probabilities) = outputs[self.output.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|error| error.to_string())?;
        validate_f32_output(
            NativeOutputRole::Vad,
            shape,
            probabilities,
            ExpectedCardinality::Exact(frames),
        )?;
        Ok(probabilities.to_vec())
    }
}
