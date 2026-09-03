// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! A self-contained valid frontend fixture for Transcription semantic tests.

use gigaam_audio::{FrontendMode, FrontendProcessor};
use gigaam_model_package::ModelPackage;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMPORARY_PACKAGE: AtomicU64 = AtomicU64::new(0);

struct TemporaryPackage {
    root: PathBuf,
}

impl TemporaryPackage {
    fn new(sample_rate: u32) -> Result<Self, String> {
        let sequence = NEXT_TEMPORARY_PACKAGE.fetch_add(1, Ordering::Relaxed);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../target/gigaam-transcription-test-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root)
            .map_err(|error| format!("create temporary model package: {error}"))?;
        fs::write(root.join("config.kv"), config(sample_rate))
            .map_err(|error| format!("write temporary model configuration: {error}"))?;
        Ok(Self { root })
    }

    fn write_f32(&self, path: &str, dimensions: &[usize], values: &[f32]) -> Result<(), String> {
        let expected = dimensions.iter().try_fold(1_usize, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or_else(|| "test f32 dimensions overflow usize".to_owned())
        })?;
        if values.len() != expected {
            return Err(format!(
                "test f32 values have length {}, expected {expected}",
                values.len()
            ));
        }
        let dimension_count = u32::try_from(dimensions.len())
            .map_err(|_| "test f32 dimension count exceeds u32".to_owned())?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&dimension_count.to_le_bytes());
        for dimension in dimensions {
            let value = u32::try_from(*dimension)
                .map_err(|_| "test f32 dimension exceeds u32".to_owned())?;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(self.root.join(path), bytes)
            .map_err(|error| format!("write temporary f32 asset: {error}"))
    }
}

impl Drop for TemporaryPackage {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            panic!(
                "test-owned temporary transcription package cleanup must succeed for {}: {error}",
                self.root.display()
            );
        }
    }
}

pub fn frontend() -> Arc<FrontendProcessor> {
    frontend_at_sample_rate(16)
}

pub fn frontend_at_sample_rate(sample_rate: u32) -> Arc<FrontendProcessor> {
    let package =
        TemporaryPackage::new(sample_rate).expect("test frontend package creation must succeed");
    package
        .write_f32("stft_window.f32", &[4], &[1.0, 1.0, 1.0, 1.0])
        .expect("test frontend window asset must write");
    package
        .write_f32("mel_fbank.f32", &[3, 2], &[1.0, 0.0, 0.5, 0.5, 0.0, 1.0])
        .expect("test frontend filterbank asset must write");
    let model = ModelPackage::open(&package.root).expect("test frontend model package must open");
    Arc::new(
        FrontendProcessor::new(
            model.frontend(),
            model
                .frontend_weights()
                .expect("test frontend package must expose its weights"),
            FrontendMode::Scalar,
        )
        .expect("test frontend weights and definition must be compatible"),
    )
}

fn config(sample_rate: u32) -> String {
    [
        "format_version=1".to_owned(),
        format!("sample_rate={sample_rate}"),
        "n_mels=2".to_owned(),
        "hop_length=2".to_owned(),
        "n_fft=4".to_owned(),
        "center=false".to_owned(),
        "log_clamp_min=0.0001".to_owned(),
        "log_clamp_max=1000.0".to_owned(),
        "frames_per_sec=8.0".to_owned(),
        "ctc.vocab=ctc_vocab.txt".to_owned(),
        "ctc.blank_id=0".to_owned(),
        "ctc.encoder_fp32=ctc_fp32.onnx".to_owned(),
        "ctc.input_names=features,feature_lengths".to_owned(),
        "ctc.output_names=log_probs,encoded_lengths".to_owned(),
        "rnnt.vocab=rnnt_vocab.txt".to_owned(),
        "rnnt.blank_id=0".to_owned(),
        "rnnt.pred_hidden=1".to_owned(),
        "rnnt.max_symbols_per_step=1".to_owned(),
        "rnnt.encoder_fp16io32=rnnt_fp16io32.onnx".to_owned(),
        "rnnt.decoder_fp16=rnnt_decoder_fp16.onnx".to_owned(),
        "rnnt.joint_fp16=rnnt_joint_fp16.onnx".to_owned(),
        "rnnt.encoder_fp32=rnnt_fp32.onnx".to_owned(),
        "rnnt.decoder_fp32=rnnt_decoder_fp32.onnx".to_owned(),
        "rnnt.joint_fp32=rnnt_joint_fp32.onnx".to_owned(),
        "ctc.encoder_fp16io32=ctc_fp16io32.onnx".to_owned(),
        "ctc.out_dim=1".to_owned(),
        "ctc.out_layout=t_d".to_owned(),
        "rnnt.out_dim=1".to_owned(),
        "rnnt.out_layout=d_t".to_owned(),
        "rnnt.input_names=audio_signal,length".to_owned(),
        "rnnt.output_names=encoded,encoded_len".to_owned(),
        "rnnt.decoder_inputs=x,hi,ci".to_owned(),
        "rnnt.decoder_outputs=dec,ho,co".to_owned(),
        "rnnt.joint_inputs=enc,dec".to_owned(),
        "rnnt.joint_outputs=joint".to_owned(),
        "vad.model=vad.onnx".to_owned(),
    ]
    .join("\n")
}
