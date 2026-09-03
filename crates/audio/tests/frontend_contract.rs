// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

use gigaam_audio::{FrontendMode, FrontendProcessor};
use gigaam_model_package::ModelPackage;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_PACK: AtomicU64 = AtomicU64::new(0);

struct TempPack {
    root: PathBuf,
}

impl TempPack {
    fn new(config: String) -> Result<Self, String> {
        let sequence = NEXT_TEMP_PACK.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "gigaam-audio-frontend-test-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root).map_err(|error| format!("create temporary package: {error}"))?;
        fs::write(root.join("config.kv"), config)
            .map_err(|error| format!("write temporary configuration: {error}"))?;
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
        let mut bytes = Vec::new();
        let ndim = u32::try_from(dimensions.len())
            .map_err(|_| "test f32 dimension count exceeds u32".to_owned())?;
        bytes.extend_from_slice(&ndim.to_le_bytes());
        for dimension in dimensions {
            let dimension = u32::try_from(*dimension)
                .map_err(|_| "test f32 dimension exceeds u32".to_owned())?;
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(self.root.join(path), bytes)
            .map_err(|error| format!("write temporary f32 asset: {error}"))
    }

    fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for TempPack {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            panic!(
                "test-owned temporary frontend package cleanup must succeed for {}: {error}",
                self.root.display()
            );
        }
    }
}

fn config(n_fft: usize, clamp_min: &str) -> String {
    [
        "format_version=1".to_owned(),
        "sample_rate=16".to_owned(),
        "n_mels=2".to_owned(),
        "hop_length=2".to_owned(),
        format!("n_fft={n_fft}"),
        "center=false".to_owned(),
        format!("log_clamp_min={clamp_min}"),
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

fn processor(pack: &TempPack, mode: FrontendMode) -> Result<FrontendProcessor, String> {
    let opened = ModelPackage::open(pack.path()).map_err(|error| error.to_string())?;
    let weights = opened
        .frontend_weights()
        .map_err(|error| error.to_string())?;
    FrontendProcessor::new(opened.frontend(), weights, mode)
}

fn write_valid_assets(pack: &TempPack) -> Result<(), String> {
    pack.write_f32("stft_window.f32", &[4], &[1.0, 1.0, 1.0, 1.0])?;
    pack.write_f32("mel_fbank.f32", &[3, 2], &[1.0, 0.0, 0.5, 0.5, 0.0, 1.0])
}

#[test]
fn scalar_and_batched_frontends_match_and_short_clips_have_zero_frames() -> Result<(), String> {
    let pack = TempPack::new(config(4, "0.0001"))?;
    write_valid_assets(&pack)?;
    let scalar = processor(&pack, FrontendMode::Scalar)?;
    let batched = processor(&pack, FrontendMode::Batched)?;
    let samples = [0.0_f32, 0.25, -0.5, 0.75, 0.5, -0.25, 0.125, 0.0];
    let scalar_features = scalar.log_mel(&samples)?;
    let batched_features = batched.log_mel(&samples)?;
    assert_eq!(scalar_features.mel_bins(), 2);
    assert_eq!(scalar_features.frames(), 3);
    assert_eq!(scalar_features.frames(), batched_features.frames());
    for (scalar, batched) in scalar_features
        .values()
        .iter()
        .zip(batched_features.values())
    {
        assert!((scalar - batched).abs() < 1e-5);
    }
    assert_eq!(scalar.log_mel(&[0.0, 0.0, 0.0])?.frames(), 0);
    Ok(())
}

#[test]
fn frontend_refuses_invalid_mode_dimensions_weights_clamps_and_fft_plan() -> Result<(), String> {
    assert!(FrontendMode::parse("scalar ").is_err());

    let invalid_dimensions = TempPack::new(config(4, "0.0001"))?;
    invalid_dimensions.write_f32("stft_window.f32", &[3], &[1.0, 1.0, 1.0])?;
    invalid_dimensions.write_f32("mel_fbank.f32", &[3, 2], &[1.0; 6])?;
    assert!(processor(&invalid_dimensions, FrontendMode::Scalar).is_err());

    let nonfinite_weights = TempPack::new(config(4, "0.0001"))?;
    nonfinite_weights.write_f32("stft_window.f32", &[4], &[1.0, 1.0, f32::NAN, 1.0])?;
    nonfinite_weights.write_f32("mel_fbank.f32", &[3, 2], &[1.0; 6])?;
    assert!(processor(&nonfinite_weights, FrontendMode::Scalar).is_err());

    let empty_band = TempPack::new(config(4, "0.0001"))?;
    empty_band.write_f32("stft_window.f32", &[4], &[1.0; 4])?;
    empty_band.write_f32("mel_fbank.f32", &[3, 2], &[0.0; 6])?;
    assert!(processor(&empty_band, FrontendMode::Scalar).is_err());

    let unsupported_plan = TempPack::new(config(34, "0.0001"))?;
    unsupported_plan.write_f32("stft_window.f32", &[34], &[1.0; 34])?;
    unsupported_plan.write_f32("mel_fbank.f32", &[18, 2], &[1.0; 36])?;
    assert!(processor(&unsupported_plan, FrontendMode::Scalar).is_err());

    let converted_clamp = TempPack::new(config(4, "1e-50"))?;
    write_valid_assets(&converted_clamp)?;
    assert!(processor(&converted_clamp, FrontendMode::Scalar).is_err());
    Ok(())
}
