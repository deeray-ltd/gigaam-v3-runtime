// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! ONNX Runtime encoder sessions for the one resolved execution-provider plan.

use crate::contracts::Encoder;
use crate::native_output::{
    ExpectedCardinality, NativeOutputRole, validate_f16_output, validate_f32_output,
};
use crate::ort_assignment::{self, CudaAssignmentEvidence};
#[cfg(feature = "cuda")]
use crate::provider::CudaArena;
use crate::provider::{Device, EncoderRole, ProviderPlan};
use gigaam_audio::FeatureMatrixView;
use gigaam_model_package::{EncoderArtifact, EncoderPrecision, ModelPackage, OutputLayout};
#[cfg(feature = "cuda")]
use ort::ep::CUDA;
#[cfg(feature = "cuda")]
use ort::ep::cuda::ConvAlgorithmSearch;
use ort::session::Session;
use ort::value::Tensor;

/// One native encoder session built for one exact provider plan.
pub struct OrtEncoder {
    session: Session,
    in_names: [String; 2],
    out_name: String,
    out_dim: usize,
    /// `false` is `[1, time, dimension]`; `true` is `[1, dimension, time]`.
    dimension_first: bool,
    precision: EncoderPrecision,
    assignment_evidence: Option<CudaAssignmentEvidence>,
}

impl OrtEncoder {
    /// Opens the selected CTC encoder artifact for one resolved provider plan.
    pub fn ctc(
        pack: &ModelPackage,
        plan: &ProviderPlan,
        precision: EncoderPrecision,
    ) -> Result<Self, String> {
        check_artifact_format(pack, plan)?;
        let artifact = pack
            .ctc_encoder(precision)
            .map_err(|error| error.to_string())?;
        Self::open(artifact, plan, EncoderRole::Ctc)
    }

    /// Opens the selected RNN-T encoder artifact for one resolved provider plan.
    pub fn rnnt(
        pack: &ModelPackage,
        plan: &ProviderPlan,
        precision: EncoderPrecision,
    ) -> Result<Self, String> {
        check_artifact_format(pack, plan)?;
        let artifact = pack
            .rnnt_encoder(precision)
            .map_err(|error| error.to_string())?;
        Self::open(artifact, plan, EncoderRole::Rnnt)
    }

    /// Opens one typed encoder artifact only through its declared process role.
    fn open(
        artifact: EncoderArtifact,
        plan: &ProviderPlan,
        role: EncoderRole,
    ) -> Result<Self, String> {
        let path = artifact.artifact().path();
        let tensor_contract = artifact.tensor_contract();
        let out_dim = artifact.output_dimension();
        let precision = artifact.precision();
        let dimension_first = match artifact.output_layout() {
            OutputLayout::TimeThenDimension => false,
            OutputLayout::DimensionThenTime => true,
        };
        if plan.device() == Device::Cuda {
            plan.preflight_cuda_role(role)?;
        }
        let mut builder = Session::builder().map_err(|error| error.to_string())?;
        if plan.config.memory_pattern == crate::provider::MemoryPattern::Disabled {
            builder = builder
                .with_memory_pattern(false)
                .map_err(|error| error.to_string())?;
        }
        if let Some(threads) = plan.config.intra_threads {
            builder = builder
                .with_intra_threads(threads.get())
                .map_err(|error| error.to_string())?;
        }

        // CPU uses the built-in CPU provider. CUDA registers only CUDA and deliberately retains
        // CPU placement while assignment evidence is collected; TensorRT alone disables CPU
        // fallback because it has no CUDA-assignment observation contract.
        match plan.device() {
            Device::Cpu => {}
            Device::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    builder = builder
                        .with_config_entry("session.record_ep_graph_assignment_info", "1")
                        .map_err(|error| error.to_string())?
                        .with_execution_providers([Self::cuda_ep(plan.config.cuda_arena)
                            .build()
                            .error_on_failure()])
                        .map_err(|error| error.to_string())?;
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(
                        "cuda EP is unavailable in this build; rebuild with --features cuda".into(),
                    );
                }
            }
            Device::Tensorrt => {
                #[cfg(feature = "tensorrt")]
                {
                    let mut tensorrt = ort::ep::TensorRT::default().with_fp16(matches!(
                        precision,
                        EncoderPrecision::Fp16Io32 | EncoderPrecision::Fp16
                    ));
                    if let Some(directory) = &plan.config.tensorrt.cache_dir {
                        tensorrt = tensorrt
                            .with_engine_cache(true)
                            .with_engine_cache_path(directory.display().to_string());
                    }
                    if let Some(profile) = &plan.config.tensorrt.profile {
                        tensorrt = tensorrt
                            .with_profile_min_shapes(profile.min.ort_value())
                            .with_profile_opt_shapes(profile.opt.ort_value())
                            .with_profile_max_shapes(profile.max.ort_value());
                    }
                    builder = builder
                        .with_execution_providers([tensorrt.build().error_on_failure()])
                        .map_err(|error| error.to_string())?
                        .with_disable_cpu_fallback()
                        .map_err(|error| error.to_string())?;
                }
                #[cfg(not(feature = "tensorrt"))]
                {
                    return Err("tensorrt EP requires building with --features tensorrt".into());
                }
            }
        }

        let session = builder
            .commit_from_file(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let assignment_evidence = match plan.device() {
            Device::Cuda => {
                let evidence = ort_assignment::inspect(&session)?;
                plan.verify_cuda_assignment(role, evidence.fingerprint())?;
                eprintln!(
                    "CUDA encoder {role:?} CPU operator assignments: {:?}",
                    evidence.cpu_operator_types()
                );
                Some(evidence)
            }
            Device::Cpu | Device::Tensorrt => None,
        };
        Ok(Self {
            session,
            in_names: [
                tensor_contract.data_input().to_owned(),
                tensor_contract.length_input().to_owned(),
            ],
            out_name: tensor_contract.data_output().to_owned(),
            out_dim,
            dimension_first,
            precision,
            assignment_evidence,
        })
    }

    /// CUDA EP configuration. Heuristic convolution search avoids an exhaustive first shape search.
    #[cfg(feature = "cuda")]
    fn cuda_ep(arena: CudaArena) -> CUDA {
        let mut cuda = CUDA::default().with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic);
        if arena == CudaArena::SameAsRequested {
            cuda = cuda.with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested);
        }
        cuda
    }

    /// The selected encoder graph precision, including its native I/O boundary.
    pub const fn precision(&self) -> EncoderPrecision {
        self.precision
    }

    /// Returns owned CUDA graph-assignment evidence after the session passed its startup gate.
    pub const fn assignment_evidence(&self) -> Option<&CudaAssignmentEvidence> {
        self.assignment_evidence.as_ref()
    }
}

pub(crate) enum EncodedRnntWindow {
    F32(Vec<f32>, usize),
    F16(Vec<half::f16>, usize),
}

impl OrtEncoder {
    pub(crate) fn forward_rnnt(
        &mut self,
        features: FeatureMatrixView<'_>,
    ) -> Result<EncodedRnntWindow, String> {
        let mel_bins = features.mel_bins();
        let frames = features.frames();
        let frame_count = i64::try_from(frames)
            .map_err(|_| format!("encoder input length {frames} exceeds i64"))?;
        let length_tensor =
            Tensor::from_array(([1usize], vec![frame_count])).map_err(|error| error.to_string())?;
        if self.precision == EncoderPrecision::Fp16 {
            let values: Vec<half::f16> = features
                .values()
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect();
            if values.iter().any(|value| !value.is_finite()) {
                return Err("encoder FP16 input contains non-finite or out-of-range values".into());
            }
            let feature_tensor = Tensor::from_array(([1usize, mel_bins, frames], values))
                .map_err(|error| error.to_string())?;
            let outputs = self.session.run(ort::inputs![self.in_names[0].as_str() => feature_tensor, self.in_names[1].as_str() => length_tensor]).map_err(|error| error.to_string())?;
            let (shape, data) = outputs[self.out_name.as_str()]
                .try_extract_tensor::<half::f16>()
                .map_err(|error| error.to_string())?;
            validate_f16_output(
                NativeOutputRole::Encoder,
                shape,
                data,
                ExpectedCardinality::ShapeDerived,
            )?;
            let (values, count) = reorder_encoder(shape, data, self.out_dim, self.dimension_first)?;
            Ok(EncodedRnntWindow::F16(values, count))
        } else {
            let feature_tensor =
                Tensor::from_array(([1usize, mel_bins, frames], features.values().to_vec()))
                    .map_err(|error| error.to_string())?;
            let outputs = self.session.run(ort::inputs![self.in_names[0].as_str() => feature_tensor, self.in_names[1].as_str() => length_tensor]).map_err(|error| error.to_string())?;
            let (shape, data) = outputs[self.out_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|error| error.to_string())?;
            validate_f32_output(
                NativeOutputRole::Encoder,
                shape,
                data,
                ExpectedCardinality::ShapeDerived,
            )?;
            let (values, count) = reorder_encoder(shape, data, self.out_dim, self.dimension_first)?;
            Ok(EncodedRnntWindow::F32(values, count))
        }
    }
}

fn reorder_encoder<T: Copy>(
    shape: &[i64],
    data: &[T],
    out_dim: usize,
    dimension_first: bool,
) -> Result<(Vec<T>, usize), String> {
    let dimensions: Vec<usize> = shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension)
                .map_err(|_| format!("encoder output dimension {dimension} is invalid"))
        })
        .collect::<Result<_, _>>()?;
    match (dimension_first, dimensions.as_slice()) {
        (false, [1, output_frames, dimension]) if *dimension == out_dim => {
            Ok((data.to_vec(), *output_frames))
        }
        (true, [1, dimension, output_frames]) if *dimension == out_dim => {
            let mut values = Vec::with_capacity(data.len());
            for frame in 0..*output_frames {
                for dimension_index in 0..*dimension {
                    values.push(data[dimension_index * output_frames + frame]);
                }
            }
            Ok((values, *output_frames))
        }
        _ => Err(format!(
            "encoder: unexpected output shape {dimensions:?}, expected dimension {out_dim}"
        )),
    }
}

impl Encoder for OrtEncoder {
    fn out_dim(&self) -> usize {
        self.out_dim
    }
    fn forward(&mut self, features: FeatureMatrixView<'_>) -> Result<(Vec<f32>, usize), String> {
        match self.forward_rnnt(features)? {
            EncodedRnntWindow::F32(values, frames) => Ok((values, frames)),
            EncodedRnntWindow::F16(values, frames) => {
                Ok((values.into_iter().map(f32::from).collect(), frames))
            }
        }
    }
}

fn check_artifact_format(pack: &ModelPackage, plan: &ProviderPlan) -> Result<(), String> {
    use gigaam_model_package::ArtifactFormat;
    match (plan.device(), pack.artifact_format()) {
        (Device::Tensorrt, ArtifactFormat::TrtEpContext) => Ok(()),
        (Device::Tensorrt, _) => Err("TensorRT requires a prepared EP context package".into()),
        (_, ArtifactFormat::TrtEpContext) => {
            Err("TensorRT EP context package requires TensorRT".into())
        }
        (_, ArtifactFormat::Onnx) => Ok(()),
    }
}
