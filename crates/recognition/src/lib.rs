// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Recognition contracts, pure CTC/RNN-T/speech-probability algorithms, ONNX Runtime execution
//! adapters, and dedicated decoder scheduling. Process configuration and transport remain outside
//! this crate.

mod contracts;
mod execution;
mod native_output;
mod scheduler;
mod tokenizer;

mod assembly;
pub mod ctc;
pub mod ctc_beam;
mod ort_assignment;
mod ort_encoder;
mod ort_rnnt;
mod ort_runtime;
mod ort_vad;
mod provider;
pub mod rnnt;
pub mod rnnt_beam;
pub mod vad;

pub use assembly::{
    DirectRecognizer, RnntBeamSelection, RnntPrecisionSelection, VariantDecodeSelection,
    VariantPrecision, validate_variant_mode,
};
pub use contracts::{
    Decoded, DecodedFrameLabelsInput, DecodedTokensInput, DecodingMode, Encoder, FrameLabelKind,
    FrameLabels, FrameRate, ScoreDetail, SilenceCapability, SpeechProbabilityDetector, Token,
    WindowDecoder, Word,
};
pub use execution::{ExecutionControl, ExecutionState};
pub use ort_assignment::CudaAssignmentEvidence;
pub use ort_encoder::OrtEncoder;
pub use ort_runtime::{NativeRuntime, init_runtime};
pub use ort_vad::Vad;
pub use provider::{
    CudaArena, CudaAssignmentEnvironment, CudaAssignmentFingerprint, CudaAssignmentPolicy, Device,
    EncoderRole, MemoryPattern, OrtConfig, OrtEnvironment, ProviderPlan, RequiredEncoderRoles,
};
pub use scheduler::{ExecutionScheduler, ScheduledRecognizer};

pub use tokenizer::{ProjectedWord, TokenTextProjection, TokenTextProjector};
