// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Direct assembly of the current CTC and RNN-T recognizer implementations.

use crate::NativeRuntime;
use crate::contracts::{Decoded, FrameRate, WindowDecoder};
use crate::ctc::{CtcConstructionNotice, CtcDecoder};
use crate::ort_assignment::CudaAssignmentEvidence;
use crate::ort_encoder::OrtEncoder;
use crate::ort_rnnt::RnntDecoder;
use crate::provider::ProviderPlan;
use gigaam_audio::FeatureMatrixView;
use gigaam_model_package::{EncoderPrecision, ModelPackage};

/// Precision of the complete RNN-T neural pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RnntPrecisionSelection {
    Fp32,
    Fp16,
}

/// A concrete selected recognizer. Process adapters choose this once and keep its provider plan
/// outside the decoder interface.
pub struct DirectRecognizer(Implementation);

enum Implementation {
    Ctc {
        decoder: Box<CtcDecoder<OrtEncoder>>,
        precision: EncoderPrecision,
        assignment_evidence: Option<CudaAssignmentEvidence>,
    },
    Rnnt(Box<RnntDecoder>),
}

impl DirectRecognizer {
    pub fn ctc(
        pack: &ModelPackage,
        runtime: &NativeRuntime,
        plan: &ProviderPlan,
        precision: EncoderPrecision,
    ) -> Result<Self, String> {
        let encoder = OrtEncoder::ctc(pack, runtime, plan, precision)?;
        let assignment_evidence = encoder.assignment_evidence().cloned();
        Ok(Self(Implementation::Ctc {
            decoder: Box::new(CtcDecoder::from_pack(pack, encoder)?),
            precision,
            assignment_evidence,
        }))
    }

    pub fn rnnt(
        pack: &ModelPackage,
        runtime: &NativeRuntime,
        plan: &ProviderPlan,
        precision: EncoderPrecision,
    ) -> Result<Self, String> {
        Ok(Self(Implementation::Rnnt(Box::new(
            RnntDecoder::from_pack(pack, (runtime, plan), precision, false)?,
        ))))
    }

    /// Opens a recognizer with the selected precision for all RNN-T neural roles.
    pub fn rnnt_with_precision(
        pack: &ModelPackage,
        runtime: &NativeRuntime,
        plan: &ProviderPlan,
        selection: RnntPrecisionSelection,
    ) -> Result<Self, String> {
        match selection {
            RnntPrecisionSelection::Fp32 => Self::rnnt(pack, runtime, plan, EncoderPrecision::Fp32),
            RnntPrecisionSelection::Fp16 => {
                let precision = if pack.schema_version().value() == 1 {
                    EncoderPrecision::Fp16Io32
                } else {
                    EncoderPrecision::Fp16
                };
                Ok(Self(Implementation::Rnnt(Box::new(
                    RnntDecoder::from_pack(pack, (runtime, plan), precision, true)?,
                ))))
            }
        }
    }

    pub const fn encoder_precision(&self) -> EncoderPrecision {
        match &self.0 {
            Implementation::Ctc { precision, .. } => *precision,
            Implementation::Rnnt(decoder) => decoder.encoder_precision(),
        }
    }

    pub const fn ctc_construction_notice(&self) -> Option<CtcConstructionNotice> {
        match &self.0 {
            Implementation::Ctc { decoder, .. } => decoder.construction_notice(),
            Implementation::Rnnt(_) => None,
        }
    }

    /// Returns CUDA graph-assignment evidence after the selected encoder passed startup verification.
    pub fn assignment_evidence(&self) -> Option<&CudaAssignmentEvidence> {
        match &self.0 {
            Implementation::Ctc {
                assignment_evidence,
                ..
            } => assignment_evidence.as_ref(),
            Implementation::Rnnt(decoder) => decoder.assignment_evidence(),
        }
    }
}

impl WindowDecoder for DirectRecognizer {
    fn silence_capability(&self) -> crate::SilenceCapability {
        match &self.0 {
            Implementation::Ctc { decoder, .. } => decoder.silence_capability(),
            Implementation::Rnnt(decoder) => decoder.silence_capability(),
        }
    }

    fn frame_rate(&self) -> FrameRate {
        match &self.0 {
            Implementation::Ctc { decoder, .. } => decoder.frame_rate(),
            Implementation::Rnnt(decoder) => decoder.frame_rate(),
        }
    }

    fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
        match &mut self.0 {
            Implementation::Ctc { decoder, .. } => decoder.decode(features),
            Implementation::Rnnt(decoder) => decoder.decode(features),
        }
    }

    fn decode_with_detail(
        &mut self,
        features: FeatureMatrixView<'_>,
        detail: crate::contracts::ScoreDetail,
    ) -> Result<Decoded, String> {
        match &mut self.0 {
            Implementation::Ctc { decoder, .. } => decoder.decode_with_detail(features, detail),
            Implementation::Rnnt(decoder) => decoder.decode_with_detail(features, detail),
        }
    }
}
