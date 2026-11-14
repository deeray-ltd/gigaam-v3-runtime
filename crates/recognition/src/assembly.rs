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
use gigaam_model_package::{EncoderPrecision, ModelFamily, ModelPackage, VariantPackage};

/// Precision of the complete RNN-T neural pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RnntPrecisionSelection {
    Fp32,
    Fp16,
}

/// Precision choices for the one selected V3 family.
#[derive(Clone, Copy, Debug)]
pub struct VariantPrecision {
    pub encoder: EncoderPrecision,
    pub rnnt_heads_fp16: bool,
}

/// Precision and decoding policy for one V3 variant.
#[derive(Clone, Copy, Debug)]
pub struct VariantDecodeSelection {
    pub precision: VariantPrecision,
    pub mode: crate::DecodingMode,
}

/// Explicit precision and width for one selected V3 RNN-T beam decoder.
#[derive(Clone, Copy, Debug)]
pub struct RnntBeamSelection {
    /// Precision of the selected encoder and RNN-T heads.
    pub precision: VariantPrecision,
    /// Positive maximum number of retained prefixes; no automatic width is chosen.
    pub width: usize,
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

fn validate_ctc_beam_width(width: std::num::NonZeroUsize, blank: usize) -> Result<(), String> {
    let vocabulary = blank.checked_add(1).ok_or("CTC vocabulary overflow")?;
    width
        .get()
        .checked_mul(vocabulary)
        .and_then(|count| count.checked_add(width.get()))
        .ok_or("CTC candidate count overflow")?;
    Ok(())
}

/// Validates selected-width arithmetic against a V3 package before native work.
pub fn validate_variant_mode(
    pack: &VariantPackage,
    mode: crate::DecodingMode,
) -> Result<(), String> {
    let crate::DecodingMode::Beam { width } = mode else {
        return Ok(());
    };
    match pack.artifacts() {
        gigaam_model_package::VariantArtifacts::Ctc(_) => {
            validate_ctc_beam_width(width, pack.tokenizer().blank_id())
        }
        gigaam_model_package::VariantArtifacts::Rnnt(definition) => {
            let vocabulary = pack
                .tokenizer()
                .blank_id()
                .checked_add(1)
                .ok_or("RNN-T vocabulary overflow")?;
            width
                .get()
                .checked_mul(vocabulary)
                .ok_or("RNN-T candidate count overflow")?;
            definition
                .max_symbols_per_step()
                .checked_add(1)
                .and_then(|count| count.checked_mul(width.get()))
                .ok_or("RNN-T completed count overflow")?;
            Ok(())
        }
    }
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

    /// Constructs the single family declared by a validated V3 package.
    pub fn variant(
        pack: &VariantPackage,
        runtime: &NativeRuntime,
        plan: &ProviderPlan,
        selection: VariantPrecision,
    ) -> Result<Self, String> {
        let precision = selection.encoder;
        let full_rnnt_fp16 = selection.rnnt_heads_fp16;
        match pack.variant().family() {
            ModelFamily::Ctc => {
                if full_rnnt_fp16 {
                    return Err("RNN-T head precision is invalid for CTC".into());
                }
                // Validate tokenizer capability before native session construction.
                let projector = crate::TokenTextProjector::new(
                    pack.load_tokenizer().map_err(|error| error.to_string())?,
                )?;
                let encoder = OrtEncoder::variant(pack, runtime, plan, precision)?;
                let assignment_evidence = encoder.assignment_evidence().cloned();
                Ok(Self(Implementation::Ctc {
                    decoder: Box::new(CtcDecoder::from_variant(pack, encoder, projector)?),
                    precision,
                    assignment_evidence,
                }))
            }
            ModelFamily::Rnnt => Ok(Self(Implementation::Rnnt(Box::new(
                RnntDecoder::from_variant(pack, (runtime, plan), precision, full_rnnt_fp16)?,
            )))),
        }
    }

    pub fn variant_with_mode(
        pack: &VariantPackage,
        runtime: &NativeRuntime,
        plan: &ProviderPlan,
        selection: VariantDecodeSelection,
    ) -> Result<Self, String> {
        validate_variant_mode(pack, selection.mode)?;
        match (pack.variant().family(), selection.mode) {
            (_, crate::DecodingMode::Greedy) => {
                Self::variant(pack, runtime, plan, selection.precision)
            }
            (ModelFamily::Rnnt, crate::DecodingMode::Beam { width }) => Self::variant_rnnt_beam(
                pack,
                runtime,
                plan,
                RnntBeamSelection {
                    precision: selection.precision,
                    width: width.get(),
                },
            ),
            (ModelFamily::Ctc, crate::DecodingMode::Beam { .. }) => {
                if selection.precision.rnnt_heads_fp16 {
                    return Err("RNN-T head precision is invalid for CTC".into());
                }
                let projector = crate::TokenTextProjector::new(
                    pack.load_tokenizer().map_err(|error| error.to_string())?,
                )?;
                let encoder =
                    OrtEncoder::variant(pack, runtime, plan, selection.precision.encoder)?;
                let assignment_evidence = encoder.assignment_evidence().cloned();
                let decoder =
                    CtcDecoder::from_variant(pack, encoder, projector)?.with_mode(selection.mode);
                Ok(Self(Implementation::Ctc {
                    decoder: Box::new(decoder),
                    precision: selection.precision.encoder,
                    assignment_evidence,
                }))
            }
        }
    }

    /// Constructs a V3 RNN-T recognizer with an explicitly selected beam width.
    ///
    /// The selected package supplies the tokenizer and symbol cap. The decoder reuses its
    /// encoder, prediction and joint sessions and returns representative token scores and
    /// emission frames. Beam width one is not guaranteed to match greedy decoding. Native
    /// calls may finish after cooperative cancellation is requested.
    pub fn variant_rnnt_beam(
        pack: &VariantPackage,
        runtime: &NativeRuntime,
        plan: &ProviderPlan,
        selection: RnntBeamSelection,
    ) -> Result<Self, String> {
        Ok(Self(Implementation::Rnnt(Box::new(
            RnntDecoder::from_variant_beam(pack, (runtime, plan), selection)?,
        ))))
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
    fn decode_in_context(
        &mut self,
        features: FeatureMatrixView<'_>,
        detail: crate::contracts::ScoreDetail,
        control: &crate::ExecutionControl,
    ) -> Result<Decoded, String> {
        match &mut self.0 {
            Implementation::Ctc { decoder, .. } => {
                decoder.decode_in_context(features, detail, control)
            }
            Implementation::Rnnt(decoder) => decoder.decode_in_context(features, detail, control),
        }
    }
}

#[cfg(test)]
mod mode_tests {
    #[test]
    fn ctc_candidate_arithmetic_refuses_overflow_without_an_arbitrary_width_limit() {
        let one = std::num::NonZeroUsize::new(1).expect("positive");
        let maximum = std::num::NonZeroUsize::new(usize::MAX).expect("positive");
        assert!(super::validate_ctc_beam_width(one, 256).is_ok());
        assert!(super::validate_ctc_beam_width(maximum, 256).is_err());
    }
}
