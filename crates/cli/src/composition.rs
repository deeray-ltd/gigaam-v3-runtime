// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Direct package, native-capability, and frontend composition for the offline CLI.

use crate::configuration::RuntimeConfig;
use crate::projection;
use gigaam_audio::{FrontendMode, FrontendProcessor, SampleRate, resample_audio};
use gigaam_model_package::{
    EncoderPrecision, FrontendDefinition, FrontendWeights, ModelPackage, PackageError, VariantId,
    VariantPackage,
};
use gigaam_recognition::{
    DirectRecognizer, ExecutionControl, NativeRuntime, OrtEncoder, ProviderPlan, Vad, init_runtime,
};
use gigaam_transcription::{
    EndpointDetector, EndpointSource, OriginalChannel, StreamChannelFactory, StreamConfig,
    StreamSetup,
};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

pub(crate) enum Package {
    Legacy(Box<ModelPackage>),
    Variant(Box<VariantPackage>),
}
impl Package {
    pub(crate) fn frontend(&self) -> &FrontendDefinition {
        match self {
            Self::Legacy(pack) => pack.frontend(),
            Self::Variant(pack) => pack.frontend(),
        }
    }
    fn frontend_weights(&self) -> Result<FrontendWeights, PackageError> {
        match self {
            Self::Legacy(pack) => pack.frontend_weights(),
            Self::Variant(pack) => pack.frontend_weights(),
        }
    }
    pub(crate) fn ctc_vocabulary(&self) -> Result<Vec<String>, PackageError> {
        match self {
            Self::Legacy(pack) => pack.ctc_vocabulary(),
            Self::Variant(pack) => pack.vocabulary(),
        }
    }
    pub(crate) fn require_ctc(&self) -> Result<(), String> {
        match self {
            Self::Legacy(_) => Ok(()),
            Self::Variant(pack)
                if pack.variant().family() == gigaam_model_package::ModelFamily::Ctc =>
            {
                Ok(())
            }
            Self::Variant(_) => Err("command requires a CTC variant".into()),
        }
    }
    pub(crate) fn space_piece(&self) -> &'static str {
        match self {
            Self::Variant(pack)
                if pack.tokenizer().kind() == gigaam_model_package::TokenizerKind::Characters =>
            {
                " "
            }
            _ => "▁",
        }
    }
    pub(crate) fn blank_id(&self) -> usize {
        match self {
            Self::Legacy(pack) => pack.ctc().blank_id(),
            Self::Variant(pack) => pack.tokenizer().blank_id(),
        }
    }
}

pub(crate) fn resolve_variant(
    request: &crate::grammar::RuntimeRequest,
) -> Result<Option<VariantId>, String> {
    let environment = std::env::var("ASR_MODEL_VARIANT");
    let environment = match environment {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(format!("ASR_MODEL_VARIANT: {error}")),
    };
    let cli = request
        .variant
        .as_deref()
        .map(str::parse::<VariantId>)
        .transpose()
        .map_err(|e| e.to_string())?;
    let env = environment
        .as_deref()
        .map(str::parse::<VariantId>)
        .transpose()
        .map_err(|e| e.to_string())?;
    match (cli, env) {
        (Some(a), Some(b)) if a != b => Err("--variant conflicts with ASR_MODEL_VARIANT".into()),
        (Some(value), _) | (_, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

pub(crate) fn open_package(path: &Path, variant: Option<VariantId>) -> Result<Package, String> {
    match variant {
        None => ModelPackage::open(path)
            .map(Box::new)
            .map(Package::Legacy)
            .map_err(|e| format!("model: {e}")),
        Some(variant) => {
            let pack = VariantPackage::open(path).map_err(|e| format!("model: {e}"))?;
            if pack.variant() != variant {
                return Err("selected variant does not match model package".into());
            }
            Ok(Package::Variant(Box::new(pack)))
        }
    }
}

pub(crate) fn package_sample_rate(package: &Package) -> Result<SampleRate, String> {
    SampleRate::from_usize(package.frontend().sample_rate(), "model")
}

/// Initializes the requested dynamic ONNX Runtime library after every applicable policy check.
pub(crate) fn initialize_runtime(runtime: &RuntimeConfig) -> Result<NativeRuntime, String> {
    init_runtime(runtime.dylib())
}

/// Resolves package-owned frontend weights into one typed frontend implementation.
pub(crate) fn frontend_for_package(
    package: &Package,
    mode: FrontendMode,
) -> Result<Arc<FrontendProcessor>, String> {
    let weights = package
        .frontend_weights()
        .map_err(|error| error.to_string())?;
    Ok(Arc::new(FrontendProcessor::new(
        package.frontend(),
        weights,
        mode,
    )?))
}

/// Resamples Audio-owned decoded input to the already validated model rate.
pub(crate) fn resample_to(
    audio: gigaam_audio::DecodedAudio,
    sample_rate: SampleRate,
) -> Result<gigaam_audio::DecodedAudio, String> {
    resample_audio(audio, sample_rate)
}

/// Builds one exact direct CTC recognizer for the selected provider and precision.
pub(crate) fn ctc_recognizer(
    package: &Package,
    runtime: &RuntimeConfig,
    native: &NativeRuntime,
    precision: EncoderPrecision,
) -> Result<DirectRecognizer, String> {
    match package {
        Package::Legacy(pack) => DirectRecognizer::ctc(pack, native, runtime.plan(), precision),
        Package::Variant(pack) => {
            if pack.variant().family() != gigaam_model_package::ModelFamily::Ctc {
                return Err("command requires a CTC variant".into());
            }
            DirectRecognizer::variant(
                pack,
                native,
                runtime.plan(),
                gigaam_recognition::VariantPrecision {
                    encoder: precision,
                    rnnt_heads_fp16: false,
                },
            )
        }
    }
}

/// Builds one exact direct RNN-T recognizer for the selected provider and precision.
pub(crate) fn rnnt_recognizer(
    package: &Package,
    context: (&RuntimeConfig, &NativeRuntime),
    precision: EncoderPrecision,
    selection: Option<gigaam_recognition::RnntPrecisionSelection>,
) -> Result<DirectRecognizer, String> {
    let (runtime, native) = context;
    match package {
        Package::Legacy(pack) => match selection {
            Some(selection) => {
                DirectRecognizer::rnnt_with_precision(pack, native, runtime.plan(), selection)
            }
            None => DirectRecognizer::rnnt(pack, native, runtime.plan(), precision),
        },
        Package::Variant(pack) => {
            if pack.variant().family() != gigaam_model_package::ModelFamily::Rnnt {
                return Err("command requires an RNN-T variant".into());
            }
            let (encoder, rnnt_heads_fp16) = match selection {
                Some(gigaam_recognition::RnntPrecisionSelection::Fp32) => {
                    (EncoderPrecision::Fp32, false)
                }
                Some(gigaam_recognition::RnntPrecisionSelection::Fp16) => {
                    (EncoderPrecision::Fp16, true)
                }
                None => (precision, false),
            };
            DirectRecognizer::variant(
                pack,
                native,
                runtime.plan(),
                gigaam_recognition::VariantPrecision {
                    encoder,
                    rnnt_heads_fp16,
                },
            )
        }
    }
}

/// Builds one exact CTC encoder for the selected provider and precision.
pub(crate) fn ctc_encoder(
    package: &Package,
    runtime: &RuntimeConfig,
    native: &NativeRuntime,
    precision: EncoderPrecision,
) -> Result<OrtEncoder, String> {
    match package {
        Package::Legacy(pack) => OrtEncoder::ctc(pack, native, runtime.plan(), precision),
        Package::Variant(pack) => {
            if pack.variant().family() != gigaam_model_package::ModelFamily::Ctc {
                return Err("benchmark requires a CTC variant".into());
            }
            OrtEncoder::variant(pack, native, runtime.plan(), precision)
        }
    }
}

/// Builds the package-selected CPU-only VAD capability.
pub(crate) fn vad(
    package: &Package,
    native: &NativeRuntime,
    intra_threads: Option<NonZeroUsize>,
) -> Result<Vad, String> {
    match package {
        Package::Legacy(pack) => Vad::from_pack(pack, native, intra_threads),
        Package::Variant(pack) => Vad::from_variant(pack, native, intra_threads),
    }
}

/// Creates channel-local direct-recognition stream capabilities for offline dialog execution.
pub(crate) struct DialogStreamFactory<'a> {
    package: &'a Package,
    plan: &'a ProviderPlan,
    native: &'a NativeRuntime,
    frontend: Arc<FrontendProcessor>,
    precision: EncoderPrecision,
}

impl<'a> DialogStreamFactory<'a> {
    /// Groups package, provider, and frontend capabilities for one dialog workflow.
    pub(crate) fn new(
        package: &'a Package,
        context: (&'a RuntimeConfig, &'a NativeRuntime),
        frontend: Arc<FrontendProcessor>,
        precision: EncoderPrecision,
    ) -> Self {
        let (runtime, native) = context;
        Self {
            package,
            plan: runtime.plan(),
            frontend,
            native,
            precision,
        }
    }
}

impl StreamChannelFactory for DialogStreamFactory<'_> {
    type Decoder = DirectRecognizer;
    type Detector = Vad;

    fn create_stream(
        &mut self,
        _channel: OriginalChannel,
        config: StreamConfig,
    ) -> Result<StreamSetup<Self::Decoder, Self::Detector>, String> {
        let decoder = match self.package {
            Package::Legacy(pack) => {
                DirectRecognizer::ctc(pack, self.native, self.plan, self.precision)
            }
            Package::Variant(pack) => DirectRecognizer::variant(
                pack,
                self.native,
                self.plan,
                gigaam_recognition::VariantPrecision {
                    encoder: self.precision,
                    rnnt_heads_fp16: false,
                },
            ),
        }
        .map_err(|error| format!("decoder: {error}"))?;
        eprintln!(
            "# dialog CTC encoder: {}",
            projection::precision_label(decoder.encoder_precision())
        );
        projection::emit_ctc_construction_notice(&decoder);
        projection::emit_unverified_cuda_assignment(
            self.plan,
            gigaam_recognition::EncoderRole::Ctc,
            decoder.assignment_evidence(),
        );
        let detector = match config.endpoint_source() {
            EndpointSource::Manual => EndpointDetector::Manual,
            EndpointSource::Blank => EndpointDetector::Blank,
            EndpointSource::Vad => EndpointDetector::Vad(
                vad(self.package, self.native, self.plan.intra_threads())
                    .map_err(|error| format!("vad: {error}"))?,
            ),
        };
        Ok(StreamSetup {
            frontend: Arc::clone(&self.frontend),
            decoder,
            config,
            detector,
            control: ExecutionControl::without_deadline(),
        })
    }
}
