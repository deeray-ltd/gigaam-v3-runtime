// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Direct package, native-capability, and frontend composition for the offline CLI.

use crate::configuration::RuntimeConfig;
use crate::projection;
use gigaam_audio::{FrontendMode, FrontendProcessor, SampleRate, resample_audio};
use gigaam_model_package::{EncoderPrecision, ModelPackage};
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

/// Opens exactly the explicit package path selected by the grammar.
pub(crate) fn open_package(path: &Path) -> Result<ModelPackage, String> {
    ModelPackage::open(path).map_err(|error| format!("model: {error}"))
}

/// Binds package frontend metadata to Audio's checked rate type.
pub(crate) fn package_sample_rate(package: &ModelPackage) -> Result<SampleRate, String> {
    SampleRate::from_usize(package.frontend().sample_rate(), "model")
}

/// Initializes the requested dynamic ONNX Runtime library after every applicable policy check.
pub(crate) fn initialize_runtime(runtime: &RuntimeConfig) -> Result<NativeRuntime, String> {
    init_runtime(runtime.dylib())
}

/// Resolves package-owned frontend weights into one typed frontend implementation.
pub(crate) fn frontend_for_package(
    package: &ModelPackage,
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
    package: &ModelPackage,
    runtime: &RuntimeConfig,
    native: &NativeRuntime,
    precision: EncoderPrecision,
) -> Result<DirectRecognizer, String> {
    DirectRecognizer::ctc(package, native, runtime.plan(), precision)
}

/// Builds one exact direct RNN-T recognizer for the selected provider and precision.
pub(crate) fn rnnt_recognizer(
    package: &ModelPackage,
    context: (&RuntimeConfig, &NativeRuntime),
    precision: EncoderPrecision,
    selection: Option<gigaam_recognition::RnntPrecisionSelection>,
) -> Result<DirectRecognizer, String> {
    let (runtime, native) = context;
    match selection {
        Some(selection) => {
            DirectRecognizer::rnnt_with_precision(package, native, runtime.plan(), selection)
        }
        None => DirectRecognizer::rnnt(package, native, runtime.plan(), precision),
    }
}

/// Builds one exact CTC encoder for the selected provider and precision.
pub(crate) fn ctc_encoder(
    package: &ModelPackage,
    runtime: &RuntimeConfig,
    native: &NativeRuntime,
    precision: EncoderPrecision,
) -> Result<OrtEncoder, String> {
    OrtEncoder::ctc(package, native, runtime.plan(), precision)
}

/// Builds the package-selected CPU-only VAD capability.
pub(crate) fn vad(
    package: &ModelPackage,
    native: &NativeRuntime,
    intra_threads: Option<NonZeroUsize>,
) -> Result<Vad, String> {
    Vad::from_pack(package, native, intra_threads)
}

/// Creates channel-local direct-recognition stream capabilities for offline dialog execution.
pub(crate) struct DialogStreamFactory<'a> {
    package: &'a ModelPackage,
    plan: &'a ProviderPlan,
    native: &'a NativeRuntime,
    frontend: Arc<FrontendProcessor>,
    precision: EncoderPrecision,
}

impl<'a> DialogStreamFactory<'a> {
    /// Groups package, provider, and frontend capabilities for one dialog workflow.
    pub(crate) fn new(
        package: &'a ModelPackage,
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
        let decoder = DirectRecognizer::ctc(self.package, self.native, self.plan, self.precision)
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
                Vad::from_pack(self.package, self.native, self.plan.intra_threads())
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
