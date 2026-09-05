// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Offline batch-transcription command adapter.

use crate::composition;
use crate::configuration;
use crate::grammar::{
    ChannelOutput, DecoderChoice, TranscribeInvocation, TranscribeOutput, TurnOutput,
};
use crate::projection::{
    self, EvidenceAudioHeader, EvidenceModelHeader, EvidenceWriter, TranscribeSummary,
    evidence_token,
};
use gigaam_audio::load;
use gigaam_model_package::ModelPackage;
use gigaam_primitives::f64_to_f32;
use gigaam_recognition::{ExecutionControl, FrameRate, RequiredEncoderRoles};
use gigaam_transcription::{
    BatchChannelPolicy, BatchConfig, BatchSetup, MultiChannelBatchOptions, MultiChannelBatchSetup,
    MultiChannelBatchTranscriber, TurnGap, WindowEvidenceObserver, subsampling_factor,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Executes a complete direct batch transcription without a service scheduler.
pub(crate) fn run(invocation: TranscribeInvocation) -> Result<(), String> {
    let required_roles = match invocation.decoder {
        DecoderChoice::Ctc => RequiredEncoderRoles::ctc(),
        DecoderChoice::Rnnt => RequiredEncoderRoles::rnnt(),
    };
    let evidence_path = configuration::evidence_path(invocation.evidence.clone())?;
    if evidence_path.is_some() && invocation.decoder == DecoderChoice::Rnnt {
        return Err("--evidence requires CTC; it does not support --rnnt".into());
    }
    let runtime = configuration::runtime(&invocation.runtime, required_roles)
        .map_err(|error| format!("runtime configuration: {error}"))?;
    let options = MultiChannelBatchOptions::new(
        batch_channel_policy(invocation.output),
        TurnGap::new(invocation.turn_gap_seconds)?,
    );
    let trace_mode = configuration::trace_mode()?;
    let started = Instant::now();
    let frontend_mode =
        configuration::frontend_mode().map_err(|error| format!("frontend: {error}"))?;
    let mut observations = projection::observations(trace_mode);

    let package = composition::open_package(&invocation.model)?;
    let sample_rate = composition::package_sample_rate(&package)?;
    let config = BatchConfig::new(
        sample_rate,
        invocation.window_seconds,
        invocation.overlap_seconds,
        invocation.padding,
    )?;
    composition::initialize_runtime(&runtime)?;
    let frontend = composition::frontend_for_package(&package, frontend_mode)
        .map_err(|error| format!("frontend: {error}"))?;
    let decoder = match invocation.decoder {
        DecoderChoice::Ctc => {
            let decoder = composition::ctc_recognizer(&package, &runtime, invocation.precision)
                .map_err(|error| format!("decoder: {error}"))?;
            projection::emit_ctc_construction_notice(&decoder);
            projection::emit_unverified_cuda_assignment(
                runtime.plan(),
                gigaam_recognition::EncoderRole::Ctc,
                decoder.assignment_evidence(),
            );
            decoder
        }
        DecoderChoice::Rnnt => {
            let decoder = composition::rnnt_recognizer(&package, &runtime, invocation.precision)
                .map_err(|error| format!("RNNT decoder: {error}"))?;
            projection::emit_unverified_cuda_assignment(
                runtime.plan(),
                gigaam_recognition::EncoderRole::Rnnt,
                decoder.assignment_evidence(),
            );
            decoder
        }
    };

    let evidence: Option<Arc<EvidenceWriter>> = match &evidence_path {
        Some(path) => Some(Arc::new(open_evidence_writer(
            path,
            &invocation.model,
            &package,
        )?)),
        None => None,
    };
    if let Some(evidence) = &evidence {
        observations =
            observations.with_evidence(Arc::clone(evidence) as Arc<dyn WindowEvidenceObserver>);
    }

    let mut transcriber = MultiChannelBatchTranscriber::new(MultiChannelBatchSetup::new(
        BatchSetup {
            frontend,
            decoder,
            config,
            control: ExecutionControl::without_deadline(),
            observations,
        },
        options,
    ))?;
    transcriber
        .warmup()
        .map_err(|error| format!("warmup: {error}"))?;
    let loading_elapsed = started.elapsed();

    let loaded = load(&invocation.input)?;
    let source_rate = loaded.sample_rate().hertz();
    let audio = composition::resample_to(loaded, config.sample_rate())
        .map_err(|error| format!("resample: {error}"))?;
    if let Some(evidence) = &evidence {
        let audio_name = evidence_token(&invocation.input, "audio")?;
        evidence.write_audio_header(EvidenceAudioHeader {
            audio_name: &audio_name,
            sample_rate: audio.sample_rate().hertz(),
            samples: audio.frames(),
            total_seconds: audio.duration_seconds(),
        })?;
    }
    let inference_started = Instant::now();
    let result = transcriber
        .transcribe(&audio)
        .map_err(|error| error.to_string())?;
    let inference_elapsed = inference_started.elapsed();
    if let Some(evidence) = &evidence {
        evidence.finish()?;
    }
    projection::transcribe(TranscribeSummary {
        result: &result,
        audio: &audio,
        source_rate,
        config,
        output: invocation.output,
        precision: invocation.precision,
        window_seconds: invocation.window_seconds,
        overlap_seconds: invocation.overlap_seconds,
        loading_elapsed,
        inference_elapsed,
    });
    Ok(())
}

/// Opens the evidence file and writes its `evidence v1` and `model` header lines from the
/// package's own typed frontend and CTC definitions. The subsampling factor is derived, never a
/// hard-coded constant, through the exact function the batch window loop uses internally.
fn open_evidence_writer(
    path: &Path,
    model_path: &Path,
    package: &ModelPackage,
) -> Result<EvidenceWriter, String> {
    let model_id = evidence_token(model_path, "model")?;
    let frame_rate = FrameRate::new(f64_to_f32(package.frontend().frames_per_second()))
        .map_err(|error| format!("model: {error}"))?;
    let subsampling = subsampling_factor(
        package.frontend().sample_rate(),
        package.frontend().hop_length(),
        frame_rate,
    )
    .map_err(|error| format!("model: {error}"))?;
    let vocabulary = package
        .ctc_vocabulary()
        .map_err(|error| format!("model: {error}"))?;
    let space = vocabulary.iter().position(|piece| piece == "\u{2581}");
    EvidenceWriter::create(
        path,
        EvidenceModelHeader {
            model_id: &model_id,
            frames_per_sec: frame_rate.get(),
            blank: package.ctc().blank_id(),
            vocab_size: vocabulary.len(),
            space,
            subsampling,
        },
    )
}

/// Maps every independent CLI output choice to the shared batch channel-selection policy.
fn batch_channel_policy(output: TranscribeOutput) -> BatchChannelPolicy {
    match (output.channels, output.turns) {
        (ChannelOutput::Combined, TurnOutput::Omitted) => BatchChannelPolicy::single_output(),
        (ChannelOutput::Combined, TurnOutput::Included)
        | (ChannelOutput::Split, TurnOutput::Omitted)
        | (ChannelOutput::Split, TurnOutput::Included) => BatchChannelPolicy::separate_channels(),
    }
}

#[cfg(test)]
mod tests {
    use super::batch_channel_policy;
    use crate::grammar::{ChannelOutput, TranscribeOutput, TurnOutput, WordOutput};
    use gigaam_transcription::BatchChannelPolicy;

    #[test]
    fn cli_output_choices_exhaustively_map_to_the_typed_batch_policy() {
        for (output, expected) in [
            (
                TranscribeOutput {
                    words: WordOutput::Text,
                    channels: ChannelOutput::Combined,
                    turns: TurnOutput::Omitted,
                },
                BatchChannelPolicy::single_output(),
            ),
            (
                TranscribeOutput {
                    words: WordOutput::Text,
                    channels: ChannelOutput::Combined,
                    turns: TurnOutput::Included,
                },
                BatchChannelPolicy::separate_channels(),
            ),
            (
                TranscribeOutput {
                    words: WordOutput::Text,
                    channels: ChannelOutput::Split,
                    turns: TurnOutput::Omitted,
                },
                BatchChannelPolicy::separate_channels(),
            ),
            (
                TranscribeOutput {
                    words: WordOutput::Text,
                    channels: ChannelOutput::Split,
                    turns: TurnOutput::Included,
                },
                BatchChannelPolicy::separate_channels(),
            ),
        ] {
            assert_eq!(batch_channel_policy(output), expected);
        }
    }
}
