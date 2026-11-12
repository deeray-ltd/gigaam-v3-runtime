// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Process-stream, diagnostic, trace, and exit-code projection for the offline CLI.

mod evidence;
pub(crate) mod stream;

pub(crate) use evidence::{
    EvidenceAudioHeader, EvidenceModelHeader, EvidenceWriter, evidence_token,
};

use crate::configuration::TraceMode;
use crate::grammar::{
    AppliedConfigOutput, GrammarFailure, TranscribeOutput, TurnOutput, WordOutput,
};
use crate::numeric::count_to_f32;
use gigaam_audio::DecodedAudio;
use gigaam_model_package::EncoderPrecision;
use gigaam_recognition::{CudaAssignmentEvidence, DirectRecognizer, EncoderRole, ProviderPlan};
use gigaam_transcription::{
    BackchannelMark, BatchConfig, DuplicateClassification, MultiChannelBatchResult,
    ObservationMode, OfflineDialogFailure, OfflineDialogFailureOrigin, OfflineDialogResult,
    SpeechSegment, StreamConfig, VadPolicyConfig, WindowTiming, WindowTimingObserver,
    words_to_text,
};
use std::sync::Arc;
use std::time::Duration;

/// Emits optional applied policy after the command's established success projection.
pub(crate) fn applied_stream_config(output: AppliedConfigOutput, config: &StreamConfig) {
    if let Some(record) = stream::applied_config_record(output, config) {
        eprintln!("{record}");
    }
}

/// A terminal CLI error with its externally visible status class.
#[derive(Debug)]
pub(crate) enum CliFailure {
    OsArguments,
    Grammar(String),
    Command(String),
}

impl From<GrammarFailure> for CliFailure {
    fn from(value: GrammarFailure) -> Self {
        match value {
            GrammarFailure::NonUtf8Arguments => Self::OsArguments,
            GrammarFailure::Syntax(message) => Self::Grammar(message),
        }
    }
}

/// Performs the single terminal stderr/status mapping for one completed invocation.
pub(crate) fn terminal(result: Result<(), CliFailure>) {
    match result {
        Ok(()) => {}
        Err(CliFailure::OsArguments) => {
            eprintln!("CLI arguments: CLI arguments must contain UTF-8 text");
            std::process::exit(1);
        }
        Err(CliFailure::Grammar(message)) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
        Err(CliFailure::Command(message)) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    }
}

/// Creates the process-owned synchronous trace observer from a typed trace setting.
pub(crate) fn observations(mode: TraceMode) -> ObservationMode {
    match mode {
        TraceMode::Disabled => ObservationMode::disabled(),
        TraceMode::Enabled => ObservationMode::enabled(Arc::new(CliWindowObserver)),
    }
}

/// Emits the established CTC-construction message at the process boundary.
pub(crate) fn emit_ctc_construction_notice(decoder: &DirectRecognizer) {
    if let Some(notice) = decoder.ctc_construction_notice() {
        eprintln!("{}", notice.message());
    }
}

/// Emits the operator-requested unverified CUDA evidence before any warmup or inference.
pub(crate) fn emit_unverified_cuda_assignment(
    plan: &ProviderPlan,
    role: EncoderRole,
    evidence: Option<&CudaAssignmentEvidence>,
) {
    if let Some(policy) = plan.cuda_assignment_policy()
        && policy.is_allow_unverified()
    {
        let evidence = evidence.expect(
            "allow-unverified CUDA encoder construction must retain graph-assignment evidence",
        );
        eprintln!(
            "# CUDA assignment unverified role={} sha256={} cpu_nodes={} cuda_nodes={}",
            role.as_str(),
            evidence.fingerprint().to_hex(),
            evidence.cpu_assignments(),
            evidence.cuda_assignments()
        );
    }
}

struct CliWindowObserver;

impl WindowTimingObserver for CliWindowObserver {
    fn observe(&self, observation: WindowTiming) {
        eprintln!(
            "{}",
            render_window_timing(
                observation.offset_sec(),
                observation.frames(),
                observation.encoder_seconds(),
            )
        );
    }
}

/// Renders one synchronous per-window trace observation at the process boundary.
fn render_window_timing(offset_seconds: f32, frames: usize, encoder_seconds: f64) -> String {
    format!(
        "#   window {offset_seconds:7.2}s frames {frames:5} encoder {:6.1} ms",
        encoder_seconds * 1_000.0
    )
}

/// Values needed to preserve batch-transcription process output after a successful workflow.
pub(crate) struct TranscribeSummary<'a> {
    pub(crate) result: &'a MultiChannelBatchResult,
    pub(crate) audio: &'a DecodedAudio,
    pub(crate) source_rate: u32,
    pub(crate) config: BatchConfig,
    pub(crate) output: TranscribeOutput,
    pub(crate) precision: EncoderPrecision,
    pub(crate) rnnt_head_precision: Option<gigaam_recognition::RnntPrecisionSelection>,
    pub(crate) window_seconds: f32,
    pub(crate) overlap_seconds: f32,
    pub(crate) loading_elapsed: Duration,
    pub(crate) inference_elapsed: Duration,
    pub(crate) vad_policy: Option<VadPolicyConfig>,
}

/// Emits the exact established transcript and timing projections for batch transcription.
pub(crate) fn transcribe(summary: TranscribeSummary<'_>) {
    let channels = summary.result.channels();
    if summary.output.turns == TurnOutput::Included && channels.len() > 1 {
        for turn in summary.result.turns() {
            println!(
                "[channel {} {:.2}–{:.2}] {}",
                turn.channel(),
                turn.start(),
                turn.end(),
                turn.text()
            );
        }
    } else {
        for channel in channels {
            if channels.len() > 1 {
                println!("[channel {}]", channel.channel());
            }
            println!("{}", words_to_text(channel.words()));
            if summary.output.words == WordOutput::Words {
                for word in channel.words() {
                    println!("  {:8.2} {:8.2}  {}", word.start(), word.end(), word.text());
                }
            }
        }
    }
    let timings = summary.result.timings();
    let dual_mono = summary.result.duplicate() == DuplicateClassification::DualMono;
    if let Some(policy) = summary.vad_policy {
        vad_policy(policy);
        eprintln!("# VAD: {:.3} s", timings.vad_seconds());
    }
    eprintln!(
        "# stages: frontend {:.3} s | encoder {:.3} s | decoder+stitching {:.3} s",
        timings.frontend_seconds(),
        timings.encoder_seconds(),
        timings.decode_seconds()
    );
    eprintln!(
        "# {:.2} s audio ({} Hz → {}), channels {}, dual-mono={}, {} window {}/{} s | load {:.2} s | inference {:.3} s | RTF {:.4}",
        summary.audio.duration_seconds(),
        summary.source_rate,
        summary.config.sample_rate().hertz(),
        summary.result.source_channels().get(),
        dual_mono,
        transcribe_precision_label(summary.precision, summary.rnnt_head_precision),
        summary.window_seconds,
        summary.overlap_seconds,
        summary.loading_elapsed.as_secs_f32(),
        summary.inference_elapsed.as_secs_f32(),
        summary.inference_elapsed.as_secs_f32() / summary.audio.duration_seconds()
    );
}

/// Values needed to project a completed encoder benchmark.
pub(crate) struct BenchmarkSummary {
    pub(crate) window_seconds: f32,
    pub(crate) alternate_window_seconds: Option<f32>,
    pub(crate) precision: EncoderPrecision,
    pub(crate) device: gigaam_recognition::Device,
    pub(crate) gap_milliseconds: u64,
    pub(crate) frontend_median_milliseconds: f64,
    pub(crate) encoder_median_milliseconds: f64,
    pub(crate) encoder_minimum_milliseconds: f64,
    pub(crate) encoder_maximum_milliseconds: f64,
    pub(crate) iterations: usize,
}

/// Emits the established one-line benchmark summary.
pub(crate) fn benchmark(summary: BenchmarkSummary) {
    println!(
        "window {} s (alternates with {:?}), {}, {:?}, pause {} ms: frontend median {:.1} ms | encoder median {:.1} ms (min {:.1}, max {:.1}) | {} iterations",
        summary.window_seconds,
        summary.alternate_window_seconds,
        precision_label(summary.precision),
        summary.device,
        summary.gap_milliseconds,
        summary.frontend_median_milliseconds,
        summary.encoder_median_milliseconds,
        summary.encoder_minimum_milliseconds,
        summary.encoder_maximum_milliseconds,
        summary.iterations
    );
}

/// Values needed to project fixed-domain VAD segments and their summary.
pub(crate) struct VadSummary<'a> {
    pub(crate) segments: &'a [SpeechSegment],
    pub(crate) sample_rate: usize,
    pub(crate) total_samples: usize,
    pub(crate) elapsed: Duration,
    pub(crate) channel: Option<usize>,
}

/// Prints the exact validated fixed-domain policy used by an enabled VAD operation.
pub(crate) fn vad_policy(policy: VadPolicyConfig) {
    eprintln!(
        "# VAD policy: sample_rate={} threshold={} neg_threshold={} min_speech_ms={} min_silence_ms={} speech_pad_ms={}",
        gigaam_recognition::vad::VAD_SR,
        policy.thresholds().speech().get(),
        policy.thresholds().silence().get(),
        policy.durations().minimum_speech_milliseconds(),
        policy.durations().minimum_silence_milliseconds(),
        policy.padding().milliseconds(),
    );
}

pub(crate) fn vad_selection(source_channels: usize, selected: impl Iterator<Item = usize>) {
    let selected = selected
        .map(|index| index.to_string())
        .collect::<Vec<_>>()
        .join(",");
    eprintln!("# VAD channels: source={source_channels} selected={selected}");
}

/// Emits VAD segments followed by the established aggregate timing projection.
pub(crate) fn vad(summary: VadSummary<'_>) -> Result<(), String> {
    if let Some(channel) = summary.channel {
        println!("[channel {channel}]");
    }
    let sample_rate = count_to_f32(summary.sample_rate);
    let mut speech = 0usize;
    for segment in summary.segments {
        let duration = segment
            .end_sample()
            .checked_sub(segment.start_sample())
            .ok_or_else(|| "validated speech segments have ordered sample bounds".to_owned())?;
        println!(
            "{:8.2} – {:8.2}  ({:.2} s)",
            count_to_f32(segment.start_sample()) / sample_rate,
            count_to_f32(segment.end_sample()) / sample_rate,
            count_to_f32(duration) / sample_rate
        );
        speech = speech
            .checked_add(duration)
            .ok_or_else(|| "speech-segment lengths overflow the aggregate".to_owned())?;
    }
    let total = count_to_f32(summary.total_samples) / sample_rate;
    eprintln!(
        "# speech segments: {}, speech {:.2}/{:.2} s ({:.0}%), VAD {:.3} s (RTF {:.4})",
        summary.segments.len(),
        count_to_f32(speech) / sample_rate,
        total,
        100.0 * count_to_f32(speech) / count_to_f32(summary.total_samples.max(1)),
        summary.elapsed.as_secs_f32(),
        summary.elapsed.as_secs_f32() / total.max(1e-6)
    );
    Ok(())
}

/// Emits an offline-dialogue result and its channel summary.
pub(crate) fn dialog(result: &OfflineDialogResult, channels: usize) {
    for turn in result.dialogue() {
        let backchannel = match turn.backchannel() {
            BackchannelMark::Yes => " [bc]",
            BackchannelMark::No => "",
        };
        println!(
            "[channel {} {:.2}–{:.2}]{} {}",
            turn.channel(),
            turn.start(),
            turn.end(),
            backchannel,
            turn.text()
        );
    }
    eprintln!(
        "# channels {} (active {}), turns {}",
        channels,
        result.active_channels().len(),
        result.dialogue().len()
    );
}

/// Restores the established CLI context for an offline-dialogue workflow failure.
pub(crate) fn dialog_failure_message(failure: &OfflineDialogFailure) -> String {
    match failure.origin() {
        OfflineDialogFailureOrigin::InputValidation => failure.error().to_owned(),
        OfflineDialogFailureOrigin::ChannelSelection => {
            format!("channel selection: {}", failure.error())
        }
        OfflineDialogFailureOrigin::Factory => failure.error().to_owned(),
        OfflineDialogFailureOrigin::SessionConstruction => failure.error().to_owned(),
        OfflineDialogFailureOrigin::SessionWarmup => format!("warmup: {}", failure.error()),
        OfflineDialogFailureOrigin::SessionPush => failure.error().to_owned(),
        OfflineDialogFailureOrigin::SessionFlush => failure.error().to_owned(),
        OfflineDialogFailureOrigin::Snapshot => format!("dialog snapshot: {}", failure.error()),
        OfflineDialogFailureOrigin::Dialog => format!("dialog merge: {}", failure.error()),
        OfflineDialogFailureOrigin::ResultValidation => failure.error().to_owned(),
    }
}

fn transcribe_precision_label(
    precision: EncoderPrecision,
    rnnt_heads: Option<gigaam_recognition::RnntPrecisionSelection>,
) -> String {
    let encoder = match precision {
        EncoderPrecision::Fp32 => "fp32",
        EncoderPrecision::Fp16Io32 => "fp16io32",
        EncoderPrecision::Fp16 => "fp16-native",
    };
    match rnnt_heads {
        Some(gigaam_recognition::RnntPrecisionSelection::Fp16) => {
            format!("encoder {encoder}, RNN-T heads fp16")
        }
        Some(gigaam_recognition::RnntPrecisionSelection::Fp32) => {
            format!("encoder {encoder}, RNN-T heads fp32")
        }
        None => format!("encoder {encoder}"),
    }
}

pub(super) fn precision_label(precision: EncoderPrecision) -> &'static str {
    match precision {
        EncoderPrecision::Fp32 => "fp32",
        EncoderPrecision::Fp16Io32 => "fp16",
        EncoderPrecision::Fp16 => "fp16-native",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dialog_failure_message, precision_label, render_window_timing, transcribe_precision_label,
    };
    use gigaam_model_package::EncoderPrecision;
    use gigaam_transcription::{OfflineDialogFailure, OfflineDialogFailureOrigin};

    #[test]
    fn window_timing_trace_projection_preserves_the_exact_line() {
        assert_eq!(
            render_window_timing(12.34, 7, 0.0125),
            "#   window   12.34s frames     7 encoder   12.5 ms"
        );
    }

    #[test]
    fn transcribe_precision_reports_encoder_boundary_and_rnnt_heads() {
        assert_eq!(
            transcribe_precision_label(
                EncoderPrecision::Fp16Io32,
                Some(gigaam_recognition::RnntPrecisionSelection::Fp16)
            ),
            "encoder fp16io32, RNN-T heads fp16"
        );
        assert_eq!(
            transcribe_precision_label(
                EncoderPrecision::Fp16,
                Some(gigaam_recognition::RnntPrecisionSelection::Fp16)
            ),
            "encoder fp16-native, RNN-T heads fp16"
        );
        assert_eq!(
            transcribe_precision_label(
                EncoderPrecision::Fp32,
                Some(gigaam_recognition::RnntPrecisionSelection::Fp32)
            ),
            "encoder fp32, RNN-T heads fp32"
        );
    }

    #[test]
    fn precision_labels_preserve_each_supported_encoder_choice() {
        assert_eq!(precision_label(EncoderPrecision::Fp32), "fp32");
        assert_eq!(precision_label(EncoderPrecision::Fp16Io32), "fp16");
    }

    #[test]
    fn dialog_failure_projection_preserves_the_established_stage_diagnostics() -> Result<(), String>
    {
        for (origin, expected) in [
            (OfflineDialogFailureOrigin::InputValidation, "raw failure"),
            (
                OfflineDialogFailureOrigin::ChannelSelection,
                "channel selection: raw failure",
            ),
            (OfflineDialogFailureOrigin::Factory, "raw failure"),
            (
                OfflineDialogFailureOrigin::SessionConstruction,
                "raw failure",
            ),
            (
                OfflineDialogFailureOrigin::SessionWarmup,
                "warmup: raw failure",
            ),
            (OfflineDialogFailureOrigin::SessionPush, "raw failure"),
            (OfflineDialogFailureOrigin::SessionFlush, "raw failure"),
            (
                OfflineDialogFailureOrigin::Snapshot,
                "dialog snapshot: raw failure",
            ),
            (
                OfflineDialogFailureOrigin::Dialog,
                "dialog merge: raw failure",
            ),
            (OfflineDialogFailureOrigin::ResultValidation, "raw failure"),
        ] {
            let failure = OfflineDialogFailure::new(origin, "raw failure".into(), Vec::new())?;
            assert_eq!(dialog_failure_message(&failure), expected);
        }
        Ok(())
    }
}
