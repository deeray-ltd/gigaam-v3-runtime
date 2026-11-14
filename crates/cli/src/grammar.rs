// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! UTF-8 command-line grammar and validated invocation values for the offline CLI.

use gigaam_audio::SampleRate;
use gigaam_model_package::EncoderPrecision;
use gigaam_recognition::RnntPrecisionSelection;
use gigaam_transcription::{
    EndpointRules, EndpointSource, OfflineChannelPolicy, PadPolicy, StreamConfig, StreamLockPolicy,
    VadDurations, VadPadding, VadPolicyConfig, VadProbability, VadThresholds,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::PathBuf;

/// One grammar failure, classified before terminal status projection.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum GrammarFailure {
    /// An operating-system argument cannot be represented by the public UTF-8 grammar.
    NonUtf8Arguments,
    /// A syntactic or scalar command-line value is invalid.
    Syntax(String),
}

/// One complete command invocation with all syntax and scalar values validated once.
#[derive(Clone, Debug)]
pub(crate) enum Invocation {
    Transcribe(TranscribeInvocation),
    Bench(BenchInvocation),
    Stream(StreamInvocation),
    Vad(VadInvocation),
    Dialog(DialogInvocation),
}

/// Provider and ONNX Runtime path choices shared by applicable commands.
#[derive(Clone, Debug)]
pub(crate) struct RuntimeRequest {
    pub(crate) provider: Option<String>,
    pub(crate) variant: Option<String>,
    pub(crate) decoding: Option<String>,
    pub(crate) beam_width: Option<String>,
    pub(crate) ort_dylib: Option<PathBuf>,
}

/// The exact transcript-detail level selected by the user.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WordOutput {
    Text,
    Words,
}

/// The exact channel line projection selected by the user.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChannelOutput {
    Combined,
    Split,
}

/// The exact turn projection selected by the user.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TurnOutput {
    Omitted,
    Included,
}

/// Complete independent batch-output choices from one transcribe invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TranscribeOutput {
    pub(crate) words: WordOutput,
    pub(crate) channels: ChannelOutput,
    pub(crate) turns: TurnOutput,
}

/// Validated batch-transcription input and command choices.
#[derive(Clone, Debug)]
pub(crate) struct TranscribeInvocation {
    pub(crate) input: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) runtime: RuntimeRequest,
    pub(crate) precision: EncoderPrecision,
    pub(crate) ctc_precision: Option<EncoderPrecision>,
    pub(crate) decoder: DecoderChoice,
    pub(crate) rnnt_asserted: bool,
    pub(crate) rnnt_precision: Option<RnntPrecisionSelection>,
    pub(crate) window_seconds: f32,
    pub(crate) overlap_seconds: f32,
    pub(crate) padding: PadPolicy,
    pub(crate) output: TranscribeOutput,
    pub(crate) turn_gap_seconds: f32,
    pub(crate) evidence: Option<PathBuf>,
    pub(crate) evidence_version: EvidenceVersion,
    pub(crate) vad: Option<VadPolicyArguments>,
}

/// The exact decoder family selected for offline batch transcription.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecoderChoice {
    Ctc,
    Rnnt,
}

/// Explicit evidence-file schema; v1 remains the default for existing callers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EvidenceVersion {
    V1,
    V2,
    V3,
}

/// Validated encoder benchmark input and command choices.
#[derive(Clone, Debug)]
pub(crate) struct BenchInvocation {
    pub(crate) audio: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) runtime: RuntimeRequest,
    pub(crate) precision: EncoderPrecision,
    pub(crate) ctc_precision: Option<EncoderPrecision>,
    pub(crate) window_seconds: f32,
    pub(crate) iterations: usize,
    pub(crate) gap_milliseconds: u64,
    pub(crate) alternate_window_seconds: Option<f32>,
}

/// Whether the stream simulator prints each client-visible event as it is applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StreamEventOutput {
    Hidden,
    Shown,
}

/// Whether a successful streaming command reports its bound policy on stderr.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AppliedConfigOutput {
    Hidden,
    Shown,
}

/// Scalar-validated stream controls, bound together at the selected package's model rate.
#[derive(Clone, Debug)]
pub(crate) struct StreamPolicyArguments {
    window_seconds: f32,
    overlap_seconds: f32,
    step_seconds: f32,
    horizon_seconds: f32,
    endpoint: EndpointSource,
    silence_milliseconds: Option<f32>,
    no_speech_milliseconds: Option<f32>,
    vad_threshold: Option<VadProbability>,
    seam_after_seconds: Option<f32>,
    keep_silence_seconds: Option<f32>,
    lock_policy: StreamLockPolicy,
}

impl StreamPolicyArguments {
    pub(crate) fn bind(&self, sample_rate: SampleRate) -> Result<StreamConfig, String> {
        let base = StreamConfig::checked_default(sample_rate)?;
        let mut changes = StreamConfig::timing_changes()
            .with_window_sec(self.window_seconds)?
            .with_overlap_sec(self.overlap_seconds)?
            .with_step_sec(self.step_seconds)?
            .with_horizon_sec(self.horizon_seconds)?;
        if let Some(value) = self.seam_after_seconds {
            changes = changes.with_seam_after_sec(value)?;
        }
        if let Some(value) = self.keep_silence_seconds {
            changes = changes.with_keep_silence_sec(value)?;
        }
        let mut config = changes.apply(base)?.with_lock_policy(self.lock_policy)?;
        let defaults = config.rules();
        let after_speech = match self.silence_milliseconds {
            Some(value) => value / 1_000.0,
            None => defaults.after_speech_seconds(),
        };
        let without_speech = match self.no_speech_milliseconds {
            Some(value) => value / 1_000.0,
            None => defaults.without_speech_seconds(),
        };
        config = config.with_rules(EndpointRules::new(after_speech, without_speech)?)?;
        if let Some(threshold) = self.vad_threshold {
            config = config.with_vad_threshold(threshold)?;
        }
        config.with_endpoint_source(self.endpoint)
    }
}

/// Validated stream-simulator input and command choices.
#[derive(Clone, Debug)]
pub(crate) struct StreamInvocation {
    pub(crate) input: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) runtime: RuntimeRequest,
    pub(crate) precision: EncoderPrecision,
    pub(crate) ctc_precision: Option<EncoderPrecision>,
    pub(crate) chunk_milliseconds: usize,
    pub(crate) stream_policy: StreamPolicyArguments,
    pub(crate) reference: Option<PathBuf>,
    pub(crate) events: StreamEventOutput,
    pub(crate) applied_config: AppliedConfigOutput,
}

/// Validated fixed-domain VAD input and command choices.
#[derive(Clone, Debug)]
pub(crate) struct VadInvocation {
    pub(crate) input: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) runtime: RuntimeRequest,
    pub(crate) policy: VadPolicyArguments,
    pub(crate) channels: VadChannels,
}

/// Scalar-validated VAD policy, bound to the library's checked fixed-rate counts before native work.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VadPolicyArguments {
    speech_threshold: f32,
    silence_threshold: f32,
    minimum_speech_milliseconds: f32,
    minimum_silence_milliseconds: f32,
    speech_padding_milliseconds: f32,
}

impl VadPolicyArguments {
    pub(crate) fn bind(self) -> Result<VadPolicyConfig, String> {
        let thresholds = VadThresholds::new(
            VadProbability::new(self.speech_threshold)?,
            VadProbability::new(self.silence_threshold)?,
        )
        .map_err(|error| format!("VAD thresholds: {error}"))?;
        let durations = VadDurations::new(
            self.minimum_speech_milliseconds,
            self.minimum_silence_milliseconds,
        )
        .map_err(|error| format!("VAD durations: {error}"))?;
        let padding = VadPadding::new(self.speech_padding_milliseconds)
            .map_err(|error| format!("VAD padding: {error}"))?;
        Ok(VadPolicyConfig::new(thresholds, durations, padding))
    }
}

/// Standalone VAD selects physical channels without batch correlation or deduplication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum VadChannels {
    First,
    All,
    Selected(Vec<usize>),
}

impl VadChannels {
    pub(crate) fn is_explicit(&self) -> bool {
        !matches!(self, Self::First)
    }

    pub(crate) fn resolve(&self, source_channels: usize) -> Result<Vec<usize>, String> {
        if source_channels == 0 {
            return Err("VAD audio must contain at least one channel".into());
        }
        let selected = match self {
            Self::First => vec![0],
            Self::All => (0..source_channels).collect(),
            Self::Selected(indices) => indices.clone(),
        };
        if selected.is_empty() || selected.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("--channels must select unique physical indices in ascending order".into());
        }
        for index in &selected {
            if *index >= source_channels {
                return Err(format!(
                    "--channels index {index} is outside source channel count {source_channels}"
                ));
            }
        }
        Ok(selected)
    }
}

/// Validated offline-dialog input and command choices.
#[derive(Clone, Debug)]
pub(crate) struct DialogInvocation {
    pub(crate) ctc_precision: Option<EncoderPrecision>,
    pub(crate) input: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) runtime: RuntimeRequest,
    pub(crate) turn_gap_seconds: f32,
    pub(crate) stream_policy: StreamPolicyArguments,
    pub(crate) channel_policy: OfflineChannelPolicy,
    pub(crate) backchannel_maximum_milliseconds: f32,
    pub(crate) applied_config: AppliedConfigOutput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Command {
    Transcribe,
    Bench,
    Stream,
    Vad,
    Dialog,
}

impl Command {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "transcribe" => Ok(Self::Transcribe),
            "bench" => Ok(Self::Bench),
            "stream" => Ok(Self::Stream),
            "vad" => Ok(Self::Vad),
            "dialog" => Ok(Self::Dialog),
            other => Err(format!("unknown command {other:?}")),
        }
    }

    fn requires_file(self) -> bool {
        self != Self::Bench
    }

    fn value_option(self, option: &str) -> bool {
        if option == "--variant" {
            return true;
        }
        if option == "--decoding" || option == "--beam-width" {
            return matches!(self, Self::Transcribe | Self::Stream | Self::Dialog);
        }
        if option == "--ctc-precision" {
            return match self {
                Self::Transcribe | Self::Bench | Self::Stream | Self::Dialog => true,
                Self::Vad => false,
            };
        }
        match self {
            Self::Transcribe => matches!(
                option,
                "--model"
                    | "--rnnt-precision"
                    | "--ep"
                    | "--ort-dylib"
                    | "--window"
                    | "--overlap"
                    | "--turn-gap"
                    | "--evidence"
                    | "--evidence-version"
                    | "--vad-threshold"
                    | "--vad-neg-threshold"
                    | "--vad-min-speech-ms"
                    | "--vad-min-silence-ms"
                    | "--vad-speech-pad-ms"
            ),
            Self::Bench => matches!(
                option,
                "--audio"
                    | "--model"
                    | "--ep"
                    | "--ort-dylib"
                    | "--window"
                    | "--iters"
                    | "--gap-ms"
                    | "--alt-window"
            ),
            Self::Stream => matches!(
                option,
                "--model"
                    | "--ep"
                    | "--ort-dylib"
                    | "--chunk-ms"
                    | "--step-ms"
                    | "--horizon"
                    | "--window"
                    | "--overlap"
                    | "--endpoint"
                    | "--silence-ms"
                    | "--no-speech-ms"
                    | "--vad-threshold"
                    | "--seam-after"
                    | "--keep-silence"
                    | "--ref"
            ),
            Self::Vad => matches!(
                option,
                "--model"
                    | "--ort-dylib"
                    | "--threshold"
                    | "--neg-threshold"
                    | "--min-speech-ms"
                    | "--min-silence-ms"
                    | "--speech-pad-ms"
                    | "--channels"
            ),
            Self::Dialog => matches!(
                option,
                "--model"
                    | "--ep"
                    | "--ort-dylib"
                    | "--turn-gap"
                    | "--endpoint"
                    | "--window"
                    | "--overlap"
                    | "--step-ms"
                    | "--horizon"
                    | "--silence-ms"
                    | "--no-speech-ms"
                    | "--vad-threshold"
                    | "--seam-after"
                    | "--keep-silence"
                    | "--backchannel-max-ms"
            ),
        }
    }

    fn flag_option(self, option: &str) -> bool {
        match self {
            Self::Transcribe => matches!(
                option,
                "--fp16"
                    | "--rnnt"
                    | "--pad"
                    | "--words"
                    | "--split-channels"
                    | "--turns"
                    | "--vad"
            ),
            Self::Bench => option == "--fp16",
            Self::Stream => {
                matches!(
                    option,
                    "--fp16" | "--lock" | "--events" | "--applied-config"
                )
            }
            Self::Vad => false,
            Self::Dialog => matches!(option, "--no-dedup" | "--lock" | "--applied-config"),
        }
    }
}

struct ParsedArguments {
    input: Option<PathBuf>,
    values: BTreeMap<String, String>,
    flags: BTreeSet<String>,
}

impl ParsedArguments {
    fn value(&self, option: &str) -> Option<&str> {
        self.values.get(option).map(String::as_str)
    }

    fn has_flag(&self, option: &str) -> bool {
        self.flags.contains(option)
    }

    fn required_input(&self, command: &str) -> Result<PathBuf, String> {
        self.input
            .clone()
            .ok_or_else(|| format!("{command} requires an input file"))
    }

    fn runtime(&self) -> RuntimeRequest {
        RuntimeRequest {
            provider: self.value("--ep").map(str::to_owned),
            variant: self.value("--variant").map(str::to_owned),
            decoding: self.value("--decoding").map(str::to_owned),
            beam_width: self.value("--beam-width").map(str::to_owned),
            ort_dylib: self.value("--ort-dylib").map(PathBuf::from),
        }
    }

    fn model(&self) -> PathBuf {
        match self.value("--model") {
            Some(value) => PathBuf::from(value),
            None => PathBuf::from("model"),
        }
    }

    fn ctc_precision(&self) -> Result<Option<EncoderPrecision>, String> {
        match self.value("--ctc-precision") {
            None => Ok(None),
            Some("fp32") => Ok(Some(EncoderPrecision::Fp32)),
            Some("fp16io32") => Ok(Some(EncoderPrecision::Fp16Io32)),
            Some("fp16") => Ok(Some(EncoderPrecision::Fp16)),
            Some(_) => Err("--ctc-precision must be fp32, fp16io32 or fp16".into()),
        }
    }

    fn precision(&self) -> EncoderPrecision {
        match self.has_flag("--fp16") {
            true => EncoderPrecision::Fp16Io32,
            false => EncoderPrecision::Fp32,
        }
    }
}

/// Converts OS arguments to UTF-8 once, then builds one fully typed invocation.
pub(crate) fn parse(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<Invocation, GrammarFailure> {
    let values = arguments
        .into_iter()
        .map(|argument| {
            argument
                .into_string()
                .map_err(|_| GrammarFailure::NonUtf8Arguments)
        })
        .collect::<Result<Vec<_>, _>>()?;
    parse_text(&values).map_err(GrammarFailure::Syntax)
}

fn parse_text(arguments: &[String]) -> Result<Invocation, String> {
    let command_text = arguments
        .first()
        .ok_or_else(|| "a command is required (transcribe|bench|stream|vad|dialog)".to_owned())?;
    let command = Command::parse(command_text)?;
    let parsed = parse_options(command, command_text, arguments)?;
    match command {
        Command::Transcribe => transcribe(parsed),
        Command::Bench => bench(parsed),
        Command::Stream => stream(parsed),
        Command::Vad => vad(parsed),
        Command::Dialog => dialog(parsed),
    }
}

fn parse_options(
    command: Command,
    command_text: &str,
    arguments: &[String],
) -> Result<ParsedArguments, String> {
    let mut position = 1;
    let input = if command.requires_file() {
        let file = arguments
            .get(position)
            .ok_or_else(|| format!("{command_text} requires an input file"))?;
        if file.is_empty() || file.starts_with('-') {
            return Err(format!("{command_text} requires an input file"));
        }
        position += 1;
        Some(PathBuf::from(file))
    } else {
        None
    };

    let mut values = BTreeMap::new();
    let mut flags = BTreeSet::new();
    while let Some(option) = arguments.get(position) {
        if command.value_option(option) {
            if values.contains_key(option) || flags.contains(option) {
                return Err(format!("{option} may be configured only once"));
            }
            let value = arguments
                .get(position + 1)
                .ok_or_else(|| format!("{option} requires a value"))?;
            if value.is_empty() || value.starts_with("--") {
                return Err(format!("{option} requires a non-empty value"));
            }
            values.insert(option.clone(), value.clone());
            position += 2;
            continue;
        }
        if command.flag_option(option) {
            if values.contains_key(option) || !flags.insert(option.clone()) {
                return Err(format!("{option} may be configured only once"));
            }
            position += 1;
            continue;
        }
        if option.starts_with('-') {
            return Err(format!("unknown option {option}"));
        }
        return Err(format!("unexpected positional argument {option:?}"));
    }
    Ok(ParsedArguments {
        input,
        values,
        flags,
    })
}

fn transcribe(parsed: ParsedArguments) -> Result<Invocation, String> {
    let window_seconds = positive_f32(&parsed, "--window", 30.0)?;
    let overlap_seconds = nonnegative_f32(&parsed, "--overlap", 6.0)?;
    if overlap_seconds >= window_seconds {
        return Err("--overlap must be smaller than --window".into());
    }
    let turn_gap_seconds = positive_f32(&parsed, "--turn-gap", 1.0)?;
    let words = match parsed.has_flag("--words") {
        true => WordOutput::Words,
        false => WordOutput::Text,
    };
    let turns = match parsed.has_flag("--turns") {
        true => TurnOutput::Included,
        false => TurnOutput::Omitted,
    };
    if words == WordOutput::Words && turns == TurnOutput::Included {
        return Err("--words conflicts with --turns".into());
    }
    let channels = match parsed.has_flag("--split-channels") {
        true => ChannelOutput::Split,
        false => ChannelOutput::Combined,
    };
    let padding = match parsed.has_flag("--pad") {
        true => PadPolicy::PadToWindow,
        false => PadPolicy::Exact,
    };
    let decoder = match parsed.has_flag("--rnnt") {
        true => DecoderChoice::Rnnt,
        false => DecoderChoice::Ctc,
    };
    if decoder == DecoderChoice::Rnnt && parsed.value("--ctc-precision").is_some() {
        return Err("--ctc-precision does not apply to --rnnt".into());
    }
    let rnnt_precision = match parsed.value("--rnnt-precision") {
        None => None,
        Some("fp32") => Some(RnntPrecisionSelection::Fp32),
        Some("fp16") => Some(RnntPrecisionSelection::Fp16),
        Some(_) => return Err("--rnnt-precision must be fp32 or fp16".into()),
    };
    if rnnt_precision.is_some() && parsed.has_flag("--fp16") {
        return Err("--rnnt-precision conflicts with --fp16".into());
    }
    let evidence = parsed.value("--evidence").map(PathBuf::from);
    let evidence_version = match parsed.value("--evidence-version") {
        None => EvidenceVersion::V1,
        Some(_) if evidence.is_none() => {
            return Err("--evidence-version requires --evidence".into());
        }
        Some("1") => EvidenceVersion::V1,
        Some("2") => EvidenceVersion::V2,
        Some("3") => EvidenceVersion::V3,
        Some(_) => return Err("--evidence-version must be 1, 2 or 3".into()),
    };
    let vad = if parsed.has_flag("--vad") {
        Some(vad_policy(&parsed, VadOptionNames::BATCH)?)
    } else {
        for option in VadOptionNames::BATCH.all() {
            if parsed.value(option).is_some() {
                return Err(format!("{option} requires --vad"));
            }
        }
        None
    };
    Ok(Invocation::Transcribe(TranscribeInvocation {
        input: parsed.required_input("transcribe")?,
        model: parsed.model(),
        runtime: parsed.runtime(),
        precision: parsed.precision(),
        ctc_precision: parsed.ctc_precision()?,
        decoder,
        rnnt_asserted: parsed.has_flag("--rnnt"),
        rnnt_precision,
        window_seconds,
        overlap_seconds,
        padding,
        output: TranscribeOutput {
            words,
            channels,
            turns,
        },
        turn_gap_seconds,
        evidence,
        evidence_version,
        vad,
    }))
}

fn bench(parsed: ParsedArguments) -> Result<Invocation, String> {
    let alternate_window_seconds = match parsed.value("--alt-window") {
        Some(value) => Some(parse_positive_f32(value, "--alt-window")?),
        None => None,
    };
    let audio = match parsed.value("--audio") {
        Some(value) => PathBuf::from(value),
        None => PathBuf::from("fixtures/long_example.wav"),
    };
    Ok(Invocation::Bench(BenchInvocation {
        audio,
        model: parsed.model(),
        runtime: parsed.runtime(),
        precision: parsed.precision(),
        ctc_precision: parsed.ctc_precision()?,
        window_seconds: positive_f32(&parsed, "--window", 30.0)?,
        iterations: positive_usize(&parsed, "--iters", 20)?,
        gap_milliseconds: nonnegative_u64(&parsed, "--gap-ms", 0)?,
        alternate_window_seconds,
    }))
}

fn stream_policy(
    parsed: &ParsedArguments,
    default_horizon: f32,
) -> Result<StreamPolicyArguments, String> {
    let window_seconds = positive_f32(parsed, "--window", 30.0)?;
    let overlap_seconds = nonnegative_f32(parsed, "--overlap", 6.0)?;
    if overlap_seconds >= window_seconds {
        return Err("--overlap must be smaller than --window".into());
    }
    let endpoint = endpoint_source(parsed.value("--endpoint"))?;
    let lock_policy = match parsed.has_flag("--lock") {
        true => StreamLockPolicy::CommitStable,
        false => StreamLockPolicy::Advisory,
    };
    let vad_threshold = parsed
        .value("--vad-threshold")
        .map(|value| {
            if endpoint != EndpointSource::Vad {
                return Err("--vad-threshold requires --endpoint vad".into());
            }
            let value = parse_nonnegative_f32(value, "--vad-threshold")?;
            VadProbability::new(value).map_err(|error| format!("--vad-threshold: {error}"))
        })
        .transpose()?;
    let optional_nonnegative = |option| {
        parsed
            .value(option)
            .map(|value| parse_nonnegative_f32(value, option))
            .transpose()
    };
    Ok(StreamPolicyArguments {
        window_seconds,
        overlap_seconds,
        step_seconds: positive_f32(parsed, "--step-ms", 500.0)? / 1_000.0,
        horizon_seconds: positive_f32(parsed, "--horizon", default_horizon)?,
        endpoint,
        silence_milliseconds: optional_nonnegative("--silence-ms")?,
        no_speech_milliseconds: optional_nonnegative("--no-speech-ms")?,
        vad_threshold,
        seam_after_seconds: optional_nonnegative("--seam-after")?,
        keep_silence_seconds: optional_nonnegative("--keep-silence")?,
        lock_policy,
    })
}

fn applied_config_output(parsed: &ParsedArguments) -> AppliedConfigOutput {
    match parsed.has_flag("--applied-config") {
        true => AppliedConfigOutput::Shown,
        false => AppliedConfigOutput::Hidden,
    }
}

fn stream(parsed: ParsedArguments) -> Result<Invocation, String> {
    let stream_policy = stream_policy(&parsed, 4.0)?;
    let events = match parsed.has_flag("--events") {
        true => StreamEventOutput::Shown,
        false => StreamEventOutput::Hidden,
    };
    Ok(Invocation::Stream(StreamInvocation {
        input: parsed.required_input("stream")?,
        model: parsed.model(),
        runtime: parsed.runtime(),
        precision: parsed.precision(),
        ctc_precision: parsed.ctc_precision()?,
        chunk_milliseconds: positive_usize(&parsed, "--chunk-ms", 100)?,
        stream_policy,
        reference: parsed.value("--ref").map(PathBuf::from),
        events,
        applied_config: applied_config_output(&parsed),
    }))
}

fn vad(parsed: ParsedArguments) -> Result<Invocation, String> {
    Ok(Invocation::Vad(VadInvocation {
        input: parsed.required_input("vad")?,
        model: parsed.model(),
        runtime: parsed.runtime(),
        policy: vad_policy(&parsed, VadOptionNames::STANDALONE)?,
        channels: vad_channels(parsed.value("--channels"))?,
    }))
}

struct VadOptionNames {
    speech: &'static str,
    silence: &'static str,
    minimum_speech: &'static str,
    minimum_silence: &'static str,
    padding: &'static str,
}

impl VadOptionNames {
    const BATCH: Self = Self {
        speech: "--vad-threshold",
        silence: "--vad-neg-threshold",
        minimum_speech: "--vad-min-speech-ms",
        minimum_silence: "--vad-min-silence-ms",
        padding: "--vad-speech-pad-ms",
    };
    const STANDALONE: Self = Self {
        speech: "--threshold",
        silence: "--neg-threshold",
        minimum_speech: "--min-speech-ms",
        minimum_silence: "--min-silence-ms",
        padding: "--speech-pad-ms",
    };

    const fn all(&self) -> [&'static str; 5] {
        [
            self.speech,
            self.silence,
            self.minimum_speech,
            self.minimum_silence,
            self.padding,
        ]
    }
}

fn vad_policy(
    parsed: &ParsedArguments,
    names: VadOptionNames,
) -> Result<VadPolicyArguments, String> {
    let defaults = VadPolicyConfig::default();
    let speech_threshold = bounded_f32(
        parsed,
        names.speech,
        defaults.thresholds().speech().get(),
        0.0,
        1.0,
    )?;
    let silence_threshold = bounded_f32(
        parsed,
        names.silence,
        defaults.thresholds().silence().get(),
        0.0,
        1.0,
    )?;
    if silence_threshold > speech_threshold {
        return Err(format!(
            "{} must not exceed {}",
            names.silence, names.speech
        ));
    }
    Ok(VadPolicyArguments {
        speech_threshold,
        silence_threshold,
        minimum_speech_milliseconds: nonnegative_f32(
            parsed,
            names.minimum_speech,
            defaults.durations().minimum_speech_milliseconds(),
        )?,
        minimum_silence_milliseconds: nonnegative_f32(
            parsed,
            names.minimum_silence,
            defaults.durations().minimum_silence_milliseconds(),
        )?,
        speech_padding_milliseconds: nonnegative_f32(
            parsed,
            names.padding,
            defaults.padding().milliseconds(),
        )?,
    })
}

fn vad_channels(value: Option<&str>) -> Result<VadChannels, String> {
    let Some(value) = value else {
        return Ok(VadChannels::First);
    };
    if value == "all" {
        return Ok(VadChannels::All);
    }
    let mut selected = BTreeSet::new();
    for value in value.split(',') {
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(
                "--channels must be all or comma-separated physical channel indices".into(),
            );
        }
        let index = value
            .parse::<usize>()
            .map_err(|_| format!("--channels: invalid channel index {value:?}"))?;
        if !selected.insert(index) {
            return Err(format!("--channels repeats physical channel index {index}"));
        }
    }
    Ok(VadChannels::Selected(selected.into_iter().collect()))
}

fn dialog(parsed: ParsedArguments) -> Result<Invocation, String> {
    let channel_policy = match parsed.has_flag("--no-dedup") {
        true => OfflineChannelPolicy::Disabled,
        false => OfflineChannelPolicy::DialogDeduplication,
    };
    Ok(Invocation::Dialog(DialogInvocation {
        ctc_precision: parsed.ctc_precision()?,
        input: parsed.required_input("dialog")?,
        model: parsed.model(),
        runtime: parsed.runtime(),
        turn_gap_seconds: positive_f32(&parsed, "--turn-gap", 0.8)?,
        stream_policy: stream_policy(&parsed, 5.0)?,
        channel_policy,
        backchannel_maximum_milliseconds: nonnegative_f32(&parsed, "--backchannel-max-ms", 0.0)?,
        applied_config: applied_config_output(&parsed),
    }))
}

fn endpoint_source(value: Option<&str>) -> Result<EndpointSource, String> {
    match value {
        None | Some("blank") => Ok(EndpointSource::Blank),
        Some("vad") => Ok(EndpointSource::Vad),
        Some(other) => Err(format!("--endpoint must be blank or vad, got {other:?}")),
    }
}

fn positive_f32(parsed: &ParsedArguments, option: &str, default: f32) -> Result<f32, String> {
    match parsed.value(option) {
        Some(value) => parse_positive_f32(value, option),
        None => Ok(default),
    }
}

fn nonnegative_f32(parsed: &ParsedArguments, option: &str, default: f32) -> Result<f32, String> {
    match parsed.value(option) {
        Some(value) => parse_nonnegative_f32(value, option),
        None => Ok(default),
    }
}

fn bounded_f32(
    parsed: &ParsedArguments,
    option: &str,
    default: f32,
    minimum: f32,
    maximum: f32,
) -> Result<f32, String> {
    let value = match parsed.value(option) {
        Some(value) => parse_finite_f32(value, option)?,
        None => default,
    };
    if !(minimum..=maximum).contains(&value) {
        return Err(format!("{option} must be in {minimum}..={maximum}"));
    }
    Ok(value)
}

fn parse_positive_f32(value: &str, option: &str) -> Result<f32, String> {
    let parsed = parse_finite_f32(value, option)?;
    if parsed <= 0.0 {
        return Err(format!("{option} must be finite and greater than zero"));
    }
    Ok(parsed)
}

fn parse_nonnegative_f32(value: &str, option: &str) -> Result<f32, String> {
    let parsed = parse_finite_f32(value, option)?;
    if parsed < 0.0 {
        return Err(format!("{option} must be finite and non-negative"));
    }
    Ok(parsed)
}

fn parse_finite_f32(value: &str, option: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("{option}: invalid value {value:?}"))?;
    if !parsed.is_finite() {
        return Err(format!("{option} must be finite"));
    }
    Ok(parsed)
}

fn positive_usize(parsed: &ParsedArguments, option: &str, default: usize) -> Result<usize, String> {
    match parsed.value(option) {
        Some(value) => {
            let parsed = value
                .parse::<usize>()
                .map_err(|_| format!("{option}: invalid value {value:?}"))?;
            if parsed == 0 {
                return Err(format!("{option} must be greater than zero"));
            }
            Ok(parsed)
        }
        None => Ok(default),
    }
}

fn nonnegative_u64(parsed: &ParsedArguments, option: &str, default: u64) -> Result<u64, String> {
    match parsed.value(option) {
        Some(value) => value
            .parse::<u64>()
            .map_err(|_| format!("{option}: invalid value {value:?}")),
        None => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChannelOutput, Command, EndpointSource, GrammarFailure, Invocation, OfflineChannelPolicy,
        TranscribeOutput, TurnOutput, WordOutput, endpoint_source, parse_text,
    };

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn streaming_arguments(
        command: &str,
        extra: &[&str],
    ) -> (super::StreamPolicyArguments, super::AppliedConfigOutput) {
        let mut values = arguments(&[command, "audio.wav"]);
        values.extend(arguments(extra));
        match parse_text(&values).expect("the fixture streaming arguments are valid") {
            Invocation::Stream(value) => (value.stream_policy, value.applied_config),
            Invocation::Dialog(value) => (value.stream_policy, value.applied_config),
            other => panic!("a streaming command must produce stream or dialog: {other:?}"),
        }
    }

    #[test]
    fn ctc_selector_grammar_is_scoped_and_strict() {
        for command in ["transcribe", "bench", "stream", "dialog"] {
            let mut base = vec![command];
            if command != "bench" {
                base.push("audio.wav");
            }
            for value in ["fp32", "fp16io32", "fp16"] {
                let mut args = base.clone();
                args.extend(["--ctc-precision", value]);
                assert!(parse_text(&arguments(&args)).is_ok());
                args.extend(["--ctc-precision", value]);
                assert!(parse_text(&arguments(&args)).is_err());
            }
            for value in ["", "FP16", " fp16", "unknown"] {
                let mut args = base.clone();
                args.extend(["--ctc-precision", value]);
                assert!(parse_text(&arguments(&args)).is_err());
            }
            let mut missing = base.clone();
            missing.push("--ctc-precision");
            assert!(parse_text(&arguments(&missing)).is_err());
        }
        assert!(
            parse_text(&arguments(&[
                "transcribe",
                "audio.wav",
                "--rnnt",
                "--ctc-precision",
                "fp32"
            ]))
            .is_err()
        );
        assert!(parse_text(&arguments(&["vad", "audio.wav", "--ctc-precision", "fp32"])).is_err());
        assert!(parse_text(&arguments(&["dialog", "audio.wav", "--fp16"])).is_err());
    }

    #[test]
    fn rnnt_precision_option_requires_rnnt_and_excludes_legacy_fp16() {
        assert!(
            parse_text(&arguments(&[
                "transcribe",
                "audio.wav",
                "--rnnt",
                "--rnnt-precision",
                "fp16"
            ]))
            .is_ok()
        );
        assert!(
            parse_text(&arguments(&[
                "transcribe",
                "audio.wav",
                "--rnnt-precision",
                "fp16"
            ]))
            .is_ok()
        );
        assert!(
            parse_text(&arguments(&[
                "transcribe",
                "audio.wav",
                "--rnnt",
                "--fp16",
                "--rnnt-precision",
                "fp32"
            ]))
            .is_err()
        );
        assert!(
            parse_text(&arguments(&[
                "stream",
                "audio.wav",
                "--rnnt-precision",
                "fp16"
            ]))
            .is_err()
        );
    }

    #[test]
    fn streaming_commands_preserve_their_distinct_default_horizons_and_hidden_policy() {
        let rate = gigaam_audio::SampleRate::new(16_000).expect("fixture rate is valid");
        for (command, horizon) in [("stream", 4.0), ("dialog", 5.0)] {
            let (arguments, output) = streaming_arguments(command, &[]);
            let expected = gigaam_transcription::StreamConfig::checked_default(rate)
                .and_then(|config| config.with_horizon_sec(horizon))
                .expect("the established defaults are valid");
            assert_eq!(arguments.bind(rate), Ok(expected));
            assert_eq!(output, super::AppliedConfigOutput::Hidden);
        }
    }

    #[test]
    fn both_streaming_commands_bind_every_control_and_coupled_window_changes() {
        let rate = gigaam_audio::SampleRate::new(16_000).expect("fixture rate is valid");
        for command in ["stream", "dialog"] {
            let (arguments, output) = streaming_arguments(
                command,
                &[
                    "--endpoint",
                    "vad",
                    "--vad-threshold",
                    "0.625",
                    "--window",
                    "1",
                    "--overlap",
                    "0.25",
                    "--step-ms",
                    "125.5",
                    "--horizon",
                    "0.75",
                    "--silence-ms",
                    "125.25",
                    "--no-speech-ms",
                    "250.5",
                    "--seam-after",
                    "0.375",
                    "--keep-silence",
                    "0.125",
                    "--lock",
                    "--applied-config",
                ],
            );
            let actual = arguments.bind(rate).expect("the combined policy is valid");
            assert_eq!(actual.sample_rate(), rate);
            assert_eq!(actual.endpoint_source(), EndpointSource::Vad);
            assert_eq!(actual.vad_threshold().get(), 0.625);
            assert_eq!(actual.window_sec(), 1.0);
            assert_eq!(actual.overlap_sec(), 0.25);
            assert_eq!(actual.step_sec(), 125.5 / 1_000.0);
            assert_eq!(actual.horizon_sec(), 0.75);
            assert_eq!(actual.rules().after_speech_seconds(), 125.25 / 1_000.0);
            assert_eq!(actual.rules().without_speech_seconds(), 250.5 / 1_000.0);
            assert_eq!(actual.seam_after_sec(), 0.375);
            assert_eq!(actual.keep_silence_sec(), 0.125);
            assert_eq!(
                actual.lock_policy(),
                gigaam_transcription::StreamLockPolicy::CommitStable
            );
            assert_eq!(output, super::AppliedConfigOutput::Shown);
        }
    }

    #[test]
    fn blank_endpoint_accepts_zero_silence_rules_and_retention_but_not_vad_threshold() {
        let rate = gigaam_audio::SampleRate::new(8_000).expect("fixture rate is valid");
        for command in ["stream", "dialog"] {
            let (arguments, _) = streaming_arguments(
                command,
                &[
                    "--silence-ms",
                    "0",
                    "--no-speech-ms",
                    "0",
                    "--seam-after",
                    "0",
                    "--keep-silence",
                    "0",
                    "--overlap",
                    "0",
                ],
            );
            let actual = arguments
                .bind(rate)
                .expect("blank endpoint accepts a non-16k rate");
            assert_eq!(actual.rules().after_speech_seconds(), 0.0);
            assert_eq!(actual.rules().without_speech_seconds(), 0.0);
            assert_eq!(actual.seam_after_sec(), 0.0);
            assert_eq!(actual.keep_silence_sec(), 0.0);
            assert_eq!(actual.overlap_sec(), 0.0);
            assert!(parse_text(&arguments_for(command, &["--vad-threshold", "0.5"])).is_err());
        }
    }

    fn arguments_for(command: &str, extra: &[&str]) -> Vec<String> {
        let mut values = arguments(&[command, "audio.wav"]);
        values.extend(arguments(extra));
        values
    }

    #[test]
    fn streaming_controls_refuse_invalid_scalars_duplicates_and_bound_sample_counts() {
        let rate = gigaam_audio::SampleRate::new(16_000).expect("fixture rate is valid");
        for command in ["stream", "dialog"] {
            for extra in [
                vec!["--window", "0"],
                vec!["--overlap", "30"],
                vec!["--step-ms", "0"],
                vec!["--horizon", "0"],
                vec!["--silence-ms", "-1"],
                vec!["--no-speech-ms", "NaN"],
                vec!["--seam-after", "inf"],
                vec!["--keep-silence", "-0.1"],
                vec!["--endpoint", "vad", "--vad-threshold", "1.1"],
                vec!["--endpoint", "vad", "--vad-threshold", "NaN"],
                vec!["--no-speech-ms", "1", "--no-speech-ms", "2"],
                vec!["--applied-config", "--applied-config"],
                vec!["--lock", "false"],
            ] {
                assert!(
                    parse_text(&arguments_for(command, &extra)).is_err(),
                    "{command} {extra:?}"
                );
            }
            for extra in [
                vec!["--keep-silence", "3.4028235e38"],
                vec!["--step-ms", "0.0000001"],
            ] {
                let (arguments, _) = streaming_arguments(command, &extra);
                assert!(arguments.bind(rate).is_err(), "{command} {extra:?}");
            }
            let (arguments, _) = streaming_arguments(command, &["--endpoint", "vad"]);
            assert!(
                arguments
                    .bind(gigaam_audio::SampleRate::new(8_000).expect("fixture rate is valid"))
                    .is_err()
            );
        }
    }

    #[test]
    fn batch_vad_is_opt_in_and_matches_standalone_policy_defaults() {
        let batch = match parse_text(&arguments(&["transcribe", "audio.wav", "--vad"])) {
            Ok(Invocation::Transcribe(value)) => value,
            other => panic!("valid batch VAD must parse: {other:?}"),
        };
        let standalone = match parse_text(&arguments(&["vad", "audio.wav"])) {
            Ok(Invocation::Vad(value)) => value,
            other => panic!("valid standalone VAD must parse: {other:?}"),
        };
        let expected = gigaam_transcription::VadPolicyConfig::default();
        assert_eq!(
            batch.vad.expect("--vad enables a policy").bind(),
            Ok(expected)
        );
        assert_eq!(standalone.policy.bind(), Ok(expected));
        match parse_text(&arguments(&["transcribe", "audio.wav"])) {
            Ok(Invocation::Transcribe(value)) => assert!(value.vad.is_none()),
            other => panic!("default batch command must parse: {other:?}"),
        }
        for option in super::VadOptionNames::BATCH.all() {
            assert!(parse_text(&arguments(&["transcribe", "audio.wav", option, "0.5"])).is_err());
        }
    }

    #[test]
    fn vad_durations_accept_zero_and_fractions_but_refuse_invalid_domains_and_overflow() {
        let parsed = parse_text(&arguments(&[
            "vad",
            "audio.wav",
            "--min-speech-ms",
            "0",
            "--min-silence-ms",
            "0.0625",
            "--speech-pad-ms",
            "0.125",
        ]));
        match parsed {
            Ok(Invocation::Vad(value)) => {
                let policy = value
                    .policy
                    .bind()
                    .expect("finite tiny nonnegative durations are representable");
                assert_eq!(policy.durations().minimum_speech_milliseconds(), 0.0);
                assert_eq!(policy.durations().minimum_silence_milliseconds(), 0.0625);
                assert_eq!(policy.padding().milliseconds(), 0.125);
            }
            other => panic!("zero and fractional milliseconds must parse: {other:?}"),
        }
        for (option, value) in [
            ("--threshold", "NaN"),
            ("--threshold", "1.1"),
            ("--neg-threshold", "0.6"),
            ("--min-speech-ms", "-1"),
            ("--min-silence-ms", "inf"),
            ("--speech-pad-ms", "-0.1"),
        ] {
            assert!(parse_text(&arguments(&["vad", "audio.wav", option, value])).is_err());
        }
        for option in ["--min-speech-ms", "--min-silence-ms", "--speech-pad-ms"] {
            match parse_text(&arguments(&[
                "vad",
                "audio.wav",
                option,
                &f32::MAX.to_string(),
            ])) {
                Ok(Invocation::Vad(value)) => assert!(value.policy.bind().is_err()),
                other => {
                    panic!("finite scalar validation precedes checked sample binding: {other:?}")
                }
            }
        }
    }

    #[test]
    fn physical_vad_selection_is_unique_sorted_and_source_validated() {
        use super::{VadChannels, vad_channels};
        assert_eq!(vad_channels(None), Ok(VadChannels::First));
        assert_eq!(vad_channels(Some("all")), Ok(VadChannels::All));
        let selection = vad_channels(Some("2,0")).expect("distinct physical indices are valid");
        assert_eq!(selection.resolve(3), Ok(vec![0, 2]));
        assert!(selection.resolve(2).is_err());
        assert_eq!(VadChannels::All.resolve(3), Ok(vec![0, 1, 2]));
        assert_eq!(VadChannels::First.resolve(3), Ok(vec![0]));
        for value in ["", "0,0", "0,00", "0,", "-1", "+1", "0, 1", "all,1", "1.0"] {
            assert!(vad_channels(Some(value)).is_err(), "{value:?} must refuse");
        }
    }

    #[test]
    fn command_grammar_accepts_one_complete_form_per_command() {
        for values in [
            vec!["transcribe", "audio.wav"],
            vec!["transcribe", "audio.wav", "--evidence", "out.evidence.txt"],
            vec!["bench", "--iters", "1"],
            vec![
                "stream",
                "audio.wav",
                "--chunk-ms",
                "20",
                "--endpoint",
                "vad",
            ],
            vec!["vad", "audio.wav", "--threshold", "0.5"],
            vec!["dialog", "audio.wav", "--turn-gap", "0.8"],
        ] {
            assert!(
                parse_text(&arguments(&values)).is_ok(),
                "{values:?} must parse"
            );
        }
    }

    #[test]
    fn command_grammar_refuses_unknown_duplicate_missing_and_invalid_values() {
        for values in [
            vec!["transcribe", "audio.wav", "--cpu"],
            vec![
                "transcribe",
                "audio.wav",
                "--window",
                "30",
                "--window",
                "20",
            ],
            vec!["stream", "audio.wav", "--chunk-ms"],
            vec!["stream", "audio.wav", "--horizon", "NaN"],
            vec!["bench", "--iters", "0"],
            vec!["dialog", "audio.wav", "--endpoint", "gpu"],
            vec!["vad", "audio.wav", "--threshold", "1.1"],
            vec!["transcribe", "audio.wav", "--words", "--turns"],
            vec!["transcribe", "audio.wav", "--evidence"],
            vec![
                "transcribe",
                "audio.wav",
                "--evidence",
                "out.evidence.txt",
                "--evidence",
                "again.evidence.txt",
            ],
        ] {
            assert!(
                parse_text(&arguments(&values)).is_err(),
                "invalid invocation {values:?} must be refused"
            );
        }
    }

    #[test]
    fn endpoint_source_refuses_unrecognized_values_without_changing_legal_choices() {
        assert_eq!(endpoint_source(None), Ok(EndpointSource::Blank));
        assert_eq!(endpoint_source(Some("blank")), Ok(EndpointSource::Blank));
        assert_eq!(endpoint_source(Some("vad")), Ok(EndpointSource::Vad));
        assert_eq!(
            endpoint_source(Some("gpu")),
            Err("--endpoint must be blank or vad, got \"gpu\"".into())
        );
    }

    #[test]
    fn parsed_output_choices_remain_independent_and_complete() {
        let invocation = parse_text(&arguments(&[
            "transcribe",
            "audio.wav",
            "--split-channels",
            "--turns",
        ]));
        match invocation {
            Ok(Invocation::Transcribe(invocation)) => {
                assert_eq!(
                    invocation.output,
                    TranscribeOutput {
                        words: WordOutput::Text,
                        channels: ChannelOutput::Split,
                        turns: TurnOutput::Included,
                    }
                );
            }
            Ok(_) => panic!("transcribe grammar must produce a transcribe invocation"),
            Err(error) => panic!("valid transcribe grammar must succeed: {error}"),
        }
    }

    #[test]
    fn evidence_option_is_absent_by_default_and_carries_its_value_when_given() {
        match parse_text(&arguments(&["transcribe", "audio.wav"])) {
            Ok(Invocation::Transcribe(invocation)) => assert_eq!(invocation.evidence, None),
            Ok(_) => panic!("transcribe grammar must produce a transcribe invocation"),
            Err(error) => panic!("valid transcribe grammar must succeed: {error}"),
        }
        match parse_text(&arguments(&[
            "transcribe",
            "audio.wav",
            "--evidence",
            "out.evidence.txt",
        ])) {
            Ok(Invocation::Transcribe(invocation)) => {
                assert_eq!(
                    invocation.evidence,
                    Some(std::path::PathBuf::from("out.evidence.txt"))
                );
            }
            Ok(_) => panic!("transcribe grammar must produce a transcribe invocation"),
            Err(error) => panic!("valid transcribe grammar must succeed: {error}"),
        }
    }

    #[test]
    fn evidence_version_is_explicit_strict_and_requires_a_destination() {
        for (suffix, expected) in [
            (vec![], super::EvidenceVersion::V1),
            (vec!["--evidence-version", "1"], super::EvidenceVersion::V1),
            (vec!["--evidence-version", "2"], super::EvidenceVersion::V2),
            (vec!["--evidence-version", "3"], super::EvidenceVersion::V3),
        ] {
            let mut values = vec!["transcribe", "audio.wav", "--evidence", "out.txt"];
            values.extend(suffix);
            match parse_text(&arguments(&values)) {
                Ok(Invocation::Transcribe(value)) => assert_eq!(value.evidence_version, expected),
                other => panic!("supported evidence schema must parse: {other:?}"),
            }
        }
        for values in [
            vec!["transcribe", "audio.wav", "--evidence-version", "2"],
            vec![
                "transcribe",
                "audio.wav",
                "--evidence",
                "out.txt",
                "--evidence-version",
                "4",
            ],
        ] {
            assert!(parse_text(&arguments(&values)).is_err());
        }
    }

    #[test]
    fn dialog_deduplication_flag_maps_to_the_offline_selection_contract() {
        for (values, expected) in [
            (
                vec!["dialog", "audio.wav"],
                OfflineChannelPolicy::DialogDeduplication,
            ),
            (
                vec!["dialog", "audio.wav", "--no-dedup"],
                OfflineChannelPolicy::Disabled,
            ),
        ] {
            match parse_text(&arguments(&values)) {
                Ok(Invocation::Dialog(invocation)) => {
                    assert_eq!(invocation.channel_policy, expected)
                }
                Ok(_) => panic!("dialog grammar must produce a dialog invocation"),
                Err(error) => panic!("valid dialog grammar must succeed: {error}"),
            }
        }
    }

    #[test]
    fn grammar_failure_keeps_utf8_and_syntax_classes_distinct() {
        assert_eq!(
            GrammarFailure::Syntax("syntax".into()),
            GrammarFailure::Syntax("syntax".into())
        );
        assert_ne!(
            GrammarFailure::NonUtf8Arguments,
            GrammarFailure::Syntax("syntax".into())
        );
        assert_eq!(Command::parse("stream"), Ok(Command::Stream));
    }
}
