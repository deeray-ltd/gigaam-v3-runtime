// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! UTF-8 command-line grammar and validated invocation values for the offline CLI.

use gigaam_model_package::EncoderPrecision;
use gigaam_transcription::{EndpointSource, OfflineChannelPolicy, PadPolicy, StreamLockPolicy};
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
    pub(crate) decoder: DecoderChoice,
    pub(crate) window_seconds: f32,
    pub(crate) overlap_seconds: f32,
    pub(crate) padding: PadPolicy,
    pub(crate) output: TranscribeOutput,
    pub(crate) turn_gap_seconds: f32,
}

/// The exact decoder family selected for offline batch transcription.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecoderChoice {
    Ctc,
    Rnnt,
}

/// Validated encoder benchmark input and command choices.
#[derive(Clone, Debug)]
pub(crate) struct BenchInvocation {
    pub(crate) audio: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) runtime: RuntimeRequest,
    pub(crate) precision: EncoderPrecision,
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

/// Validated stream-simulator input and command choices.
#[derive(Clone, Debug)]
pub(crate) struct StreamInvocation {
    pub(crate) input: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) runtime: RuntimeRequest,
    pub(crate) precision: EncoderPrecision,
    pub(crate) chunk_milliseconds: usize,
    pub(crate) step_seconds: f32,
    pub(crate) horizon_seconds: f32,
    pub(crate) window_seconds: f32,
    pub(crate) overlap_seconds: f32,
    pub(crate) endpoint: EndpointSource,
    pub(crate) silence_milliseconds: Option<f32>,
    pub(crate) reference: Option<PathBuf>,
    pub(crate) lock_policy: StreamLockPolicy,
    pub(crate) events: StreamEventOutput,
}

/// Validated fixed-domain VAD input and command choices.
#[derive(Clone, Debug)]
pub(crate) struct VadInvocation {
    pub(crate) input: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) runtime: RuntimeRequest,
    pub(crate) speech_threshold: f32,
    pub(crate) silence_threshold: f32,
    pub(crate) minimum_speech_milliseconds: usize,
    pub(crate) minimum_silence_milliseconds: usize,
    pub(crate) speech_padding_milliseconds: usize,
}

/// Validated offline-dialog input and command choices.
#[derive(Clone, Debug)]
pub(crate) struct DialogInvocation {
    pub(crate) input: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) runtime: RuntimeRequest,
    pub(crate) turn_gap_seconds: f32,
    pub(crate) endpoint: EndpointSource,
    pub(crate) channel_policy: OfflineChannelPolicy,
    pub(crate) backchannel_maximum_milliseconds: f32,
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
        match self {
            Self::Transcribe => matches!(
                option,
                "--model" | "--ep" | "--ort-dylib" | "--window" | "--overlap" | "--turn-gap"
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
            ),
            Self::Dialog => matches!(
                option,
                "--model"
                    | "--ep"
                    | "--ort-dylib"
                    | "--turn-gap"
                    | "--endpoint"
                    | "--backchannel-max-ms"
            ),
        }
    }

    fn flag_option(self, option: &str) -> bool {
        match self {
            Self::Transcribe => matches!(
                option,
                "--fp16" | "--rnnt" | "--pad" | "--words" | "--split-channels" | "--turns"
            ),
            Self::Bench => option == "--fp16",
            Self::Stream => matches!(option, "--fp16" | "--lock" | "--events"),
            Self::Vad => false,
            Self::Dialog => option == "--no-dedup",
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
            ort_dylib: self.value("--ort-dylib").map(PathBuf::from),
        }
    }

    fn model(&self) -> PathBuf {
        match self.value("--model") {
            Some(value) => PathBuf::from(value),
            None => PathBuf::from("model"),
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
    Ok(Invocation::Transcribe(TranscribeInvocation {
        input: parsed.required_input("transcribe")?,
        model: parsed.model(),
        runtime: parsed.runtime(),
        precision: parsed.precision(),
        decoder,
        window_seconds,
        overlap_seconds,
        padding,
        output: TranscribeOutput {
            words,
            channels,
            turns,
        },
        turn_gap_seconds,
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
        window_seconds: positive_f32(&parsed, "--window", 30.0)?,
        iterations: positive_usize(&parsed, "--iters", 20)?,
        gap_milliseconds: nonnegative_u64(&parsed, "--gap-ms", 0)?,
        alternate_window_seconds,
    }))
}

fn stream(parsed: ParsedArguments) -> Result<Invocation, String> {
    let window_seconds = positive_f32(&parsed, "--window", 30.0)?;
    let overlap_seconds = nonnegative_f32(&parsed, "--overlap", 6.0)?;
    if overlap_seconds >= window_seconds {
        return Err("--overlap must be smaller than --window".into());
    }
    let endpoint = endpoint_source(parsed.value("--endpoint"))?;
    let lock_policy = match parsed.has_flag("--lock") {
        true => StreamLockPolicy::CommitStable,
        false => StreamLockPolicy::Advisory,
    };
    let events = match parsed.has_flag("--events") {
        true => StreamEventOutput::Shown,
        false => StreamEventOutput::Hidden,
    };
    Ok(Invocation::Stream(StreamInvocation {
        input: parsed.required_input("stream")?,
        model: parsed.model(),
        runtime: parsed.runtime(),
        precision: parsed.precision(),
        chunk_milliseconds: positive_usize(&parsed, "--chunk-ms", 100)?,
        step_seconds: positive_f32(&parsed, "--step-ms", 500.0)? / 1_000.0,
        horizon_seconds: positive_f32(&parsed, "--horizon", 4.0)?,
        window_seconds,
        overlap_seconds,
        endpoint,
        silence_milliseconds: parsed
            .value("--silence-ms")
            .map(|value| parse_nonnegative_f32(value, "--silence-ms"))
            .transpose()?,
        reference: parsed.value("--ref").map(PathBuf::from),
        lock_policy,
        events,
    }))
}

fn vad(parsed: ParsedArguments) -> Result<Invocation, String> {
    let speech_threshold = bounded_f32(&parsed, "--threshold", 0.5, 0.0, 1.0)?;
    let silence_threshold = bounded_f32(&parsed, "--neg-threshold", 0.35, 0.0, 1.0)?;
    if silence_threshold > speech_threshold {
        return Err("--neg-threshold must not exceed --threshold".into());
    }
    Ok(Invocation::Vad(VadInvocation {
        input: parsed.required_input("vad")?,
        model: parsed.model(),
        runtime: parsed.runtime(),
        speech_threshold,
        silence_threshold,
        minimum_speech_milliseconds: positive_usize(&parsed, "--min-speech-ms", 250)?,
        minimum_silence_milliseconds: positive_usize(&parsed, "--min-silence-ms", 100)?,
        speech_padding_milliseconds: nonnegative_usize(&parsed, "--speech-pad-ms", 30)?,
    }))
}

fn dialog(parsed: ParsedArguments) -> Result<Invocation, String> {
    let channel_policy = match parsed.has_flag("--no-dedup") {
        true => OfflineChannelPolicy::Disabled,
        false => OfflineChannelPolicy::DialogDeduplication,
    };
    Ok(Invocation::Dialog(DialogInvocation {
        input: parsed.required_input("dialog")?,
        model: parsed.model(),
        runtime: parsed.runtime(),
        turn_gap_seconds: positive_f32(&parsed, "--turn-gap", 0.8)?,
        endpoint: endpoint_source(parsed.value("--endpoint"))?,
        channel_policy,
        backchannel_maximum_milliseconds: nonnegative_f32(&parsed, "--backchannel-max-ms", 0.0)?,
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

fn nonnegative_usize(
    parsed: &ParsedArguments,
    option: &str,
    default: usize,
) -> Result<usize, String> {
    match parsed.value(option) {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{option}: invalid value {value:?}")),
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

    #[test]
    fn command_grammar_accepts_one_complete_form_per_command() {
        for values in [
            vec!["transcribe", "audio.wav"],
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
