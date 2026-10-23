// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Versioned window-evidence file writer for `transcribe --evidence`.
//!
//! Plain UTF-8 text, ASCII content only, one line per record, fields separated by single
//! spaces: an `evidence v1` line, a `model` line, an `audio` line, then one
//! `window`/`labels`/`mask`/`tokens`/`words`/`end` block per successfully decoded window. Token
//! ids are written as bare integers; SentencePiece vocabulary text never crosses this boundary.
//! The `words` line carries the same token-index spans the decoder's own word rule produced, one
//! `first-last` pair per word in word order, both ends inclusive indices into that window's
//! `tokens` line; a window with no words writes `words` with no pairs after it. Transcription
//! never writes files or reads the environment; this module is the CLI's own process-boundary
//! adapter around the crate's typed `WindowEvidenceObserver` port.
//! Explicit v2 also records physical channels and sample crop origins, applied VAD policy,
//! and successful selected-channel coverage including zero-window channels. Its closing
//! record is structural evidence, not an atomic publication or durability guarantee.

use crate::grammar::EvidenceVersion;
use gigaam_recognition::{Token, vad::VAD_SR};
use gigaam_transcription::{
    DuplicateClassification, MultiChannelBatchResult, VadPolicyConfig, WindowEvidence,
    WindowEvidenceObserver, WindowKind,
};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

/// Extracts one path's final component as a validated evidence-file field: non-empty, ASCII, and free of
/// whitespace, since the format splits fields on single spaces.
pub(crate) fn evidence_token(path: &Path, field: &str) -> Result<String, String> {
    let name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| {
            format!(
                "--evidence {field}: {} has no representable name",
                path.display()
            )
        })?;
    if !name.bytes().all(|byte| byte.is_ascii_graphic()) {
        return Err(format!(
            "--evidence {field}: name {name:?} must be non-empty ASCII with no whitespace"
        ));
    }
    Ok(name.to_owned())
}

/// Named inputs for the one-time `model` header line.
pub(crate) struct EvidenceModelHeader<'a> {
    pub(crate) model_id: &'a str,
    pub(crate) frames_per_sec: f32,
    pub(crate) blank: usize,
    pub(crate) vocab_size: usize,
    pub(crate) space: Option<usize>,
    pub(crate) subsampling: usize,
}

/// Named inputs for the one-time `audio` header line.
pub(crate) struct EvidenceAudioHeader<'a> {
    pub(crate) audio_name: &'a str,
    pub(crate) sample_rate: u32,
    pub(crate) samples: usize,
    pub(crate) total_seconds: f32,
}

/// One decoded window's fields in the exact order the `window`/`labels`/`mask`/`tokens` lines
/// need them. Decoupled from `WindowEvidence` so the wire format can be exercised with literal
/// values in a test, without depending on transcription's internal construction path.
struct WindowRecord<'a> {
    channel: Option<usize>,
    frontend_origin_sample: usize,
    index: usize,
    kind: WindowKind,
    start_sec: f32,
    end_sec: f32,
    first_sample: usize,
    first_feature_frame: usize,
    feature_frames: usize,
    residue: usize,
    origin: usize,
    output_frames: usize,
    keep_until_sec: f32,
    labels: &'a [usize],
    mask: &'a [bool],
    tokens: &'a [Token],
    word_spans: &'a [(usize, usize)],
}

/// A file-writing observer that formats each successfully decoded window in the selected schema.
/// Window writes are best-effort while transcription runs, matching the infallible observer
/// port; `finish` reports the first write failure, if any, and flushes the file.
pub(crate) struct EvidenceWriter {
    writer: Mutex<BufWriter<File>>,
    failure: Mutex<Option<String>>,
    version: EvidenceVersion,
    emitted_windows: Mutex<BTreeMap<usize, usize>>,
}

impl EvidenceWriter {
    /// Creates the evidence file at an already-validated path and writes the selected version and
    /// `model` header lines.
    pub(crate) fn create(
        path: &Path,
        header: EvidenceModelHeader<'_>,
        version: EvidenceVersion,
    ) -> Result<Self, String> {
        let file = File::create(path)
            .map_err(|error| format!("--evidence {}: {error}", path.display()))?;
        let mut writer = BufWriter::new(file);
        write_model_header(&mut writer, &header, version)
            .map_err(|error| format!("--evidence {}: {error}", path.display()))?;
        Ok(Self {
            writer: Mutex::new(writer),
            failure: Mutex::new(None),
            version,
            emitted_windows: Mutex::new(BTreeMap::new()),
        })
    }

    /// Writes the one-time `audio` header line once the resampled input's properties are known.
    pub(crate) fn write_audio_header(&self, header: EvidenceAudioHeader<'_>) -> Result<(), String> {
        let mut writer = self
            .writer
            .lock()
            .expect("evidence writer lock must not be poisoned");
        write_audio_header_line(&mut *writer, &header).map_err(|error| error.to_string())
    }

    pub(crate) fn write_vad_policy(&self, policy: Option<VadPolicyConfig>) -> Result<(), String> {
        if self.version == EvidenceVersion::V1 {
            return Ok(());
        }
        let mut writer = self
            .writer
            .lock()
            .expect("evidence writer lock must not be poisoned");
        write_vad_policy_line(&mut *writer, policy)
            .map_err(|error| format!("--evidence write: {error}"))
    }

    /// Flushes the file and returns the first window-write failure observed during decoding, if
    /// any occurred.
    pub(crate) fn finish(&self, result: &MultiChannelBatchResult) -> Result<(), String> {
        if let Some(failure) = self
            .failure
            .lock()
            .expect("evidence failure lock must not be poisoned")
            .clone()
        {
            return Err(format!("--evidence write: {failure}"));
        }
        let mut writer = self
            .writer
            .lock()
            .expect("evidence writer lock must not be poisoned");
        match self.version {
            EvidenceVersion::V1 => writer.flush(),
            EvidenceVersion::V2 => {
                let selected = result
                    .channels()
                    .iter()
                    .map(|channel| channel.channel())
                    .collect::<Vec<_>>();
                let counts = self
                    .emitted_windows
                    .lock()
                    .expect("evidence count lock must not be poisoned");
                write_completion(
                    &mut *writer,
                    &Coverage {
                        source_channels: result.source_channels().get(),
                        selected: &selected,
                        duplicate: result.duplicate(),
                        emitted_windows: &counts,
                    },
                )
            }
        }
        .map_err(|error| format!("--evidence write: {error}"))
    }

    fn record_failure(&self, error: String) {
        let mut failure = self
            .failure
            .lock()
            .expect("evidence failure lock must not be poisoned");
        if failure.is_none() {
            *failure = Some(error);
        }
    }
}

impl WindowEvidenceObserver for EvidenceWriter {
    fn observe(&self, observation: WindowEvidence) {
        let record = WindowRecord {
            channel: observation.channel().map(|channel| channel.index()),
            frontend_origin_sample: observation.frontend_origin_sample(),
            index: observation.index(),
            kind: observation.kind(),
            start_sec: observation.start_sec(),
            end_sec: observation.end_sec(),
            first_sample: observation.first_sample(),
            first_feature_frame: observation.first_feature_frame(),
            feature_frames: observation.feature_frames(),
            residue: observation.residue(),
            origin: observation.origin(),
            output_frames: observation.output_frames(),
            keep_until_sec: observation.keep_until_sec(),
            labels: observation.labels(),
            mask: observation.mask(),
            tokens: observation.tokens(),
            word_spans: observation.word_spans(),
        };
        let mut writer = self
            .writer
            .lock()
            .expect("evidence writer lock must not be poisoned");
        if let Err(error) = write_window(&mut *writer, &record, self.version) {
            drop(writer);
            self.record_failure(error.to_string());
            return;
        }
        if self.version == EvidenceVersion::V2 {
            let channel = record
                .channel
                .expect("v2 window serialization validates physical identity");
            let mut counts = self
                .emitted_windows
                .lock()
                .expect("evidence count lock must not be poisoned");
            let count = counts.entry(channel).or_insert(0);
            match count.checked_add(1) {
                Some(next) => *count = next,
                None => self.record_failure("emitted window count overflows".into()),
            }
        }
    }
}

fn write_model_header(
    writer: &mut dyn Write,
    header: &EvidenceModelHeader<'_>,
    version: EvidenceVersion,
) -> io::Result<()> {
    let version = match version {
        EvidenceVersion::V1 => "v1",
        EvidenceVersion::V2 => "v2",
    };
    writeln!(writer, "evidence {version}")?;
    writeln!(
        writer,
        "model {} frames_per_sec {} blank {} vocab_size {} space {} subsampling {}",
        header.model_id,
        header.frames_per_sec,
        header.blank,
        header.vocab_size,
        space_field(header.space),
        header.subsampling,
    )
}

fn write_audio_header_line(
    writer: &mut dyn Write,
    header: &EvidenceAudioHeader<'_>,
) -> io::Result<()> {
    writeln!(
        writer,
        "audio {} sample_rate {} samples {} total_seconds {}",
        header.audio_name, header.sample_rate, header.samples, header.total_seconds,
    )
}

fn write_vad_policy_line(
    writer: &mut dyn Write,
    policy: Option<VadPolicyConfig>,
) -> io::Result<()> {
    match policy {
        None => writeln!(writer, "vad disabled"),
        Some(policy) => writeln!(
            writer,
            "vad enabled sample_rate {} threshold {} neg_threshold {} min_speech_ms {} min_silence_ms {} speech_pad_ms {}",
            VAD_SR,
            policy.thresholds().speech().get(),
            policy.thresholds().silence().get(),
            policy.durations().minimum_speech_milliseconds(),
            policy.durations().minimum_silence_milliseconds(),
            policy.padding().milliseconds()
        ),
    }
}

struct Coverage<'a> {
    source_channels: usize,
    selected: &'a [usize],
    duplicate: DuplicateClassification,
    emitted_windows: &'a BTreeMap<usize, usize>,
}

/// A closing record describes successful structural completion, not durable file publication.
/// Previously buffered records must flush first; a late final-write/flush failure still fails CLI.
fn write_completion(writer: &mut dyn Write, coverage: &Coverage<'_>) -> io::Result<()> {
    if coverage.selected.is_empty()
        || coverage
            .selected
            .iter()
            .any(|channel| *channel >= coverage.source_channels)
        || coverage.selected.windows(2).any(|pair| pair[0] >= pair[1])
        || coverage
            .emitted_windows
            .keys()
            .any(|channel| coverage.selected.binary_search(channel).is_err())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "evidence windows do not match successful selected-channel coverage",
        ));
    }
    writer.flush()?;
    let duplicate = match coverage.duplicate {
        DuplicateClassification::DualMono => "dual-mono",
        DuplicateClassification::NotDualMono => "not-dual-mono",
    };
    write!(
        writer,
        "coverage source_channels {} selected_channels",
        coverage.source_channels
    )?;
    for channel in coverage.selected {
        write!(writer, " {channel}")?;
    }
    writeln!(writer, " duplicate {duplicate}")?;
    for channel in coverage.selected {
        let count = coverage.emitted_windows.get(channel).copied().unwrap_or(0);
        writeln!(writer, "channel_windows {channel} emitted {count}")?;
    }
    writer.flush()?;
    writeln!(writer, "complete")?;
    writer.flush()
}

fn write_window(
    writer: &mut dyn Write,
    record: &WindowRecord<'_>,
    version: EvidenceVersion,
) -> io::Result<()> {
    if version == EvidenceVersion::V2 && record.channel.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "evidence v2 requires a physical source channel",
        ));
    }
    write!(writer, "window {}", record.index)?;
    if version == EvidenceVersion::V2 {
        write!(
            writer,
            " channel {} frontend_origin_sample {}",
            record
                .channel
                .expect("v2 channel was validated before writing"),
            record.frontend_origin_sample
        )?;
    }
    writeln!(
        writer,
        " kind {} start_sec {} end_sec {} first_sample {} first_feature_frame {} feature_frames {} residue {} origin {} output_frames {} keep_until_sec {}",
        window_kind_field(record.kind),
        record.start_sec,
        record.end_sec,
        record.first_sample,
        record.first_feature_frame,
        record.feature_frames,
        record.residue,
        record.origin,
        record.output_frames,
        record.keep_until_sec,
    )?;
    write!(writer, "labels")?;
    for label in record.labels {
        write!(writer, " {label}")?;
    }
    writeln!(writer)?;
    write!(writer, "mask")?;
    for silent in record.mask {
        write!(writer, " {}", u8::from(*silent))?;
    }
    writeln!(writer)?;
    write!(writer, "tokens")?;
    for token in record.tokens {
        write!(writer, " {}@{}", token.id(), token.frame())?;
    }
    writeln!(writer)?;
    write!(writer, "words")?;
    for (first, last) in record.word_spans {
        write!(writer, " {first}-{last}")?;
    }
    writeln!(writer)?;
    writeln!(writer, "end")
}

fn window_kind_field(kind: WindowKind) -> &'static str {
    match kind {
        WindowKind::Body => "body",
        WindowKind::Tail => "tail",
    }
}

fn space_field(space: Option<usize>) -> String {
    match space {
        Some(space) => space.to_string(),
        None => "none".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EvidenceAudioHeader, EvidenceModelHeader, WindowRecord, evidence_token,
        write_audio_header_line, write_model_header, write_window,
    };
    use crate::grammar::EvidenceVersion;
    use gigaam_recognition::Token;
    use gigaam_transcription::WindowKind;
    use std::path::Path;

    #[test]
    fn v2_preserves_local_geometry_and_closes_unequal_selected_channel_coverage() {
        let mut output = Vec::new();
        let record = WindowRecord {
            channel: Some(1),
            frontend_origin_sample: 511,
            index: 0,
            kind: WindowKind::Tail,
            start_sec: 0.0319375,
            end_sec: 0.0640625,
            first_sample: 511,
            first_feature_frame: 0,
            feature_frames: 256,
            residue: 0,
            origin: 0,
            output_frames: 2,
            keep_until_sec: 0.0640625,
            labels: &[1, 0],
            mask: &[false, true],
            tokens: &[Token::new(1, 0)],
            word_spans: &[(0, 0)],
        };
        write_window(&mut output, &record, EvidenceVersion::V2)
            .expect("in-memory v2 window must write");
        let counts = std::collections::BTreeMap::from([(1, 2), (2, 1)]);
        super::write_completion(
            &mut output,
            &super::Coverage {
                source_channels: 3,
                selected: &[0, 1, 2],
                duplicate: gigaam_transcription::DuplicateClassification::NotDualMono,
                emitted_windows: &counts,
            },
        )
        .expect("successful coverage includes the silent selected channel");
        let text = String::from_utf8(output).expect("evidence is ASCII");
        let fields = text
            .lines()
            .next()
            .expect("window line exists")
            .split_whitespace()
            .collect::<Vec<_>>();
        assert_eq!(field(&fields, "channel"), "1");
        assert_eq!(field(&fields, "frontend_origin_sample"), "511");
        assert_eq!(field(&fields, "first_sample"), "511");
        assert_eq!(field(&fields, "first_feature_frame"), "0");
        assert!(text.contains("tokens 1@0\nwords 0-0\n"));
        assert!(text.ends_with("coverage source_channels 3 selected_channels 0 1 2 duplicate not-dual-mono\nchannel_windows 0 emitted 0\nchannel_windows 1 emitted 2\nchannel_windows 2 emitted 1\ncomplete\n"));

        let unknown = WindowRecord {
            channel: None,
            ..record
        };
        let mut refused = Vec::new();
        assert!(write_window(&mut refused, &unknown, EvidenceVersion::V2).is_err());
        assert!(
            refused.is_empty(),
            "unknown physical identity refuses before a partial window line"
        );
        let mut old = Vec::new();
        let mut labelled = Vec::new();
        write_window(&mut old, &unknown, EvidenceVersion::V1)
            .expect("v1 retains its existing fields");
        write_window(
            &mut labelled,
            &WindowRecord {
                channel: Some(2),
                frontend_origin_sample: 0,
                ..unknown
            },
            EvidenceVersion::V1,
        )
        .expect("v1 ignores newly available library metadata");
        assert_eq!(old, labelled, "new metadata must not change v1 bytes");
    }

    #[test]
    fn v2_completion_refuses_unknown_channels_and_known_flush_failures() {
        use std::io::{self, Write};
        struct FailFlush {
            bytes: Vec<u8>,
            flushes: usize,
            fail_at: usize,
        }
        impl Write for FailFlush {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.bytes.extend(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                self.flushes += 1;
                if self.flushes == self.fail_at {
                    Err(io::Error::other("injected flush refusal"))
                } else {
                    Ok(())
                }
            }
        }
        let counts = std::collections::BTreeMap::new();
        let coverage = super::Coverage {
            source_channels: 2,
            selected: &[0],
            duplicate: gigaam_transcription::DuplicateClassification::DualMono,
            emitted_windows: &counts,
        };
        for fail_at in [1, 2, 3] {
            let mut writer = FailFlush {
                bytes: Vec::new(),
                flushes: 0,
                fail_at,
            };
            assert!(super::write_completion(&mut writer, &coverage).is_err());
            let text = String::from_utf8(writer.bytes).expect("written fields are ASCII");
            assert_eq!(
                text.ends_with("complete\n"),
                fail_at == 3,
                "prior known failures suppress complete; a late failure is not atomic publication"
            );
        }
        let unknown_counts = std::collections::BTreeMap::from([(1, 1)]);
        let mut output = Vec::new();
        assert!(
            super::write_completion(
                &mut output,
                &super::Coverage {
                    emitted_windows: &unknown_counts,
                    ..coverage
                }
            )
            .is_err()
        );
        assert!(output.is_empty());
    }

    #[test]
    fn v2_applied_policy_records_disabled_and_exact_enabled_values() {
        let mut output = Vec::new();
        super::write_vad_policy_line(&mut output, None).expect("disabled policy line writes");
        super::write_vad_policy_line(
            &mut output,
            Some(gigaam_transcription::VadPolicyConfig::default()),
        )
        .expect("enabled policy line writes");
        assert_eq!(
            String::from_utf8(output).expect("policy is ASCII"),
            "vad disabled\nvad enabled sample_rate 16000 threshold 0.5 neg_threshold 0.35 min_speech_ms 250 min_silence_ms 100 speech_pad_ms 30\n"
        );
    }

    #[test]
    fn evidence_token_accepts_a_plain_ascii_basename_and_refuses_embedded_whitespace() {
        assert_eq!(
            evidence_token(Path::new("/tmp/model"), "model"),
            Ok("model".to_owned())
        );
        assert_eq!(
            evidence_token(Path::new("call.wav"), "audio"),
            Ok("call.wav".to_owned())
        );
        assert!(evidence_token(Path::new("has space.wav"), "audio").is_err());
    }

    /// A minimal evidence-file reader, independent of the writer, that only needs to prove the writer's
    /// output round-trips: it does not need to be a general-purpose parser.
    struct ParsedWindow {
        index: usize,
        kind: String,
        start_sec: f32,
        end_sec: f32,
        first_sample: usize,
        first_feature_frame: usize,
        feature_frames: usize,
        residue: usize,
        origin: usize,
        output_frames: usize,
        keep_until_sec: f32,
        labels: Vec<usize>,
        mask: Vec<bool>,
        tokens: Vec<(usize, usize)>,
        words: Vec<(usize, usize)>,
    }

    fn field<'a>(pairs: &[&'a str], key: &str) -> &'a str {
        let position = pairs
            .iter()
            .position(|candidate| *candidate == key)
            .unwrap_or_else(|| panic!("evidence line must contain field {key:?}"));
        pairs[position + 1]
    }

    fn parse_usize(value: &str) -> usize {
        value
            .parse()
            .unwrap_or_else(|_| panic!("evidence field {value:?} must be an unsigned integer"))
    }

    fn parse_f32(value: &str) -> f32 {
        value
            .parse()
            .unwrap_or_else(|_| panic!("evidence field {value:?} must be a finite number"))
    }

    struct ParsedEvidence {
        model_id: String,
        model_pairs: Vec<(String, String)>,
        audio_name: String,
        audio_pairs: Vec<(String, String)>,
        window: ParsedWindow,
    }

    fn parse_evidence(text: &str) -> ParsedEvidence {
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("evidence v1"));

        let model_line = lines.next().expect("evidence must have a model line");
        let model_fields: Vec<&str> = model_line.split_whitespace().collect();
        assert_eq!(model_fields[0], "model");
        let model_id = model_fields[1].to_owned();
        let model_pairs: Vec<(String, String)> = model_fields[2..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (pair[0].to_owned(), pair[1].to_owned()))
            .collect();

        let audio_line = lines.next().expect("evidence must have an audio line");
        let audio_fields: Vec<&str> = audio_line.split_whitespace().collect();
        assert_eq!(audio_fields[0], "audio");
        let audio_name = audio_fields[1].to_owned();
        let audio_pairs: Vec<(String, String)> = audio_fields[2..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (pair[0].to_owned(), pair[1].to_owned()))
            .collect();

        let window_line = lines.next().expect("evidence must have a window line");
        let window_fields: Vec<&str> = window_line.split_whitespace().collect();
        assert_eq!(window_fields[0], "window");
        let index = parse_usize(window_fields[1]);
        let rest = &window_fields[2..];

        let labels_line = lines.next().expect("evidence must have a labels line");
        let mut labels_fields = labels_line.split_whitespace();
        assert_eq!(labels_fields.next(), Some("labels"));
        let labels: Vec<usize> = labels_fields.map(parse_usize).collect();

        let mask_line = lines.next().expect("evidence must have a mask line");
        let mut mask_fields = mask_line.split_whitespace();
        assert_eq!(mask_fields.next(), Some("mask"));
        let mask: Vec<bool> = mask_fields
            .map(|value| match value {
                "0" => false,
                "1" => true,
                other => panic!("evidence mask value must be 0 or 1, got {other:?}"),
            })
            .collect();

        let tokens_line = lines.next().expect("evidence must have a tokens line");
        let mut tokens_fields = tokens_line.split_whitespace();
        assert_eq!(tokens_fields.next(), Some("tokens"));
        let tokens: Vec<(usize, usize)> = tokens_fields
            .map(|entry| {
                let (id, frame) = entry
                    .split_once('@')
                    .unwrap_or_else(|| panic!("evidence token {entry:?} must contain '@'"));
                (parse_usize(id), parse_usize(frame))
            })
            .collect();

        let words_line = lines.next().expect("evidence must have a words line");
        let mut words_fields = words_line.split_whitespace();
        assert_eq!(words_fields.next(), Some("words"));
        let words: Vec<(usize, usize)> = words_fields
            .map(|entry| {
                let (first, last) = entry
                    .split_once('-')
                    .unwrap_or_else(|| panic!("evidence word span {entry:?} must contain '-'"));
                (parse_usize(first), parse_usize(last))
            })
            .collect();

        assert_eq!(lines.next(), Some("end"));
        assert_eq!(
            lines.next(),
            None,
            "the synthetic record has exactly one window"
        );

        let window = ParsedWindow {
            index,
            kind: field(rest, "kind").to_owned(),
            start_sec: parse_f32(field(rest, "start_sec")),
            end_sec: parse_f32(field(rest, "end_sec")),
            first_sample: parse_usize(field(rest, "first_sample")),
            first_feature_frame: parse_usize(field(rest, "first_feature_frame")),
            feature_frames: parse_usize(field(rest, "feature_frames")),
            residue: parse_usize(field(rest, "residue")),
            origin: parse_usize(field(rest, "origin")),
            output_frames: parse_usize(field(rest, "output_frames")),
            keep_until_sec: parse_f32(field(rest, "keep_until_sec")),
            labels,
            mask,
            tokens,
            words,
        };
        ParsedEvidence {
            model_id,
            model_pairs,
            audio_name,
            audio_pairs,
            window,
        }
    }

    #[test]
    fn writer_output_for_a_synthetic_record_parses_back_exactly() {
        let mut buffer = Vec::new();
        write_model_header(
            &mut buffer,
            &EvidenceModelHeader {
                model_id: "model",
                frames_per_sec: 25.0,
                blank: 256,
                vocab_size: 257,
                space: Some(3),
                subsampling: 4,
            },
            EvidenceVersion::V1,
        )
        .expect("writing the model header to an in-memory buffer must succeed");
        write_audio_header_line(
            &mut buffer,
            &EvidenceAudioHeader {
                audio_name: "call.wav",
                sample_rate: 16_000,
                samples: 1_140_000,
                total_seconds: 71.25,
            },
        )
        .expect("writing the audio header to an in-memory buffer must succeed");
        let record = WindowRecord {
            channel: None,
            frontend_origin_sample: 0,
            index: 2,
            kind: WindowKind::Tail,
            start_sec: 41.25,
            end_sec: 71.25,
            first_sample: 660_000,
            first_feature_frame: 4123,
            feature_frames: 751,
            residue: 3,
            origin: 1030,
            output_frames: 188,
            keep_until_sec: 71.25,
            labels: &[256, 1, 1, 0, 2],
            mask: &[true, false, false, true, false],
            tokens: &[Token::new(1, 1), Token::new(2, 4)],
            word_spans: &[(0, 0), (1, 1)],
        };
        write_window(&mut buffer, &record, EvidenceVersion::V1)
            .expect("writing one window record to an in-memory buffer must succeed");

        let text = String::from_utf8(buffer).expect("evidence output must be UTF-8");
        assert!(text.is_ascii(), "evidence output must be ASCII-only");
        let parsed = parse_evidence(&text);

        assert_eq!(parsed.model_id, "model");
        assert_eq!(
            parsed.model_pairs,
            vec![
                ("frames_per_sec".to_owned(), "25".to_owned()),
                ("blank".to_owned(), "256".to_owned()),
                ("vocab_size".to_owned(), "257".to_owned()),
                ("space".to_owned(), "3".to_owned()),
                ("subsampling".to_owned(), "4".to_owned()),
            ]
        );
        assert_eq!(parsed.audio_name, "call.wav");
        assert_eq!(
            parsed.audio_pairs,
            vec![
                ("sample_rate".to_owned(), "16000".to_owned()),
                ("samples".to_owned(), "1140000".to_owned()),
                ("total_seconds".to_owned(), "71.25".to_owned()),
            ]
        );
        let window = parsed.window;
        assert_eq!(window.index, 2);
        assert_eq!(window.kind, "tail");
        assert_eq!((window.start_sec, window.end_sec), (41.25, 71.25));
        assert_eq!(window.first_sample, 660_000);
        assert_eq!(window.first_feature_frame, 4123);
        assert_eq!(window.feature_frames, 751);
        assert_eq!(window.residue, 3);
        assert_eq!(window.origin, 1030);
        assert_eq!(window.output_frames, 188);
        assert_eq!(window.keep_until_sec, 71.25);
        assert_eq!(window.labels, vec![256, 1, 1, 0, 2]);
        assert_eq!(window.mask, vec![true, false, false, true, false]);
        assert_eq!(window.tokens, vec![(1, 1), (2, 4)]);
        assert_eq!(window.words, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn writer_output_for_a_window_with_no_words_writes_an_empty_words_line() {
        let mut buffer = Vec::new();
        let record = WindowRecord {
            channel: None,
            frontend_origin_sample: 0,
            index: 0,
            kind: WindowKind::Body,
            start_sec: 0.0,
            end_sec: 1.0,
            first_sample: 0,
            first_feature_frame: 0,
            feature_frames: 4,
            residue: 0,
            origin: 0,
            output_frames: 2,
            keep_until_sec: 1.0,
            labels: &[0, 0],
            mask: &[true, true],
            tokens: &[],
            word_spans: &[],
        };
        write_window(&mut buffer, &record, EvidenceVersion::V1)
            .expect("writing a wordless window record to an in-memory buffer must succeed");
        let text = String::from_utf8(buffer).expect("evidence output must be UTF-8");
        assert!(
            text.lines().any(|line| line == "words"),
            "a window with no words must write a bare words line"
        );
    }

    #[test]
    fn model_header_writes_none_for_a_vocabulary_without_a_standalone_space_token() {
        let mut buffer = Vec::new();
        write_model_header(
            &mut buffer,
            &EvidenceModelHeader {
                model_id: "model",
                frames_per_sec: 25.0,
                blank: 0,
                vocab_size: 32,
                space: None,
                subsampling: 4,
            },
            EvidenceVersion::V1,
        )
        .expect("writing the model header to an in-memory buffer must succeed");
        let text = String::from_utf8(buffer).expect("evidence output must be UTF-8");
        let parsed = parse_evidence(&format!(
            "{text}audio a sample_rate 1 samples 1 total_seconds 1\nwindow 0 kind body start_sec 0 end_sec 1 first_sample 0 first_feature_frame 0 feature_frames 1 residue 0 origin 0 output_frames 0 keep_until_sec 1\nlabels\nmask\ntokens\nwords\nend\n"
        ));
        assert_eq!(
            parsed
                .model_pairs
                .iter()
                .find(|(key, _)| key == "space")
                .map(|(_, value)| value.as_str()),
            Some("none")
        );
    }
}
