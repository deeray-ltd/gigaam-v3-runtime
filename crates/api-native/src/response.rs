// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Build `POST /v1/transcribe` JSON responses to the contract schema with pure functions over
//! Transcription application results and HTTP wire envelopes using the service JSON writer.
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use gigaam_service_host::wire::json::{JSON_CONTENT_TYPE, Json, error_object, obj};
use gigaam_transcription::{
    BatchChannelSelection, ChannelLines, DuplicateClassification, OriginalChannel,
    SourceChannelCount, StoredLine, StoredWords, TranscriptWord, TurnGap, TurnLines,
    VadPolicyConfig,
};
#[cfg(test)]
use gigaam_transcription::{ChannelTranscript, Turn, words_to_text};
use std::io::{Read, Seek, Write};

/// Query request parameters that affect response shape.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Options {
    pub words: bool,
    pub segments: bool,
    pub turns: bool,
    pub turn_gap: f32,
    pub split: bool,
}

/// Fully named input to response shaping. It prevents positional ambiguity at the
/// public HTTP boundary and keeps the response constructor below the argument-count limit.
#[cfg(test)]
pub(crate) struct ResponseInput<'a> {
    pub contract: ResponseContract,
    pub channels: &'a [ChannelTranscript],
    pub segments: &'a [Turn],
    pub turns: &'a [Turn],
    pub options: Options,
    pub duration_sec: f32,
    pub sample_rate_in: u32,
    pub source_channels: SourceChannelCount,
    pub duplicate: DuplicateClassification,
    pub model: &'a str,
    pub variant: &'a str,
    pub rtf: f32,
}

/// Metadata associated with a request-owned transcript replay.
pub(crate) struct StoredResponseInput<'a> {
    pub contract: ResponseContract,
    pub selection: &'a BatchChannelSelection,
    pub options: Options,
    pub duration_sec: f32,
    pub sample_rate_in: u32,
    pub model: &'a str,
    pub variant: &'a str,
    pub rtf: f32,
}

struct WireMetadata<'a> {
    contract: ResponseContract,
    duration_sec: f32,
    sample_rate_in: u32,
    source_channels: SourceChannelCount,
    duplicate: DuplicateClassification,
    model: &'a str,
    variant: &'a str,
    rtf: f32,
}

/// An explicitly selected response contract, carrying the policy actually used by execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResponseVersion {
    V1,
    V2,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ResponseContract {
    pub version: ResponseVersion,
    pub vad: Option<VadPolicyConfig>,
}

impl ResponseContract {
    fn applied_vad(self) -> Json {
        let Some(policy) = self.vad else {
            return obj(vec![("enabled", Json::Bool(false))]);
        };
        obj(vec![
            ("enabled", Json::Bool(true)),
            (
                "sample_rate",
                Json::Int(
                    i64::try_from(gigaam_recognition::vad::VAD_SR)
                        .expect("the fixed 16000 Hz VAD rate fits i64"),
                ),
            ),
            (
                "threshold",
                Json::Num(f64::from(policy.thresholds().speech().get())),
            ),
            (
                "neg_threshold",
                Json::Num(f64::from(policy.thresholds().silence().get())),
            ),
            (
                "min_speech_ms",
                Json::Num(f64::from(policy.durations().minimum_speech_milliseconds())),
            ),
            (
                "min_silence_ms",
                Json::Num(f64::from(policy.durations().minimum_silence_milliseconds())),
            ),
            (
                "speech_pad_ms",
                Json::Num(f64::from(policy.padding().milliseconds())),
            ),
        ])
    }
}

/// Stable v2 refusal categories. A broad input refusal does not imply a permanent bad file.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TranscribeErrorCode {
    InvalidRequest,
    UnsupportedCapability,
    Unavailable,
    Overloaded,
    Draining,
    Timeout,
}

impl TranscribeErrorCode {
    fn name(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::UnsupportedCapability => "unsupported_capability",
            Self::Unavailable => "unavailable",
            Self::Overloaded => "overloaded",
            Self::Draining => "draining",
            Self::Timeout => "timeout",
        }
    }
}

impl Default for Options {
    fn default() -> Self {
        Options {
            words: false,
            segments: false,
            turns: false,
            turn_gap: 1.0,
            split: false,
        }
    }
}

fn word_json(w: &TranscriptWord, channel: Option<usize>) -> Json {
    let mut f = vec![
        ("start", Json::round(w.start(), 2)),
        ("end", Json::round(w.end(), 2)),
        ("text", Json::str(w.text().to_owned())),
    ];
    if let Some(c) = channel {
        f.push(("channel", Json::Int(channel_index(c))));
    }
    obj(f)
}

#[derive(Clone, Copy)]
enum LineKind {
    Segment,
    Turn,
}

impl LineKind {
    fn fields(self, channel: usize, start: f32, end: f32) -> [(&'static str, Json); 3] {
        let channel = ("channel", Json::Int(channel_index(channel)));
        let start = ("start", Json::round(start, 2));
        let end = ("end", Json::round(end, 2));
        match self {
            Self::Segment => [start, end, channel],
            Self::Turn => [channel, start, end],
        }
    }
}

#[cfg(test)]
fn line_json(kind: LineKind, line: &Turn) -> Json {
    let mut fields = kind
        .fields(line.channel(), line.start(), line.end())
        .to_vec();
    fields.push(("text", Json::str(line.text())));
    obj(fields)
}

fn channel_index(value: usize) -> i64 {
    match i64::try_from(value) {
        Ok(value) => value,
        Err(_) => panic!("a channel index from one addressable response collection must fit i64"),
    }
}

/// Precomputed Transcription projections are serialized without reconstructing application state.
#[cfg(test)]
pub(crate) fn transcribe_response(input: ResponseInput<'_>) -> Json {
    let ResponseInput {
        contract,
        channels,
        segments,
        turns,
        options: opt,
        duration_sec,
        sample_rate_in,
        source_channels,
        duplicate,
        model,
        variant,
        rtf,
    } = input;
    let mut f = metadata_fields(WireMetadata {
        contract,
        duration_sec,
        sample_rate_in,
        source_channels,
        duplicate,
        model,
        variant,
        rtf,
    });

    let multi = channels.len() > 1;
    // Text is shared for one channel/merge and an array by channel for split.
    if opt.split && multi {
        let texts: Vec<Json> = channels
            .iter()
            .map(|channel| Json::str(words_to_text(channel.words())))
            .collect();
        f.push(("channels_text", Json::Array(texts)));
    } else {
        let channel = channels
            .first()
            .expect("a validated multi-channel batch result always retains one channel");
        let text = words_to_text(channel.words());
        f.push(("text", Json::str(text)));
    }

    if opt.words {
        let mut arr: Vec<Json> = Vec::new();
        for channel in channels {
            for word in channel.words() {
                arr.push(word_json(
                    word,
                    if multi { Some(channel.channel()) } else { None },
                ));
            }
        }
        f.push(("words", Json::Array(arr)));
    }
    if opt.segments {
        f.push((
            "segments",
            Json::Array(
                segments
                    .iter()
                    .map(|line| line_json(LineKind::Segment, line))
                    .collect(),
            ),
        ));
    }
    if opt.turns && multi {
        let ts: Vec<Json> = turns
            .iter()
            .map(|line| line_json(LineKind::Turn, line))
            .collect();
        f.push(("turns", Json::Array(ts)));
    }
    obj(f)
}

fn metadata_fields(input: WireMetadata<'_>) -> Vec<(&'static str, Json)> {
    let mut fields = vec![
        ("duration_sec", Json::round(input.duration_sec, 2)),
        ("sample_rate_in", Json::Int(i64::from(input.sample_rate_in))),
        (
            "channels",
            Json::Int(channel_index(input.source_channels.get())),
        ),
        (
            "dual_mono",
            Json::Bool(matches!(input.duplicate, DuplicateClassification::DualMono)),
        ),
        ("model", Json::str(input.model)),
        ("rtf", Json::round(input.rtf, 4)),
    ];
    if input.contract.version == ResponseVersion::V2 {
        fields.push(("schema_version", Json::Int(2)));
        fields.push(("variant", Json::str(input.variant)));
        fields.push(("applied", obj(vec![("vad", input.contract.applied_vad())])));
    }
    fields
}

/// Writes the complete native envelope from ordered request-owned records.
pub(crate) fn write_stored(
    input: StoredResponseInput<'_>,
    words: &mut StoredWords<impl Read + Seek>,
    output: &mut impl Write,
) -> Result<(), String> {
    let channels = input.selection.selection().channels();
    let multi = channels.len() > 1;
    let options = input.options;
    let gap = TurnGap::new(options.turn_gap)?;
    let fields = metadata_fields(WireMetadata {
        contract: input.contract,
        duration_sec: input.duration_sec,
        sample_rate_in: input.sample_rate_in,
        source_channels: input.selection.source_channels(),
        duplicate: input.selection.duplicate(),
        model: input.model,
        variant: input.variant,
        rtf: input.rtf,
    });

    output.write_all(b"{").map_err(wire_error)?;
    write_fields(output, fields)?;

    if options.split && multi {
        output
            .write_all(b",\"channels_text\":[")
            .map_err(wire_error)?;
        for (index, channel) in channels.iter().enumerate() {
            write_comma(output, index != 0)?;
            let mut cursor = words.cursor(channel.index())?;
            write_text(
                output,
                std::iter::from_fn(|| words.next(&mut cursor).transpose()),
            )?;
        }
        output.write_all(b"]").map_err(wire_error)?;
    } else {
        let first = channels
            .first()
            .ok_or_else(|| "native response requires a retained channel".to_owned())?;
        output.write_all(b",\"text\":").map_err(wire_error)?;
        let mut cursor = words.cursor(first.index())?;
        write_text(
            output,
            std::iter::from_fn(|| words.next(&mut cursor).transpose()),
        )?;
    }

    if options.words {
        output.write_all(b",\"words\":[").map_err(wire_error)?;
        let mut wrote_word = false;
        for channel in channels {
            let mut cursor = words.cursor(channel.index())?;
            while let Some(word) = words.next(&mut cursor)? {
                write_comma(output, wrote_word)?;
                let identity = multi.then_some(channel.index());
                write!(output, "{}", word_json(&word, identity)).map_err(wire_error)?;
                wrote_word = true;
            }
        }
        output.write_all(b"]").map_err(wire_error)?;
    }
    if options.segments {
        write_segments(channels, gap, words, output)?;
    }
    if options.turns && multi {
        write_turns(channels, gap, words, output)?;
    }
    output.write_all(b"}").map_err(wire_error)
}

fn write_fields(
    output: &mut impl Write,
    fields: impl IntoIterator<Item = (&'static str, Json)>,
) -> Result<(), String> {
    for (index, (name, value)) in fields.into_iter().enumerate() {
        write_comma(output, index != 0)?;
        write!(output, "\"{name}\":{value}").map_err(wire_error)?;
    }
    Ok(())
}

fn write_comma(output: &mut impl Write, needed: bool) -> Result<(), String> {
    if needed {
        output.write_all(b",").map_err(wire_error)?;
    }
    Ok(())
}

fn write_text(
    output: &mut impl Write,
    words: impl Iterator<Item = Result<TranscriptWord, String>>,
) -> Result<(), String> {
    output.write_all(b"\"").map_err(wire_error)?;
    for (index, word) in words.enumerate() {
        let word = word?;
        if index != 0 {
            output.write_all(b" ").map_err(wire_error)?;
        }
        // Reuse the schema writer's exact escaping. Only one word is encoded at once.
        let quoted = Json::str(word.text()).to_string();
        output
            .write_all(&quoted.as_bytes()[1..quoted.len() - 1])
            .map_err(wire_error)?;
    }
    output.write_all(b"\"").map_err(wire_error)
}

fn write_line(
    line: StoredLine,
    kind: LineKind,
    words: &mut StoredWords<impl Read + Seek>,
    output: &mut impl Write,
) -> Result<(), String> {
    output.write_all(b"{").map_err(wire_error)?;
    write_fields(
        output,
        kind.fields(line.channel(), line.start(), line.end()),
    )?;
    output.write_all(b",\"text\":").map_err(wire_error)?;
    write_text(output, line.words(words))?;
    output.write_all(b"}").map_err(wire_error)
}

fn write_segments(
    channels: &[OriginalChannel],
    gap: TurnGap,
    words: &mut StoredWords<impl Read + Seek>,
    output: &mut impl Write,
) -> Result<(), String> {
    output.write_all(b",\"segments\":[").map_err(wire_error)?;
    let mut wrote_segment = false;
    for &channel in channels {
        let mut scan = ChannelLines::new(channel, words, gap)?;
        while let Some(line) = scan.next(words)? {
            write_comma(output, wrote_segment)?;
            write_line(line, LineKind::Segment, words, output)?;
            wrote_segment = true;
        }
    }
    output.write_all(b"]").map_err(wire_error)
}

fn write_turns(
    channels: &[OriginalChannel],
    gap: TurnGap,
    words: &mut StoredWords<impl Read + Seek>,
    output: &mut impl Write,
) -> Result<(), String> {
    output.write_all(b",\"turns\":[").map_err(wire_error)?;
    let mut turns = TurnLines::new(channels, words, gap)?;
    let mut wrote_turn = false;
    while let Some(line) = turns.next(words)? {
        write_comma(output, wrote_turn)?;
        write_line(line, LineKind::Turn, words, output)?;
        wrote_turn = true;
    }
    output.write_all(b"]").map_err(wire_error)
}

fn wire_error(error: std::io::Error) -> String {
    format!("native response I/O: {error}")
}

pub(crate) fn json_err(code: StatusCode, message: impl Into<String>) -> Response {
    (
        code,
        [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
        error_object(message),
    )
        .into_response()
}

pub(crate) fn transcribe_error_response(
    version: ResponseVersion,
    status: StatusCode,
    code: TranscribeErrorCode,
    message: impl Into<String>,
) -> Response {
    let message = message.into();
    match version {
        ResponseVersion::V1 => json_err(status, message),
        ResponseVersion::V2 => (
            status,
            [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
            obj(vec![
                ("schema_version", Json::Int(2)),
                (
                    "error",
                    obj(vec![
                        ("code", Json::str(code.name())),
                        ("message", Json::str(message)),
                    ]),
                ),
            ])
            .to_string(),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::{StatusCode, header};
    use gigaam_audio::ChannelAudio;
    use gigaam_transcription::{BatchChannelPolicy, WordRecords, select_batch_channels, turns};
    use std::io::{self, Cursor};

    fn w(t: &str, s: f32, e: f32) -> TranscriptWord {
        TranscriptWord::new(t.into(), s, e).expect("test word timestamps are ordered and finite")
    }

    fn channel(index: usize, words: Vec<TranscriptWord>) -> ChannelTranscript {
        ChannelTranscript::new(index, words).expect("test channel words are ordered by start time")
    }

    fn projections(channels: &[ChannelTranscript], gap_seconds: f32) -> (Vec<Turn>, Vec<Turn>) {
        let gap = TurnGap::new(gap_seconds).expect("test turn gap is valid");
        let segments = channels
            .iter()
            .flat_map(|channel| {
                turns(std::slice::from_ref(channel), gap).expect("test channel transcript is valid")
            })
            .collect();
        let turns = turns(channels, gap).expect("test channel transcripts are valid");
        (segments, turns)
    }

    // This also proves that a whole long line is never handed to the output writer.
    #[derive(Default)]
    struct ChunkLimitedOutput(Vec<u8>);

    impl Write for ChunkLimitedOutput {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > 256 {
                return Err(io::Error::other("unexpected unbounded response write"));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn assert_stored_matches_collected(
        selection: &BatchChannelSelection,
        channels: &[ChannelTranscript],
        options: Options,
        contract: ResponseContract,
    ) -> Result<(), String> {
        let mut records =
            WordRecords::new(Cursor::new(Vec::new()), selection.source_channels().get())?;
        for channel in channels {
            for word in channel.words() {
                records.push(channel.channel(), word.clone())?;
            }
        }
        let mut words = records.finish()?;
        let mut output = ChunkLimitedOutput::default();
        write_stored(
            StoredResponseInput {
                contract,
                selection,
                options,
                duration_sec: 1.996,
                sample_rate_in: 44_100,
                model: "ctc",
                variant: "v3_ctc",
                rtf: 0.012345,
            },
            &mut words,
            &mut output,
        )?;
        let (segments, turns) = projections(channels, options.turn_gap);
        let expected = transcribe_response(ResponseInput {
            contract,
            channels,
            segments: &segments,
            turns: &turns,
            options,
            duration_sec: 1.996,
            sample_rate_in: 44_100,
            source_channels: selection.source_channels(),
            duplicate: selection.duplicate(),
            model: "ctc",
            variant: "v3_ctc",
            rtf: 0.012345,
        });
        assert_eq!(output.0, expected.to_string().as_bytes());
        Ok(())
    }

    #[test]
    fn stored_schema_matches_all_collected_shapes_and_f32_gap_boundaries() -> Result<(), String> {
        let first = ChannelAudio::new(vec![1.0, 0.0, -1.0, 0.0])?;
        let second = ChannelAudio::new(vec![0.0, 1.0, 0.0, -1.0])?;
        let dual = select_batch_channels(
            &[first.view(), first.view()],
            BatchChannelPolicy::separate_channels(),
        )?;
        let stereo = select_batch_channels(
            &[first.view(), second.view()],
            BatchChannelPolicy::separate_channels(),
        )?;
        let single = select_batch_channels(
            &[first.view(), second.view()],
            BatchChannelPolicy::single_output(),
        )?;
        let lines = [
            channel(
                0,
                vec![
                    w("два\\\"\n", 0.0, 0.91),
                    w("overlap", 0.125, 0.25),
                    w("split", 0.375, 0.5),
                    w("below", 0.625_f32.next_down(), 0.71),
                ],
            ),
            channel(
                1,
                vec![
                    w("tie", 0.0, 0.08),
                    w("later", 0.5, 0.5),
                    w("above", 0.625_f32.next_up(), 0.75),
                ],
            ),
        ];
        let empty = [channel(0, Vec::new()), channel(1, Vec::new())];
        for (selection, channels) in [
            (&stereo, lines.as_slice()),
            (&stereo, empty.as_slice()),
            (&dual, &lines[..1]),
            (&single, &lines[..1]),
        ] {
            for contract in [
                ResponseContract {
                    version: ResponseVersion::V1,
                    vad: None,
                },
                ResponseContract {
                    version: ResponseVersion::V2,
                    vad: None,
                },
                ResponseContract {
                    version: ResponseVersion::V2,
                    vad: Some(VadPolicyConfig::default()),
                },
            ] {
                for flags in 0..16 {
                    let options = Options {
                        words: flags & 1 != 0,
                        segments: flags & 2 != 0,
                        turns: flags & 4 != 0,
                        split: flags & 8 != 0,
                        turn_gap: 0.125,
                    };
                    assert_stored_matches_collected(selection, channels, options, contract)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn stored_projection_streams_long_segment_and_turn_text() -> Result<(), String> {
        let first = ChannelAudio::new(vec![1.0, 0.0, -1.0, 0.0])?;
        let second = ChannelAudio::new(vec![0.0, 1.0, 0.0, -1.0])?;
        let selection = select_batch_channels(
            &[first.view(), second.view()],
            BatchChannelPolicy::separate_channels(),
        )?;
        let channels = [
            channel(0, vec![w("bounded", 0.0, 2000.0); 4_096]),
            channel(1, vec![w("independent", 0.1, 0.2)]),
        ];
        assert_stored_matches_collected(
            &selection,
            &channels,
            Options {
                split: true,
                words: true,
                segments: true,
                turns: true,
                turn_gap: 1.0,
            },
            ResponseContract {
                version: ResponseVersion::V2,
                vad: None,
            },
        )
    }

    #[test]
    fn mono_minimal() {
        let ch = vec![channel(0, vec![w("hello", 0.0, 0.4), w("world", 0.5, 0.8)])];
        let (segments, turns) = projections(&ch, 1.0);
        let j = transcribe_response(ResponseInput {
            contract: ResponseContract {
                version: ResponseVersion::V1,
                vad: None,
            },
            channels: &ch,
            segments: &segments,
            turns: &turns,
            options: Options::default(),
            duration_sec: 0.8,
            sample_rate_in: 16000,
            source_channels: SourceChannelCount::new(1).expect("one source channel is valid"),
            duplicate: DuplicateClassification::NotDualMono,
            model: "ctc",
            variant: "v3_ctc",
            rtf: 0.01,
        });
        assert_eq!(
            j.to_string(),
            r#"{"duration_sec":0.8,"sample_rate_in":16000,"channels":1,"dual_mono":false,"model":"ctc","rtf":0.01,"text":"hello world"}"#
        );
    }
    #[test]
    fn stereo_turns_and_words() {
        let ch = vec![
            channel(0, vec![w("hello", 0.0, 0.4)]),
            channel(1, vec![w("yes", 1.0, 1.3)]),
        ];
        let opt = Options {
            words: true,
            turns: true,
            split: true,
            ..Options::default()
        };
        let (segments, turns) = projections(&ch, 1.0);
        let j = transcribe_response(ResponseInput {
            contract: ResponseContract {
                version: ResponseVersion::V1,
                vad: None,
            },
            channels: &ch,
            segments: &segments,
            turns: &turns,
            options: opt,
            duration_sec: 1.3,
            sample_rate_in: 8000,
            source_channels: SourceChannelCount::new(2).expect("two source channels are valid"),
            duplicate: DuplicateClassification::NotDualMono,
            model: "ctc",
            variant: "v3_ctc",
            rtf: 0.02,
        });
        assert_eq!(
            j.to_string(),
            r#"{"duration_sec":1.3,"sample_rate_in":8000,"channels":2,"dual_mono":false,"model":"ctc","rtf":0.02,"channels_text":["hello","yes"],"words":[{"start":0,"end":0.4,"text":"hello","channel":0},{"start":1,"end":1.3,"text":"yes","channel":1}],"turns":[{"channel":0,"start":0,"end":0.4,"text":"hello"},{"channel":1,"start":1,"end":1.3,"text":"yes"}]}"#
        );
    }

    #[test]
    fn response_serializes_supplied_segments_and_turns_without_reconstructing_them() {
        let channels = vec![
            channel(0, vec![w("first", 0.0, 0.1), w("later", 2.0, 2.1)]),
            channel(1, vec![w("middle", 0.5, 0.6)]),
        ];
        let (segments, turns) = projections(&channels, 1.0);
        let options = Options {
            segments: true,
            turns: true,
            split: true,
            turn_gap: 5.0,
            ..Options::default()
        };
        let rendered = transcribe_response(ResponseInput {
            contract: ResponseContract {
                version: ResponseVersion::V1,
                vad: None,
            },
            channels: &channels,
            segments: &segments,
            turns: &turns,
            options,
            duration_sec: 2.1,
            sample_rate_in: 16_000,
            source_channels: SourceChannelCount::new(2).expect("two source channels are valid"),
            duplicate: DuplicateClassification::NotDualMono,
            model: "ctc",
            variant: "v3_ctc",
            rtf: 0.01,
        })
        .to_string();

        assert_eq!(
            rendered,
            r#"{"duration_sec":2.1,"sample_rate_in":16000,"channels":2,"dual_mono":false,"model":"ctc","rtf":0.01,"channels_text":["first later","middle"],"segments":[{"start":0,"end":0.1,"channel":0,"text":"first"},{"start":2,"end":2.1,"channel":0,"text":"later"},{"start":0.5,"end":0.6,"channel":1,"text":"middle"}],"turns":[{"channel":0,"start":0,"end":0.1,"text":"first"},{"channel":1,"start":0.5,"end":0.6,"text":"middle"},{"channel":0,"start":2,"end":2.1,"text":"later"}]}"#
        );
    }

    #[tokio::test]
    async fn http_errors_use_the_shared_json_content_type_and_error_object() {
        let response = json_err(StatusCode::BAD_REQUEST, "invalid audio");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&axum::http::HeaderValue::from_static(
                "application/json; charset=utf-8",
            ))
        );
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("HTTP error response body is available"),
            "{\"error\":\"invalid audio\"}"
        );
    }

    #[tokio::test]
    async fn v2_error_codes_preserve_status_and_escape_diagnostic_text() {
        for (status, code, name) in [
            (
                StatusCode::BAD_REQUEST,
                TranscribeErrorCode::InvalidRequest,
                "invalid_request",
            ),
            (
                StatusCode::BAD_REQUEST,
                TranscribeErrorCode::UnsupportedCapability,
                "unsupported_capability",
            ),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                TranscribeErrorCode::Unavailable,
                "unavailable",
            ),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                TranscribeErrorCode::Overloaded,
                "overloaded",
            ),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                TranscribeErrorCode::Draining,
                "draining",
            ),
            (
                StatusCode::GATEWAY_TIMEOUT,
                TranscribeErrorCode::Timeout,
                "timeout",
            ),
        ] {
            let response = transcribe_error_response(
                ResponseVersion::V2,
                status,
                code,
                "diagnostic \"quoted\"\n",
            );
            assert_eq!(response.status(), status);
            assert_eq!(
                response
                    .headers()
                    .get(header::CONTENT_TYPE)
                    .expect("application errors have a content type"),
                JSON_CONTENT_TYPE
            );
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("the complete error body is available");
            assert_eq!(
                body,
                format!(
                    "{{\"schema_version\":2,\"error\":{{\"code\":\"{name}\",\"message\":\"diagnostic \\\"quoted\\\"\\n\"}}}}"
                )
            );
        }
    }
}
