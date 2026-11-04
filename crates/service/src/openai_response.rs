// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! OpenAI-compatible batch response projection over retained transcription results.

use std::io::Write;

use axum::{
    http::header,
    response::{IntoResponse, Response},
};
use flate2::{Compression, write::ZlibEncoder};
use gigaam_transcription::{ChannelTranscript, MultiChannelBatchResult, TurnGap, words_to_text};

use crate::json::{JSON_CONTENT_TYPE, Json, obj};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Format {
    Json,
    Text,
    Verbose,
}

pub(crate) struct ProjectionInput<'a> {
    pub result: &'a MultiChannelBatchResult,
    pub duration: f32,
    pub format: Format,
    pub word: bool,
    pub segment: bool,
    pub gap: TurnGap,
}

pub(crate) fn project(input: ProjectionInput<'_>) -> Result<Response, String> {
    let [channel] = input.result.channels() else {
        return Err("OpenAI output requires exactly one selected channel".into());
    };
    project_channel(
        channel,
        input.duration,
        input.format,
        input.word,
        input.segment,
        input.gap,
    )
}

fn project_channel(
    channel: &ChannelTranscript,
    duration: f32,
    format: Format,
    word: bool,
    segment: bool,
    gap: TurnGap,
) -> Result<Response, String> {
    if !duration.is_finite() || duration < 0.0 {
        return Err("invalid audio duration".into());
    }
    let text = words_to_text(channel.words());
    if format == Format::Text {
        return Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response());
    }
    let mut fields = vec![("text", Json::str(text))];
    if format == Format::Verbose {
        fields.insert(0, ("task", Json::str("transcribe")));
        fields.insert(1, ("language", Json::str("russian")));
        fields.insert(2, ("duration", Json::Num(f64::from(duration))));
        if word {
            let mut words = Vec::new();
            for item in channel.words() {
                if !item.start().is_finite() || !item.end().is_finite() {
                    return Err("invalid word timing".into());
                }
                words.push(obj(vec![
                    ("word", Json::str(item.text())),
                    ("start", Json::Num(f64::from(item.start()))),
                    ("end", Json::Num(f64::from(item.end()))),
                ]));
            }
            fields.push(("words", Json::Array(words)));
        }
        if segment {
            let mut segments = Vec::new();
            for (id, item) in channel.scored_segments(gap)?.iter().enumerate() {
                let average = item.average_log_probability();
                let no_speech = item.no_speech_probability();
                if !item.start().is_finite()
                    || !item.end().is_finite()
                    || !average.is_finite()
                    || !no_speech.is_finite()
                    || !(0.0..=1.0).contains(&no_speech)
                {
                    return Err("invalid scored segment metadata".into());
                }
                let tokens = item
                    .tokens()
                    .iter()
                    .map(|token| Json::Usize(token.id()))
                    .collect();
                segments.push(obj(vec![
                    ("id", Json::Usize(id)),
                    ("seek", Json::Usize(item.seek())),
                    ("start", Json::Num(f64::from(item.start()))),
                    ("end", Json::Num(f64::from(item.end()))),
                    ("text", Json::str(item.text())),
                    ("tokens", Json::Array(tokens)),
                    ("temperature", Json::Num(0.0)),
                    ("avg_logprob", Json::Num(average)),
                    (
                        "compression_ratio",
                        Json::Num(compression_ratio(item.text())?),
                    ),
                    ("no_speech_prob", Json::Num(no_speech)),
                ]));
            }
            fields.push(("segments", Json::Array(segments)));
        }
    }
    Ok((
        [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
        obj(fields).to_string(),
    )
        .into_response())
}

fn compressed(text: &str) -> Result<Vec<u8>, String> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
    encoder
        .write_all(text.as_bytes())
        .map_err(|error| format!("compression failed: {error}"))?;
    encoder
        .finish()
        .map_err(|error| format!("compression failed: {error}"))
}

fn compression_ratio(text: &str) -> Result<f64, String> {
    if text.is_empty() {
        return Ok(0.0);
    }
    let bytes = compressed(text)?;
    let input = u32::try_from(text.len()).map_err(|_| "text length overflows".to_owned())?;
    let output =
        u32::try_from(bytes.len()).map_err(|_| "compressed length overflows".to_owned())?;
    if output == 0 {
        return Err("empty compressed output".into());
    }
    Ok(f64::from(input) / f64::from(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    #[tokio::test]
    async fn requested_verbose_shape_and_missing_scores() -> Result<(), String> {
        use gigaam_transcription::TranscriptWord;
        let channel =
            ChannelTranscript::new(0, vec![TranscriptWord::new("Привет".into(), 0.25, 0.75)?])?;
        let gap = TurnGap::new(1.0)?;
        let response = project_channel(&channel, 1.0, Format::Verbose, true, false, gap)?;
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .map_err(|e| e.to_string())?;
        assert_eq!(
            std::str::from_utf8(&body).map_err(|e| e.to_string())?,
            "{\"task\":\"transcribe\",\"language\":\"russian\",\"duration\":1,\"text\":\"Привет\",\"words\":[{\"word\":\"Привет\",\"start\":0.25,\"end\":0.75}]}"
        );
        assert!(project_channel(&channel, 1.0, Format::Verbose, true, true, gap).is_err());
        let empty = ChannelTranscript::new(0, vec![])?;
        let response = project_channel(&empty, 0.0, Format::Verbose, true, true, gap)?;
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .map_err(|e| e.to_string())?;
        assert_eq!(
            std::str::from_utf8(&body).map_err(|e| e.to_string())?,
            "{\"task\":\"transcribe\",\"language\":\"russian\",\"duration\":0,\"text\":\"\",\"words\":[],\"segments\":[]}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn ordinary_formats_have_only_their_documented_body() -> Result<(), String> {
        use gigaam_transcription::TranscriptWord;
        let channel =
            ChannelTranscript::new(0, vec![TranscriptWord::new("hello".into(), 0.0, 0.5)?])?;
        let gap = TurnGap::new(1.0)?;
        for (format, expected) in [
            (Format::Json, "{\"text\":\"hello\"}"),
            (Format::Text, "hello"),
        ] {
            let response = project_channel(&channel, 0.5, format, false, false, gap)?;
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .map_err(|e| e.to_string())?;
            assert_eq!(
                std::str::from_utf8(&body).map_err(|e| e.to_string())?,
                expected
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn scored_batch_projects_actual_segment_metadata() -> Result<(), String> {
        use gigaam_audio::{
            ChannelAudio, ChannelAudioView, DecodedAudio, FeatureMatrixView, SampleRate,
        };
        use gigaam_recognition::{
            Decoded, DecodedTokensInput, ExecutionControl, FrameRate, ScoreDetail,
            SpeechProbabilityDetector, Token, WindowDecoder, Word,
        };
        use gigaam_transcription::{
            BatchChannelPolicy, BatchConfig, BatchSetup, MultiChannelBatchOptions,
            MultiChannelBatchSetup, MultiChannelBatchTranscriber, ObservationMode, PadPolicy,
        };

        struct Decoder;
        impl WindowDecoder for Decoder {
            fn frame_rate(&self) -> FrameRate {
                match FrameRate::new(50.0) {
                    Ok(value) => value,
                    Err(error) => panic!("invalid fixture rate: {error}"),
                }
            }
            fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
                self.decode_with_detail(features, ScoreDetail::None)
            }
            fn decode_with_detail(
                &mut self,
                _: FeatureMatrixView<'_>,
                detail: ScoreDetail,
            ) -> Result<Decoded, String> {
                if detail != ScoreDetail::TokenLogProbabilities {
                    return Err("fixture requires scores".into());
                }
                Decoded::new_with_tokens(DecodedTokensInput {
                    words: vec![
                        Word::new("hello".into(), 0.25, 0.5)?,
                        Word::new("world".into(), 0.5, 0.75)?,
                    ],
                    silence: vec![false],
                    output_frames: 1,
                    encoder_seconds: 0.0,
                    tokens: vec![
                        Token::scored(7, 0, -0.25)?,
                        Token::scored(8, 0, -0.5)?,
                        Token::scored(9, 0, -0.75)?,
                    ],
                    word_spans: vec![(0, 0), (2, 2)],
                })
            }
        }
        struct Detector;
        impl SpeechProbabilityDetector for Detector {
            fn probabilities(&mut self, audio: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
                Ok(vec![0.5; audio.samples().len().div_ceil(512)])
            }
        }
        let fixture = crate::tests::assembly_fixture()?;
        let rate = SampleRate::new(16000)?;
        let gap = TurnGap::new(1.0)?;
        let batch = MultiChannelBatchTranscriber::new(MultiChannelBatchSetup::new(
            BatchSetup {
                frontend: fixture.frontend.clone(),
                decoder: Decoder,
                config: BatchConfig::new(rate, 1.0, 0.0, PadPolicy::Exact)?,
                control: ExecutionControl::without_deadline(),
                observations: ObservationMode::disabled(),
            },
            MultiChannelBatchOptions::new(BatchChannelPolicy::single_output(), gap),
        ))?;
        let audio = DecodedAudio::new(rate, vec![ChannelAudio::new(vec![0.25; 16000])?])?;
        let result = batch
            .transcribe_scored(&audio, &mut Detector, None)
            .map_err(|e| format!("{e:?}"))?;
        let response = project(ProjectionInput {
            result: &result,
            duration: 1.0,
            format: Format::Verbose,
            word: true,
            segment: true,
            gap,
        })?;
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .map_err(|e| e.to_string())?;
        let text = std::str::from_utf8(&body).map_err(|e| e.to_string())?;
        assert_eq!(
            text,
            format!(
                "{{\"task\":\"transcribe\",\"language\":\"russian\",\"duration\":1,\"text\":\"hello world\",\"words\":[{{\"word\":\"hello\",\"start\":0.25,\"end\":0.5}},{{\"word\":\"world\",\"start\":0.5,\"end\":0.75}}],\"segments\":[{{\"id\":0,\"seek\":0,\"start\":0.25,\"end\":0.75,\"text\":\"hello world\",\"tokens\":[7,8,9],\"temperature\":0,\"avg_logprob\":-0.5,\"compression_ratio\":{},\"no_speech_prob\":0.5}}]}}",
                11.0_f64 / 19.0
            )
        );
        Ok(())
    }

    #[test]
    fn compression_roundtrips_and_uses_utf8_byte_lengths() -> Result<(), String> {
        for text in [
            "",
            "hello",
            "Привет, мир!",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ] {
            let encoded = compressed(text)?;
            let mut decoded = String::new();
            ZlibDecoder::new(encoded.as_slice())
                .read_to_string(&mut decoded)
                .map_err(|e| e.to_string())?;
            assert_eq!(decoded, text);
            let expected = if text.is_empty() {
                0.0
            } else {
                f64::from(u32::try_from(text.len()).map_err(|e| e.to_string())?)
                    / f64::from(u32::try_from(encoded.len()).map_err(|e| e.to_string())?)
            };
            assert_eq!(compression_ratio(text)?, expected);
        }
        assert_eq!(compressed("")?, vec![120, 156, 3, 0, 0, 0, 0, 1]);
        assert_eq!(
            compressed("Привет, мир!")?,
            vec![
                120, 156, 187, 48, 255, 98, 195, 133, 29, 23, 54, 93, 216, 122, 177, 73, 71, 225,
                194, 158, 11, 59, 46, 54, 40, 2, 0, 157, 147, 13, 117
            ]
        );
        assert_eq!(
            compressed("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")?,
            vec![120, 156, 75, 76, 36, 14, 0, 0, 54, 235, 15, 41]
        );
        Ok(())
    }
}
