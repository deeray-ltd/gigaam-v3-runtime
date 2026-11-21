// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

use crate::openai_response::{Format, StoredProjectionInput, write_stored};
use gigaam_transcription::{TurnGap, WordRecords};
use std::io::Cursor;

struct ProjectionInput<'a> {
    pub result: &'a gigaam_transcription::MultiChannelBatchResult,
    pub duration: f32,
    pub format: Format,
    pub word: bool,
    pub segment: bool,
    pub gap: TurnGap,
}

fn project(input: ProjectionInput<'_>) -> Result<axum::response::Response, String> {
    let [channel] = input.result.channels() else {
        return Err("OpenAI output requires exactly one selected channel".into());
    };
    crate::openai_response::project_channel(
        channel,
        input.duration,
        input.format,
        input.word,
        input.segment,
        input.gap,
    )
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
        fn silence_capability(&self) -> gigaam_recognition::SilenceCapability {
            gigaam_recognition::SilenceCapability::LegacyBlank
        }
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
                    Word::new("Привет,".into(), 0.25, 0.5)?,
                    Word::new("мир!".into(), 0.5, 0.75)?,
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
            "{{\"task\":\"transcribe\",\"language\":\"russian\",\"duration\":1,\"text\":\"Привет, мир!\",\"words\":[{{\"word\":\"Привет,\",\"start\":0.25,\"end\":0.5}},{{\"word\":\"мир!\",\"start\":0.5,\"end\":0.75}}],\"segments\":[{{\"id\":0,\"seek\":0,\"start\":0.25,\"end\":0.75,\"text\":\"Привет, мир!\",\"tokens\":[7,8,9],\"temperature\":0,\"avg_logprob\":-0.5,\"compression_ratio\":{},\"no_speech_prob\":0.5}}]}}",
            21.0_f64 / 31.0
        )
    );

    let [channel] = result.channels() else {
        return Err("fixture must produce one channel".into());
    };
    let mut records = WordRecords::new(Cursor::new(Vec::new()), 1)?;
    for item in channel.words() {
        records.push(0, item.clone())?;
    }
    let mut stored = records.finish()?;
    for format in [Format::Json, Format::Text, Format::Verbose] {
        for (word, segment) in [(false, false), (true, false), (false, true), (true, true)] {
            let retained = project(ProjectionInput {
                result: &result,
                duration: 1.0,
                format,
                word,
                segment,
                gap,
            })?;
            let retained_body = axum::body::to_bytes(retained.into_body(), usize::MAX)
                .await
                .map_err(|error| error.to_string())?;
            let mut bounded_body = Vec::new();
            write_stored(
                &mut stored,
                StoredProjectionInput {
                    duration: 1.0,
                    format,
                    word,
                    segment,
                    gap,
                },
                &mut bounded_body,
            )?;
            assert_eq!(
                bounded_body, retained_body,
                "{format:?}, words={word}, segments={segment}"
            );
        }
    }

    Ok(())
}
