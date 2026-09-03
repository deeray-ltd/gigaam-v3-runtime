// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Decoder-format and stereo-dialogue acceptance through Transcription.

mod common;

use gigaam_audio::{
    FrontendProcessor, SampleRate, channel_correlation, load, read_wav, resample_audio,
};
use gigaam_model_package::{EncoderPrecision, ModelPackage};
use gigaam_recognition::{DirectRecognizer, ExecutionControl};
use gigaam_transcription::{
    BatchConfig, BatchSetup, BatchTranscriber, ChannelTranscript, ObservationMode, PadPolicy,
    TurnGap, merge, turns, word_edits,
};
use std::sync::Arc;

fn transcribe(
    pack: &ModelPackage,
    frontend: Arc<FrontendProcessor>,
    path: &str,
) -> (String, u32, usize) {
    let decoder = DirectRecognizer::ctc(pack, &common::cpu_plan(), EncoderPrecision::Fp32)
        .expect("the CPU CTC test session must open");
    let sample_rate = frontend.sample_rate();
    let mut batch = BatchTranscriber::new(BatchSetup {
        frontend,
        decoder,
        config: BatchConfig::new(sample_rate, 100.0, 6.0, PadPolicy::Exact)
            .expect("the documented batch window must be valid"),
        control: ExecutionControl::without_deadline(),
        observations: ObservationMode::disabled(),
    })
    .expect("the documented batch setup must be valid");
    let loaded = load(&common::root().join(path)).expect("the fixture must decode");
    let rate = loaded.sample_rate().hertz();
    let channels = loaded.channels().len();
    let audio = resample_audio(
        loaded,
        SampleRate::new(16000).expect("fixed test rate is valid"),
    )
    .expect("the fixture must resample");
    let text = batch
        .transcribe_channel(&audio.channels()[0])
        .expect("the fixture must transcribe")
        .text()
        .to_owned();
    (text, rate, channels)
}

fn golden() -> String {
    std::fs::read_to_string(common::root().join("fixtures/example.ctc.gold.txt"))
        .expect("the CTC golden text must be available")
}

#[test]
fn telephony_wavs_8k_alaw_ulaw_pcm() {
    common::init_ort();
    let pack =
        ModelPackage::open(&common::root().join("model")).expect("the test model pack must open");
    let frontend = common::frontend(&pack);
    for fixture in [
        "fixtures/example_8k_ulaw.wav",
        "fixtures/example_8k_alaw.wav",
        "fixtures/example_8k.wav",
    ] {
        let (text, rate, _) = transcribe(&pack, frontend.clone(), fixture);
        let edits = word_edits(&golden(), &text);
        eprintln!("{fixture}: {rate} Hz, edits {edits}");
        assert!(edits <= 2, "{fixture}: {edits} edits\n{text}");
    }
}

#[test]
fn stereo_channels_are_separate() {
    let audio = read_wav(&common::root().join("fixtures/example_stereo.wav"))
        .expect("the stereo fixture must decode");
    assert_eq!(audio.channels().len(), 2);
    let correlation = channel_correlation(audio.channels()[0].view(), audio.channels()[1].view())
        .expect("equal decoded stereo channels are correlatable");
    eprintln!("channel correlation {correlation:.3}");
    assert!(
        correlation < 0.98,
        "shifted copy must not count as dual-mono"
    );
}

#[test]
fn flac_is_bit_exact() {
    let wav = read_wav(&common::root().join("fixtures/example.wav"))
        .expect("the WAV fixture must decode");
    let flac =
        load(&common::root().join("fixtures/example.flac")).expect("the FLAC fixture must decode");
    assert_eq!(flac.sample_rate().hertz(), 16000);
    assert_eq!(flac.channels()[0].len(), wav.channels()[0].len());
    let maximum_delta = flac.channels()[0]
        .samples()
        .iter()
        .zip(wav.channels()[0].samples())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        maximum_delta < 1e-6,
        "FLAC must match WAV bit-for-bit, max|Δ|={maximum_delta}"
    );
}

#[test]
fn lossy_formats_keep_text() {
    common::init_ort();
    let pack =
        ModelPackage::open(&common::root().join("model")).expect("the test model pack must open");
    let frontend = common::frontend(&pack);
    for fixture in [
        "fixtures/example.ogg",
        "fixtures/example.opus",
        "fixtures/example.mp3",
    ] {
        let (text, rate, channels) = transcribe(&pack, frontend.clone(), fixture);
        let edits = word_edits(&golden(), &text);
        eprintln!("{fixture}: {rate} Hz, channels {channels}, edits {edits}");
        assert!(edits <= 2, "{fixture}: {edits} edits\n{text}");
    }
}

#[test]
fn stereo_turns_preserve_channel_identity_and_word_count() {
    common::init_ort();
    let pack =
        ModelPackage::open(&common::root().join("model")).expect("the test model pack must open");
    let audio = read_wav(&common::root().join("fixtures/example_stereo.wav"))
        .expect("the stereo fixture must decode");
    let mut transcripts = Vec::new();
    for (channel, samples) in audio.channels().iter().enumerate() {
        let decoder = DirectRecognizer::ctc(&pack, &common::cpu_plan(), EncoderPrecision::Fp32)
            .expect("the CPU CTC test session must open");
        let frontend = common::frontend(&pack);
        let sample_rate = frontend.sample_rate();
        let mut batch = BatchTranscriber::new(BatchSetup {
            frontend,
            decoder,
            config: BatchConfig::new(sample_rate, 100.0, 6.0, PadPolicy::Exact)
                .expect("the documented batch window must be valid"),
            control: ExecutionControl::without_deadline(),
            observations: ObservationMode::disabled(),
        })
        .expect("the batch setup must be valid");
        transcripts.push(
            ChannelTranscript::new(
                channel,
                batch
                    .transcribe_channel(samples)
                    .expect("the stereo fixture channel must transcribe")
                    .into_words(),
            )
            .expect("transcribed channel words must remain ordered"),
        );
    }
    let merged = merge(&transcripts).expect("unique stereo identities must merge");
    assert!(
        merged
            .windows(2)
            .all(|pair| pair[0].word().start() <= pair[1].word().start()),
        "timeline is not ordered by time"
    );
    let turns = turns(
        &transcripts,
        TurnGap::new(1.0).expect("test turn gap must be valid"),
    )
    .expect("stereo transcripts must produce turns");
    let word_count: usize = turns.iter().map(|turn| turn.words().len()).sum();
    let expected: usize = transcripts
        .iter()
        .map(|channel| channel.words().len())
        .sum();
    assert_eq!(word_count, expected, "turns lose or duplicate words");
    assert!(turns.iter().all(|turn| {
        turn.words()
            .iter()
            .all(|word| word.channel() == turn.channel())
    }));
}
