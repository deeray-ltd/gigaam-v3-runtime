// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! End-to-end CPU batch-transcription acceptance against the CTC golden text.

mod common;

use gigaam_audio::{SampleRate, read_wav};
use gigaam_model_package::{EncoderPrecision, ModelPackage};
use gigaam_recognition::{DirectRecognizer, ExecutionControl};
use gigaam_transcription::{
    BatchConfig, BatchSetup, BatchTranscriber, ObservationMode, PadPolicy, normalize_word,
};

fn batch(
    pack: &ModelPackage,
    precision: EncoderPrecision,
    config: BatchConfig,
) -> BatchTranscriber<DirectRecognizer> {
    let decoder = DirectRecognizer::ctc(pack, &common::cpu_plan(), precision)
        .expect("the CTC decoder must match the test pack");
    BatchTranscriber::new(BatchSetup {
        frontend: common::frontend(pack),
        decoder,
        config,
        control: ExecutionControl::without_deadline(),
        observations: ObservationMode::disabled(),
    })
    .expect("the documented batch setup must be valid")
}

fn golden(pack: &ModelPackage, precision: EncoderPrecision, config: BatchConfig) -> String {
    let root = common::root();
    let audio = read_wav(&root.join("fixtures/example.wav")).expect("the test WAV must decode");
    batch(pack, precision, config)
        .transcribe_channel(&audio.channels()[0])
        .expect("the test WAV must transcribe")
        .text()
        .to_owned()
}

fn model_rate(pack: &ModelPackage) -> SampleRate {
    SampleRate::from_usize(pack.frontend().sample_rate(), "test model rate")
        .expect("test model package rate is representable")
}

#[test]
fn cpu_end_to_end_matches_golden_text() {
    common::init_ort();
    let root = common::root();
    let pack = ModelPackage::open(&root.join("model")).expect("the test model pack must open");
    let text = golden(
        &pack,
        EncoderPrecision::Fp32,
        BatchConfig::new(model_rate(&pack), 100.0, 6.0, PadPolicy::Exact)
            .expect("the documented batch window must be valid"),
    );
    let expected = std::fs::read_to_string(root.join("fixtures/example.ctc.gold.txt"))
        .expect("the CTC golden text must be available");
    assert_eq!(text, expected);
}

#[test]
fn cpu_fp16_graph_with_fp32_io_matches_golden_text() {
    common::init_ort();
    let root = common::root();
    let pack = ModelPackage::open(&root.join("model")).expect("the test model pack must open");
    let text = golden(
        &pack,
        EncoderPrecision::Fp16Io32,
        BatchConfig::new(model_rate(&pack), 100.0, 6.0, PadPolicy::Exact)
            .expect("the documented batch window must be valid"),
    );
    let expected = std::fs::read_to_string(root.join("fixtures/example.ctc.gold.txt"))
        .expect("the CTC golden text must be available");
    assert_eq!(text, expected);
}

#[test]
fn pad_to_window_keeps_words_and_trailing_punctuation() {
    common::init_ort();
    let root = common::root();
    let pack = ModelPackage::open(&root.join("model")).expect("the test model pack must open");
    let text = golden(
        &pack,
        EncoderPrecision::Fp32,
        BatchConfig::new(model_rate(&pack), 30.0, 6.0, PadPolicy::PadToWindow)
            .expect("the stream window must be valid"),
    );
    let expected = std::fs::read_to_string(root.join("fixtures/example.ctc.gold.txt"))
        .expect("the CTC golden text must be available");
    let expected_words: Vec<String> = expected.split_whitespace().map(normalize_word).collect();
    let actual_words: Vec<String> = text.split_whitespace().map(normalize_word).collect();
    assert_eq!(
        actual_words, expected_words,
        "padding changed words:\n{text}"
    );
    assert!(
        text.trim_end().ends_with('.'),
        "trailing punctuation must be preserved: {text}"
    );
}
