// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! CPU streaming acceptance: client patches, endpointing, and batch agreement.

mod common;

use gigaam_audio::{ChannelAudio, SampleRate, read_wav};
use gigaam_model_package::{EncoderPrecision, ModelPackage};
use gigaam_recognition::{DirectRecognizer, ExecutionControl, Vad};
use gigaam_transcription::{
    BatchConfig, BatchSetup, BatchTranscriber, EndpointDetector, FinalReason, ObservationMode,
    PadPolicy, StreamConfig, StreamEvent, StreamLockPolicy, StreamSession, StreamSetup, StreamWord,
    WordStability, word_count, word_edits,
};

type Session = StreamSession<DirectRecognizer, Vad>;

fn model_rate(pack: &ModelPackage) -> SampleRate {
    SampleRate::from_usize(pack.frontend().sample_rate(), "test model rate")
        .expect("test model package rate is representable")
}

fn default_config(pack: &ModelPackage) -> StreamConfig {
    StreamConfig::checked_default(model_rate(pack))
        .expect("default stream configuration must be valid")
}

fn cpu_session(pack: &ModelPackage, config: StreamConfig) -> Session {
    let decoder = DirectRecognizer::ctc(pack, &common::cpu_plan(), EncoderPrecision::Fp32)
        .expect("the CTC decoder must match the test pack");
    StreamSession::new(StreamSetup {
        frontend: common::frontend(pack),
        decoder,
        config,
        detector: EndpointDetector::Blank,
        control: ExecutionControl::without_deadline(),
    })
    .expect("the stream session configuration must be valid")
}

fn apply(events: Vec<StreamEvent>, client: &mut Vec<StreamWord>) -> usize {
    let mut endpoints = 0;
    for event in events {
        match event {
            StreamEvent::Words(event) => {
                client.truncate(event.revise_from());
                client.extend_from_slice(event.words());
            }
            StreamEvent::Stable(event) => {
                for word in &mut client[..event.upto()] {
                    *word = word.clone().with_stability(WordStability::Stable);
                }
            }
            StreamEvent::Final(event) => {
                for word in &mut client[..event.upto()] {
                    *word = word.clone().with_stability(WordStability::Stable);
                }
                if event.reason() == FinalReason::Endpoint {
                    endpoints += 1;
                }
            }
        }
    }
    endpoints
}

fn run(mut session: Session, samples: &[f32], chunk: usize) -> Vec<StreamWord> {
    let mut client = Vec::new();
    for part in samples.chunks(chunk) {
        let channel = ChannelAudio::new(part.to_vec()).expect("fixture samples must be finite");
        apply(
            session
                .push(&channel)
                .expect("stream input chunks must decode"),
            &mut client,
        );
        assert_eq!(client, session.transcript(), "client diverged from session");
    }
    apply(
        session.flush().expect("stream finalization must decode"),
        &mut client,
    );
    assert_eq!(client, session.transcript());
    client
}

#[test]
fn stream_matches_batch_and_patches_are_consistent() {
    common::init_ort();
    let root = common::root();
    let pack = ModelPackage::open(&root.join("model")).expect("the test model pack must open");
    let audio = read_wav(&root.join("fixtures/example.wav")).expect("the test WAV must decode");
    let decoder = DirectRecognizer::ctc(&pack, &common::cpu_plan(), EncoderPrecision::Fp32)
        .expect("the CTC decoder must match the test pack");
    let mut batch = BatchTranscriber::new(BatchSetup {
        frontend: common::frontend(&pack),
        decoder,
        config: BatchConfig::new(model_rate(&pack), 30.0, 6.0, PadPolicy::PadToWindow)
            .expect("the documented stream window must be valid"),
        control: ExecutionControl::without_deadline(),
        observations: ObservationMode::disabled(),
    })
    .expect("the batch setup must be valid");
    let expected = batch
        .transcribe_channel(&audio.channels()[0])
        .expect("the test clip must transcribe in batch mode");
    let client = run(
        cpu_session(&pack, default_config(&pack)),
        audio.channels()[0].samples(),
        1600,
    );
    assert!(
        client
            .iter()
            .all(|word| word.stability() == WordStability::Stable)
    );
    let text = client
        .iter()
        .map(StreamWord::text)
        .collect::<Vec<_>>()
        .join(" ");
    let edits = word_edits(expected.text(), &text);
    eprintln!("batch: {}\nstream: {text}\nedits {edits}", expected.text());
    assert!(edits <= 1, "stream diverged from batch: {edits} edits");
}

#[test]
fn stream_forced_cuts_lock_no_adjacent_duplicates() {
    common::init_ort();
    let root = common::root();
    let pack = ModelPackage::open(&root.join("model")).expect("the test model pack must open");
    let audio =
        read_wav(&root.join("fixtures/long_example.wav")).expect("the long test WAV must decode");
    let config = StreamConfig::timing_changes()
        .with_horizon_sec(5.0)
        .expect("test horizon must be valid")
        .with_step_sec(0.5)
        .expect("test step must be valid")
        .apply(default_config(&pack))
        .expect("combined stream timing must be valid")
        .with_lock_policy(StreamLockPolicy::CommitStable)
        .expect("lock policy must preserve a valid configuration");
    let client = run(
        cpu_session(&pack, config),
        audio.channels()[0].samples(),
        1600,
    );
    let duplicates: Vec<&str> = client
        .windows(2)
        .filter(|pair| {
            gigaam_transcription::normalize_word(pair[0].text())
                == gigaam_transcription::normalize_word(pair[1].text())
                && (pair[0].start() - pair[1].start()).abs() < 0.15
        })
        .map(|pair| pair[0].text())
        .collect();
    assert!(
        duplicates.is_empty(),
        "duplicate words at committed boundary: {duplicates:?}"
    );
    assert!(
        client
            .iter()
            .all(|word| word.stability() == WordStability::Stable)
    );
}

#[test]
fn stream_endpoint_fires_on_silence_and_keeps_both_parts() {
    common::init_ort();
    let root = common::root();
    let pack = ModelPackage::open(&root.join("model")).expect("the test model pack must open");
    let example = read_wav(&root.join("fixtures/example.wav")).expect("the test WAV must decode");
    let mut samples = example.channels()[0].samples().to_vec();
    samples.extend(std::iter::repeat_n(0.0_f32, 32000));
    samples.extend_from_slice(example.channels()[0].samples());
    let mut session = cpu_session(&pack, default_config(&pack));
    let mut client = Vec::new();
    let mut endpoints = 0;
    for part in samples.chunks(1600) {
        let channel = ChannelAudio::new(part.to_vec()).expect("fixture samples must be finite");
        endpoints += apply(
            session
                .push(&channel)
                .expect("stream input chunks must decode"),
            &mut client,
        );
    }
    endpoints += apply(
        session.flush().expect("stream finalization must decode"),
        &mut client,
    );
    let words = word_count(
        &client
            .iter()
            .map(StreamWord::text)
            .collect::<Vec<_>>()
            .join(" "),
    );
    let single = word_count(
        &std::fs::read_to_string(root.join("fixtures/example.ctc.gold.txt"))
            .expect("the CTC golden text must be available"),
    );
    assert!(endpoints >= 1, "2 s pause must close a segment");
    assert!(
        words >= single + single / 2,
        "both parts must appear in text: {words} vs {single}"
    );
}
