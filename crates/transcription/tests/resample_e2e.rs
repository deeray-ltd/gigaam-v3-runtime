// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! End-to-end resampler acceptance through the Transcription batch workflow.

mod common;

use gigaam_audio::{ChannelAudio, RatePair, Resampler, ResamplerConfig, SampleRate, read_wav};
use gigaam_model_package::{EncoderPrecision, ModelPackage};
use gigaam_recognition::{DirectRecognizer, ExecutionControl};
use gigaam_transcription::{
    BatchConfig, BatchSetup, BatchTranscriber, ObservationMode, PadPolicy, word_edits,
};

fn resampler(input: u32, output: u32) -> Resampler {
    let pair = RatePair::new(
        SampleRate::new(input).expect("fixed test input rate is valid"),
        SampleRate::new(output).expect("fixed test output rate is valid"),
    )
    .expect("fixed test rate ratio is supported");
    Resampler::new(ResamplerConfig::new(pair)).expect("fixed test resampler must construct")
}

#[test]
fn telephony_roundtrip_keeps_text() {
    common::init_ort();
    let root = common::root();
    let pack = ModelPackage::open(&root.join("model")).expect("the test model pack must open");
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
    let audio = read_wav(&root.join("fixtures/example.wav")).expect("the test WAV must decode");
    let down = resampler(16000, 8000)
        .process(audio.channels()[0].samples())
        .expect("test downsample must succeed");
    let up = resampler(8000, 16000)
        .process(&down)
        .expect("test upsample must succeed");
    let up_len = i64::try_from(up.len()).expect("resampled length must fit i64");
    let source_len = i64::try_from(audio.channels()[0].len()).expect("source length must fit i64");
    assert!((up_len - source_len).abs() <= 2);
    let resampled = ChannelAudio::new(up).expect("resampler output must remain finite");
    let transcript = batch
        .transcribe_channel(&resampled)
        .expect("the resampled fixture must transcribe");
    let expected = std::fs::read_to_string(root.join("fixtures/example.ctc.gold.txt"))
        .expect("the CTC golden text must be available");
    let edits = word_edits(&expected, transcript.text());
    eprintln!(
        "after 16→8→16 kHz: word edits = {edits}\n  {}",
        transcript.text()
    );
    assert!(edits <= 2, "telephone path degraded text: {edits} edits");
}
