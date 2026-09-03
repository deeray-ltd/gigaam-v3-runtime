// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! RNN-T batch transcription acceptance against the model golden output.

mod common;

use gigaam_audio::read_wav;
use gigaam_model_package::{EncoderPrecision, ModelPackage};
use gigaam_recognition::{DirectRecognizer, ExecutionControl};
use gigaam_transcription::{
    BatchConfig, BatchSetup, BatchTranscriber, ObservationMode, PadPolicy, word_edits,
};

#[test]
fn rnnt_reproduces_golden_text() {
    common::init_ort();
    let root = common::root();
    let pack = ModelPackage::open(&root.join("model")).expect("the test model pack must open");
    for clip in ["example", "long_example"] {
        let decoder = DirectRecognizer::rnnt(&pack, &common::cpu_plan(), EncoderPrecision::Fp32)
            .expect("the RNNT decoder must match the test pack");
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
        let audio = read_wav(&root.join(format!("fixtures/{clip}.wav")))
            .expect("the RNNT fixture must decode");
        let transcript = batch
            .transcribe_channel(&audio.channels()[0])
            .expect("the RNNT fixture must transcribe");
        let expected = std::fs::read_to_string(root.join(format!("fixtures/{clip}.rnnt.gold.txt")))
            .expect("the RNNT golden text must be available");
        let edits = word_edits(&expected, transcript.text());
        eprintln!(
            "{clip} RNNT: edits against golden output {edits}\n  {}",
            transcript.text()
        );
        assert_eq!(
            transcript.text().trim(),
            expected.trim(),
            "{clip}: RNNT diverged from golden output"
        );
    }
}
