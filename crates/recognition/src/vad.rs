// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Provider-independent speech-probability mask decisions.

use crate::contracts::SpeechProbabilityDetector;
use gigaam_audio::ChannelAudioView;

/// Current Silero VAD delivery sample rate in Hz.
pub const VAD_SR: usize = 16_000;
/// Current Silero VAD mask frame rate.
pub const VAD_FPS: f32 = 31.25;

/// Per-frame silence mask: `true` when `P(speech) < threshold`.
///
/// Finite inputs preserve the current comparison behavior; non-finite input refuses before a mask
/// is produced.
pub fn mask(probabilities: &[f32], threshold: f32) -> Result<Vec<bool>, String> {
    if !threshold.is_finite() {
        return Err("VAD silence threshold must be finite".into());
    }
    for (index, probability) in probabilities.iter().enumerate() {
        if !probability.is_finite() {
            return Err(format!(
                "VAD speech probability at frame {index} must be finite"
            ));
        }
    }
    Ok(probabilities
        .iter()
        .map(|&probability| probability < threshold)
        .collect())
}

/// Obtains probabilities from an injected execution port and projects their neutral silence mask.
/// Endpoint hysteresis and segments remain outside Recognition.
pub fn detect_mask<D: SpeechProbabilityDetector>(
    detector: &mut D,
    audio: ChannelAudioView<'_>,
    threshold: f32,
) -> Result<Vec<bool>, String> {
    let probabilities = detector.probabilities(audio)?;
    mask(&probabilities, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gigaam_audio::ChannelAudio;

    struct FakeDetector {
        expected_audio: ChannelAudio,
        probabilities: Vec<f32>,
    }

    impl SpeechProbabilityDetector for FakeDetector {
        fn probabilities(&mut self, audio: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
            if audio.samples() != self.expected_audio.samples() {
                return Err("fake detector received unexpected audio".into());
            }
            Ok(self.probabilities.clone())
        }
    }

    #[test]
    fn injected_probability_port_projects_current_silence_mask() {
        let mut detector = FakeDetector {
            expected_audio: ChannelAudio::new(vec![0.1, -0.1])
                .expect("test audio samples are finite"),
            probabilities: vec![0.2, 0.5, 0.7],
        };
        let audio = ChannelAudio::new(vec![0.1, -0.1]).expect("test audio samples are finite");
        let silence = detect_mask(&mut detector, audio.view(), 0.5)
            .expect("fake detector returns the declared probabilities");
        assert_eq!(silence, vec![true, false, false]);
    }

    #[test]
    fn mask_refuses_nonfinite_thresholds_and_probabilities_at_every_position() {
        for (name, invalid) in [
            ("NaN", f32::NAN),
            ("positive infinity", f32::INFINITY),
            ("negative infinity", f32::NEG_INFINITY),
        ] {
            assert!(
                mask(&[0.2, 0.5, 0.7], invalid).is_err(),
                "{name} threshold must refuse"
            );
            for position in [0, 1, 2] {
                let mut probabilities = vec![0.2, 0.5, 0.7];
                probabilities[position] = invalid;
                assert!(
                    mask(&probabilities, 0.5).is_err(),
                    "{name} at VAD probability position {position} must refuse"
                );
            }
        }
    }

    #[test]
    fn finite_mask_preserves_current_threshold_comparison() {
        let mask = mask(&[0.2, 0.5, 0.7], 0.5)
            .expect("finite VAD inputs satisfy the current mask contract");
        assert_eq!(mask, vec![true, false, false]);
    }

    #[test]
    fn detect_mask_propagates_invalid_probabilities() {
        let mut detector = FakeDetector {
            expected_audio: ChannelAudio::new(vec![0.1, -0.1])
                .expect("test audio samples are finite"),
            probabilities: vec![f32::NAN],
        };
        let audio = ChannelAudio::new(vec![0.1, -0.1]).expect("test audio samples are finite");

        assert!(detect_mask(&mut detector, audio.view(), 0.5).is_err());
    }
}
