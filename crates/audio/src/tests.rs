// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

use crate::{
    ChannelAudio, ChannelAudioView, EncodedAudio, FeatureMatrix, FeatureMatrixView, FrontendMode,
    SampleRate,
};

#[test]
fn typed_input_contracts_refuse_invalid_values() {
    assert!(SampleRate::new(0).is_err());
    assert!(ChannelAudio::new(vec![f32::NAN]).is_err());
    assert!(EncodedAudio::new(Vec::new(), None).is_err());
    assert!(EncodedAudio::new(vec![1], Some(String::new())).is_err());
    assert!(EncodedAudio::new(vec![1], Some("mp-3".into())).is_err());
    assert!(FrontendMode::parse("invalid").is_err());
}

#[test]
fn feature_frame_range_preserves_validated_mel_time_layout() {
    let matrix = FeatureMatrix::from_values(2, 3, vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        .expect("test feature matrix dimensions and values are valid");
    let range = matrix
        .frame_range(1, 3)
        .expect("a contained frame range is valid");
    assert_eq!(range.mel_bins(), 2);
    assert_eq!(range.frames(), 2);
    assert_eq!(range.values(), &[2.0, 3.0, 20.0, 30.0]);
    assert!(matrix.frame_range(3, 1).is_err());
    assert!(matrix.frame_range(0, 4).is_err());
}

#[test]
fn borrowed_audio_and_feature_views_preserve_audio_validation_and_layout() {
    let audio = ChannelAudio::new(vec![0.25, -0.5]).expect("finite test audio is valid");
    assert_eq!(audio.view().samples(), &[0.25, -0.5]);
    assert!(ChannelAudioView::new(&[f32::NAN]).is_err());

    let matrix = FeatureMatrix::from_values(2, 1, vec![0.25, -0.5])
        .expect("finite test features have the declared shape");
    let view = matrix.view();
    assert_eq!(view.mel_bins(), 2);
    assert_eq!(view.frames(), 1);
    assert_eq!(view.values(), &[0.25, -0.5]);
    assert!(FeatureMatrixView::new(2, 1, &[0.25]).is_err());
    assert!(FeatureMatrixView::new(1, 1, &[f32::NAN]).is_err());
}
