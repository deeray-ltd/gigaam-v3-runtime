// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Coded-frame geometry for the interleaved PCM layouts accepted by Symphonia.

use std::num::NonZeroUsize;
use symphonia::core::codecs::audio::{AudioCodecParameters, well_known::*};

#[derive(Clone, Copy)]
pub(crate) struct PcmFrameGeometry {
    pub(crate) frame_bytes: NonZeroUsize,
}

impl PcmFrameGeometry {
    pub(crate) fn from_params(params: &AudioCodecParameters) -> Result<Option<Self>, String> {
        let width: usize = match params.codec {
            CODEC_ID_PCM_S8 | CODEC_ID_PCM_U8 | CODEC_ID_PCM_ALAW | CODEC_ID_PCM_MULAW => 1,
            CODEC_ID_PCM_S16LE | CODEC_ID_PCM_S16BE | CODEC_ID_PCM_U16LE | CODEC_ID_PCM_U16BE => 2,
            CODEC_ID_PCM_S24LE | CODEC_ID_PCM_S24BE | CODEC_ID_PCM_U24LE | CODEC_ID_PCM_U24BE => 3,
            CODEC_ID_PCM_S32LE | CODEC_ID_PCM_S32BE | CODEC_ID_PCM_U32LE | CODEC_ID_PCM_U32BE
            | CODEC_ID_PCM_F32LE | CODEC_ID_PCM_F32BE => 4,
            CODEC_ID_PCM_F64LE | CODEC_ID_PCM_F64BE => 8,
            _ => return Ok(None),
        };
        let Some(channels) = params.channels.as_ref() else {
            return Ok(None);
        };
        let frame_bytes = width
            .checked_mul(channels.count())
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| "PCM coded frame width is invalid".to_owned())?;
        Ok(Some(Self { frame_bytes }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symphonia::core::{
        audio::Channels,
        codecs::audio::{AudioCodecId, AudioDecoderOptions},
    };
    use symphonia::core::{
        packet::Packet,
        units::{Duration, Timestamp},
    };

    #[test]
    fn geometry_matches_every_linked_interleaved_decoder_and_excludes_planar() -> Result<(), String>
    {
        let cases: &[(AudioCodecId, usize)] = &[
            (CODEC_ID_PCM_S8, 1),
            (CODEC_ID_PCM_U8, 1),
            (CODEC_ID_PCM_ALAW, 1),
            (CODEC_ID_PCM_MULAW, 1),
            (CODEC_ID_PCM_S16LE, 2),
            (CODEC_ID_PCM_S16BE, 2),
            (CODEC_ID_PCM_U16LE, 2),
            (CODEC_ID_PCM_U16BE, 2),
            (CODEC_ID_PCM_S24LE, 3),
            (CODEC_ID_PCM_S24BE, 3),
            (CODEC_ID_PCM_U24LE, 3),
            (CODEC_ID_PCM_U24BE, 3),
            (CODEC_ID_PCM_S32LE, 4),
            (CODEC_ID_PCM_S32BE, 4),
            (CODEC_ID_PCM_U32LE, 4),
            (CODEC_ID_PCM_U32BE, 4),
            (CODEC_ID_PCM_F32LE, 4),
            (CODEC_ID_PCM_F32BE, 4),
            (CODEC_ID_PCM_F64LE, 8),
            (CODEC_ID_PCM_F64BE, 8),
        ];
        for &(codec, width) in cases {
            let mut params = AudioCodecParameters::new();
            params
                .for_codec(codec)
                .with_channels(Channels::Discrete(2))
                .with_sample_rate(8000);
            let geometry =
                PcmFrameGeometry::from_params(&params)?.ok_or("supported PCM geometry")?;
            assert_eq!(geometry.frame_bytes.get(), width * 2);
            let mut decoder = symphonia::default::get_codecs()
                .make_audio_decoder(&params, &AudioDecoderOptions::default())
                .map_err(|error| error.to_string())?;
            let packet = Packet::new(1, Timestamp::ZERO, Duration::new(1), vec![0_u8; width * 2]);
            let decoded = decoder.decode(&packet).map_err(|error| error.to_string())?;
            assert_eq!(decoded.frames(), 1);
        }
        for codec in [CODEC_ID_PCM_S16LE_PLANAR, CODEC_ID_PCM_F32LE_PLANAR] {
            let mut params = AudioCodecParameters::new();
            params
                .for_codec(codec)
                .with_channels(Channels::Discrete(2))
                .with_sample_rate(8000);
            assert!(PcmFrameGeometry::from_params(&params)?.is_none());
            assert!(
                symphonia::default::get_codecs()
                    .make_audio_decoder(&params, &AudioDecoderOptions::default())
                    .is_err()
            );
        }
        Ok(())
    }
}
