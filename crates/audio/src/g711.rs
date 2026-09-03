// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! G.711 A-law and mu-law conversion into signed 16-bit PCM.

const SIGN_BIT: u8 = 0x80;
const QUANT_MASK: u8 = 0x0f;
const SEG_SHIFT: u8 = 4;
const SEG_MASK: u8 = 0x70;
const BIAS: i32 = 0x84;

fn pcm_i16(value: i32) -> i16 {
    match i16::try_from(value) {
        Ok(sample) => sample,
        Err(_) => panic!("G.711 decoding formulas always produce signed 16-bit PCM"),
    }
}

pub(crate) fn ulaw_to_i16(value: u8) -> i16 {
    let value = !value;
    let mut magnitude = i32::from(value & QUANT_MASK) << 3;
    magnitude += BIAS;
    magnitude <<= i32::from((value & SEG_MASK) >> SEG_SHIFT);
    pcm_i16(if value & SIGN_BIT != 0 {
        BIAS - magnitude
    } else {
        magnitude - BIAS
    })
}

pub(crate) fn alaw_to_i16(value: u8) -> i16 {
    let value = value ^ 0x55;
    let mut magnitude = i32::from(value & QUANT_MASK) << 4;
    let segment = i32::from((value & SEG_MASK) >> SEG_SHIFT);
    match segment {
        0 => magnitude += 8,
        1 => magnitude += 0x108,
        _ => {
            magnitude += 0x108;
            magnitude <<= segment - 1;
        }
    }
    pcm_i16(if value & SIGN_BIT != 0 {
        magnitude
    } else {
        -magnitude
    })
}

#[cfg(test)]
mod tests {
    use super::{alaw_to_i16, ulaw_to_i16};

    #[test]
    fn ulaw_reference_points() {
        assert_eq!(ulaw_to_i16(0xFF), 0);
        assert_eq!(ulaw_to_i16(0x7F), 0);
        assert_eq!(ulaw_to_i16(0x00), -32124);
        assert_eq!(ulaw_to_i16(0x80), 32124);
    }

    #[test]
    fn alaw_reference_points() {
        assert_eq!(alaw_to_i16(0x55), -8);
        assert_eq!(alaw_to_i16(0xD5), 8);
        assert_eq!(alaw_to_i16(0x2A), -32256);
        assert_eq!(alaw_to_i16(0xAA), 32256);
    }
}
