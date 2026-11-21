// Copyright (c) 2026 Yuriy Krasilnikov and Deeray Ltd.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use symphonia_common::xiph::audio::vorbis::vorbis_channels_to_channels;

pub(crate) const IDENTIFICATION_BYTES: usize = 30;

/// Bounds the prefix used by the Symphonia 0.6.1 Vorbis audio decoder. Matroska supplies
/// the identification and setup packets concatenated in `extra_data`. The decoder
/// derives its geometry from the identification packet itself.
pub(crate) fn audio_packet_prefix_len(extra_data: &[u8]) -> Option<u64> {
    let identification = extra_data.get(..IDENTIFICATION_BYTES)?;
    let channels = identification[11];
    let block_sizes = identification[28];
    let short_block_exp = block_sizes & 0x0f;
    let long_block_exp = block_sizes >> 4;
    if !identification.starts_with(b"\x01vorbis\0\0\0\0")
        || vorbis_channels_to_channels(channels).is_none()
        || identification[12..16] == [0; 4]
        || !(6..=13).contains(&short_block_exp)
        || !(short_block_exp..=13).contains(&long_block_exp)
        || identification[29] != 1
    {
        return None;
    }

    // The packet type, mode index and two window flags use at most nine bits.
    const PACKET_HEADER_BITS: u64 = 1 + 6 + 2;
    const CODEWORD_BITS: u64 = 32;
    // Floor 0 can read 63 amplitude bits, a five-bit book index, and up to 255 codewords.
    // Floor 1 uses at most 17 flag/endpoint bits and 31 + 63 codewords per channel.
    const FLOOR_BITS_PER_CHANNEL: u64 = 63 + 5 + 255 * CODEWORD_BITS;
    // Submaps partition the channels. Even when residue type 2 combines their spectra,
    // each channel contributes at most half a block of positions. There is at most one
    // classification per partition and one codeword per position per pass for eight passes.
    const RESIDUE_BITS_PER_POSITION: u64 = (1 + 8) * CODEWORD_BITS;
    // BitReaderRtl can fill its 64-bit reservoir ahead of the bits consumed.
    const LOOKAHEAD_BITS: u64 = 64;

    let channels = u64::from(channels);
    let max_block_size = 1_u64.checked_shl(u32::from(long_block_exp))?;
    let floor_bits = channels.checked_mul(FLOOR_BITS_PER_CHANNEL)?;
    let residue_bits = (max_block_size / 2)
        .checked_mul(channels)?
        .checked_mul(RESIDUE_BITS_PER_POSITION)?;
    let packet_bits = (PACKET_HEADER_BITS + LOOKAHEAD_BITS)
        .checked_add(floor_bits)?
        .checked_add(residue_bits)?;
    Some(packet_bits.div_ceil(8))
}
