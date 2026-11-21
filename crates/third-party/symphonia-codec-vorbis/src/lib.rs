// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![warn(rust_2018_idioms)]
#![forbid(unsafe_code)]
// The following lints are allowed in all Symphonia crates. Please see clippy.toml for their
// justification.
#![allow(clippy::comparison_chain)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_range_contains)]
// Disable to better express the specification.
#![allow(clippy::collapsible_else_if)]

use symphonia_core::audio::{
    AsGenericAudioBufferRef, AudioBuffer, AudioMut, AudioSpec, GenericAudioBufferRef,
};
use symphonia_core::codecs::CodecInfo;
use symphonia_core::codecs::audio::well_known::CODEC_ID_VORBIS;
use symphonia_core::codecs::audio::{AudioCodecParameters, AudioDecoderOptions};
use symphonia_core::codecs::audio::{AudioDecoder, FinalizeResult};
use symphonia_core::codecs::registry::{RegisterableAudioDecoder, SupportedAudioCodec};
use symphonia_core::dsp::mdct::Imdct;
use symphonia_core::errors::{Result, decode_error, unsupported_error};
use symphonia_core::io::{
    BitReaderRtl, BitStreamRtl, BufReader, FiniteBitStream, ReadBitsRtl, ReadBytes,
};
use symphonia_core::packet::PacketRef;
use symphonia_core::support_audio_codec;

use symphonia_common::xiph::audio::vorbis::*;

use log::{debug, warn};

mod codebook;
mod common;
mod dsp;
mod floor;
mod residue;
mod window;

use codebook::VorbisCodebook;
use common::*;
use dsp::*;
use floor::*;
use residue::*;
use window::Windows;

/// Vorbis decoder.
pub struct VorbisDecoder {
    /// Decoder options.
    opts: AudioDecoderOptions,
    /// Codec paramters.
    params: AudioCodecParameters,
    /// Identity header.
    ident: IdentHeader,
    /// Codebooks (max. 256).
    codebooks: Vec<VorbisCodebook>,
    /// Floors (max. 64).
    floors: Vec<Box<dyn Floor>>,
    /// Residues (max. 64).
    residues: Vec<Residue>,
    /// Modes (max. 64).
    modes: Vec<Mode>,
    /// Mappings (max. 64).
    mappings: Vec<Mapping>,
    /// DSP.
    dsp: Dsp,
    /// Output buffer.
    buf: AudioBuffer<f32>,
}

fn read_extradata(buf: &[u8]) -> Result<(IdentHeader, Setup)> {
    let is_xiph_laced = !buf.is_empty() && buf[0] == XIPH_LACED_LEADING_HEADER;
    if is_xiph_laced {
        let (pkt_ident, pkt_setup) = unpack_xiph_laced_extradata(buf)?;
        let ident = read_ident_header(&mut BufReader::new(pkt_ident))?;
        let setup = read_setup(&mut BufReader::new(pkt_setup), &ident)?;
        Ok((ident, setup))
    }
    else {
        let mut reader = BufReader::new(buf);
        let ident = read_ident_header(&mut reader)?;
        let setup = read_setup(&mut reader, &ident)?;
        Ok((ident, setup))
    }
}

/// Read the significant bytes of a Vorbis setup packet through its framing bit.
///
/// The caller supplies the identification packet and a reader positioned at the setup
/// packet's first byte. Bound the reader to the declared packet extent. This parses the
/// setup syntax without retaining trailing bytes or building the decoder's codebooks.
/// The returned length includes the byte containing the framing bit.
pub fn vorbis_setup_prefix_len<R: ReadBytes>(identification: &[u8], setup: &mut R) -> Result<u64> {
    let ident = read_ident_header(&mut BufReader::new(identification))?;
    let start = setup.pos();
    read_setup_header(setup)?;
    read_setup_body::<_, false>(&mut BitStreamRtl::new(setup), &ident)?;
    match setup.pos().checked_sub(start) {
        Some(len) => Ok(len),
        None => decode_error("vorbis: setup packet reader moved backwards"),
    }
}

impl VorbisDecoder {
    pub fn try_new(params: &AudioCodecParameters, opts: &AudioDecoderOptions) -> Result<Self> {
        // This decoder only supports Vorbis.
        if params.codec != CODEC_ID_VORBIS {
            return unsupported_error("vorbis: invalid codec");
        }

        // Get the extra data (mandatory).
        let extra_data = match params.extra_data.as_ref() {
            Some(buf) => buf,
            _ => return unsupported_error("vorbis: missing extra data"),
        };

        // The extra data contains the identification and setup headers.
        let (ident, setup) = read_extradata(extra_data)?;

        // Initialize static DSP data.
        let windows = Windows::new(1 << ident.bs0_exp, 1 << ident.bs1_exp);

        // Initialize dynamic DSP for each channel.
        let dsp_channels =
            (0..ident.n_channels).map(|_| DspChannel::new(ident.bs0_exp, ident.bs1_exp)).collect();

        // Map the channels
        let channels = match vorbis_channels_to_channels(ident.n_channels) {
            Some(channels) => channels,
            _ => return unsupported_error("vorbis: unknown channel map (fix me)"),
        };

        // Initialize the output buffer.
        let spec = AudioSpec::new(ident.sample_rate, channels);

        let imdct_short = Imdct::new((1 << ident.bs0_exp) >> 1);
        let imdct_long = Imdct::new((1 << ident.bs1_exp) >> 1);

        // TODO: Should this be half the block size?
        let duration = 1 << ident.bs1_exp;

        let dsp =
            Dsp { windows, channels: dsp_channels, imdct_short, imdct_long, prev_block_flag: None };

        Ok(VorbisDecoder {
            opts: *opts,
            params: params.clone(),
            ident,
            codebooks: setup.codebooks,
            floors: setup.floors,
            residues: setup.residues,
            modes: setup.modes,
            mappings: setup.mappings,
            dsp,
            buf: AudioBuffer::new(spec, duration),
        })
    }

    fn decode_inner(&mut self, packet: &PacketRef<'_>) -> Result<()> {
        let mut bs = BitReaderRtl::new(packet.data);

        // Section 4.3.1 - Packet Type, Mode, and Window Decode

        // First bit must be 0 to indicate audio packet.
        if bs.read_bool()? {
            return decode_error("vorbis: not an audio packet");
        }

        let num_modes = self.modes.len() - 1;

        let mode_number = bs.read_bits_leq32(common::ilog(num_modes as u32))? as usize;

        if mode_number >= self.modes.len() {
            return decode_error("vorbis: invalid packet mode number");
        }

        let mode = &self.modes[mode_number];
        let mapping = &self.mappings[usize::from(mode.mapping)];

        let (bs_exp, imdct) = if mode.block_flag {
            // This packet (block) uses a long window. Do not use the window flags since they may
            // be wrong.
            let _prev_window_flag = bs.read_bool()?;
            let _next_window_flag = bs.read_bool()?;

            (self.ident.bs1_exp, &mut self.dsp.imdct_long)
        }
        else {
            // This packet (block) uses a short window.
            (self.ident.bs0_exp, &mut self.dsp.imdct_short)
        };

        // Block, and half-block size
        let n = 1 << bs_exp;
        let n2 = n >> 1;

        // Section 4.3.2 - Floor Curve Decode

        // Read the floors from the packet. There is one floor per audio channel. Each mapping will
        // have one multiplex (submap number) per audio channel. Therefore, iterate over all
        // muxes in the mapping, and read the floor.
        for (&submap_num, ch) in mapping.multiplex.iter().zip(&mut self.dsp.channels) {
            let submap = &mapping.submaps[submap_num as usize];
            let floor = &mut self.floors[submap.floor as usize];

            // Read the floor from the bitstream.
            floor.read_channel(&mut bs, &self.codebooks)?;

            ch.do_not_decode = floor.is_unused();

            if !ch.do_not_decode {
                // Since the same floor can be used by multiple channels and thus overwrite the
                // data just read from the bitstream, synthesize the floor curve for this channel
                // now and save it for audio synthesis later.
                floor.synthesis(bs_exp, &mut ch.floor)?;
            }
            else {
                // If the channel is unused, zero the floor vector.
                ch.floor[..n2].fill(0.0);
            }
        }

        // Section 4.3.3 - Non-zero Vector Propagate

        // If within a pair of coupled channels, one channel has an unused floor (do_not_decode
        // is true for that channel), but the other channel is used, then both channels must have
        // do_not_decode unset.
        for couple in &mapping.couplings {
            let magnitude_ch_idx = usize::from(couple.magnitude_ch);
            let angle_ch_idx = usize::from(couple.angle_ch);

            if self.dsp.channels[magnitude_ch_idx].do_not_decode
                != self.dsp.channels[angle_ch_idx].do_not_decode
            {
                self.dsp.channels[magnitude_ch_idx].do_not_decode = false;
                self.dsp.channels[angle_ch_idx].do_not_decode = false;
            }
        }

        // Section 4.3.4 - Residue Decode

        for (submap_idx, submap) in mapping.submaps.iter().enumerate() {
            let mut residue_channels: BitSet256 = Default::default();

            // Find the channels using this submap.
            for (c, &ch_submap_idx) in mapping.multiplex.iter().enumerate() {
                if submap_idx == usize::from(ch_submap_idx) {
                    residue_channels.set(c)
                }
            }

            let residue = &mut self.residues[submap.residue as usize];

            residue.read_residue(
                &mut bs,
                bs_exp,
                &self.codebooks,
                &residue_channels,
                &mut self.dsp.channels,
            )?;
        }

        // Section 4.3.5 - Inverse Coupling

        for coupling in mapping.couplings.iter().rev() {
            debug_assert!(coupling.magnitude_ch != coupling.angle_ch);

            // Get mutable reference to each channel in the pair.
            let (magnitude_ch, angle_ch) = if coupling.magnitude_ch < coupling.angle_ch {
                // Magnitude channel index < angle channel index.
                let (a, b) = self.dsp.channels.split_at_mut(coupling.angle_ch as usize);
                (&mut a[coupling.magnitude_ch as usize], &mut b[0])
            }
            else {
                // Angle channel index < magnitude channel index.
                let (a, b) = self.dsp.channels.split_at_mut(coupling.magnitude_ch as usize);
                (&mut b[0], &mut a[coupling.angle_ch as usize])
            };

            for (m, a) in magnitude_ch.residue[..n2].iter_mut().zip(&mut angle_ch.residue[..n2]) {
                let (new_m, new_a) = if *m > 0.0 {
                    if *a > 0.0 { (*m, *m - *a) } else { (*m + *a, *m) }
                }
                else {
                    if *a > 0.0 { (*m, *m + *a) } else { (*m - *a, *m) }
                };

                *m = new_m;
                *a = new_a;
            }
        }

        // Section 4.3.6 - Dot Product

        for channel in self.dsp.channels.iter_mut() {
            // If the channel is marked as do not decode, the floor vector is all 0. Therefore the
            // dot product will be 0.
            if channel.do_not_decode {
                continue;
            }

            for (f, r) in channel.floor[..n2].iter_mut().zip(&mut channel.residue[..n2]) {
                *f *= *r;
            }
        }

        // Combined Section 4.3.7 and 4.3.8 - Inverse MDCT and Overlap-add (Synthesis)
        self.buf.clear();

        // Calculate the output length and reserve space in the output buffer. If there was no
        // previous packet, then use the current block size so the rendered length is half the
        // current block size.
        let prev_block_flag = self.dsp.prev_block_flag.unwrap_or(mode.block_flag);

        let prev_block_n =
            1 << if prev_block_flag { self.ident.bs1_exp } else { self.ident.bs0_exp };

        self.buf.render_uninit(Some((prev_block_n + n) / 4));

        // Render all the audio channels.
        for (i, channel) in self.dsp.channels.iter_mut().enumerate() {
            let ch_idx = map_vorbis_channel(self.ident.n_channels, i);
            let plane = match self.buf.plane_mut(ch_idx) {
                Some(p) => p,
                None => return decode_error("vorbis: missing audio plane"),
            };
            channel.synth(mode.block_flag, prev_block_flag, &self.dsp.windows, imdct, plane);
        }

        // Trim gaps.
        if self.opts.gapless {
            if self.dsp.prev_block_flag.is_none() {
                // The first packet after a decoder reset is silenced when trimming gaps.
                self.buf.clear();
            }
            else {
                self.buf.trim(packet.trim_start.get() as usize, packet.trim_end.get() as usize);
            }
        }

        // Update the previous block flag.
        self.dsp.prev_block_flag = Some(mode.block_flag);

        Ok(())
    }
}

impl AudioDecoder for VorbisDecoder {
    fn reset(&mut self) {
        self.dsp.reset();
    }

    fn codec_info(&self) -> &CodecInfo {
        // Only one codec is supported.
        &Self::supported_codecs().first().expect("at least one codec registered").info
    }

    fn codec_params(&self) -> &AudioCodecParameters {
        &self.params
    }

    fn decode_ref(&mut self, packet: &PacketRef<'_>) -> Result<GenericAudioBufferRef<'_>> {
        if let Err(e) = self.decode_inner(packet) {
            self.buf.clear();
            Err(e)
        }
        else {
            Ok(self.buf.as_generic_audio_buffer_ref())
        }
    }

    fn finalize(&mut self) -> FinalizeResult {
        Default::default()
    }

    fn last_decoded(&self) -> GenericAudioBufferRef<'_> {
        self.buf.as_generic_audio_buffer_ref()
    }
}

impl RegisterableAudioDecoder for VorbisDecoder {
    fn try_registry_new(
        params: &AudioCodecParameters,
        opts: &AudioDecoderOptions,
    ) -> Result<Box<dyn AudioDecoder>>
    where
        Self: Sized,
    {
        Ok(Box::new(VorbisDecoder::try_new(params, opts)?))
    }

    fn supported_codecs() -> &'static [SupportedAudioCodec] {
        &[support_audio_codec!(CODEC_ID_VORBIS, "vorbis", "Vorbis")]
    }
}

#[derive(Debug)]
struct IdentHeader {
    n_channels: u8,
    sample_rate: u32,
    bs0_exp: u8,
    bs1_exp: u8,
}

/// The packet type for an identification header.
const VORBIS_PACKET_TYPE_IDENTIFICATION: u8 = 1;
/// The packet type for a setup header.
const VORBIS_PACKET_TYPE_SETUP: u8 = 5;

/// The common header packet signature.
const VORBIS_HEADER_PACKET_SIGNATURE: &[u8] = b"vorbis";

/// The Vorbis version supported by this mapper.
const VORBIS_VERSION: u32 = 0;

/// The minimum block size (64) expressed as a power-of-2 exponent.
const VORBIS_BLOCKSIZE_MIN: u8 = 6;
/// The maximum block size (8192) expressed as a power-of-2 exponent.
const VORBIS_BLOCKSIZE_MAX: u8 = 13;

fn read_ident_header<B: ReadBytes>(reader: &mut B) -> Result<IdentHeader> {
    // The packet type must be an identification header.
    let packet_type = reader.read_u8()?;

    if packet_type != VORBIS_PACKET_TYPE_IDENTIFICATION {
        return decode_error("vorbis: invalid packet type for identification header");
    }

    // Next, the header packet signature must be correct.
    let mut packet_sig_buf = [0; 6];
    reader.read_buf_exact(&mut packet_sig_buf)?;

    if packet_sig_buf != VORBIS_HEADER_PACKET_SIGNATURE {
        return decode_error("vorbis: invalid header signature");
    }

    // Next, the Vorbis version must be 0.
    let version = reader.read_u32()?;

    if version != VORBIS_VERSION {
        return unsupported_error("vorbis: only vorbis 1 is supported");
    }

    // Next, the number of channels and sample rate must be non-zero.
    let n_channels = reader.read_u8()?;

    if n_channels == 0 {
        return decode_error("vorbis: number of channels cannot be 0");
    }

    // This is a Symphonia limitation.
    if n_channels > 32 {
        return unsupported_error("vorbis: only a maximum of 32 channels are supported");
    }

    let sample_rate = reader.read_u32()?;

    if sample_rate == 0 {
        return decode_error("vorbis: sample rate cannot be 0");
    }

    // Read the bitrate range.
    let _bitrate_max = reader.read_u32()?;
    let _bitrate_nom = reader.read_u32()?;
    let _bitrate_min = reader.read_u32()?;

    // Next, blocksize_0 and blocksize_1 are packed into a single byte.
    let block_sizes = reader.read_u8()?;

    let bs0_exp = (block_sizes & 0x0f) >> 0;
    let bs1_exp = (block_sizes & 0xf0) >> 4;

    // The block sizes must not exceed the bounds.
    if bs0_exp < VORBIS_BLOCKSIZE_MIN || bs0_exp > VORBIS_BLOCKSIZE_MAX {
        return decode_error("vorbis: blocksize_0 out-of-bounds");
    }

    if bs1_exp < VORBIS_BLOCKSIZE_MIN || bs1_exp > VORBIS_BLOCKSIZE_MAX {
        return decode_error("vorbis: blocksize_1 out-of-bounds");
    }

    // Blocksize_0 must be >= blocksize_1
    if bs0_exp > bs1_exp {
        return decode_error("vorbis: blocksize_0 exceeds blocksize_1");
    }

    // Framing flag must be set.
    if reader.read_u8()? != 0x1 {
        return decode_error("vorbis: ident header framing flag unset");
    }

    Ok(IdentHeader { n_channels, sample_rate, bs0_exp, bs1_exp })
}

struct Setup {
    codebooks: Vec<VorbisCodebook>,
    floors: Vec<Box<dyn Floor>>,
    residues: Vec<Residue>,
    mappings: Vec<Mapping>,
    modes: Vec<Mode>,
}

fn read_setup(reader: &mut BufReader<'_>, ident: &IdentHeader) -> Result<Setup> {
    read_setup_header(reader)?;
    let mut bs = BitReaderRtl::new(reader.read_buf_bytes_available_ref());
    let setup = read_setup_body::<_, true>(&mut bs, ident)?;

    // Extra bits less than a byte (< 8 bits) are expected padding to align to
    // a byte boundary. Only log if there are leftover full bytes.
    if bs.bits_left() >= 8 {
        debug!("vorbis: leftover bits in setup head extra data");
    }

    Ok(setup)
}

fn read_setup_header<R: ReadBytes>(reader: &mut R) -> Result<()> {
    // The packet type must be an setup header.
    let packet_type = reader.read_u8()?;

    if packet_type != VORBIS_PACKET_TYPE_SETUP {
        return decode_error("vorbis: invalid packet type for setup header");
    }

    // Next, the setup packet signature must be correct.
    let mut packet_sig_buf = [0; 6];
    reader.read_buf_exact(&mut packet_sig_buf)?;

    if packet_sig_buf != VORBIS_HEADER_PACKET_SIGNATURE {
        return decode_error("vorbis: invalid setup header signature");
    }

    Ok(())
}

fn read_setup_body<B: ReadBitsRtl, const BUILD_TABLES: bool>(
    bs: &mut B,
    ident: &IdentHeader,
) -> Result<Setup> {
    // Read codebooks.
    let codebook_count = bs.read_bits_leq32(8)? + 1;
    let mut codebooks = Vec::new();
    for _ in 0..codebook_count {
        if let Some(codebook) = VorbisCodebook::read::<_, BUILD_TABLES>(bs)? {
            codebooks.push(codebook);
        }
    }
    // A count of 256 has no representable nonzero upper bound in the existing setup
    // readers. Those readers reject references in that case.
    let max_codebook = u8::try_from(codebook_count).unwrap_or(0);

    // Read time-domain transforms (placeholders in Vorbis 1).
    read_time_domain_transforms(bs)?;

    // Read floors.
    let floors = read_floors(bs, ident.bs0_exp, ident.bs1_exp, max_codebook, BUILD_TABLES)?;

    // Read residues.
    let residues = read_residues(bs, max_codebook)?;

    // Read channel mappings.
    let mappings = read_mappings(bs, ident.n_channels, floors.len() as u8, residues.len() as u8)?;

    // Read modes.
    let modes = read_modes(bs, mappings.len() as u8)?;

    // Framing flag must be set.
    if !bs.read_bool()? {
        return decode_error("vorbis: setup header framing flag unset");
    }

    Ok(Setup { codebooks, floors, residues, mappings, modes })
}

fn read_time_domain_transforms<B: ReadBitsRtl>(bs: &mut B) -> Result<()> {
    let count = bs.read_bits_leq32(6)? + 1;

    for _ in 0..count {
        // All these values are placeholders and must be 0.
        if bs.read_bits_leq32(16)? != 0 {
            return decode_error("vorbis: invalid time domain tranform");
        }
    }

    Ok(())
}

fn read_floors<B: ReadBitsRtl>(
    bs: &mut B,
    bs0_exp: u8,
    bs1_exp: u8,
    max_codebook: u8,
    build_tables: bool,
) -> Result<Vec<Box<dyn Floor>>> {
    let count = bs.read_bits_leq32(6)? + 1;
    (0..count).map(|_| read_floor(bs, bs0_exp, bs1_exp, max_codebook, build_tables)).collect()
}

fn read_floor<B: ReadBitsRtl>(
    bs: &mut B,
    bs0_exp: u8,
    bs1_exp: u8,
    max_codebook: u8,
    build_tables: bool,
) -> Result<Box<dyn Floor>> {
    let floor_type = bs.read_bits_leq32(16)?;

    match floor_type {
        0 => Floor0::try_read(bs, bs0_exp, bs1_exp, max_codebook, build_tables),
        1 => Floor1::try_read(bs, max_codebook),
        _ => decode_error("vorbis: invalid floor type"),
    }
}

fn read_residues<B: ReadBitsRtl>(bs: &mut B, max_codebook: u8) -> Result<Vec<Residue>> {
    let count = bs.read_bits_leq32(6)? + 1;
    (0..count).map(|_| read_residue(bs, max_codebook)).collect()
}

fn read_residue<B: ReadBitsRtl>(bs: &mut B, max_codebook: u8) -> Result<Residue> {
    let residue_type = bs.read_bits_leq32(16)? as u16;

    match residue_type {
        0..=2 => Residue::try_read(bs, residue_type, max_codebook),
        _ => decode_error("vorbis: invalid residue type"),
    }
}

fn read_mappings<B: ReadBitsRtl>(
    bs: &mut B,
    audio_channels: u8,
    max_floor: u8,
    max_residue: u8,
) -> Result<Vec<Mapping>> {
    let count = bs.read_bits_leq32(6)? + 1;
    (0..count).map(|_| read_mapping(bs, audio_channels, max_floor, max_residue)).collect()
}

fn read_mapping<B: ReadBitsRtl>(
    bs: &mut B,
    audio_channels: u8,
    max_floor: u8,
    max_residue: u8,
) -> Result<Mapping> {
    let mapping_type = bs.read_bits_leq32(16)?;

    match mapping_type {
        0 => read_mapping_type0(bs, audio_channels, max_floor, max_residue),
        _ => decode_error("vorbis: invalid mapping type"),
    }
}

fn read_modes<B: ReadBitsRtl>(bs: &mut B, max_mapping: u8) -> Result<Vec<Mode>> {
    let count = bs.read_bits_leq32(6)? + 1;
    (0..count).map(|_| read_mode(bs, max_mapping)).collect()
}

#[derive(Debug)]
struct ChannelCouple {
    magnitude_ch: u8,
    angle_ch: u8,
}

#[derive(Debug)]
struct SubMap {
    floor: u8,
    residue: u8,
}

#[derive(Debug)]
struct Mapping {
    couplings: Vec<ChannelCouple>,
    multiplex: Vec<u8>,
    submaps: Vec<SubMap>,
}

fn read_mapping_type0<B: ReadBitsRtl>(
    bs: &mut B,
    audio_channels: u8,
    max_floor: u8,
    max_residue: u8,
) -> Result<Mapping> {
    let num_submaps = if bs.read_bool()? { bs.read_bits_leq32(4)? as u8 + 1 } else { 1 };

    let mut couplings = Vec::new();

    if bs.read_bool()? {
        // Number of channel couplings (up-to 256).
        let coupling_steps = bs.read_bits_leq32(8)? as u16 + 1;

        // Reserve space.
        couplings.reserve_exact(usize::from(coupling_steps));

        // The maximum channel number.
        let max_ch = audio_channels - 1;

        // The number of bits to read for the magnitude and angle channel numbers. Never exceeds 8.
        let coupling_bits = ilog(u32::from(max_ch));
        debug_assert!(coupling_bits <= 8);

        // Read each channel coupling.
        for _ in 0..coupling_steps {
            let magnitude_ch = bs.read_bits_leq32(coupling_bits)? as u8;
            let angle_ch = bs.read_bits_leq32(coupling_bits)? as u8;

            // Ensure the channels to be coupled are not the same, and that neither channel number
            // exceeds the maximum channel in the stream.
            if magnitude_ch == angle_ch || magnitude_ch > max_ch || angle_ch > max_ch {
                return decode_error("vorbis: invalid channel coupling");
            }

            couplings.push(ChannelCouple { magnitude_ch, angle_ch });
        }
    }

    if bs.read_bits_leq32(2)? != 0 {
        return decode_error("vorbis: reserved mapping bits non-zero");
    }

    let mut multiplex = Vec::with_capacity(usize::from(audio_channels));

    // If the number of submaps is > 1 read the multiplex numbers from the bitstream, otherwise
    // they're all 0.
    if num_submaps > 1 {
        for _ in 0..audio_channels {
            let mux = bs.read_bits_leq32(4)? as u8;

            if mux >= num_submaps {
                return decode_error("vorbis: invalid channel multiplex");
            }

            multiplex.push(mux);
        }
    }
    else {
        multiplex.resize(usize::from(audio_channels), 0);
    }

    let mut submaps = Vec::with_capacity(usize::from(num_submaps));

    for _ in 0..num_submaps {
        // Unused.
        let _ = bs.read_bits_leq32(8)?;

        // The floor to use.
        let floor = bs.read_bits_leq32(8)? as u8;

        if floor >= max_floor {
            return decode_error("vorbis: invalid floor for mapping");
        }

        // The residue to use.
        let residue = bs.read_bits_leq32(8)? as u8;

        if residue >= max_residue {
            return decode_error("vorbis: invalid residue for mapping");
        }

        submaps.push(SubMap { floor, residue });
    }

    let mapping = Mapping { couplings, multiplex, submaps };

    Ok(mapping)
}

#[derive(Debug)]
struct Mode {
    block_flag: bool,
    mapping: u8,
}

fn read_mode<B: ReadBitsRtl>(bs: &mut B, max_mapping: u8) -> Result<Mode> {
    let block_flag = bs.read_bool()?;
    let window_type = bs.read_bits_leq32(16)? as u16;
    let transform_type = bs.read_bits_leq32(16)? as u16;
    let mapping = bs.read_bits_leq32(8)? as u8;

    // Only window type 0 is allowed in Vorbis 1 (section 4.2.4).
    if window_type != 0 {
        return decode_error("vorbis: invalid window type for mode");
    }

    // Only transform type 0 is allowed in Vorbis 1 (section 4.2.4).
    if transform_type != 0 {
        return decode_error("vorbis: invalid transform type for mode");
    }

    // Mapping number must be exist.
    if mapping >= max_mapping {
        return decode_error("vorbis: invalid mode mapping");
    }

    let mode = Mode { block_flag, mapping };

    Ok(mode)
}

#[cfg(test)]
mod setup_prefix_tests {
    use super::*;

    fn identification() -> Vec<u8> {
        let mut bytes = b"\x01vorbis\0\0\0\0\x01".to_vec();
        bytes.extend(48_000_u32.to_le_bytes());
        bytes.extend([0; 12]); // Bitrate hints.
        bytes.extend([0x66, 1]); // Equal 64-sample blocks and the framing bit.
        bytes
    }

    fn setup() -> Vec<u8> {
        let mut bytes = b"\x05vorbis".to_vec();
        let mut position = bytes.len() * 8;
        let mut write = |value: u32, width: usize| {
            for bit in 0..width {
                if position % 8 == 0 {
                    bytes.push(0);
                }
                if value & (1 << bit) != 0 {
                    bytes[position / 8] |= 1 << (position % 8);
                }
                position += 1;
            }
        };
        write(0, 8); // One codebook.
        write(0x564342, 24);
        write(1, 16); // Vector dimension.
        write(256, 24);
        write(1, 1); // Ordered lengths.
        write(7, 5); // Length 8.
        write(256, 9); // 256 codewords of length 8.
        write(2, 4); // Vector lookup type 2.
        write(0, 32);
        write(0, 32);
        write(15, 4); // 16-bit multiplicands.
        write(0, 1);
        for value in 0..256 {
            write(value, 16);
        }
        write(0, 6); // One time-domain transform.
        write(0, 16);
        write(0, 6); // One floor.
        write(1, 16); // Floor 1.
        write(0, 5); // No partitions.
        write(0, 2);
        write(6, 4);
        write(0, 6); // One residue.
        write(0, 16);
        for _ in 0..3 {
            write(0, 24);
        }
        write(0, 6);
        write(0, 8);
        write(0, 4); // No residue vectors.
        write(0, 6); // One mapping.
        write(0, 16);
        write(0, 4); // Default submap, no coupling and reserved bits.
        write(0, 24); // Time transform, floor and residue indices.
        write(0, 6); // One mode.
        write(0, 1);
        write(0, 16);
        write(0, 16);
        write(0, 8);
        write(1, 1); // Framing bit.
        bytes
    }

    #[test]
    fn read_only_the_setup_prefix_used_by_the_decoder() -> Result<()> {
        let ident = identification();
        let valid = setup();
        let mut padded = valid.clone();
        padded.resize(valid.len() + 4096, 0xa5);
        for bytes in [&valid, &padded] {
            let mut reader = BufReader::new(bytes);
            let prefix = vorbis_setup_prefix_len(&ident, &mut reader)?;
            assert_eq!(usize::try_from(prefix).ok(), Some(valid.len()));
            assert_eq!(reader.pos(), prefix);

            let (_, decoded_setup) = read_extradata(&[ident.as_slice(), bytes].concat())?;
            assert_eq!(decoded_setup.codebooks.len(), 1);
            assert_eq!(decoded_setup.modes.len(), 1);
        }

        let truncated = &valid[..valid.len() - 1];
        assert!(vorbis_setup_prefix_len(&ident, &mut BufReader::new(truncated)).is_err());
        assert!(read_extradata(&[ident.as_slice(), truncated].concat()).is_err());

        padded[8] = 0; // Invalid first codebook synchronization word.
        let mut invalid = BufReader::new(&padded);
        assert!(vorbis_setup_prefix_len(&ident, &mut invalid).is_err());
        assert_eq!(invalid.pos(), 11);
        assert!(read_extradata(&[ident.as_slice(), &padded].concat()).is_err());
        Ok(())
    }
}

/// Map a Vorbis channel index to an audio buffer channel index given the channel map implied by the
/// total number of channels.
///
/// See channel map as defined in section 4.3.9 of the Vorbis I specification.
fn map_vorbis_channel(num_channels: u8, ch: usize) -> usize {
    // This pre-condition should always be true.
    assert!(ch < usize::from(num_channels));

    let mapped_ch: u8 = match num_channels {
        1 => [0][ch],                      // FL
        2 => [0, 1][ch],                   // FL, FR
        3 => [0, 2, 1][ch],                // FL, FC, FR
        4 => [0, 1, 2, 3][ch],             // FL, FR, RL, RR
        5 => [0, 2, 1, 3, 4][ch],          // FL, FC, FR, RL, RR
        6 => [0, 2, 1, 4, 5, 3][ch],       // FL, FC, FR, RL, RR, LFE
        7 => [0, 2, 1, 5, 6, 4, 3][ch],    // FL, FC, FR, SL, SR, RC, LFE
        8 => [0, 2, 1, 6, 7, 4, 5, 3][ch], // FL, FC, FR, SL, SR, RL, RR, LFE
        _ => return ch,
    };

    usize::from(mapped_ch)
}
