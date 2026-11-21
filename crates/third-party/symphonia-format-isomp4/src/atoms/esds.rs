// Symphonia
// Copyright (c) 2019-2026 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use symphonia_core::codecs::CodecId;
use symphonia_core::codecs::video::{VIDEO_EXTRA_DATA_ID_NULL, VideoExtraData};
use symphonia_core::errors::Error;
use symphonia_core::io::{BufReader, ReadBytes};

use symphonia_common::mpeg::audio::*;
use symphonia_common::mpeg::formats::*;

use crate::atoms::stsd::{AudioSampleEntry, VisualSampleEntry};
use crate::atoms::{Atom, AtomHeader, AtomIterator, ReadAtom, Result, decode_error};

/// Elementary stream descriptor atom.
#[allow(dead_code)]
#[derive(Debug)]
pub struct EsdsAtom {
    /// Elementary stream descriptor.
    es_desc: ESDescriptor,
}

impl Atom for EsdsAtom {
    fn read<R: ReadAtom>(it: &mut AtomIterator<R>, _header: &AtomHeader) -> Result<Self> {
        let (_, _) = it.read_extended_header()?;

        // The ES descriptor occupies the entire atom.
        let size = it
            .data_left()?
            .ok_or(Error::DecodeError("isomp4 (esds): expected atom size to be known"))?;

        if size <= MIN_OBJECT_DESCRIPTOR_SIZE {
            return decode_error("isomp4 (esds): missing es descriptor in esds");
        }

        let es_desc = match it.read_scoped_batch_atom_data(size, |reader| read_es_descriptor(reader, true))? {
            Some(descriptor) => descriptor,
            None => {
                let size = usize::try_from(size)
                    .map_err(|_| Error::LimitError("isomp4 (esds): atom size exceeds usize"))?;
                let buf = it.read_boxed_slice_exact(size)?;
                read_es_descriptor(&mut BufReader::new(&buf), false)?
            }
        };
        Ok(EsdsAtom { es_desc })
    }
}

fn read_es_descriptor<R: ReadBytes>(reader: &mut R, bound_aac_config: bool) -> Result<ESDescriptor> {
    let (tag, len) = read_object_descriptor_header(reader)?;
    if tag != ClassTag::EsDescriptor {
        return decode_error("isomp4 (esds): missing es descriptor in esds");
    }
    if bound_aac_config {
        Ok(ESDescriptor::read_with_bounded_aac_config(reader, len)?)
    } else {
        Ok(ESDescriptor::read(reader, len)?)
    }
}

impl EsdsAtom {
    /// If the elementary stream descriptor describes an audio stream, populate the provided
    /// audio sample entry.
    pub fn fill_audio_sample_entry(self, entry: &mut AudioSampleEntry) -> Result<()> {
        match codec_id_from_object_type_indication(self.es_desc.dec_config.object_type_indication) {
            Some(CodecId::Audio(id)) => {
                // Object type indication identified an audio codec.
                entry.codec_id = id;
            }
            Some(_) => {
                // Object type indication identified a non-audio codec. This is unexpected.
                return decode_error("isomp4 (esds): expected an audio codec type");
            }
            None => {}
        }

        if let Some(ds_config) = self.es_desc.dec_config.dec_specific_info {
            // Try to read the audio specific configuration and populate the audio sample entry.
            if let Ok(asc) = AudioSpecificConfig::read(&ds_config.extra_data) {
                entry.profile = get_audio_codec_profile(&asc);
                entry.channels = asc.channels;
            }

            entry.extra_data = Some(ds_config.extra_data);
        }

        Ok(())
    }

    /// If the elementary stream descriptor describes an video stream, populate the provided
    /// video sample entry.
    pub fn fill_video_sample_entry(self, entry: &mut VisualSampleEntry) -> Result<()> {
        match codec_id_from_object_type_indication(self.es_desc.dec_config.object_type_indication) {
            Some(CodecId::Video(id)) => {
                // Object type indication identified an video codec.
                entry.codec_id = id;
            }
            Some(_) => {
                // Object type indication identified a non-video codec. This is unexpected.
                return decode_error("isomp4 (esds): expected a video codec type");
            }
            None => {}
        }

        if let Some(ds_config) = self.es_desc.dec_config.dec_specific_info {
            entry
                .extra_data
                .push(VideoExtraData { id: VIDEO_EXTRA_DATA_ID_NULL, data: ds_config.extra_data });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Cursor, Read, Seek, SeekFrom};
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    use symphonia_core::io::{MediaSource, MediaSourceStream};
    use crate::atoms::AtomType;

    struct CountedSource(Cursor<Vec<u8>>, Arc<AtomicUsize>);

    impl Read for CountedSource {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let count = self.0.read(buf)?;
            self.1.fetch_add(count, Ordering::Relaxed);
            Ok(count)
        }
    }

    impl Seek for CountedSource {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> { self.0.seek(pos) }
    }

    impl MediaSource for CountedSource {
        fn is_seekable(&self) -> bool { true }
        fn byte_len(&self) -> Option<u64> { u64::try_from(self.0.get_ref().len()).ok() }
    }

    fn descriptor(tag: u8, size: usize) -> Vec<u8> {
        let mut header = vec![tag];
        for shift in [21, 14, 7, 0] {
            let digit = u8::try_from((size >> shift) & 0x7f).expect("seven descriptor-size bits");
            header.push(if shift == 0 { digit } else { digit | 0x80 });
        }
        header
    }

    fn fixture(padding: usize, config_tail: usize, truncated: bool) -> Vec<u8> {
        let mut decoder_config = vec![0x40, 0x15];
        decoder_config.extend([0; 11]); // Buffer size and bit rates.
        decoder_config.extend(descriptor(0x05, 2 + config_tail));
        decoder_config.extend([0x11, 0x88]); // AAC-LC, 48 kHz, mono AudioSpecificConfig.
        decoder_config.resize(decoder_config.len() + config_tail, 0);
        let mut es = vec![0, 1, 0]; // ES identifier and flags.
        es.extend(descriptor(0x04, decoder_config.len()));
        es.extend(decoder_config);
        es.extend(descriptor(0xc0, padding));
        es.resize(es.len() + padding, 0); // Ignored user-private descriptor.
        es.extend(descriptor(0x06, 1));
        es.push(2); // MP4 SL configuration.
        let desc = descriptor(0x03, es.len() + usize::from(truncated));
        let size = u32::try_from(12 + desc.len() + es.len()).expect("fixture atom size");
        let mut data = size.to_be_bytes().to_vec();
        data.extend(b"esds\0\0\0\0");
        data.extend(desc);
        data.extend(es);
        data.extend(b"\0\0\0\x08free");
        data
    }

    #[test]
    fn batch_reads_esds_without_materializing_its_private_descriptor() -> symphonia_core::errors::Result<()> {
        let padding = 512 * 1024;
        for (batch, size, truncated) in [(false, 0, false), (true, padding, false), (true, 0, true)] {
            let data = fixture(size, 0, truncated);
            let length = u64::try_from(data.len()).expect("fixture source size");
            let bytes_read = Arc::new(AtomicUsize::new(0));
            let input = CountedSource(Cursor::new(data), Arc::clone(&bytes_read));
            let mut it = AtomIterator::new(MediaSourceStream::new(Box::new(input), Default::default()), Some(length));
            if batch {
                it.use_deferred_sample_sizes();
            }
            assert_eq!(it.next_header()?.unwrap().atom_type(), AtomType::Esds);
            let result = it.read_atom::<EsdsAtom>();
            if truncated {
                assert!(result.is_err());
            } else {
                let config = result?.es_desc.dec_config;
                assert_eq!(config.object_type_indication, 0x40);
                assert_eq!(config.dec_specific_info.unwrap().extra_data.as_ref(), [0x11, 0x88]);
            }
            assert_eq!(it.next_header()?.unwrap().atom_type(), AtomType::Free);
            assert!(bytes_read.load(Ordering::Relaxed) < padding);
        }
        Ok(())
    }

    #[test]
    fn batch_retains_only_the_aac_decoder_configuration_prefix()
    -> symphonia_core::errors::Result<()> {
        for (batch, tail) in [(false, 33), (true, 512 * 1024)] {
            let data = fixture(0, tail, false);
            let length = u64::try_from(data.len()).expect("fixture source size");
            let bytes_read = Arc::new(AtomicUsize::new(0));
            let input = CountedSource(Cursor::new(data), Arc::clone(&bytes_read));
            let mut it = AtomIterator::new(
                MediaSourceStream::new(Box::new(input), Default::default()), Some(length)
            );
            if batch {
                it.use_deferred_sample_sizes();
            }
            assert_eq!(it.next_header()?.expect("descriptor atom").atom_type(), AtomType::Esds);
            let info = it.read_atom::<EsdsAtom>()?.es_desc.dec_config
                .dec_specific_info.expect("fixture AAC configuration");
            assert_eq!(&info.extra_data[..2], &[0x11, 0x88]);
            assert_eq!(info.extra_data.len(), if batch { 32 } else { 2 + tail });
            assert_eq!(it.next_header()?.expect("following atom").atom_type(), AtomType::Free);
            if batch {
                assert!(bytes_read.load(Ordering::Relaxed) < tail);
            }
        }
        Ok(())
    }
}
