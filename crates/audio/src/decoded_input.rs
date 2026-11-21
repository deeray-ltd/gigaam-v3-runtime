// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Typed decoded audio shared by file and live inputs.

use std::io::{Read, Seek};
use std::num::NonZeroUsize;

use crate::{ChannelAudio, ChannelCount, DecodedAudio, SampleRate};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioMetadata {
    sample_rate: SampleRate,
    channels: ChannelCount,
}

impl AudioMetadata {
    pub const fn new(sample_rate: SampleRate, channels: ChannelCount) -> Self {
        Self {
            sample_rate,
            channels,
        }
    }

    pub const fn sample_rate(self) -> SampleRate {
        self.sample_rate
    }

    pub const fn channel_count(self) -> ChannelCount {
        self.channels
    }
}

/// A nonempty group of complete frames on the decoded source sample clock.
///
/// Frame zero is the first emitted sample after format-defined trimming. Positions
/// count emitted frames, without using container PTS or inserting gaps for
/// packets skipped by a decoder.
#[derive(Debug, PartialEq)]
pub struct DecodedChunk {
    first_frame: u64,
    end_frame: u64,
    audio: DecodedAudio,
}

impl DecodedChunk {
    pub fn new(first_frame: u64, audio: DecodedAudio) -> Result<Self, String> {
        let frames = u64::try_from(audio.frames())
            .map_err(|_| "decoded frame count exceeds u64".to_owned())?;
        let end_frame = first_frame
            .checked_add(frames)
            .ok_or_else(|| "decoded end frame overflows u64".to_owned())?;
        Ok(Self {
            first_frame,
            end_frame,
            audio,
        })
    }

    pub const fn first_frame(&self) -> u64 {
        self.first_frame
    }

    pub const fn end_frame(&self) -> u64 {
        self.end_frame
    }

    pub const fn audio(&self) -> &DecodedAudio {
        &self.audio
    }

    pub fn into_audio(self) -> DecodedAudio {
        self.audio
    }
}

/// Seekable encoded audio owned by a selected format implementation.
pub trait AudioFile: Read + Seek + Send + Sync {}
impl<T: Read + Seek + Send + Sync> AudioFile for T {}

/// Decoded PCM supplied by an additional linked format implementation.
pub trait FileChunkReader: Send {
    fn metadata(&self) -> AudioMetadata;
    fn next_audio(&mut self, max_frames: NonZeroUsize) -> Result<Option<DecodedAudio>, String>;
}

/// Pulls source-rate, physical-channel chunks from an encoded file.
pub struct FileDecoder {
    source: Box<dyn FileChunkReader>,
    metadata: AudioMetadata,
    next_frame: u64,
    state: FileState,
}

enum FileState {
    Active,
    Eof,
    Failed(String),
}

impl FileDecoder {
    pub fn from_reader(reader: Box<dyn FileChunkReader>) -> Self {
        let metadata = reader.metadata();
        Self {
            source: reader,
            metadata,
            next_frame: 0,
            state: FileState::Active,
        }
    }

    pub const fn metadata(&self) -> AudioMetadata {
        self.metadata
    }
    pub fn next_chunk(&mut self, max_frames: NonZeroUsize) -> Result<Option<DecodedChunk>, String> {
        match &self.state {
            FileState::Active => {}
            FileState::Eof => return Ok(None),
            FileState::Failed(error) => return Err(error.clone()),
        }
        let result = self.source.next_audio(max_frames);
        match result {
            Ok(Some(audio)) => {
                if audio.frames() > max_frames.get()
                    || audio.sample_rate() != self.metadata.sample_rate()
                    || audio.channels().len() != self.metadata.channel_count().get()
                {
                    let error =
                        "file decoder chunk violates its declared audio metadata or frame limit"
                            .to_owned();
                    self.state = FileState::Failed(error.clone());
                    return Err(error);
                }
                let chunk = DecodedChunk::new(self.next_frame, audio)?;
                self.next_frame = chunk.end_frame();
                Ok(Some(chunk))
            }
            Ok(None) => {
                self.state = FileState::Eof;
                Ok(None)
            }
            Err(error) => {
                self.state = FileState::Failed(error.clone());
                Err(error)
            }
        }
    }
    /// Collects the decoded remainder at its source rate and in physical channel order.
    pub fn collect_remaining(&mut self) -> Result<DecodedAudio, String> {
        let chunk_size = NonZeroUsize::new(8192).expect("chunk size is nonzero");
        collect_audio(self.metadata(), || {
            self.next_chunk(chunk_size)
                .map(|chunk| chunk.map(DecodedChunk::into_audio))
        })
    }
}

/// Collects source-rate chunks while preserving physical channel order.
pub fn collect_audio(
    metadata: AudioMetadata,
    mut next: impl FnMut() -> Result<Option<DecodedAudio>, String>,
) -> Result<DecodedAudio, String> {
    let mut channels = vec![Vec::new(); metadata.channel_count().get()];
    while let Some(audio) = next()? {
        for (destination, source) in channels.iter_mut().zip(audio.channels()) {
            destination
                .try_reserve(source.len())
                .map_err(|_| "decoded channel samples cannot reserve memory".to_owned())?;
            destination.extend_from_slice(source.samples());
        }
    }
    DecodedAudio::new(
        metadata.sample_rate(),
        channels
            .into_iter()
            .map(ChannelAudio::new)
            .collect::<Result<_, _>>()?,
    )
}
