// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Statically linked decoders for incremental live audio.

use crate::{AudioMetadata, DecodedChunk};
use std::{collections::BTreeMap, sync::Arc};

/// A build-linked live decoder factory selected before audio bytes arrive.
pub trait LiveDecoderProvider: Send + Sync {
    /// Stable identifier used by route composition and the protocol format selector.
    fn id(&self) -> &'static str;

    fn open(&self, metadata: AudioMetadata) -> Result<Box<dyn LiveChunkReader>, String>;
}

/// A decoder for a live byte stream with unknown duration.
pub trait LiveChunkReader: Send {
    fn metadata(&self) -> AudioMetadata;

    /// Accepts an arbitrary byte partition and emits available audio before EOF.
    /// Implementations keep bounded parser state and bound output for each input partition.
    fn push(&mut self, bytes: &[u8]) -> Result<Option<DecodedChunk>, String>;

    /// Validates terminal input and may emit one final chunk.
    fn finish(self: Box<Self>) -> Result<Option<DecodedChunk>, String>;
}

/// Immutable decoder choices and optional default for one live route.
#[derive(Clone, Default)]
pub struct LiveDecoders {
    providers: BTreeMap<&'static str, Arc<dyn LiveDecoderProvider>>,
    default: Option<&'static str>,
}

impl LiveDecoders {
    /// Builds a route's explicit linked provider set and selects its optional default.
    pub fn new(
        providers: Vec<Arc<dyn LiveDecoderProvider>>,
        default: Option<&str>,
    ) -> Result<Self, String> {
        let mut selected = BTreeMap::new();
        for provider in providers {
            let id = provider.id();
            if id.is_empty() || selected.insert(id, provider).is_some() {
                return Err(format!("invalid or duplicate live decoder provider: {id}"));
            }
        }
        let default = default
            .map(|id| {
                selected
                    .get_key_value(id)
                    .map(|(&name, _)| name)
                    .ok_or_else(|| format!("unknown default live decoder: {id}"))
            })
            .transpose()?;
        Ok(Self {
            providers: selected,
            default,
        })
    }

    pub fn select(&self, name: Option<&str>) -> Result<SelectedLiveDecoder, String> {
        let name = name
            .or(self.default)
            .ok_or("no default live decoder is configured")?;
        let (&id, provider) = self
            .providers
            .get_key_value(name)
            .ok_or_else(|| format!("live decoder {name} is unavailable on this route"))?;
        Ok(SelectedLiveDecoder {
            id,
            provider: Arc::clone(provider),
        })
    }
}

/// Resolved live decoder identity, available for an independent session to open.
#[derive(Clone)]
pub struct SelectedLiveDecoder {
    id: &'static str,
    provider: Arc<dyn LiveDecoderProvider>,
}

impl SelectedLiveDecoder {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub fn open(&self, metadata: AudioMetadata) -> Result<LiveDecoder, String> {
        let reader = self.provider.open(metadata)?;
        let id = self.id();
        if reader.metadata() != metadata {
            return Err(format!(
                "live decoder {id} does not match the requested audio metadata"
            ));
        }
        Ok(LiveDecoder {
            reader,
            id,
            metadata,
            next_frame: 0,
        })
    }
}

/// Live stream with checked and contiguous decoded frame positions.
pub struct LiveDecoder {
    reader: Box<dyn LiveChunkReader>,
    id: &'static str,
    metadata: AudioMetadata,
    next_frame: u64,
}

impl LiveDecoder {
    pub const fn metadata(&self) -> AudioMetadata {
        self.metadata
    }

    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Option<DecodedChunk>, String> {
        let chunk = self.reader.push(bytes)?;
        validate_chunk(chunk.as_ref(), self.id, self.metadata, self.next_frame)?;
        if let Some(chunk) = &chunk {
            self.next_frame = chunk.end_frame();
        }
        Ok(chunk)
    }

    pub fn finish(self) -> Result<Option<DecodedChunk>, String> {
        let chunk = self.reader.finish()?;
        validate_chunk(chunk.as_ref(), self.id, self.metadata, self.next_frame)?;
        Ok(chunk)
    }
}

fn validate_chunk(
    chunk: Option<&DecodedChunk>,
    id: &str,
    metadata: AudioMetadata,
    next_frame: u64,
) -> Result<(), String> {
    if let Some(chunk) = chunk {
        let audio = chunk.audio();
        if chunk.first_frame() != next_frame
            || AudioMetadata::new(audio.sample_rate(), audio.channel_count()) != metadata
        {
            return Err(format!(
                "live decoder {id} returned invalid frame coverage or metadata"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChannelAudio, ChannelCount, DecodedAudio, SampleRate};

    struct Provider {
        metadata: AudioMetadata,
        first: u64,
        tail: u64,
        output_rate: SampleRate,
    }
    impl LiveDecoderProvider for Provider {
        fn id(&self) -> &'static str {
            "fixture"
        }
        fn open(&self, _: AudioMetadata) -> Result<Box<dyn LiveChunkReader>, String> {
            Ok(Box::new(Reader {
                metadata: self.metadata,
                first: self.first,
                tail: self.tail,
                output_rate: self.output_rate,
            }))
        }
    }
    struct Reader {
        metadata: AudioMetadata,
        first: u64,
        tail: u64,
        output_rate: SampleRate,
    }
    impl Reader {
        fn chunk(&self, first: u64) -> Result<Option<DecodedChunk>, String> {
            Ok(Some(DecodedChunk::new(
                first,
                DecodedAudio::new(self.output_rate, vec![ChannelAudio::new(vec![0.25])?])?,
            )?))
        }
    }
    impl LiveChunkReader for Reader {
        fn metadata(&self) -> AudioMetadata {
            self.metadata
        }
        fn push(&mut self, _: &[u8]) -> Result<Option<DecodedChunk>, String> {
            self.chunk(self.first)
        }
        fn finish(self: Box<Self>) -> Result<Option<DecodedChunk>, String> {
            self.chunk(self.tail)
        }
    }
    fn selected(
        metadata: AudioMetadata,
        first: u64,
        tail: u64,
        output_rate: SampleRate,
    ) -> Result<SelectedLiveDecoder, String> {
        LiveDecoders::new(
            vec![Arc::new(Provider {
                metadata,
                first,
                tail,
                output_rate,
            })],
            Some("fixture"),
        )?
        .select(None)
    }
    #[test]
    fn route_default_and_names_are_entirely_explicit() -> Result<(), String> {
        let rate = SampleRate::new(16000)?;
        let metadata = AudioMetadata::new(rate, ChannelCount::new(1)?);
        let make = || -> Arc<dyn LiveDecoderProvider> {
            Arc::new(Provider {
                metadata,
                first: 0,
                tail: 1,
                output_rate: rate,
            })
        };
        struct PcmAlias(Arc<dyn LiveDecoderProvider>);
        impl LiveDecoderProvider for PcmAlias {
            fn id(&self) -> &'static str {
                "pcm16"
            }
            fn open(&self, metadata: AudioMetadata) -> Result<Box<dyn LiveChunkReader>, String> {
                self.0.open(metadata)
            }
        }
        let replacement: Arc<dyn LiveDecoderProvider> = Arc::new(PcmAlias(make()));
        let custom = LiveDecoders::new(vec![replacement], Some("pcm16"))?;
        let mut reader = custom.select(None)?.open(metadata)?;
        assert_eq!(
            reader
                .push(b"x")?
                .ok_or("custom output")?
                .audio()
                .channels()[0]
                .samples(),
            &[0.25]
        );
        let empty = LiveDecoders::new(Vec::new(), None)?;
        assert!(empty.select(None).is_err());
        assert!(empty.select(Some("pcm16")).is_err());
        assert!(LiveDecoders::new(vec![make()], Some("pcm16")).is_err());
        assert!(LiveDecoders::new(vec![make(), make()], None).is_err());
        let route = LiveDecoders::new(vec![make()], Some("fixture"))?;
        assert_eq!(route.select(None)?.id(), "fixture");
        assert!(route.select(Some("pcm16")).is_err());
        let explicit = LiveDecoders::new(vec![make()], None)?;
        assert!(explicit.select(None).is_err());
        assert_eq!(explicit.select(Some("fixture"))?.id(), "fixture");
        Ok(())
    }

    #[test]
    fn validates_open_metadata_and_frame_coverage_including_eof() -> Result<(), String> {
        let rate = SampleRate::new(16_000)?;
        let other = SampleRate::new(8_000)?;
        let metadata = AudioMetadata::new(rate, ChannelCount::new(1)?);
        let wrong = AudioMetadata::new(other, ChannelCount::new(1)?);
        assert!(selected(wrong, 0, 1, other)?.open(metadata).is_err());
        assert!(
            selected(metadata, 1, 2, rate)?
                .open(metadata)?
                .push(b"x")
                .is_err()
        );
        assert!(
            selected(metadata, 0, 1, other)?
                .open(metadata)?
                .push(b"x")
                .is_err()
        );
        let mut decoder = selected(metadata, 0, 0, rate)?.open(metadata)?;
        assert!(decoder.push(b"x")?.is_some());
        assert!(decoder.finish().is_err());
        let mut decoder = selected(metadata, 0, 1, rate)?.open(metadata)?;
        assert!(decoder.push(b"x")?.is_some());
        assert!(decoder.finish()?.is_some());
        Ok(())
    }
}
