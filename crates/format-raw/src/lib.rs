// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Headerless live audio format providers.

mod interleaved;
use interleaved::InterleavedFrameDecoder;
pub use interleaved::decode_samples;

use gigaam_audio::{
    AudioMetadata, ChannelCount, DecodedAudio, DecodedChunk, LiveChunkReader, LiveDecoderProvider,
    SampleFormat, SampleRate,
};
use std::sync::Arc;

/// Decoder for explicitly configured headerless live input.
pub struct RawDecoder {
    decoder: InterleavedFrameDecoder,
    metadata: AudioMetadata,
    next_frame: u64,
}

impl RawDecoder {
    pub const SUPPORTED_FORMATS: [SampleFormat; 4] = [
        SampleFormat::Pcm16,
        SampleFormat::F32,
        SampleFormat::Alaw,
        SampleFormat::Ulaw,
    ];

    pub fn new(
        format: SampleFormat,
        sample_rate: SampleRate,
        channels: ChannelCount,
    ) -> Result<Self, String> {
        Ok(Self {
            decoder: InterleavedFrameDecoder::new(format, channels)?,
            metadata: AudioMetadata::new(sample_rate, channels),
            next_frame: 0,
        })
    }

    pub const fn metadata(&self) -> AudioMetadata {
        self.metadata
    }

    /// Emits each available complete frame from this byte partition.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Option<DecodedChunk>, String> {
        let Some(channels) = self.decoder.push(bytes)? else {
            return Ok(None);
        };
        let chunk = DecodedChunk::new(
            self.next_frame,
            DecodedAudio::new(self.metadata.sample_rate(), channels)?,
        )?;
        self.next_frame = chunk.end_frame();
        Ok(Some(chunk))
    }

    pub fn finish(self) -> Result<(), String> {
        self.decoder.finish()
    }
}

impl LiveChunkReader for RawDecoder {
    fn metadata(&self) -> AudioMetadata {
        self.metadata()
    }
    fn push(&mut self, bytes: &[u8]) -> Result<Option<DecodedChunk>, String> {
        self.push(bytes)
    }
    fn finish(self: Box<Self>) -> Result<Option<DecodedChunk>, String> {
        RawDecoder::finish(*self)?;
        Ok(None)
    }
}

struct RawFormat(SampleFormat);
impl LiveDecoderProvider for RawFormat {
    fn id(&self) -> &'static str {
        match self.0 {
            SampleFormat::Pcm16 => "pcm16",
            SampleFormat::F32 => "f32",
            SampleFormat::Alaw => "alaw",
            SampleFormat::Ulaw => "ulaw",
        }
    }
    fn open(&self, metadata: AudioMetadata) -> Result<Box<dyn LiveChunkReader>, String> {
        Ok(Box::new(RawDecoder::new(
            self.0,
            metadata.sample_rate(),
            metadata.channel_count(),
        )?))
    }
}

/// Stock headerless format implementations, explicitly selected by an embedding process.
pub fn providers() -> Vec<Arc<dyn LiveDecoderProvider>> {
    RawDecoder::SUPPORTED_FORMATS
        .into_iter()
        .map(|format| -> Arc<dyn LiveDecoderProvider> { Arc::new(RawFormat(format)) })
        .collect()
}

/// Resolve one stock raw format for a directly composed consumer.
pub fn select(format: SampleFormat) -> Result<gigaam_audio::SelectedLiveDecoder, String> {
    let name = RawFormat(format).id();
    gigaam_audio::LiveDecoders::new(providers(), Some(name))?.select(None)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stock_providers_are_explicitly_selectable_and_stream_across_bytes() -> Result<(), String> {
        let set = gigaam_audio::LiveDecoders::new(providers(), Some("pcm16"))?;
        assert_eq!(set.select(None)?.id(), "pcm16");
        let metadata = AudioMetadata::new(SampleRate::new(8000)?, ChannelCount::new(2)?);
        for (name, bytes) in [
            ("pcm16", vec![0, 0, 0, 0]),
            ("f32", vec![0; 8]),
            ("alaw", vec![0xd5, 0xd5]),
            ("ulaw", vec![0xff, 0xff]),
        ] {
            let mut decoder = set.select(Some(name))?.open(metadata)?;
            assert!(decoder.push(&bytes[..1])?.is_none());
            let chunk = decoder.push(&bytes[1..])?.ok_or("complete frame")?;
            assert_eq!((chunk.first_frame(), chunk.end_frame()), (0, 1));
            assert_eq!(chunk.audio().channel_count().get(), 2);
            assert!(decoder.finish()?.is_none());
            let mut partial = set.select(Some(name))?.open(metadata)?;
            assert!(partial.push(&bytes[..1])?.is_none());
            assert!(partial.finish().is_err());
        }
        Ok(())
    }

    #[test]
    fn live_raw_emits_before_eof_and_tracks_positions_across_partitions() -> Result<(), String> {
        let rate = SampleRate::new(8000)?;
        let channels = ChannelCount::new(2)?;
        let mut raw = RawDecoder::new(SampleFormat::Pcm16, rate, channels)?;
        assert_eq!(raw.metadata(), AudioMetadata::new(rate, channels));
        let input = [i16::MIN, 16384, 0, -16384, i16::MAX, 0];
        let bytes: Vec<u8> = input
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        assert!(raw.push(&bytes[..3])?.is_none());
        let first = raw
            .push(&bytes[3..5])?
            .ok_or("first full frame is available before EOF")?;
        assert_eq!((first.first_frame(), first.end_frame()), (0, 1));
        assert_eq!(first.audio().channels()[0].samples(), &[-1.0]);
        assert_eq!(first.audio().channels()[1].samples(), &[0.5]);
        let tail = raw
            .push(&bytes[5..])?
            .ok_or("two additional frames are available before EOF")?;
        assert_eq!((tail.first_frame(), tail.end_frame()), (1, 3));
        assert_eq!(
            tail.audio().channels()[0].samples(),
            &[0.0, f32::from(i16::MAX) / 32768.0]
        );
        assert_eq!(tail.audio().channels()[1].samples(), &[-0.5, 0.0]);
        raw.finish()?;

        let mut partial = RawDecoder::new(SampleFormat::Pcm16, rate, channels)?;
        assert!(partial.push(&bytes[..5])?.is_some());
        assert!(partial.finish().is_err());
        Ok(())
    }
}
