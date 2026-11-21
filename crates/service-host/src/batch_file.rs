// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! A completed batch response whose storage stays leased until its final reader finishes.

use crate::batch::TranscribeError;
use crate::scratch::ScratchFile;
use axum::body::Bytes;
use std::io::{self, BufWriter, Read, Seek, Write};

const RESPONSE_BLOCK_BYTES: usize = 64 * 1024;

pub struct BatchFile {
    file: ScratchFile,
    remaining: u64,
    _delivery: tokio::sync::OwnedSemaphorePermit,
}

impl BatchFile {
    pub(crate) fn project(
        file: ScratchFile,
        delivery: tokio::sync::OwnedSemaphorePermit,
        write_response: impl FnOnce(&mut dyn Write) -> Result<(), String>,
    ) -> Result<Self, TranscribeError> {
        let mut output = BufWriter::with_capacity(RESPONSE_BLOCK_BYTES, file);
        write_response(&mut output).map_err(TranscribeError::Unavailable)?;
        let mut file = output.into_inner().map_err(|error| {
            TranscribeError::Unavailable(format!("batch response storage: {}", error.error()))
        })?;
        file.rewind().map_err(|error| {
            TranscribeError::Unavailable(format!("batch response storage: {error}"))
        })?;
        Ok(Self {
            remaining: file.len(),
            file,
            _delivery: delivery,
        })
    }

    /// Builds a demand-driven HTTP body while retaining the file and its delivery lease.
    pub fn into_body(self) -> axum::body::Body {
        axum::body::Body::from_stream(futures_util::stream::try_unfold(self, Self::read_block))
    }

    pub const fn len(&self) -> u64 {
        self.remaining
    }

    pub const fn is_empty(&self) -> bool {
        self.remaining == 0
    }

    /// Reads a single block on downstream demand. A cancelled blocking read retains the file
    /// ownership until the operating system finishes the read.
    pub async fn read_block(mut self) -> io::Result<Option<(Bytes, Self)>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        tokio::task::spawn_blocking(move || {
            let length = self
                .remaining
                .min(u64::try_from(RESPONSE_BLOCK_BYTES).map_err(io::Error::other)?);
            let mut block = vec![0; usize::try_from(length).map_err(io::Error::other)?];
            self.file.read_exact(&mut block)?;
            self.remaining -= length;
            Ok(Some((Bytes::from(block), self)))
        })
        .await
        .map_err(io::Error::other)?
    }
}

#[cfg(test)]
mod tests {
    use super::{BatchFile, RESPONSE_BLOCK_BYTES};
    use gigaam_transcription::ScratchSpace;
    use std::io::Write;
    use std::num::NonZeroU64;

    #[tokio::test]
    async fn partially_read_response_releases_storage_when_recipient_leaves() -> Result<(), String>
    {
        let size = RESPONSE_BLOCK_BYTES * 2;
        let limit = u64::try_from(size)
            .ok()
            .and_then(NonZeroU64::new)
            .ok_or_else(|| "response fixture must use a positive byte limit".to_owned())?;
        let scratch =
            ScratchSpace::new(std::env::temp_dir(), limit).map_err(|error| error.to_string())?;
        let slots = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        let response = BatchFile::project(
            scratch.create().map_err(|error| error.to_string())?,
            slots
                .clone()
                .try_acquire_owned()
                .map_err(|error| error.to_string())?,
            |output| {
                output
                    .write_all(&vec![b'a'; RESPONSE_BLOCK_BYTES])
                    .map_err(|error| error.to_string())?;
                output
                    .write_all(&vec![b'b'; RESPONSE_BLOCK_BYTES])
                    .map_err(|error| error.to_string())
            },
        )
        .map_err(crate::batch::TranscribeError::message)?;
        assert_eq!(response.len(), limit.get());
        let (block, remaining) = response
            .read_block()
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "completed response must include its first block".to_owned())?;
        assert_eq!(block.len(), RESPONSE_BLOCK_BYTES);
        assert!(block.iter().all(|&byte| byte == b'a'));
        let mut next_request = scratch.create().map_err(|error| error.to_string())?;
        assert!(next_request.write_all(b"next").is_err());
        assert!(slots.clone().try_acquire_owned().is_err());
        drop(remaining);
        assert!(slots.try_acquire_owned().is_ok());
        next_request
            .write_all(b"next")
            .map_err(|error| error.to_string())?;
        Ok(())
    }
}
