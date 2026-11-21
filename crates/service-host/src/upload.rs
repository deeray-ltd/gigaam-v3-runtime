// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Owned encoded batch input and incremental upload spooling.

use std::io::{self, Cursor, Write};

use axum::body::Bytes;
use gigaam_audio::{FileDecoder, FileDecoders};

use crate::batch::UploadReservation;
use crate::scratch::ScratchFile;

/// Encoded audio owned by the batch request until decoding completes.
#[derive(Debug)]
pub struct BatchInput {
    storage: Storage,
    len: usize,
}

#[derive(Debug)]
enum Storage {
    Memory(Bytes),
    Spool(ScratchFile),
}

impl BatchInput {
    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn open(
        self,
        decoders: FileDecoders,
        format_hint: Option<&str>,
    ) -> Result<FileDecoder, String> {
        match self.storage {
            Storage::Memory(bytes) => decoders.open(Box::new(Cursor::new(bytes)), format_hint),
            Storage::Spool(file) => decoders.open(Box::new(file), format_hint),
        }
    }
}

impl From<Vec<u8>> for BatchInput {
    fn from(bytes: Vec<u8>) -> Self {
        Bytes::from(bytes).into()
    }
}

impl From<Bytes> for BatchInput {
    fn from(bytes: Bytes) -> Self {
        Self {
            len: bytes.len(),
            storage: Storage::Memory(bytes),
        }
    }
}

impl From<BatchUpload> for BatchInput {
    fn from(upload: BatchUpload) -> Self {
        Self {
            len: upload.len,
            storage: Storage::Spool(upload.file),
        }
    }
}

/// A network upload being written to quota-accounted scratch storage.
#[derive(Debug)]
pub struct BatchUpload {
    file: ScratchFile,
    len: usize,
    _reader: UploadReservation,
}

impl BatchUpload {
    pub(crate) const fn new(file: ScratchFile, reader: UploadReservation) -> Self {
        Self {
            file,
            len: 0,
            _reader: reader,
        }
    }

    /// Writes one chunk before allowing the caller to read another. The worker
    /// retains ownership of the file and its quota if the future is cancelled.
    pub async fn append(mut self, chunk: Bytes) -> io::Result<Self> {
        if chunk.is_empty() {
            return Ok(self);
        }
        let len = self.len.checked_add(chunk.len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "batch upload length overflow")
        })?;
        tokio::task::spawn_blocking(move || {
            self.file.write_all(&chunk)?;
            self.len = len;
            Ok(self)
        })
        .await
        .map_err(io::Error::other)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scratch::ScratchSpace;
    use std::num::{NonZeroU64, NonZeroUsize};

    #[tokio::test]
    async fn upload_transfers_exact_bytes_and_quota_to_decoder()
    -> Result<(), Box<dyn std::error::Error>> {
        // Mono PCM16 WAV containing one sample of +0.25 at 16 kHz.
        let wav = Bytes::from_static(
            b"RIFF\x26\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\x02\0\0\0\0\x20",
        );
        let limit = NonZeroU64::new(u64::try_from(wav.len())?).ok_or("nonempty test WAV")?;
        let scratch = ScratchSpace::new(std::env::temp_dir(), limit)?;
        let reader = UploadReservation(std::sync::Arc::new(
            std::sync::Arc::new(tokio::sync::Semaphore::new(1))
                .acquire_owned()
                .await?,
        ));
        let upload = BatchUpload::new(scratch.create()?, reader.clone())
            .append(wav.slice(..13))
            .await?
            .append(wav.slice(13..45))
            .await?
            .append(wav.slice(45..))
            .await?;
        let input = BatchInput::from(upload);
        assert_eq!(input.len(), wav.len());

        let mut decoder = input.open(
            FileDecoders::new(vec![std::sync::Arc::new(gigaam_format_wav::Wave)])?,
            None,
        )?;
        let bound = NonZeroUsize::new(1).ok_or("one frame is nonzero")?;
        let chunk = decoder
            .next_chunk(bound)?
            .ok_or("test WAV contains one frame")?;
        assert_eq!(chunk.audio().channels()[0].samples(), &[0.25]);
        assert!(decoder.next_chunk(bound)?.is_none());

        let refused = BatchUpload::new(scratch.create()?, reader.clone())
            .append(Bytes::from_static(b"x"))
            .await
            .expect_err("the decoder still owns the original upload");
        assert_eq!(refused.kind(), io::ErrorKind::StorageFull);
        drop(decoder);
        BatchUpload::new(scratch.create()?, reader)
            .append(wav)
            .await?;
        Ok(())
    }
}
