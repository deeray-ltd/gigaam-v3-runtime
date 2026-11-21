// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Private batch files sharing a quota on logical file bytes.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::num::NonZeroU64;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct ScratchSpace {
    directory: Arc<PathBuf>,
    quota: Arc<Quota>,
}

impl ScratchSpace {
    /// One byte ledger for uploads, decoded PCM, observations and completed responses.
    pub fn from_settings(directory: Option<PathBuf>, byte_limit: Option<&str>) -> io::Result<Self> {
        const DEFAULT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
        let limit = match byte_limit {
            Some(value) => value.parse::<NonZeroU64>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "scratch capacity must be a positive byte count",
                )
            })?,
            None => NonZeroU64::new(DEFAULT_BYTES).expect("the default scratch quota is positive"),
        };
        Self::new(directory.unwrap_or_else(std::env::temp_dir), limit)
    }

    /// The configured directory must already exist. Its owner and any account
    /// allowed to replace entries belonging to this process must be trusted.
    pub fn new(directory: PathBuf, limit: NonZeroU64) -> io::Result<Self> {
        let directory = directory.canonicalize()?;
        if !directory.metadata()?.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                "batch scratch path must be a directory",
            ));
        }
        Ok(Self {
            directory: Arc::new(directory),
            quota: Arc::new(Quota {
                limit: limit.get(),
                used: AtomicU64::new(0),
            }),
        })
    }

    pub fn create(&self) -> io::Result<ScratchFile> {
        for _ in 0..128 {
            let serial = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
            let path = self
                .directory
                .join(format!(".gigaam-batch-{}-{serial}", std::process::id()));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            options.mode(0o600);
            let file = match options.open(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            };
            // No request data is written until this descriptor is the only
            // owner. The kernel reclaims the file when the descriptor closes.
            fs::remove_file(path)?;
            return Ok(ScratchFile {
                file,
                charge: Charge {
                    quota: self.quota.clone(),
                    bytes: 0,
                },
                position: 0,
            });
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not create a unique batch scratch file",
        ))
    }
}

#[derive(Debug)]
struct Quota {
    limit: u64,
    used: AtomicU64,
}

impl Quota {
    fn reserve(&self, bytes: u64) -> io::Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.used
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(bytes).filter(|&total| total <= self.limit)
            })
            .map_err(|used| {
                io::Error::new(
                    io::ErrorKind::StorageFull,
                    ScratchQuotaExceeded {
                        requested: bytes,
                        available: self.limit - used,
                    },
                )
            })?;
        Ok(())
    }

    fn release(&self, bytes: u64) {
        if bytes != 0 {
            self.used.fetch_sub(bytes, Ordering::Relaxed);
        }
    }
}

#[derive(Debug)]
pub(crate) struct ScratchQuotaExceeded {
    requested: u64,
    available: u64,
}

impl fmt::Display for ScratchQuotaExceeded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "batch scratch quota exceeded: requested {} additional bytes with {} available",
            self.requested, self.available
        )
    }
}

impl std::error::Error for ScratchQuotaExceeded {}

#[derive(Debug)]
struct Charge {
    quota: Arc<Quota>,
    bytes: u64,
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.quota.release(self.bytes);
    }
}

#[derive(Debug)]
pub struct ScratchFile {
    // Fields drop in declaration order. Close the only descriptor before
    // allowing another file to acquire these bytes.
    file: File,
    charge: Charge,
    position: u64,
}

impl ScratchFile {
    pub const fn len(&self) -> u64 {
        self.charge.bytes
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn write_bytes(
        &mut self,
        buffer: &[u8],
        write: impl FnOnce(&mut File, &[u8]) -> io::Result<usize>,
    ) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let requested = u64::try_from(buffer.len()).map_err(io::Error::other)?;
        let requested_end = self.position.checked_add(requested).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "batch scratch offset overflow")
        })?;
        let old_len = self.len();
        let reserved = requested_end.saturating_sub(old_len);
        self.charge.quota.reserve(reserved)?;

        let result = write(&mut self.file, buffer);
        // Write guarantees that an error writes no bytes. A short write only
        // advances by the reported count, so release any unused growth.
        let written = match &result {
            Ok(bytes) => u64::try_from(*bytes)
                .expect("bytes written cannot exceed the validated buffer size"),
            Err(_) => 0,
        };
        self.position += written;
        if written != 0 {
            self.charge.bytes = old_len.max(self.position);
        }
        self.charge.quota.release(reserved - (self.len() - old_len));
        result
    }
}

impl Read for ScratchFile {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let bytes = self.file.read(buffer)?;
        self.position += u64::try_from(bytes).expect("a Linux file read count fits in u64");
        Ok(bytes)
    }
}

impl Seek for ScratchFile {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.position = self.file.seek(position)?;
        Ok(self.position)
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(self.position)
    }
}

impl Write for ScratchFile {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.write_bytes(buffer, File::write)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;

    fn scratch(limit: u64) -> io::Result<ScratchSpace> {
        ScratchSpace::new(
            std::env::temp_dir(),
            NonZeroU64::new(limit).expect("tests use a positive byte limit"),
        )
    }

    fn assert_quota_exceeded(error: &io::Error) {
        assert_eq!(error.kind(), io::ErrorKind::StorageFull);
        assert!(
            error
                .get_ref()
                .and_then(|source| source.downcast_ref::<ScratchQuotaExceeded>())
                .is_some()
        );
    }

    #[test]
    fn shared_quota_is_released_when_the_owned_file_closes() -> io::Result<()> {
        let space = scratch(6)?;
        let mut input = space.create()?;
        input.write_all(b"abcd")?;
        let worker_space = space.clone();
        let mut worker = std::thread::spawn(move || -> io::Result<_> {
            let mut pcm = worker_space.create()?;
            pcm.write_all(b"ef")?;
            Ok(pcm)
        })
        .join()
        .expect("scratch worker should not panic")?;
        let mut response = space.create()?;
        assert_quota_exceeded(&response.write(b"g").expect_err("all quota is held"));
        assert_eq!(response.len(), 0);
        worker.rewind()?;
        let mut bytes = [0; 2];
        worker.read_exact(&mut bytes)?;
        assert_eq!(&bytes, b"ef");
        drop(worker);
        response.write_all(b"gh")?;
        assert_eq!(response.len(), 2);
        assert_eq!(input.len(), 4);
        Ok(())
    }

    #[test]
    fn seeks_and_overwrites_charge_only_logical_growth() -> io::Result<()> {
        let space = scratch(8)?;
        let mut file = space.create()?;
        #[cfg(unix)]
        {
            let metadata = file.file.metadata()?;
            assert_eq!(metadata.nlink(), 0);
            assert_eq!(metadata.mode() & 0o177, 0);
        }
        file.write_all(b"abc")?;
        file.rewind()?;
        let mut prefix = [0; 1];
        file.read_exact(&mut prefix)?;
        file.write_all(b"XY")?;
        assert_eq!(file.len(), 3);
        file.seek(SeekFrom::Start(10))?;
        assert_eq!(file.write(&[])?, 0);
        assert_quota_exceeded(&file.write(b"Z").expect_err("sparse extent exceeds quota"));
        assert_eq!(file.stream_position()?, 10);
        assert_eq!(file.len(), 3);
        file.seek(SeekFrom::Start(7))?;
        file.write_all(b"Z")?;
        assert_eq!(file.len(), file.file.metadata()?.len());
        file.rewind()?;
        let mut bytes = [0; 8];
        file.read_exact(&mut bytes)?;
        assert_eq!(&bytes, b"aXY\0\0\0\0Z");
        Ok(())
    }

    #[test]
    fn unused_reservation_from_short_or_failed_write_is_released() -> io::Result<()> {
        let space = scratch(8)?;
        let mut file = space.create()?;
        let mut other = space.create()?;
        let written = file.write_bytes(b"abcdefgh", |file, bytes| {
            assert_quota_exceeded(
                &other
                    .write(b"x")
                    .expect_err("write already reserved growth"),
            );
            file.write(&bytes[..2])
        })?;
        assert_eq!(written, 2);

        let error = file
            .write_bytes(b"abcdef", |_, _| {
                Err(io::Error::from(io::ErrorKind::PermissionDenied))
            })
            .expect_err("underlying write failed");
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(file.stream_position()?, 2);
        assert_eq!(file.len(), 2);
        other.write_all(b"123456")?;
        Ok(())
    }
}
