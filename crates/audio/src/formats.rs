// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Statically linked format selection for one file route.

use crate::{AudioFile, EncodedAudio, FileDecoder};
use std::io::{Cursor, Seek};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileFormatMatch {
    None,
    Fallback,
    Exact,
}

pub trait FileDecoderProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn identify(&self, prefix: &[u8], hint: Option<&str>) -> FileFormatMatch;
    fn open(&self, source: Box<dyn AudioFile>, hint: Option<&str>) -> Result<FileDecoder, String>;
}

#[derive(Clone)]
pub struct FileDecoders(Arc<[Arc<dyn FileDecoderProvider>]>);

impl FileDecoders {
    pub fn new(providers: Vec<Arc<dyn FileDecoderProvider>>) -> Result<Self, String> {
        for (index, provider) in providers.iter().enumerate() {
            let name = provider.name();
            if name.is_empty() || providers[..index].iter().any(|prior| prior.name() == name) {
                return Err(format!("invalid or duplicate file decoder name: {name:?}"));
            }
        }
        Ok(Self(providers.into()))
    }

    pub fn open_encoded(&self, encoded: EncodedAudio) -> Result<FileDecoder, String> {
        let (bytes, hint) = encoded.into_parts();
        self.open(Box::new(Cursor::new(bytes)), hint.as_deref())
    }

    pub fn open(
        &self,
        mut source: Box<dyn AudioFile>,
        hint: Option<&str>,
    ) -> Result<FileDecoder, String> {
        const PROBE_BYTES: usize = 4096;
        source.rewind().map_err(|error| error.to_string())?;
        let mut prefix = [0; PROBE_BYTES];
        let mut filled = 0;
        while filled < prefix.len() {
            let count = source
                .read(&mut prefix[filled..])
                .map_err(|error| error.to_string())?;
            if count == 0 {
                break;
            }
            filled = filled
                .checked_add(count)
                .filter(|&next| next <= prefix.len())
                .ok_or_else(|| "file decoder source reported an invalid byte count".to_owned())?;
        }
        source.rewind().map_err(|error| error.to_string())?;
        if filled == 0 {
            return Err("encoded audio bytes must not be empty".into());
        }
        let mut exact = None;
        let mut fallback = None;
        let mut ambiguous_fallback = false;
        for provider in self.0.iter() {
            match provider.identify(&prefix[..filled], hint) {
                FileFormatMatch::None => {}
                FileFormatMatch::Fallback => {
                    ambiguous_fallback |= fallback.replace(provider).is_some();
                }
                FileFormatMatch::Exact => {
                    if exact.replace(provider).is_some() {
                        return Err("ambiguous audio format on this route".into());
                    }
                }
            }
        }
        let provider = match (exact, fallback) {
            (Some(provider), _) => provider,
            (None, Some(provider)) if !ambiguous_fallback => provider,
            (None, Some(_)) => return Err("ambiguous fallback audio format on this route".into()),
            (None, None) => return Err("audio format is unavailable on this route".into()),
        };
        provider.open(source, hint)
    }
}
