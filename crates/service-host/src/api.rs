// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Process composition for independently linked protocol adapters.

use crate::{batch::BatchService, live::LiveService};
use axum::Router;
use std::collections::{BTreeMap, btree_map::Entry};

/// The transcription operations available to the selected protocol adapter.
pub struct ApiOperations {
    /// File upload and transcription operations.
    pub batch: BatchService,
    /// Live transcription operations.
    pub live: LiveService,
}

/// Builds the routes of one statically linked API family.
pub trait ApiAdapter: Send + Sync {
    /// The exact startup selection name for this family.
    fn name(&self) -> &'static str;

    /// Builds the selected family's router from the application operations.
    fn routes(&self, operations: ApiOperations) -> Router;
}

struct AudioRoutes<A> {
    adapter: A,
    file_decoders: gigaam_audio::FileDecoders,
    live_decoders: Option<gigaam_audio::LiveDecoders>,
}

impl<A: ApiAdapter> ApiAdapter for AudioRoutes<A> {
    fn name(&self) -> &'static str {
        self.adapter.name()
    }

    fn routes(&self, mut operations: ApiOperations) -> Router {
        operations.batch = operations
            .batch
            .with_file_decoders(self.file_decoders.clone());
        if let Some(decoders) = &self.live_decoders {
            operations.live = operations.live.with_audio_decoders(decoders.clone());
        }
        self.adapter.routes(operations)
    }
}

/// API families included by the process composition.
#[derive(Default)]
pub struct ApiRegistry {
    adapters: BTreeMap<&'static str, Box<dyn ApiAdapter>>,
}

impl ApiRegistry {
    /// Registers an adapter and rejects duplicate selection names.
    pub fn register(&mut self, adapter: impl ApiAdapter + 'static) -> Result<(), String> {
        let name = adapter.name();
        validate_name(name)?;
        match self.adapters.entry(name) {
            Entry::Occupied(_) => Err(format!("duplicate API family: {name}")),
            Entry::Vacant(entry) => {
                entry.insert(Box::new(adapter));
                Ok(())
            }
        }
    }

    /// Registers a family with the exact file formats its route may use.
    pub fn register_file_formats(
        &mut self,
        adapter: impl ApiAdapter + 'static,
        decoders: gigaam_audio::FileDecoders,
    ) -> Result<(), String> {
        self.register(AudioRoutes {
            adapter,
            file_decoders: decoders,
            live_decoders: None,
        })
    }

    /// Registers a family with explicit decoder choices for its file and live routes.
    pub fn register_audio_formats(
        &mut self,
        adapter: impl ApiAdapter + 'static,
        file_decoders: gigaam_audio::FileDecoders,
        live_decoders: gigaam_audio::LiveDecoders,
    ) -> Result<(), String> {
        self.register(AudioRoutes {
            adapter,
            file_decoders,
            live_decoders: Some(live_decoders),
        })
    }

    pub(crate) fn select(&self, name: &str) -> Result<&dyn ApiAdapter, String> {
        self.adapters.get(name).map(Box::as_ref).ok_or_else(|| {
            let available = self.adapters.keys().copied().collect::<Vec<_>>().join(", ");
            format!("ASR_API_MODE {name:?} is unavailable (linked families: {available})")
        })
    }
}

/// Checks the spelling before any model or runtime side effect.
pub(crate) fn validate_name(name: &str) -> Result<(), String> {
    if !name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err("ASR_API_MODE requires a lowercase API family name".into());
    }
    Ok(())
}
