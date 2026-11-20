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
