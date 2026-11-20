// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Statically composed speech-probability providers.

use crate::{NativeRuntime, SpeechProbabilityDetector, Vad};
use gigaam_audio::ChannelAudioView;
use gigaam_model_package::{ModelPackage, VariantPackage};
use std::{collections::BTreeMap, num::NonZeroUsize, sync::Arc};

/// Package available to a provider during preflight and detector construction.
#[derive(Clone, Copy)]
pub enum VadPackage<'a> {
    Legacy(&'a ModelPackage),
    Variant(&'a VariantPackage),
}

/// Resources available to a linked provider. Providers decide which resources they require.
#[derive(Clone, Copy)]
pub struct VadContext<'a> {
    pub package: VadPackage<'a>,
    pub native_runtime: Option<&'a NativeRuntime>,
    pub intra_threads: Option<NonZeroUsize>,
}

/// A statically linked provider; each call creates an independent detector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VadNativeRuntime {
    Required,
    NotRequired,
}

pub trait VadProvider: Send + Sync {
    fn native_runtime(&self) -> VadNativeRuntime {
        VadNativeRuntime::Required
    }
    fn id(&self) -> &'static str;
    fn preflight(&self, context: VadContext<'_>) -> Result<(), String>;
    fn create(
        &self,
        context: VadContext<'_>,
    ) -> Result<Box<dyn SpeechProbabilityDetector + Send>, String>;
}

/// Validates frame coverage and probabilities for every provider observation.
pub struct VadDetector(Box<dyn SpeechProbabilityDetector + Send>);
impl SpeechProbabilityDetector for VadDetector {
    fn probabilities(&mut self, audio: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
        let expected = audio.samples().len().div_ceil(512);
        let values = self.0.probabilities(audio)?;
        if values.len() != expected {
            return Err(format!(
                "VAD returned {} frames, expected {expected}",
                values.len()
            ));
        }
        if let Some(index) = values
            .iter()
            .position(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(format!(
                "VAD probability at frame {index} must be finite in 0..=1"
            ));
        }
        Ok(values)
    }
}

/// Registry assembled by the embedding process from linked implementations.
#[derive(Default)]
pub struct VadRegistry(BTreeMap<&'static str, Arc<dyn VadProvider>>);
impl VadRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_silero() -> Self {
        let mut registry = Self::new();
        registry
            .register(Arc::new(SileroProvider))
            .expect("unique built-in VAD provider");
        registry
    }
    pub fn register(&mut self, provider: Arc<dyn VadProvider>) -> Result<(), String> {
        let id = provider.id();
        if id.is_empty() || self.0.contains_key(id) {
            return Err(format!("invalid or duplicate VAD provider: {id}"));
        }
        self.0.insert(id, provider);
        Ok(())
    }
    pub fn select(&self, id: &str) -> Result<SelectedVadProvider, String> {
        self.0
            .get(id)
            .cloned()
            .map(SelectedVadProvider)
            .ok_or_else(|| format!("unknown VAD provider: {id}"))
    }
}

/// Resolved immutable provider selection, shareable across requests and channels.
#[derive(Clone)]
pub struct SelectedVadProvider(Arc<dyn VadProvider>);
impl SelectedVadProvider {
    pub fn native_runtime(&self) -> VadNativeRuntime {
        self.0.native_runtime()
    }
    pub fn preflight(&self, context: VadContext<'_>) -> Result<(), String> {
        self.0.preflight(context)
    }
    pub fn create(&self, context: VadContext<'_>) -> Result<VadDetector, String> {
        self.preflight(context)?;
        self.0.create(context).map(VadDetector)
    }
}

/// Select a registered process provider from ASR_VAD_PROVIDER, defaulting to Silero.
pub fn process_vad_provider(registry: &VadRegistry) -> Result<SelectedVadProvider, String> {
    let id = match std::env::var("ASR_VAD_PROVIDER") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(format!("ASR_VAD_PROVIDER: {error}")),
    };
    select_process_provider(registry, id.as_deref())
}

fn select_process_provider(
    registry: &VadRegistry,
    id: Option<&str>,
) -> Result<SelectedVadProvider, String> {
    registry.select(id.unwrap_or("silero"))
}

struct SileroProvider;
impl VadProvider for SileroProvider {
    fn id(&self) -> &'static str {
        "silero"
    }
    fn preflight(&self, context: VadContext<'_>) -> Result<(), String> {
        match context.package {
            VadPackage::Legacy(pack) => {
                pack.vad_model().map_err(|e| e.to_string())?;
            }
            VadPackage::Variant(pack) => {
                pack.vad_model()
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| "selected package does not declare VAD".to_owned())?;
            }
        }
        context
            .native_runtime
            .ok_or_else(|| "native runtime capability is required for VAD".to_owned())?;
        Ok(())
    }
    fn create(
        &self,
        context: VadContext<'_>,
    ) -> Result<Box<dyn SpeechProbabilityDetector + Send>, String> {
        let runtime = context
            .native_runtime
            .ok_or_else(|| "native runtime capability is required for VAD".to_owned())?;
        let vad = match context.package {
            VadPackage::Legacy(pack) => Vad::from_pack(pack, runtime, context.intra_threads),
            VadPackage::Variant(pack) => Vad::from_variant(pack, runtime, context.intra_threads),
        }?;
        Ok(Box::new(vad))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gigaam_audio::ChannelAudio;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Fake {
        calls: Arc<AtomicUsize>,
        values: Vec<f32>,
    }
    impl VadProvider for Fake {
        fn id(&self) -> &'static str {
            "fake"
        }
        fn native_runtime(&self) -> VadNativeRuntime {
            VadNativeRuntime::NotRequired
        }
        fn preflight(&self, _: VadContext<'_>) -> Result<(), String> {
            Ok(())
        }
        fn create(
            &self,
            _: VadContext<'_>,
        ) -> Result<Box<dyn SpeechProbabilityDetector + Send>, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeDetector(self.values.clone())))
        }
    }
    struct FakeDetector(Vec<f32>);
    impl SpeechProbabilityDetector for FakeDetector {
        fn probabilities(&mut self, _: ChannelAudioView<'_>) -> Result<Vec<f32>, String> {
            Ok(self.0.clone())
        }
    }
    fn package() -> ModelPackage {
        ModelPackage::open(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../model"
        )))
        .expect("test invariant")
    }
    #[test]
    fn registered_provider_is_selected_and_checks_observation_contract() {
        let pack = package();
        let context = VadContext {
            package: VadPackage::Legacy(&pack),
            native_runtime: None,
            intra_threads: None,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = VadRegistry::with_silero();
        registry
            .register(Arc::new(Fake {
                calls: calls.clone(),
                values: vec![0.2, 0.8],
            }))
            .expect("test invariant");
        assert!(
            registry
                .register(Arc::new(Fake {
                    calls: calls.clone(),
                    values: vec![]
                }))
                .is_err()
        );
        assert!(registry.select("missing").is_err());
        assert!(
            registry
                .select("silero")
                .expect("test invariant")
                .preflight(context)
                .is_err()
        );
        assert!(select_process_provider(&registry, None).is_ok());
        assert!(select_process_provider(&registry, Some("")).is_err());
        let selected = select_process_provider(&registry, Some("fake")).expect("test invariant");
        assert_eq!(selected.native_runtime(), VadNativeRuntime::NotRequired);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let mut detector = selected.create(context).expect("test invariant");
        let audio = ChannelAudio::new(vec![0.0; 513]).expect("test invariant");
        assert_eq!(
            detector
                .probabilities(audio.view())
                .expect("test invariant"),
            vec![0.2, 0.8]
        );
        assert!(
            detector
                .probabilities(ChannelAudio::new(vec![]).expect("test invariant").view())
                .is_err()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        for values in [vec![0.0], vec![0.0, f32::NAN], vec![0.0, 1.1]] {
            let mut invalid = VadDetector(Box::new(FakeDetector(values)));
            assert!(invalid.probabilities(audio.view()).is_err());
        }
        let mut empty = VadDetector(Box::new(FakeDetector(vec![])));
        assert!(
            empty
                .probabilities(ChannelAudio::new(vec![]).expect("test invariant").view())
                .expect("test invariant")
                .is_empty()
        );
    }
}
