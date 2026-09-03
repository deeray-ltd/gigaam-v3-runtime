// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Immutable application capabilities and policies assembled before routing starts.

use crate::admission::{RequestBodyLimit, ServiceAdmission};
use gigaam_audio::{FrontendProcessor, SampleRate};
use gigaam_model_package::ModelPackage;
use gigaam_recognition::{Device, ExecutionScheduler};
use gigaam_transcription::{BatchConfig, ObservationMode, PadPolicy, StreamConfig};
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Converts a model-package rate to Audio's validated rate while retaining package diagnostics.
pub(crate) fn model_sample_rate(value: usize) -> Result<SampleRate, String> {
    let rate = u32::try_from(value).map_err(|_| "model sample_rate exceeds u32".to_owned())?;
    SampleRate::new(rate).map_err(|_| "model sample_rate must be positive".to_owned())
}

/// All recognition resources accepted atomically by the service facade.
pub struct ServiceCapabilitiesParameters {
    pub pack: Arc<ModelPackage>,
    pub frontend: Arc<FrontendProcessor>,
    pub ctc: Arc<ExecutionScheduler>,
    pub rnnt: Option<Arc<ExecutionScheduler>>,
    pub provider: Device,
    /// Intra-op thread count applied to every ONNX Runtime session this application constructs
    /// lazily after startup, such as a per-connection VAD session. `None` keeps the library
    /// default.
    pub intra_threads: Option<NonZeroUsize>,
}

/// Mandatory and optional recognition capabilities owned by one service application.
pub struct ServiceCapabilities {
    pub(crate) pack: Arc<ModelPackage>,
    pub(crate) frontend: Arc<FrontendProcessor>,
    pub(crate) ctc: Arc<ExecutionScheduler>,
    pub(crate) rnnt: Option<Arc<ExecutionScheduler>>,
    pub(crate) provider: Device,
    pub(crate) intra_threads: Option<NonZeroUsize>,
}

impl ServiceCapabilities {
    /// Validates that the supplied frontend and model package describe the same model-rate input.
    pub fn new(parameters: ServiceCapabilitiesParameters) -> Result<Self, String> {
        let ServiceCapabilitiesParameters {
            pack,
            frontend,
            ctc,
            rnnt,
            provider,
            intra_threads,
        } = parameters;
        let package_rate = model_sample_rate(pack.frontend().sample_rate())?;
        if frontend.sample_rate() != package_rate {
            return Err("frontend sample_rate must match the model package".into());
        }
        Ok(Self {
            pack,
            frontend,
            ctc,
            rnnt,
            provider,
            intra_threads,
        })
    }
}

/// Cohesive facade input assembled after recognition and admission validation.
pub struct ServiceApplicationParameters {
    capabilities: ServiceCapabilities,
    policy: ServicePolicy,
    admission: ServiceAdmission,
    request_body_limit: RequestBodyLimit,
}

impl ServiceApplicationParameters {
    pub fn new(
        capabilities: ServiceCapabilities,
        policy: ServicePolicy,
        admission: ServiceAdmission,
        request_body_limit: RequestBodyLimit,
    ) -> Self {
        Self {
            capabilities,
            policy,
            admission,
            request_body_limit,
        }
    }

    pub(crate) fn into_assembly_parts(
        self,
    ) -> (
        ServiceCapabilities,
        ServicePolicy,
        ServiceAdmission,
        RequestBodyLimit,
    ) {
        (
            self.capabilities,
            self.policy,
            self.admission,
            self.request_body_limit,
        )
    }
}

/// Values accepted once into the immutable batch and stream policy of an application.
pub struct ServicePolicyParameters {
    pub model_sample_rate: SampleRate,
    pub window_seconds: f32,
    pub overlap_seconds: f32,
    pub dedup_default: bool,
    pub dedup_window_samples: usize,
    pub dedup_threshold: f32,
    pub observations: ObservationMode,
    pub backchannel_max_seconds: f32,
}

/// Validated public policy values shared by HTTP and WebSocket protocol adapters.
pub struct ServicePolicy {
    pub(crate) http_batch: BatchConfig,
    pub(crate) stream_base: StreamConfig,
    pub(crate) dedup_default: bool,
    pub(crate) dedup_window_samples: usize,
    pub(crate) dedup_threshold: f32,
    pub(crate) observations: ObservationMode,
    pub(crate) backchannel_max_seconds: f32,
}

impl ServicePolicy {
    pub fn new(parameters: ServicePolicyParameters) -> Result<Self, String> {
        if !parameters.window_seconds.is_finite() || parameters.window_seconds <= 0.0 {
            return Err("service window must be finite and greater than zero".into());
        }
        if !parameters.overlap_seconds.is_finite() || parameters.overlap_seconds < 0.0 {
            return Err("service overlap must be finite and non-negative".into());
        }
        if parameters.overlap_seconds >= parameters.window_seconds {
            return Err("service overlap must be shorter than the window".into());
        }
        if parameters.dedup_window_samples == 0 {
            return Err("service deduplication window must be greater than zero".into());
        }
        if !parameters.dedup_threshold.is_finite()
            || !(0.0..=1.0).contains(&parameters.dedup_threshold)
        {
            return Err("service deduplication threshold must be finite in 0..=1".into());
        }
        if !parameters.backchannel_max_seconds.is_finite()
            || parameters.backchannel_max_seconds < 0.0
        {
            return Err("service backchannel maximum must be finite and non-negative".into());
        }
        let http_batch = BatchConfig::new(
            parameters.model_sample_rate,
            parameters.window_seconds,
            parameters.overlap_seconds,
            PadPolicy::Exact,
        )?;
        let stream_base = StreamConfig::checked_default(parameters.model_sample_rate)?;
        Ok(Self {
            http_batch,
            stream_base,
            dedup_default: parameters.dedup_default,
            dedup_window_samples: parameters.dedup_window_samples,
            dedup_threshold: parameters.dedup_threshold,
            observations: parameters.observations,
            backchannel_max_seconds: parameters.backchannel_max_seconds,
        })
    }

    pub(crate) const fn sample_rate(&self) -> SampleRate {
        self.http_batch.sample_rate()
    }
}

/// Private immutable state shared by the directional protocol adapters.
pub(crate) struct ApplicationContext {
    pub(crate) capabilities: ServiceCapabilities,
    pub(crate) policy: ServicePolicy,
}

impl ApplicationContext {
    pub(crate) fn new(
        capabilities: ServiceCapabilities,
        policy: ServicePolicy,
    ) -> Result<Self, String> {
        if policy.sample_rate() != capabilities.frontend.sample_rate()
            || policy.stream_base.sample_rate() != capabilities.frontend.sample_rate()
        {
            return Err("service policy sample rate must match the frontend".into());
        }
        Ok(Self {
            capabilities,
            policy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::model_sample_rate;

    #[test]
    fn model_sample_rate_converts_the_package_rate_for_application_policy() {
        assert_eq!(
            model_sample_rate(16_000)
                .expect("the documented model sample rate must be valid")
                .hertz(),
            16_000
        );
    }
}
