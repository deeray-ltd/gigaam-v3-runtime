// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Stock GigaAM v3 service composition.

use gigaam_api_native::NativeApi;
use gigaam_api_openai::OpenAiApi;
use gigaam_service_host::{ApiRegistry, run_with_registries};

pub use gigaam_service_host::{
    RequestBodyLimit, ServiceAdmission, ServiceApplicationParameters, ServiceCapabilities,
    ServiceCapabilitiesParameters, ServicePolicy, ServicePolicyParameters,
};

/// An independently runnable application composed with the stock API families.
pub struct ServiceApplication(gigaam_service_host::ServiceApplication);

impl ServiceApplication {
    /// Assembles the service with the stock native and OpenAI-compatible adapters.
    pub fn assemble(parameters: ServiceApplicationParameters) -> Result<Self, String> {
        gigaam_service_host::ServiceApplication::assemble(parameters, &stock_apis()?).map(Self)
    }

    /// Transfers the assembled application to an embedding HTTP server.
    pub fn into_router(self) -> axum::Router {
        self.0.into_router()
    }
}

fn stock_apis() -> Result<ApiRegistry, String> {
    let mut apis = ApiRegistry::default();
    apis.register(NativeApi)?;
    apis.register(OpenAiApi)?;
    Ok(apis)
}

/// Runs the stock service process.
pub fn run_process() {
    run_with_vad_registry(gigaam_recognition::VadRegistry::with_silero());
}

/// Runs the stock API families with the VAD providers linked by the embedding process.
pub fn run_with_vad_registry(vad: gigaam_recognition::VadRegistry) {
    match stock_apis() {
        Ok(apis) => run_with_registries(vad, &apis),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}
