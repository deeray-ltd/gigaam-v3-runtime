// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! OpenAI-compatible file transcription and model discovery routes.

mod http;
mod openai;
mod openai_response;

use axum::Router;
use gigaam_service_host::{ApiAdapter, ApiOperations};

/// The OpenAI-compatible file API family.
pub struct OpenAiApi;

impl ApiAdapter for OpenAiApi {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn routes(&self, operations: ApiOperations) -> Router {
        http::routes(operations.batch)
    }
}
