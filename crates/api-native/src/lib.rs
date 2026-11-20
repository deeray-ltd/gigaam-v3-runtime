// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Native file and WebSocket routes for GigaAM v3 Runtime.

mod http;
mod response;
mod stream_response;
mod ws;

use axum::{Router, routing::get};
use gigaam_service_host::{ApiAdapter, ApiOperations};

/// The native HTTP and WebSocket API family.
pub struct NativeApi;

impl ApiAdapter for NativeApi {
    fn name(&self) -> &'static str {
        "native"
    }

    fn routes(&self, operations: ApiOperations) -> Router {
        http::routes(operations.batch).merge(
            Router::new()
                .route("/v1/stream", get(ws::stream))
                .with_state(ws::WsState::new(operations.live)),
        )
    }
}
