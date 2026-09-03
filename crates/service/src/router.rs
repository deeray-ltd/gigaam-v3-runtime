// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Route composition from endpoint-owned, narrow state values.

use crate::{health, http, metrics, ws};
use axum::Router;
use axum::routing::get;

/// The facade supplies independently assembled endpoint state; routers never share a catch-all
/// application context.
pub(crate) struct RouterParameters {
    pub(crate) http: http::HttpState,
    pub(crate) ws: ws::WsState,
    pub(crate) health: health::HealthState,
    pub(crate) metrics: metrics::MetricsState,
}

pub(crate) fn router(parameters: RouterParameters) -> Router {
    let RouterParameters {
        http,
        ws,
        health,
        metrics,
    } = parameters;
    Router::new()
        .merge(http::routes(http).into_router())
        .route("/v1/stream", get(ws::stream).with_state(ws))
        .route("/health", get(health::health).with_state(health.clone()))
        .route("/livez", get(health::livez))
        .route("/readyz", get(health::readyz).with_state(health))
        .route("/metrics", get(metrics::endpoint).with_state(metrics))
}

#[cfg(test)]
mod tests {
    use crate::RequestBodyLimit;
    use crate::http::body_limited_test_routes;
    use axum::Router;
    use axum::body::Bytes;
    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::routing::post;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    async fn bounded_body(State(arrivals): State<Arc<AtomicUsize>>, body: Bytes) -> StatusCode {
        arrivals.fetch_add(body.len(), Ordering::Relaxed);
        StatusCode::OK
    }

    #[tokio::test]
    async fn opaque_http_routes_enforce_the_validated_body_boundary_at_the_real_listener() {
        let limit = RequestBodyLimit::new(3).expect("a positive body boundary is valid");
        let arrivals = Arc::new(AtomicUsize::new(0));
        let routes = body_limited_test_routes(
            Router::new()
                .route("/body", post(bounded_body))
                .with_state(Arc::clone(&arrivals)),
            limit,
        )
        .into_router();
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("the loopback listener must bind");
        let address = listener
            .local_addr()
            .expect("the loopback listener has an address");
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, routes)
                .with_graceful_shutdown(async {
                    shutdown_rx
                        .await
                        .expect("the test sends one listener shutdown signal");
                })
                .await
        });

        async fn request(address: std::net::SocketAddr, body: &[u8]) -> Result<String, String> {
            let mut stream = TcpStream::connect(address)
                .await
                .map_err(|error| format!("connect body-limit client: {error}"))?;
            let head = format!(
                "POST /body HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            stream
                .write_all(head.as_bytes())
                .await
                .map_err(|error| format!("write body-limit headers: {error}"))?;
            stream
                .write_all(body)
                .await
                .map_err(|error| format!("write body-limit bytes: {error}"))?;
            let mut response = Vec::new();
            stream
                .read_to_end(&mut response)
                .await
                .map_err(|error| format!("read body-limit response: {error}"))?;
            String::from_utf8(response)
                .map_err(|error| format!("body-limit response must be UTF-8: {error}"))
        }

        let exact = request(address, b"abc")
            .await
            .expect("a body exactly at the configured limit completes");
        assert!(
            !exact.starts_with("HTTP/1.1 413"),
            "the exact-size request must reach the handler: {exact}"
        );
        assert_eq!(arrivals.load(Ordering::Relaxed), 3);

        let over = request(address, b"abcd")
            .await
            .expect("an over-limit request receives an Axum response");
        assert!(
            over.starts_with("HTTP/1.1 413"),
            "one byte over must receive Axum's extractor status: {over}"
        );
        assert_eq!(
            arrivals.load(Ordering::Relaxed),
            3,
            "an over-limit body must not reach the byte-consuming handler"
        );
        shutdown_tx
            .send(())
            .expect("the listener remains live for its explicit shutdown");
        server
            .await
            .expect("the body-limit server task must not panic")
            .expect("the body-limit server must stop normally");
    }
}
