// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Wire projections of shared operational probes.
use crate::json::{JSON_CONTENT_TYPE, Json, error_object, obj};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use gigaam_recognition::Device;

/// Validated immutable capability text retained by the health endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HealthProjection {
    provider: Device,
    variant: gigaam_model_package::VariantId,
}

impl HealthProjection {
    pub(crate) const fn new(provider: Device, variant: gigaam_model_package::VariantId) -> Self {
        Self { provider, variant }
    }
}

pub fn json_ok(body: String) -> Response {
    ([(header::CONTENT_TYPE, JSON_CONTENT_TYPE)], body).into_response()
}

pub(crate) fn json_err(code: StatusCode, message: impl Into<String>) -> Response {
    (
        code,
        [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
        error_object(message),
    )
        .into_response()
}

pub(crate) fn health_response(projection: HealthProjection) -> Response {
    let models = vec![Json::str(projection.variant.as_str())];
    json_ok(
        obj(vec![
            ("status", Json::str("ok")),
            ("models", Json::Array(models)),
            ("provider", Json::str(projection.provider.as_str())),
        ])
        .to_string(),
    )
}

pub(crate) fn liveness_response() -> Response {
    json_ok(obj(vec![("status", Json::str("alive"))]).to_string())
}

/// Typed not-ready reason. The wire body names the actual cause instead of a generic refusal,
/// so a stopped recognition worker is never reported as an in-progress shutdown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadinessOutcome {
    Ready,
    Draining,
    WorkerStopped,
}

pub(crate) fn readiness_response(outcome: ReadinessOutcome) -> Response {
    match outcome {
        ReadinessOutcome::Ready => json_ok(obj(vec![("status", Json::str("ready"))]).to_string()),
        ReadinessOutcome::Draining => json_err(StatusCode::SERVICE_UNAVAILABLE, "draining"),
        ReadinessOutcome::WorkerStopped => {
            json_err(StatusCode::SERVICE_UNAVAILABLE, "worker stopped")
        }
    }
}
