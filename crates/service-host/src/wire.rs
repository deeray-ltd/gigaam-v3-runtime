// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Shared HTTP envelope helpers for linked API adapters.

pub use crate::health_response::json_ok;
pub use crate::json;
pub use crate::protocol::{QueryParameters, query_bool, query_finite_f32};
use axum::{
    http::{HeaderMap, HeaderValue},
    response::Response,
};
use gigaam_transcription::{
    VadDurations, VadPadding, VadPolicyConfig, VadProbability, VadThresholds,
};

/// A request identifier kept as both log text and a validated response header.
#[derive(Clone)]
pub struct RequestId {
    pub(crate) text: String,
    pub(crate) header: HeaderValue,
}

impl RequestId {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn from_headers(headers: &HeaderMap) -> Result<Self, String> {
        let Some(header) = headers.get("x-request-id") else {
            let text = crate::log::new_req_id();
            let header = HeaderValue::from_bytes(text.as_bytes())
                .expect("generated request identifiers contain only visible ASCII bytes");
            return Ok(Self { text, header });
        };
        let text = header
            .to_str()
            .map_err(|_| "x-request-id must contain visible ASCII text")?;
        if text.is_empty() {
            return Err("x-request-id must not be empty".into());
        }
        Ok(Self {
            text: text.to_owned(),
            header: header.clone(),
        })
    }
}

pub fn with_request_id(mut response: Response, request_id: &RequestId) -> Response {
    response
        .headers_mut()
        .insert("x-request-id", request_id.header.clone());
    response
}

pub fn bind_vad_policy(
    parameters: &QueryParameters,
    enabled_for_request: bool,
) -> Result<Option<VadPolicyConfig>, String> {
    let tuning = [
        "vad_threshold",
        "vad_neg_threshold",
        "vad_min_speech_ms",
        "vad_min_silence_ms",
        "vad_speech_pad_ms",
    ];
    let has_tuning = tuning.iter().any(|key| parameters.get(key).is_some());
    if !enabled_for_request && (parameters.get("vad").is_some() || has_tuning) {
        return Err("VAD parameters require explicit response_version=2".into());
    }
    let enabled = match parameters.get("vad") {
        None | Some("false") => false,
        Some("true") => true,
        Some(value) => return Err(format!("vad must be true or false, got {value:?}")),
    };
    if !enabled {
        if has_tuning {
            return Err("VAD tuning parameters require vad=true".into());
        }
        return Ok(None);
    }
    let defaults = VadPolicyConfig::default();
    Ok(Some(VadPolicyConfig::new(
        VadThresholds::new(
            VadProbability::new(query_finite_f32(
                parameters,
                "vad_threshold",
                defaults.thresholds().speech().get(),
            )?)?,
            VadProbability::new(query_finite_f32(
                parameters,
                "vad_neg_threshold",
                defaults.thresholds().silence().get(),
            )?)?,
        )?,
        VadDurations::new(
            query_finite_f32(
                parameters,
                "vad_min_speech_ms",
                defaults.durations().minimum_speech_milliseconds(),
            )?,
            query_finite_f32(
                parameters,
                "vad_min_silence_ms",
                defaults.durations().minimum_silence_milliseconds(),
            )?,
        )?,
        VadPadding::new(query_finite_f32(
            parameters,
            "vad_speech_pad_ms",
            defaults.padding().milliseconds(),
        )?)?,
    )))
}
