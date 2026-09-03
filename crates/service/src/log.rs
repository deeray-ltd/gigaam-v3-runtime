// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Structured access logs to stdout as JSON lines, using the service JSON writer.
use crate::json::{Json, obj};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

/// One fully serialized, newline-terminated access record owned by the caller.
pub(crate) struct AccessLine(String);

impl AccessLine {
    pub(crate) fn into_inner(self) -> String {
        self.0
    }

    #[cfg(test)]
    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Typed access records keep protocol adapters independent from JSON field construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TranscribeAccess {
    pub(crate) request_id: String,
    pub(crate) status: u16,
    pub(crate) milliseconds: u128,
    pub(crate) bytes: usize,
    pub(crate) model: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WsOpenAccess {
    pub(crate) request_id: String,
    pub(crate) channels: usize,
    pub(crate) rate: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WsCloseAccess {
    pub(crate) request_id: String,
    pub(crate) channels: usize,
    pub(crate) frames: u64,
    pub(crate) milliseconds: u128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WsErrorAccess {
    pub(crate) phase: &'static str,
    pub(crate) message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AccessEvent {
    Transcribe(TranscribeAccess),
    WsOpen(WsOpenAccess),
    WsClose(WsCloseAccess),
    WsError(WsErrorAccess),
}

/// A Unix timestamp that has been checked to fit the JSON integer boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UnixMilliseconds(i64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimestampError {
    BeforeUnixEpoch,
    OutsideJsonIntegerRange,
}

impl TimestampError {
    fn code(self) -> &'static str {
        match self {
            Self::BeforeUnixEpoch => "before_unix_epoch",
            Self::OutsideJsonIntegerRange => "outside_json_integer_range",
        }
    }
}

/// Process-unique request ID when the client did not send x-request-id.
pub fn new_req_id() -> String {
    format!("r{:x}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn timestamp_from_elapsed(elapsed: Duration) -> Result<UnixMilliseconds, TimestampError> {
    i64::try_from(elapsed.as_millis())
        .map(UnixMilliseconds)
        .map_err(|_| TimestampError::OutsideJsonIntegerRange)
}

fn now_ms() -> Result<UnixMilliseconds, TimestampError> {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => timestamp_from_elapsed(duration),
        Err(_) => Err(TimestampError::BeforeUnixEpoch),
    }
}

fn access_line(
    event: &str,
    mut fields: Vec<(&str, Json)>,
    timestamp: Result<UnixMilliseconds, TimestampError>,
) -> String {
    let mut all = Vec::with_capacity(fields.len() + 2);
    match timestamp {
        Ok(UnixMilliseconds(milliseconds)) => all.push(("ts_ms", Json::Int(milliseconds))),
        Err(error) => all.push(("timestamp_error", Json::str(error.code()))),
    }
    all.push(("event", Json::str(event.to_string())));
    all.append(&mut fields);
    obj(all).to_string()
}

/// Renders one complete typed access event with the established JSON field order and newline.
/// Valid timestamps use exact `ts_ms`; unavailable clocks use `timestamp_error`, never synthetic time.
pub(crate) fn render(event: AccessEvent) -> AccessLine {
    let timestamp = now_ms();
    let line = match event {
        AccessEvent::Transcribe(event) => access_line(
            "transcribe",
            vec![
                ("req_id", Json::str(event.request_id)),
                ("status", Json::Int(i64::from(event.status))),
                ("ms", Json::UInt128(event.milliseconds)),
                ("bytes", Json::Usize(event.bytes)),
                ("model", Json::str(event.model)),
            ],
            timestamp,
        ),
        AccessEvent::WsOpen(event) => access_line(
            "ws_open",
            vec![
                ("req_id", Json::str(event.request_id)),
                ("channels", Json::Usize(event.channels)),
                ("rate", Json::Int(i64::from(event.rate))),
            ],
            timestamp,
        ),
        AccessEvent::WsClose(event) => access_line(
            "ws_close",
            vec![
                ("req_id", Json::str(event.request_id)),
                ("channels", Json::Usize(event.channels)),
                ("frames", Json::UInt(event.frames)),
                ("ms", Json::UInt128(event.milliseconds)),
            ],
            timestamp,
        ),
        AccessEvent::WsError(event) => access_line(
            "ws_error",
            vec![
                ("phase", Json::str(event.phase)),
                ("message", Json::str(event.message)),
            ],
            timestamp,
        ),
    };
    AccessLine(format!("{line}\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn access_log_preserves_an_exact_unix_millisecond_timestamp() {
        let line = access_line(
            "transcribe",
            vec![("status", Json::Int(200))],
            Ok(UnixMilliseconds(1_725_000_001_234)),
        );

        assert_eq!(
            line,
            r#"{"ts_ms":1725000001234,"event":"transcribe","status":200}"#
        );
    }

    #[test]
    fn access_log_exposes_clock_failure_without_a_synthetic_timestamp() {
        let line = access_line(
            "transcribe",
            Vec::new(),
            Err(TimestampError::BeforeUnixEpoch),
        );

        assert_eq!(
            line,
            r#"{"timestamp_error":"before_unix_epoch","event":"transcribe"}"#
        );
    }

    #[test]
    fn timestamp_refuses_milliseconds_outside_the_json_integer_range() {
        assert_eq!(
            timestamp_from_elapsed(Duration::from_millis(u64::MAX)),
            Err(TimestampError::OutsideJsonIntegerRange)
        );
    }

    #[test]
    fn rendered_access_record_preserves_json_order_and_the_stdout_newline() {
        let line = render(AccessEvent::WsError(WsErrorAccess {
            phase: "stream",
            message: "broken".to_owned(),
        }));

        assert!(line.as_str().ends_with('\n'));
        assert!(line.as_str().contains("\"event\":\"ws_error\""));
        assert!(
            line.as_str()
                .ends_with("\"phase\":\"stream\",\"message\":\"broken\"}\n")
        );
    }
}
