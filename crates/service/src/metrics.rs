// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Metrics in Prometheus text format, implemented directly on atomics without dependencies.
//! Served on `GET /metrics`. Active counts come from one live runtime gauge source.
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use gigaam_primitives::usize_to_u64_checked;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard};

/// One scrape-time view of dynamic runtime gauges. Counters remain in `Metrics`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeGaugeSnapshot {
    pub(crate) available_http: usize,
    pub(crate) available_ws: usize,
    pub(crate) ctc_pending: u64,
    pub(crate) rnnt_pending: Option<u64>,
}

/// Reads dynamic runtime gauges exactly when the metrics endpoint is scraped.
pub(crate) trait RuntimeGaugeSource: Send + Sync {
    fn snapshot(&self) -> RuntimeGaugeSnapshot;
}

/// Narrow state of the metrics adapter; it does not own any runtime truth.
#[derive(Clone)]
pub(crate) struct MetricsState {
    metrics: Arc<Metrics>,
    gauges: Arc<dyn RuntimeGaugeSource>,
}

impl MetricsState {
    pub(crate) fn new(metrics: Arc<Metrics>, gauges: Arc<dyn RuntimeGaugeSource>) -> Self {
        Self { metrics, gauges }
    }
}

const BUCKETS: [f64; 11] = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    f64::INFINITY,
];

/// A finite, non-negative latency accepted by a histogram.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LatencySeconds(f64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LatencyError {
    NonFinite,
    Negative,
}

impl LatencySeconds {
    fn from_seconds(seconds: f64) -> Result<Self, LatencyError> {
        if !seconds.is_finite() {
            return Err(LatencyError::NonFinite);
        }
        if seconds < 0.0 {
            return Err(LatencyError::Negative);
        }
        Ok(Self(seconds))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct HistogramSnapshot {
    buckets: [u64; 11], // cumulative by `le`
    sum_seconds: f64,
    count: u64,
    rejected_non_finite: u64,
    rejected_negative: u64,
}

#[derive(Default)]
struct HistogramState {
    buckets: [u64; 11], // cumulative by `le`
    sum_seconds: f64,
    count: u64,
    rejected_non_finite: u64,
    rejected_negative: u64,
}

impl HistogramState {
    fn increment(counter: &mut u64, name: &str) {
        *counter = match counter.checked_add(1) {
            Some(next) => next,
            None => panic!("{name} must not overflow"),
        };
    }

    fn observe(&mut self, seconds: LatencySeconds) {
        for (index, &bound) in BUCKETS.iter().enumerate() {
            if seconds.0 <= bound {
                Self::increment(&mut self.buckets[index], "histogram bucket counter");
            }
        }
        self.sum_seconds += seconds.0;
        Self::increment(&mut self.count, "histogram count");
    }

    fn reject(&mut self, error: LatencyError) {
        match error {
            LatencyError::NonFinite => {
                Self::increment(
                    &mut self.rejected_non_finite,
                    "non-finite rejection counter",
                );
            }
            LatencyError::Negative => {
                Self::increment(&mut self.rejected_negative, "negative rejection counter");
            }
        }
    }

    fn snapshot(&self) -> HistogramSnapshot {
        HistogramSnapshot {
            buckets: self.buckets,
            sum_seconds: self.sum_seconds,
            count: self.count,
            rejected_non_finite: self.rejected_non_finite,
            rejected_negative: self.rejected_negative,
        }
    }
}

#[derive(Default)]
struct Hist {
    state: Mutex<HistogramState>,
}

impl Hist {
    fn state(&self) -> MutexGuard<'_, HistogramState> {
        match self.state.lock() {
            Ok(state) => state,
            Err(_) => panic!(
                "histogram state mutex is poisoned; an interrupted update invalidates coherent metrics"
            ),
        }
    }

    fn observe(&self, secs: f64) {
        let mut state = self.state();
        match LatencySeconds::from_seconds(secs) {
            Ok(seconds) => state.observe(seconds),
            Err(error) => state.reject(error),
        }
    }

    fn snapshot(&self) -> HistogramSnapshot {
        self.state().snapshot()
    }

    fn render(&self, out: &mut String, name: &str) {
        let snapshot = self.snapshot();
        let _ = writeln!(out, "# TYPE {name} histogram");
        for (i, &b) in BUCKETS.iter().enumerate() {
            let le = if b.is_infinite() {
                "+Inf".to_string()
            } else {
                format!("{b}")
            };
            let _ = writeln!(out, "{name}_bucket{{le=\"{le}\"}} {}", snapshot.buckets[i]);
        }
        let _ = writeln!(out, "{name}_sum {}", prometheus_float(snapshot.sum_seconds));
        let _ = writeln!(out, "{name}_count {}", snapshot.count);
        let _ = writeln!(out, "# TYPE {name}_rejected_total counter");
        let _ = writeln!(
            out,
            "{name}_rejected_total{{reason=\"non_finite\"}} {}",
            snapshot.rejected_non_finite
        );
        let _ = writeln!(
            out,
            "{name}_rejected_total{{reason=\"negative\"}} {}",
            snapshot.rejected_negative
        );
    }
}

fn prometheus_float(value: f64) -> String {
    if value == f64::INFINITY {
        "+Inf".to_string()
    } else {
        value.to_string()
    }
}

pub struct Metrics {
    transcribe_total: AtomicU64,
    transcribe_errors: AtomicU64,
    overload_total: AtomicU64,
    timeout_total: AtomicU64,
    ws_total: AtomicU64,
    ws_rejected: AtomicU64,
    ws_frames: AtomicU64,
    ws_channels: AtomicU64,
    ws_turn_patches: AtomicU64,
    dedup_collapsed: AtomicU64,
    ws_errors: AtomicU64,
    telemetry_queue_full: AtomicU64,
    telemetry_sink_closed: AtomicU64,
    telemetry_stdout_failures: AtomicU64,
    telemetry_stderr_failures: AtomicU64,
    lat: Hist,
    lat_frontend: Hist,
    lat_encoder: Hist,
    lat_decode: Hist,
    max_http: usize,
    max_ws: usize,
}

impl Metrics {
    pub fn new(max_http: usize, max_ws: usize) -> Metrics {
        Metrics {
            transcribe_total: AtomicU64::new(0),
            transcribe_errors: AtomicU64::new(0),
            overload_total: AtomicU64::new(0),
            timeout_total: AtomicU64::new(0),
            ws_total: AtomicU64::new(0),
            ws_rejected: AtomicU64::new(0),
            ws_frames: AtomicU64::new(0),
            ws_channels: AtomicU64::new(0),
            ws_turn_patches: AtomicU64::new(0),
            dedup_collapsed: AtomicU64::new(0),
            ws_errors: AtomicU64::new(0),
            telemetry_queue_full: AtomicU64::new(0),
            telemetry_sink_closed: AtomicU64::new(0),
            telemetry_stdout_failures: AtomicU64::new(0),
            telemetry_stderr_failures: AtomicU64::new(0),
            lat: Hist::default(),
            lat_frontend: Hist::default(),
            lat_encoder: Hist::default(),
            lat_decode: Hist::default(),
            max_http,
            max_ws,
        }
    }

    pub fn transcribe(&self) {
        self.transcribe_total.fetch_add(1, Relaxed);
    }
    pub fn transcribe_error(&self) {
        self.transcribe_errors.fetch_add(1, Relaxed);
    }
    pub fn overload(&self) {
        self.overload_total.fetch_add(1, Relaxed);
    }
    pub fn timeout(&self) {
        self.timeout_total.fetch_add(1, Relaxed);
    }
    pub fn observe_latency(&self, secs: f64) {
        self.lat.observe(secs);
    }
    pub fn ws_open(&self) {
        self.ws_total.fetch_add(1, Relaxed);
    }
    pub fn ws_reject(&self) {
        self.ws_rejected.fetch_add(1, Relaxed);
    }
    pub fn ws_frame(&self) {
        self.ws_frames.fetch_add(1, Relaxed);
    }
    pub fn ws_channels(&self, n: usize) {
        self.ws_channels.fetch_add(addressable_count(n), Relaxed);
    }
    pub fn ws_turn_patch(&self) {
        self.ws_turn_patches.fetch_add(1, Relaxed);
    }
    pub fn dedup_collapsed(&self, n: usize) {
        self.dedup_collapsed
            .fetch_add(addressable_count(n), Relaxed);
    }
    pub fn ws_error(&self) {
        self.ws_errors.fetch_add(1, Relaxed);
    }
    pub(crate) fn telemetry_queue_full(&self) {
        self.telemetry_queue_full.fetch_add(1, Relaxed);
    }
    pub(crate) fn telemetry_sink_closed(&self, count: usize) {
        self.telemetry_sink_closed
            .fetch_add(addressable_count(count), Relaxed);
    }
    pub(crate) fn telemetry_stdout_failure(&self) {
        self.telemetry_stdout_failures.fetch_add(1, Relaxed);
    }
    pub(crate) fn telemetry_stderr_failure(&self) {
        self.telemetry_stderr_failures.fetch_add(1, Relaxed);
    }
    pub fn observe_frontend(&self, secs: f64) {
        self.lat_frontend.observe(secs);
    }
    pub fn observe_encoder(&self, secs: f64) {
        self.lat_encoder.observe(secs);
    }
    pub fn observe_decode(&self, secs: f64) {
        self.lat_decode.observe(secs);
    }

    /// Prometheus text. `avail_http`/`avail_ws` are available semaphore permits used to derive active counts.
    pub fn render(
        &self,
        avail_http: usize,
        avail_ws: usize,
        ctc_pending: u64,
        rnnt_pending: Option<u64>,
    ) -> String {
        let mut s = String::new();
        let c = |o: &AtomicU64| o.load(Relaxed);
        macro_rules! counter {
            ($n:expr, $v:expr) => {{
                let _ = writeln!(s, "# TYPE {} counter\n{} {}", $n, $n, $v);
            }};
        }
        macro_rules! gauge {
            ($n:expr, $v:expr) => {{
                let _ = writeln!(s, "# TYPE {} gauge\n{} {}", $n, $n, $v);
            }};
        }
        counter!("asr_transcribe_requests_total", c(&self.transcribe_total));
        counter!("asr_transcribe_errors_total", c(&self.transcribe_errors));
        counter!("asr_overload_total", c(&self.overload_total));
        counter!("asr_timeout_total", c(&self.timeout_total));
        counter!("asr_ws_sessions_total", c(&self.ws_total));
        counter!("asr_ws_rejected_total", c(&self.ws_rejected));
        counter!("asr_ws_frames_total", c(&self.ws_frames));
        counter!("asr_ws_channels_total", c(&self.ws_channels));
        counter!("asr_ws_turn_patches_total", c(&self.ws_turn_patches));
        counter!("asr_dedup_collapsed_total", c(&self.dedup_collapsed));
        counter!("asr_ws_errors_total", c(&self.ws_errors));
        let _ = writeln!(s, "# TYPE asr_telemetry_dropped_total counter");
        let _ = writeln!(
            s,
            "asr_telemetry_dropped_total{{reason=\"queue_full\"}} {}",
            c(&self.telemetry_queue_full)
        );
        let _ = writeln!(
            s,
            "asr_telemetry_dropped_total{{reason=\"sink_closed\"}} {}",
            c(&self.telemetry_sink_closed)
        );
        let _ = writeln!(s, "# TYPE asr_telemetry_write_failures_total counter");
        let _ = writeln!(
            s,
            "asr_telemetry_write_failures_total{{destination=\"stdout\"}} {}",
            c(&self.telemetry_stdout_failures)
        );
        let _ = writeln!(
            s,
            "asr_telemetry_write_failures_total{{destination=\"stderr\"}} {}",
            c(&self.telemetry_stderr_failures)
        );
        gauge!(
            "asr_active_transcribe",
            self.max_http.saturating_sub(avail_http)
        );
        gauge!("asr_active_streams", self.max_ws.saturating_sub(avail_ws));
        gauge!("asr_max_transcribe", self.max_http);
        gauge!("asr_max_channels", self.max_ws);
        // GPU-worker queue depth by worker label: one TYPE line and labeled samples.
        let _ = writeln!(s, "# TYPE asr_worker_pending gauge");
        let _ = writeln!(s, "asr_worker_pending{{worker=\"ctc\"}} {ctc_pending}");
        if let Some(rp) = rnnt_pending {
            let _ = writeln!(s, "asr_worker_pending{{worker=\"rnnt\"}} {rp}");
        }
        let _ = writeln!(s, "# TYPE asr_build_info gauge");
        let _ = writeln!(
            s,
            "asr_build_info{{version=\"{}\"}} 1",
            env!("CARGO_PKG_VERSION")
        );
        self.lat.render(&mut s, "asr_transcribe_latency_seconds");
        self.lat_frontend.render(&mut s, "asr_frontend_seconds");
        self.lat_encoder.render(&mut s, "asr_encoder_seconds");
        self.lat_decode.render(&mut s, "asr_decode_seconds");
        s
    }
}

pub(crate) async fn endpoint(State(state): State<MetricsState>) -> Response {
    let gauges = state.gauges.snapshot();
    let body = state.metrics.render(
        gauges.available_http,
        gauges.available_ws,
        gauges.ctc_pending,
        gauges.rnnt_pending,
    );
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

fn addressable_count(value: usize) -> u64 {
    match usize_to_u64_checked(value) {
        Ok(value) => value,
        Err(_) => panic!("an addressable service collection count must fit the u64 metric counter"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{AdmissionState, ServiceAdmission};
    use axum::body::to_bytes;
    use axum::extract::State;
    use std::time::Duration;

    fn rendered_sample(rendered: &str, name: &str) -> f64 {
        let prefix = format!("{name} ");
        let line = match rendered.lines().find(|line| line.starts_with(&prefix)) {
            Some(line) => line,
            None => panic!("rendered metrics must include sample {name}"),
        };
        let value = match line.strip_prefix(&prefix) {
            Some(value) => value,
            None => panic!("located sample must retain its expected prefix"),
        };
        match value {
            "+Inf" => f64::INFINITY,
            "-Inf" => f64::NEG_INFINITY,
            _ => match value.parse::<f64>() {
                Ok(value) => value,
                Err(error) => panic!("rendered sample {name} must be numeric: {error}"),
            },
        }
    }

    #[test]
    fn histogram_sum_preserves_fractional_seconds() {
        let histogram = Hist::default();
        histogram.observe(0.012_9);

        let mut rendered = String::new();
        histogram.render(&mut rendered, "test_latency_seconds");

        assert_eq!(
            rendered_sample(&rendered, "test_latency_seconds_sum").to_bits(),
            0.012_9_f64.to_bits()
        );
        assert_eq!(
            rendered_sample(&rendered, "test_latency_seconds_count"),
            1.0
        );
        assert_eq!(
            histogram.snapshot().sum_seconds.to_bits(),
            0.012_9_f64.to_bits()
        );
        assert_eq!(
            rendered_sample(
                &rendered,
                "test_latency_seconds_rejected_total{reason=\"non_finite\"}",
            ),
            0.0
        );
    }

    #[test]
    fn rejected_latency_is_explicit_and_does_not_enter_the_histogram() {
        let histogram = Hist::default();
        histogram.observe(f64::NAN);
        histogram.observe(-0.001);

        let mut rendered = String::new();
        histogram.render(&mut rendered, "test_latency_seconds");

        assert_eq!(rendered_sample(&rendered, "test_latency_seconds_sum"), 0.0);
        assert_eq!(
            rendered_sample(&rendered, "test_latency_seconds_count"),
            0.0
        );
        assert_eq!(
            rendered_sample(
                &rendered,
                "test_latency_seconds_rejected_total{reason=\"non_finite\"}",
            ),
            1.0
        );
        assert_eq!(
            rendered_sample(
                &rendered,
                "test_latency_seconds_rejected_total{reason=\"negative\"}",
            ),
            1.0
        );
    }

    #[test]
    fn finite_latency_aggregate_reports_positive_infinity_after_f64_overflow() {
        let histogram = Hist::default();
        histogram.observe(f64::MAX);
        histogram.observe(f64::MAX);

        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.count, 2);
        assert_eq!(snapshot.sum_seconds, f64::INFINITY);

        let mut rendered = String::new();
        histogram.render(&mut rendered, "test_latency_seconds");
        assert_eq!(
            rendered_sample(&rendered, "test_latency_seconds_sum"),
            f64::INFINITY
        );
    }

    #[test]
    fn concurrent_observations_expose_only_coherent_histogram_snapshots() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::{Arc, Barrier};

        const WORKERS: usize = 4;
        const OBSERVATIONS_PER_WORKER: usize = 1_024;

        let histogram = Arc::new(Hist::default());
        let start = Arc::new(Barrier::new(WORKERS + 1));
        let completed = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..WORKERS {
                let histogram = Arc::clone(&histogram);
                let start = Arc::clone(&start);
                let completed = Arc::clone(&completed);
                scope.spawn(move || {
                    start.wait();
                    for _ in 0..OBSERVATIONS_PER_WORKER {
                        histogram.observe(0.005);
                    }
                    completed.fetch_add(1, Relaxed);
                });
            }

            start.wait();
            while completed.load(Relaxed) < WORKERS {
                let snapshot = histogram.snapshot();
                assert!(
                    snapshot
                        .buckets
                        .iter()
                        .all(|bucket| *bucket == snapshot.count)
                );
                assert!(snapshot.sum_seconds.is_finite());
                std::thread::yield_now();
            }
        });

        let expected = match u64::try_from(WORKERS * OBSERVATIONS_PER_WORKER) {
            Ok(value) => value,
            Err(_) => panic!("test observation count must fit the histogram counter"),
        };
        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.buckets, [expected; 11]);
        assert_eq!(snapshot.count, expected);
    }

    struct AdmissionGaugeSource {
        admission: Arc<AdmissionState>,
    }

    impl RuntimeGaugeSource for AdmissionGaugeSource {
        fn snapshot(&self) -> RuntimeGaugeSnapshot {
            RuntimeGaugeSnapshot {
                available_http: self.admission.available_http(),
                available_ws: self.admission.available_ws(),
                ctc_pending: 7,
                rnnt_pending: None,
            }
        }
    }

    #[tokio::test]
    async fn endpoint_scrapes_live_admission_capacity_without_inventing_an_rnnt_sample() {
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
            .expect("test admission settings are valid");
        let admission =
            Arc::new(AdmissionState::new(&settings).expect("test admission state is valid"));
        let state = MetricsState::new(
            Arc::new(Metrics::new(1, 1)),
            Arc::new(AdmissionGaugeSource {
                admission: Arc::clone(&admission),
            }),
        );
        let before = endpoint(State(state.clone())).await;
        let before = String::from_utf8(
            to_bytes(before.into_body(), usize::MAX)
                .await
                .expect("metrics body is available")
                .to_vec(),
        )
        .expect("metrics text is UTF-8");
        assert!(before.contains("asr_active_transcribe 0\n"));
        assert!(before.contains("asr_worker_pending{worker=\"ctc\"} 7\n"));
        assert!(!before.contains("asr_worker_pending{worker=\"rnnt\"}"));

        let token = admission
            .admit_http()
            .expect("running admission supplies one HTTP permit");
        let after = endpoint(State(state)).await;
        let after = String::from_utf8(
            to_bytes(after.into_body(), usize::MAX)
                .await
                .expect("metrics body is available")
                .to_vec(),
        )
        .expect("metrics text is UTF-8");
        assert!(after.contains("asr_active_transcribe 1\n"));
        drop(token);
    }

    #[test]
    fn telemetry_metric_families_start_at_zero_and_keep_their_fixed_labels() {
        let metrics = Metrics::new(1, 1);
        let initial = metrics.render(1, 1, 0, None);
        assert!(initial.contains("# TYPE asr_telemetry_dropped_total counter\n"));
        assert!(initial.contains("asr_telemetry_dropped_total{reason=\"queue_full\"} 0\n"));
        assert!(initial.contains("asr_telemetry_dropped_total{reason=\"sink_closed\"} 0\n"));
        assert!(initial.contains("# TYPE asr_telemetry_write_failures_total counter\n"));
        assert!(initial.contains("asr_telemetry_write_failures_total{destination=\"stdout\"} 0\n"));
        assert!(initial.contains("asr_telemetry_write_failures_total{destination=\"stderr\"} 0\n"));

        metrics.telemetry_queue_full();
        metrics.telemetry_sink_closed(2);
        metrics.telemetry_stdout_failure();
        metrics.telemetry_stderr_failure();
        let later = metrics.render(1, 1, 0, None);
        assert!(later.contains("asr_telemetry_dropped_total{reason=\"queue_full\"} 1\n"));
        assert!(later.contains("asr_telemetry_dropped_total{reason=\"sink_closed\"} 2\n"));
        assert!(later.contains("asr_telemetry_write_failures_total{destination=\"stdout\"} 1\n"));
        assert!(later.contains("asr_telemetry_write_failures_total{destination=\"stderr\"} 1\n"));
    }
}
