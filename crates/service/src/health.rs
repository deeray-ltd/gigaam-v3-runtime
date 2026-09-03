// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Liveness, readiness, and capability probes over the shared admission state.

use crate::admission::{AdmissionPhase, AdmissionState};
use crate::response::{
    HealthProjection, ReadinessOutcome, health_response, liveness_response, readiness_response,
};
use axum::extract::State;
use axum::response::Response;
use gigaam_recognition::ExecutionScheduler;
use std::sync::Arc;

/// Narrow typed state for immutable capability probes and the one admission lifecycle truth.
#[derive(Clone)]
pub(crate) struct HealthState {
    admission: Arc<AdmissionState>,
    ctc: Arc<ExecutionScheduler>,
    rnnt: Option<Arc<ExecutionScheduler>>,
    projection: HealthProjection,
}

impl HealthState {
    pub(crate) fn new(
        admission: Arc<AdmissionState>,
        ctc: Arc<ExecutionScheduler>,
        rnnt: Option<Arc<ExecutionScheduler>>,
        projection: HealthProjection,
    ) -> Self {
        Self {
            admission,
            ctc,
            rnnt,
            projection,
        }
    }

    /// Whether any owned recognition worker has stopped after a decoder panic. Ordinary decode
    /// errors are returned to the calling request and do not stop the worker.
    fn workers_stopped(&self) -> bool {
        self.ctc.is_stopped()
            || self
                .rnnt
                .as_ref()
                .is_some_and(|scheduler| scheduler.is_stopped())
    }
}

pub(crate) async fn health(State(state): State<HealthState>) -> Response {
    health_response(state.projection)
}

pub(crate) async fn livez() -> Response {
    liveness_response()
}

pub(crate) async fn readyz(State(state): State<HealthState>) -> Response {
    readiness_with_workers(state.admission.phase(), state.workers_stopped())
}

/// Readiness that also fails once an owned recognition worker has stopped after a decoder
/// panic; ordinary decode errors are returned to the calling request and do not stop the
/// worker. A stopped worker fails every later call and the process must be restarted. Draining
/// wins whenever the phase is Draining: a draining process with a stopped worker still reports
/// draining, not a worker failure, because it is already shutting down for its own reason.
pub(crate) fn readiness_with_workers(phase: AdmissionPhase, workers_stopped: bool) -> Response {
    let outcome = match phase {
        AdmissionPhase::Draining => ReadinessOutcome::Draining,
        AdmissionPhase::Running if workers_stopped => ReadinessOutcome::WorkerStopped,
        AdmissionPhase::Running => ReadinessOutcome::Ready,
    };
    readiness_response(outcome)
}

#[cfg(test)]
mod tests {
    use super::{HealthState, health, livez, readiness_with_workers, readyz};
    use crate::admission::ServiceAdmission;
    use crate::admission::{AdmissionPhase, AdmissionState, DrainTransition};
    use crate::response::HealthProjection;
    use axum::body::to_bytes;
    use axum::extract::State;
    use axum::http::header;
    use axum::response::Response;
    use gigaam_audio::{FeatureMatrix, FeatureMatrixView};
    use gigaam_recognition::{
        Decoded, Device, ExecutionControl, ExecutionScheduler, FrameRate, WindowDecoder,
    };
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    /// A decoder that never fails, used only to exercise a running (not stopped) worker.
    struct IdleDecoder;

    impl WindowDecoder for IdleDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(25.0).expect("the health probe test decoder frame rate is positive")
        }

        fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            Decoded::new(
                Vec::new(),
                vec![false; features.frames()],
                features.frames(),
                0.0,
            )
        }
    }

    /// A decoder that always panics, used to drive a real scheduler into its stopped state.
    struct PanicDecoder;

    impl WindowDecoder for PanicDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(25.0).expect("the health probe test decoder frame rate is positive")
        }

        fn decode(&mut self, _features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            panic!("controlled health probe decoder panic");
        }
    }

    fn idle_scheduler() -> Arc<ExecutionScheduler> {
        Arc::new(ExecutionScheduler::spawn(IdleDecoder))
    }

    fn wait_until(description: &str, condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(
                Instant::now() < deadline,
                "test deadlock guard elapsed while waiting for {description}"
            );
            thread::yield_now();
        }
    }

    #[tokio::test]
    async fn readiness_uses_the_same_monotonic_transition_as_admission() {
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
            .expect("positive test admission settings are valid");
        let admission = AdmissionState::new(&settings).expect("test admission state is valid");
        let before = readiness_with_workers(admission.phase(), false);
        assert_eq!(before.status().as_u16(), 200);
        assert_eq!(
            to_bytes(before.into_body(), usize::MAX)
                .await
                .expect("readiness response body is available"),
            "{\"status\":\"ready\"}"
        );
        assert_eq!(admission.begin_draining(), DrainTransition::Began);
        assert_eq!(admission.phase(), AdmissionPhase::Draining);
        let after = readiness_with_workers(admission.phase(), false);
        assert_eq!(after.status().as_u16(), 503);
        assert_eq!(
            to_bytes(after.into_body(), usize::MAX)
                .await
                .expect("draining readiness response body is available"),
            "{\"error\":\"draining\"}"
        );
    }

    #[tokio::test]
    async fn liveness_remains_stable_while_readiness_drains() {
        let response = livez().await;
        assert_eq!(response.status().as_u16(), 200);
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("liveness response body is available"),
            "{\"status\":\"alive\"}"
        );
    }

    #[tokio::test]
    async fn probe_triple_preserves_exact_health_liveness_and_readiness_across_drain() {
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
            .expect("positive test admission settings are valid");
        let admission = Arc::new(
            AdmissionState::new(&settings).expect("validated test admission state creates health"),
        );
        let state = HealthState::new(
            Arc::clone(&admission),
            idle_scheduler(),
            None,
            HealthProjection::new(Device::Cpu, false),
        );

        let health_before = health(State(state.clone())).await;
        assert_eq!(health_before.status().as_u16(), 200);
        assert_eq!(
            health_before.headers().get(header::CONTENT_TYPE),
            Some(&axum::http::HeaderValue::from_static(
                "application/json; charset=utf-8",
            ))
        );
        assert_eq!(
            to_bytes(health_before.into_body(), usize::MAX)
                .await
                .expect("health response body is available"),
            "{\"status\":\"ok\",\"models\":[\"ctc\"],\"provider\":\"cpu\"}"
        );
        let live_before = livez().await;
        assert_eq!(
            to_bytes(live_before.into_body(), usize::MAX)
                .await
                .expect("liveness response body is available"),
            "{\"status\":\"alive\"}"
        );
        let ready_before = readyz(State(state.clone())).await;
        assert_eq!(ready_before.status().as_u16(), 200);
        assert_eq!(
            to_bytes(ready_before.into_body(), usize::MAX)
                .await
                .expect("readiness response body is available"),
            "{\"status\":\"ready\"}"
        );

        assert_eq!(admission.begin_draining(), DrainTransition::Began);
        let health_after = health(State(state.clone())).await;
        assert_eq!(health_after.status().as_u16(), 200);
        assert_eq!(
            to_bytes(health_after.into_body(), usize::MAX)
                .await
                .expect("health response body is available"),
            "{\"status\":\"ok\",\"models\":[\"ctc\"],\"provider\":\"cpu\"}"
        );
        let live_after = livez().await;
        assert_eq!(
            to_bytes(live_after.into_body(), usize::MAX)
                .await
                .expect("liveness response body is available"),
            "{\"status\":\"alive\"}"
        );
        let ready_after = readyz(State(state)).await;
        assert_eq!(ready_after.status().as_u16(), 503);
        assert_eq!(
            to_bytes(ready_after.into_body(), usize::MAX)
                .await
                .expect("draining readiness response body is available"),
            "{\"error\":\"draining\"}"
        );
    }

    async fn readiness_body(response: Response) -> String {
        String::from_utf8(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("readiness response body is available")
                .to_vec(),
        )
        .expect("readiness response body is UTF-8")
    }

    #[tokio::test]
    async fn readiness_with_workers_names_the_typed_reason_and_draining_wins_over_a_stopped_worker()
    {
        let ready = readiness_with_workers(AdmissionPhase::Running, false);
        assert_eq!(ready.status().as_u16(), 200);
        assert_eq!(readiness_body(ready).await, "{\"status\":\"ready\"}");

        let worker_stopped = readiness_with_workers(AdmissionPhase::Running, true);
        assert_eq!(worker_stopped.status().as_u16(), 503);
        assert_eq!(
            readiness_body(worker_stopped).await,
            "{\"error\":\"worker stopped\"}"
        );

        let draining = readiness_with_workers(AdmissionPhase::Draining, false);
        assert_eq!(draining.status().as_u16(), 503);
        assert_eq!(readiness_body(draining).await, "{\"error\":\"draining\"}");

        let draining_and_stopped = readiness_with_workers(AdmissionPhase::Draining, true);
        assert_eq!(draining_and_stopped.status().as_u16(), 503);
        assert_eq!(
            readiness_body(draining_and_stopped).await,
            "{\"error\":\"draining\"}",
            "a draining process with a stopped worker must still report draining"
        );
    }

    #[tokio::test]
    async fn readyz_fails_once_a_real_recognition_worker_has_stopped_after_a_panic() {
        let settings = ServiceAdmission::new(1, 1, Duration::from_secs(1))
            .expect("positive test admission settings are valid");
        let admission =
            Arc::new(AdmissionState::new(&settings).expect("test admission state is valid"));
        let ctc = Arc::new(ExecutionScheduler::spawn(PanicDecoder));
        let state = HealthState::new(
            Arc::clone(&admission),
            Arc::clone(&ctc),
            None,
            HealthProjection::new(Device::Cpu, false),
        );

        let ready_before = readyz(State(state.clone())).await;
        assert_eq!(ready_before.status().as_u16(), 200);

        let control = ExecutionControl::without_deadline();
        let features = FeatureMatrix::from_values(1, 1, vec![0.0])
            .expect("test feature dimensions and values are valid");
        assert!(ctc.window_channel(control).decode(features.view()).is_err());
        wait_until(
            "the CTC worker to report stopped after its decoder panic",
            || ctc.is_stopped(),
        );

        let ready_after = readyz(State(state.clone())).await;
        assert_eq!(ready_after.status().as_u16(), 503);
        assert_eq!(
            to_bytes(ready_after.into_body(), usize::MAX)
                .await
                .expect("stopped-worker readiness response body is available"),
            "{\"error\":\"worker stopped\"}"
        );

        // A draining process with a stopped worker still reports draining, not a worker
        // failure: it is already shutting down for its own reason.
        assert_eq!(admission.begin_draining(), DrainTransition::Began);
        let ready_while_draining_and_stopped = readyz(State(state)).await;
        assert_eq!(ready_while_draining_and_stopped.status().as_u16(), 503);
        assert_eq!(
            to_bytes(ready_while_draining_and_stopped.into_body(), usize::MAX)
                .await
                .expect("draining-and-stopped readiness response body is available"),
            "{\"error\":\"draining\"}"
        );
    }
}
