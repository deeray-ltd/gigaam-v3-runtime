// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! One monotonic lifecycle truth and the owned HTTP/WebSocket admission tokens.

use gigaam_recognition::{ExecutionControl, ExecutionState};
use gigaam_transcription::SourceChannelCount;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};

/// Validated admission capacities and the complete HTTP work deadline.
#[derive(Debug)]
pub struct ServiceAdmission {
    max_http: NonZeroUsize,
    max_ws: NonZeroUsize,
    request_timeout: Duration,
}

impl ServiceAdmission {
    pub fn new(max_http: usize, max_ws: usize, request_timeout: Duration) -> Result<Self, String> {
        let max_http = NonZeroUsize::new(max_http)
            .ok_or_else(|| "ASR_MAX_CONCURRENCY must be a positive integer".to_owned())?;
        let max_ws = NonZeroUsize::new(max_ws)
            .ok_or_else(|| "ASR_MAX_STREAMS must be a positive integer".to_owned())?;
        if max_http.get() > Semaphore::MAX_PERMITS {
            return Err("ASR_MAX_CONCURRENCY exceeds the supported semaphore capacity".into());
        }
        if max_ws.get() > Semaphore::MAX_PERMITS {
            return Err("ASR_MAX_STREAMS exceeds the supported semaphore capacity".into());
        }
        if request_timeout.is_zero() {
            return Err("ASR_REQ_TIMEOUT_SEC must be a positive integer".into());
        }
        Ok(Self {
            max_http,
            max_ws,
            request_timeout,
        })
    }

    pub(crate) const fn max_http(&self) -> usize {
        self.max_http.get()
    }

    pub(crate) const fn max_ws(&self) -> usize {
        self.max_ws.get()
    }

    pub(crate) const fn request_timeout(&self) -> Duration {
        self.request_timeout
    }
}

/// A request-body byte boundary whose construction cannot leave a zero-sized route limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestBodyLimit(usize);

impl RequestBodyLimit {
    pub fn new(bytes: usize) -> Result<Self, String> {
        if bytes == 0 {
            return Err("service request body limit must be greater than zero".into());
        }
        Ok(Self(bytes))
    }

    pub(crate) const fn bytes(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionPhase {
    Running,
    Draining,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrainTransition {
    Began,
    AlreadyDraining,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionRefusal {
    Overloaded,
    Draining,
}

/// The sole lifecycle truth plus the two independently owned protocol capacities.
pub(crate) struct AdmissionState {
    phase_tx: watch::Sender<AdmissionPhase>,
    http: Arc<Semaphore>,
    ws: Arc<Semaphore>,
    max_http: usize,
    max_ws: usize,
}

impl AdmissionState {
    pub(crate) fn new(settings: &ServiceAdmission) -> Result<Self, String> {
        let max_http = settings.max_http();
        let max_ws = settings.max_ws();
        let http = Arc::new(Semaphore::new(max_http));
        let ws = Arc::new(Semaphore::new(max_ws));
        let (phase_tx, _phase_rx) = watch::channel(AdmissionPhase::Running);
        Ok(Self {
            phase_tx,
            http,
            ws,
            max_http,
            max_ws,
        })
    }

    pub(crate) fn phase(&self) -> AdmissionPhase {
        *self.phase_tx.borrow()
    }

    pub(crate) fn subscribe(&self) -> watch::Receiver<AdmissionPhase> {
        self.phase_tx.subscribe()
    }

    pub(crate) fn begin_draining(&self) -> DrainTransition {
        let changed = self.phase_tx.send_if_modified(|phase| match phase {
            AdmissionPhase::Running => {
                *phase = AdmissionPhase::Draining;
                true
            }
            AdmissionPhase::Draining => false,
        });
        if changed {
            self.http.close();
            self.ws.close();
            DrainTransition::Began
        } else {
            DrainTransition::AlreadyDraining
        }
    }

    /// Commits HTTP admission only after the second lifecycle read while the permit is owned.
    pub(crate) fn admit_http(&self) -> Result<HttpAdmission, AdmissionRefusal> {
        self.reserve_http()?.commit(self)
    }

    fn reserve_http(&self) -> Result<PendingHttpAdmission, AdmissionRefusal> {
        if self.phase() == AdmissionPhase::Draining {
            return Err(AdmissionRefusal::Draining);
        }
        let permit = match Arc::clone(&self.http).try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => return Err(self.refusal_after_capacity_failure()),
        };
        Ok(PendingHttpAdmission { permit })
    }

    /// Commits all declared source-channel permits together or releases all of them on a drain race.
    pub(crate) fn admit_ws(
        &self,
        channels: WsPermitCount,
    ) -> Result<WsAdmission, AdmissionRefusal> {
        self.reserve_ws(channels)?.commit(self)
    }

    fn reserve_ws(&self, channels: WsPermitCount) -> Result<PendingWsAdmission, AdmissionRefusal> {
        if self.phase() == AdmissionPhase::Draining {
            return Err(AdmissionRefusal::Draining);
        }
        let permit = match Arc::clone(&self.ws).try_acquire_many_owned(channels.0.get()) {
            Ok(permit) => permit,
            Err(_) => return Err(self.refusal_after_capacity_failure()),
        };
        Ok(PendingWsAdmission { permit })
    }

    fn refusal_after_capacity_failure(&self) -> AdmissionRefusal {
        match self.phase() {
            AdmissionPhase::Running => AdmissionRefusal::Overloaded,
            AdmissionPhase::Draining => AdmissionRefusal::Draining,
        }
    }

    pub(crate) const fn max_http(&self) -> usize {
        self.max_http
    }

    pub(crate) const fn max_ws(&self) -> usize {
        self.max_ws
    }

    pub(crate) fn available_http(&self) -> usize {
        self.http.available_permits()
    }

    pub(crate) fn available_ws(&self) -> usize {
        self.ws.available_permits()
    }

    /// Uses returned channel permits, not an independent active-session counter, as drain truth.
    pub(crate) async fn wait_for_ws_drain(&self, bound: Duration) -> WsDrainWait {
        let deadline = tokio::time::Instant::now() + bound;
        loop {
            if self.ws.available_permits() == self.max_ws {
                return WsDrainWait::Drained;
            }
            if tokio::time::Instant::now() >= deadline {
                return WsDrainWait::TimedOut;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

/// Linearizes lifecycle refusal before the process boundary emits its unchanged diagnostic.
pub(crate) fn begin_draining_then(admission: &AdmissionState, diagnostic: impl FnOnce()) {
    admission.begin_draining();
    diagnostic();
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WsDrainWait {
    Drained,
    TimedOut,
}

/// A checked nonzero `Semaphore::acquire_many` request derived from stream source channels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WsPermitCount(NonZeroU32);

impl WsPermitCount {
    pub(crate) fn from_source_channels(channels: SourceChannelCount) -> Result<Self, String> {
        let count = u32::try_from(channels.get())
            .map_err(|_| "stream channel count exceeds semaphore range".to_owned())?;
        let count = NonZeroU32::new(count)
            .ok_or_else(|| "stream channel count must be greater than zero".to_owned())?;
        let count_usize = usize::try_from(count.get())
            .map_err(|_| "stream channel count exceeds semaphore range".to_owned())?;
        if count_usize > Semaphore::MAX_PERMITS {
            return Err("stream channel count exceeds semaphore range".into());
        }
        Ok(Self(count))
    }

    #[cfg(test)]
    fn new_for_test(count: u32) -> Result<Self, String> {
        let channels = SourceChannelCount::new(
            usize::try_from(count)
                .map_err(|_| "test stream channel count must fit usize".to_owned())?,
        )?;
        Self::from_source_channels(channels)
    }
}

/// A reservation that has capacity but has not crossed the final admission read.
struct PendingHttpAdmission {
    permit: OwnedSemaphorePermit,
}

impl PendingHttpAdmission {
    fn commit(self, state: &AdmissionState) -> Result<HttpAdmission, AdmissionRefusal> {
        match state.phase() {
            AdmissionPhase::Running => Ok(HttpAdmission {
                permit: self.permit,
            }),
            AdmissionPhase::Draining => Err(AdmissionRefusal::Draining),
        }
    }
}

/// An all-channel reservation that has capacity but has not crossed the final admission read.
struct PendingWsAdmission {
    permit: OwnedSemaphorePermit,
}

impl PendingWsAdmission {
    fn commit(self, state: &AdmissionState) -> Result<WsAdmission, AdmissionRefusal> {
        match state.phase() {
            AdmissionPhase::Running => Ok(WsAdmission {
                _permit: self.permit,
            }),
            AdmissionPhase::Draining => Err(AdmissionRefusal::Draining),
        }
    }
}

/// One committed HTTP capacity reservation.
pub(crate) struct HttpAdmission {
    permit: OwnedSemaphorePermit,
}

/// One committed all-channel WebSocket capacity reservation.
pub(crate) struct WsAdmission {
    _permit: OwnedSemaphorePermit,
}

/// HTTP work owns both request execution control and its committed capacity until terminalization.
pub(crate) struct RequestWorkOwner {
    _permit: OwnedSemaphorePermit,
    control: Option<ExecutionControl>,
}

impl RequestWorkOwner {
    pub(crate) fn new(admission: HttpAdmission, control: ExecutionControl) -> Self {
        Self {
            _permit: admission.permit,
            control: Some(control),
        }
    }

    pub(crate) fn complete(mut self) -> ExecutionState {
        let control = self.take_control();
        control.complete()
    }

    pub(crate) fn fail(mut self) -> ExecutionState {
        let control = self.take_control();
        control.fail()
    }

    fn take_control(&mut self) -> ExecutionControl {
        self.control
            .take()
            .expect("request work retains execution control until its terminal acknowledgement")
    }
}

impl Drop for RequestWorkOwner {
    fn drop(&mut self) {
        if let Some(control) = self.control.take() {
            control.fail();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdmissionPhase, AdmissionRefusal, AdmissionState, DrainTransition, RequestBodyLimit,
        ServiceAdmission, WsDrainWait, WsPermitCount, begin_draining_then,
    };
    use std::sync::Arc;
    use std::time::Duration;

    fn state(http: usize, ws: usize) -> AdmissionState {
        let settings = ServiceAdmission::new(http, ws, Duration::from_secs(1))
            .expect("positive admission capacities create a test state");
        AdmissionState::new(&settings).expect("validated settings create admission state")
    }

    #[test]
    fn drain_is_monotonic_and_idempotent() {
        let state = state(1, 2);
        assert_eq!(state.phase(), AdmissionPhase::Running);
        assert_eq!(state.begin_draining(), DrainTransition::Began);
        assert_eq!(state.phase(), AdmissionPhase::Draining);
        assert_eq!(state.begin_draining(), DrainTransition::AlreadyDraining);
        assert_eq!(state.phase(), AdmissionPhase::Draining);
    }

    #[test]
    fn request_body_limit_refuses_zero_with_the_public_constructor_message() {
        assert_eq!(
            RequestBodyLimit::new(0),
            Err("service request body limit must be greater than zero".to_owned())
        );
    }

    #[test]
    fn committed_http_work_survives_drain_but_later_admission_refuses() {
        let state = state(1, 2);
        let committed = state
            .admit_http()
            .expect("running capacity commits HTTP work");
        assert_eq!(state.begin_draining(), DrainTransition::Began);
        assert!(matches!(
            state.admit_http(),
            Err(AdmissionRefusal::Draining)
        ));
        drop(committed);
        assert_eq!(state.available_http(), state.max_http());
    }

    #[test]
    fn exhausted_running_capacity_is_overload_but_drain_refuses() {
        let state = state(1, 2);
        let committed = state
            .admit_http()
            .expect("running capacity commits HTTP work");
        assert!(matches!(
            state.admit_http(),
            Err(AdmissionRefusal::Overloaded)
        ));
        assert_eq!(state.begin_draining(), DrainTransition::Began);
        assert!(matches!(
            state.admit_http(),
            Err(AdmissionRefusal::Draining)
        ));
        drop(committed);
    }

    #[test]
    fn websocket_reservation_is_all_or_none() {
        let state = state(1, 3);
        let committed = state
            .admit_ws(WsPermitCount::new_for_test(2).expect("two test permits are valid"))
            .expect("two running channel permits commit together");
        assert_eq!(state.available_ws(), 1);
        assert!(matches!(
            state.admit_ws(WsPermitCount::new_for_test(2).expect("two test permits are valid")),
            Err(AdmissionRefusal::Overloaded)
        ));
        drop(committed);
        assert_eq!(state.available_ws(), state.max_ws());
    }

    #[test]
    fn reservation_loses_a_drain_race_and_releases_capacity() {
        let state = state(1, 2);
        let pending = state
            .reserve_http()
            .expect("the first lifecycle read and reservation succeed while running");
        assert_eq!(state.available_http(), 0);
        assert_eq!(state.begin_draining(), DrainTransition::Began);
        assert!(matches!(
            pending.commit(&state),
            Err(AdmissionRefusal::Draining)
        ));
        assert_eq!(
            state.available_http(),
            state.max_http(),
            "the uncommitted reservation must release after its final drain read"
        );
    }

    #[test]
    fn websocket_reservation_loses_a_drain_race_and_releases_all_channels() {
        let state = state(1, 3);
        let permits = WsPermitCount::new_for_test(2).expect("two test permits are valid");
        let pending = state
            .reserve_ws(permits)
            .expect("running state reserves every requested channel together");
        assert_eq!(state.available_ws(), 1);
        assert_eq!(state.begin_draining(), DrainTransition::Began);
        assert!(matches!(
            pending.commit(&state),
            Err(AdmissionRefusal::Draining)
        ));
        assert_eq!(state.available_ws(), state.max_ws());
    }

    #[tokio::test]
    async fn bounded_drain_wait_counts_an_admitted_upgrade_before_its_callback_runs() {
        let state = state(1, 2);
        let token = state
            .admit_ws(WsPermitCount::new_for_test(2).expect("two test permits are valid"))
            .expect("the pre-upgrade callback token commits while running");
        assert_eq!(state.begin_draining(), DrainTransition::Began);
        assert_eq!(
            state.wait_for_ws_drain(Duration::ZERO).await,
            WsDrainWait::TimedOut,
            "a callback-owned token must remain in the bounded drain count"
        );
        drop(token);
        assert_eq!(
            state.wait_for_ws_drain(Duration::ZERO).await,
            WsDrainWait::Drained
        );
    }

    #[test]
    fn draining_precedes_the_synchronous_signal_diagnostic() {
        let state = Arc::new(state(1, 2));
        let diagnostic_started = std::sync::mpsc::sync_channel(0);
        let resume_diagnostic = std::sync::mpsc::sync_channel(0);
        let diagnostic_state = Arc::clone(&state);
        let thread = std::thread::spawn(move || {
            begin_draining_then(diagnostic_state.as_ref(), || {
                diagnostic_started
                    .0
                    .send(())
                    .expect("the test must observe the blocked diagnostic");
                resume_diagnostic
                    .1
                    .recv()
                    .expect("the test must release the blocked diagnostic");
            });
        });
        diagnostic_started
            .1
            .recv()
            .expect("the diagnostic must block after draining linearizes");
        assert_eq!(state.phase(), AdmissionPhase::Draining);
        assert!(matches!(
            state.admit_http(),
            Err(AdmissionRefusal::Draining)
        ));
        assert!(matches!(
            state.admit_ws(WsPermitCount::new_for_test(2).expect("two permits are valid")),
            Err(AdmissionRefusal::Draining)
        ));
        assert_eq!(
            crate::health::readiness_with_workers(state.phase(), false)
                .status()
                .as_u16(),
            503,
            "readiness observes the same transition while the diagnostic remains blocked"
        );
        resume_diagnostic
            .0
            .send(())
            .expect("the blocked diagnostic must be released");
        thread
            .join()
            .expect("the signal transition helper must return normally");
    }
}
