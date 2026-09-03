// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Request-owned execution lifecycle control shared by scheduled recognition work.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

const READY: u8 = 0;
const QUEUED: u8 = 1;
const RUNNING: u8 = 2;
const CANCEL_REQUESTED: u8 = 3;
const COMPLETED: u8 = 4;
const FAILED: u8 = 5;
const CANCELLED: u8 = 6;

/// Observable lifecycle state of one request-owned execution control.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionState {
    Ready,
    Queued,
    Running,
    CancelRequested,
    Completed,
    Failed,
    Cancelled,
}

impl ExecutionState {
    fn from_stored(value: u8) -> Self {
        match value {
            READY => Self::Ready,
            QUEUED => Self::Queued,
            RUNNING => Self::Running,
            CANCEL_REQUESTED => Self::CancelRequested,
            COMPLETED => Self::Completed,
            FAILED => Self::Failed,
            CANCELLED => Self::Cancelled,
            _ => panic!("execution control stores only declared lifecycle states"),
        }
    }
}

/// Cloneable control for all scheduled decoder calls made on behalf of one request.
///
/// A control reaches `Running` when its first scheduled call starts and remains there while the
/// request submits later sequential calls. The request owner, rather than an individual decoder
/// call, acknowledges the terminal success or failure.
#[derive(Clone, Debug)]
pub struct ExecutionControl {
    state: Arc<AtomicU8>,
}

impl ExecutionControl {
    /// Creates a control for a caller that owns its deadline and terminal response.
    pub fn for_request() -> Self {
        Self::new()
    }

    /// Creates a control for work whose caller deliberately has no deadline.
    pub fn without_deadline() -> Self {
        Self::new()
    }

    fn new() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(READY)),
        }
    }

    /// Returns the current lifecycle observation using acquire synchronization.
    pub fn state(&self) -> ExecutionState {
        ExecutionState::from_stored(self.state.load(Ordering::Acquire))
    }

    /// Requests cancellation. Repeating the request returns the already-observed state.
    pub fn request_cancellation(&self) -> ExecutionState {
        loop {
            let observed = self.state.load(Ordering::Acquire);
            match ExecutionState::from_stored(observed) {
                ExecutionState::Ready | ExecutionState::Queued | ExecutionState::Running => {
                    if self
                        .state
                        .compare_exchange(
                            observed,
                            CANCEL_REQUESTED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return ExecutionState::CancelRequested;
                    }
                }
                ExecutionState::CancelRequested
                | ExecutionState::Completed
                | ExecutionState::Failed
                | ExecutionState::Cancelled => return ExecutionState::from_stored(observed),
            }
        }
    }

    /// Acknowledges cancellation after the scheduler has reached a terminal job outcome.
    pub(crate) fn acknowledge_cancellation(&self) -> ExecutionState {
        loop {
            let observed = self.state.load(Ordering::Acquire);
            match ExecutionState::from_stored(observed) {
                ExecutionState::CancelRequested => {
                    if self
                        .state
                        .compare_exchange(
                            CANCEL_REQUESTED,
                            CANCELLED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return ExecutionState::Cancelled;
                    }
                }
                ExecutionState::Cancelled => return ExecutionState::Cancelled,
                ExecutionState::Ready
                | ExecutionState::Queued
                | ExecutionState::Running
                | ExecutionState::Completed
                | ExecutionState::Failed => return ExecutionState::from_stored(observed),
            }
        }
    }

    /// Completes a request. A pending cancellation is terminally acknowledged instead.
    pub fn complete(&self) -> ExecutionState {
        loop {
            let observed = self.state.load(Ordering::Acquire);
            match ExecutionState::from_stored(observed) {
                ExecutionState::Ready | ExecutionState::Queued | ExecutionState::Running => {
                    if self
                        .state
                        .compare_exchange(observed, COMPLETED, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        return ExecutionState::Completed;
                    }
                }
                ExecutionState::CancelRequested => return self.acknowledge_cancellation(),
                ExecutionState::Completed | ExecutionState::Failed | ExecutionState::Cancelled => {
                    return ExecutionState::from_stored(observed);
                }
            }
        }
    }

    /// Fails a request. A pending cancellation is terminally acknowledged instead.
    pub fn fail(&self) -> ExecutionState {
        loop {
            let observed = self.state.load(Ordering::Acquire);
            match ExecutionState::from_stored(observed) {
                ExecutionState::Ready | ExecutionState::Queued | ExecutionState::Running => {
                    if self
                        .state
                        .compare_exchange(observed, FAILED, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        return ExecutionState::Failed;
                    }
                }
                ExecutionState::CancelRequested => return self.acknowledge_cancellation(),
                ExecutionState::Completed | ExecutionState::Failed | ExecutionState::Cancelled => {
                    return ExecutionState::from_stored(observed);
                }
            }
        }
    }

    pub(crate) fn enqueue(&self) -> Result<(), String> {
        loop {
            let observed = self.state.load(Ordering::Acquire);
            match ExecutionState::from_stored(observed) {
                ExecutionState::Ready => {
                    if self
                        .state
                        .compare_exchange(READY, QUEUED, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        return Ok(());
                    }
                }
                ExecutionState::Queued | ExecutionState::Running => return Ok(()),
                ExecutionState::CancelRequested => return Err("execution cancelled".into()),
                ExecutionState::Cancelled => return Err("execution cancelled".into()),
                ExecutionState::Completed | ExecutionState::Failed => {
                    return Err("execution is already terminal".into());
                }
            }
        }
    }

    pub(crate) fn start(&self) -> Result<(), String> {
        loop {
            let observed = self.state.load(Ordering::Acquire);
            match ExecutionState::from_stored(observed) {
                ExecutionState::Ready | ExecutionState::Queued => {
                    if self
                        .state
                        .compare_exchange(observed, RUNNING, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        return Ok(());
                    }
                }
                ExecutionState::Running => return Ok(()),
                ExecutionState::CancelRequested => return Err("execution cancelled".into()),
                ExecutionState::Cancelled => return Err("execution cancelled".into()),
                ExecutionState::Completed | ExecutionState::Failed => {
                    return Err("execution is already terminal".into());
                }
            }
        }
    }

    pub(crate) fn cancellation_won(&self) -> bool {
        matches!(
            self.state(),
            ExecutionState::CancelRequested | ExecutionState::Cancelled
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutionControl, ExecutionState};

    #[test]
    fn completion_never_revives_a_cancelled_request() {
        let control = ExecutionControl::for_request();
        control.enqueue().expect("a ready request can queue work");
        control.start().expect("queued work can start");
        assert_eq!(
            control.request_cancellation(),
            ExecutionState::CancelRequested
        );
        assert_eq!(control.complete(), ExecutionState::Cancelled);
        assert_eq!(control.state(), ExecutionState::Cancelled);
    }

    #[test]
    fn ordinary_request_lifecycle_completes() {
        let control = ExecutionControl::for_request();
        control.enqueue().expect("a ready request can queue work");
        control.start().expect("queued work can start");
        assert_eq!(control.complete(), ExecutionState::Completed);
        assert_eq!(control.state(), ExecutionState::Completed);
    }

    #[test]
    fn active_failure_and_cancellation_have_precise_terminal_outcomes() {
        let failed = ExecutionControl::for_request();
        assert_eq!(failed.fail(), ExecutionState::Failed);
        assert_eq!(failed.request_cancellation(), ExecutionState::Failed);

        let cancelled = ExecutionControl::for_request();
        assert_eq!(
            cancelled.request_cancellation(),
            ExecutionState::CancelRequested
        );
        assert_eq!(cancelled.fail(), ExecutionState::Cancelled);
        assert_eq!(cancelled.state(), ExecutionState::Cancelled);
    }
}
