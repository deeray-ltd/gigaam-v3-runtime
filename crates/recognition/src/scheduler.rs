// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Dedicated FIFO recognition execution scheduler with strict streaming-tick priority.

use crate::contracts::{Decoded, FrameRate, WindowDecoder};
use crate::execution::ExecutionControl;
use gigaam_audio::{FeatureMatrix, FeatureMatrixView};
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Maximum time a scheduled call may wait before its decoder starts.
const QUEUE_WAIT_BOUND: Duration = Duration::from_secs(60);

#[derive(Clone, Copy)]
enum Priority {
    Tick,
    Window,
}

struct Job {
    features: FeatureMatrix,
    control: ExecutionControl,
    events: Sender<JobEvent>,
}

enum JobEvent {
    Started,
    Finished(Result<Decoded, String>),
}

/// Named inputs for one synchronous recognizer facade.
struct ScheduledRecognizerParts {
    shared: Arc<(Mutex<Queue>, Condvar)>,
    frame_rate: FrameRate,
    priority: Priority,
    pending: Arc<AtomicU64>,
    control: ExecutionControl,
}

#[derive(Default)]
struct Queue {
    ticks: VecDeque<Job>,
    windows: VecDeque<Job>,
    stopped: bool,
}

fn queue_lock(lock: &Mutex<Queue>) -> MutexGuard<'_, Queue> {
    match lock.lock() {
        Ok(queue) => queue,
        Err(_) => panic!("execution scheduler queue mutex is never held across a panic"),
    }
}

fn queue_count(value: usize) -> u64 {
    match u64::try_from(value) {
        Ok(value) => value,
        Err(_) => panic!("an addressable scheduler queue length must fit the pending counter"),
    }
}

fn send_event(sender: Sender<JobEvent>, event: JobEvent) {
    if sender.send(event).is_err() {
        // The caller timed out or disconnected after cancellation; its result is intentionally dropped.
    }
}

/// Dedicated owner of one non-concurrent recognition decoder.
///
/// Streaming ticks are dequeued before file windows. Each queued item carries a request-owned
/// [`ExecutionControl`], so cancellation prevents a queued decoder invocation and discards a
/// result that returns after cancellation.
pub struct ExecutionScheduler {
    shared: Arc<(Mutex<Queue>, Condvar)>,
    frame_rate: FrameRate,
    pending: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

impl ExecutionScheduler {
    /// Starts a dedicated thread for one `WindowDecoder` instance.
    pub fn spawn<D: WindowDecoder + Send + 'static>(mut decoder: D) -> Self {
        let frame_rate = decoder.frame_rate();
        let shared = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        let scheduler_shared = Arc::clone(&shared);
        let pending = Arc::new(AtomicU64::new(0));
        let scheduler_pending = Arc::clone(&pending);
        let handle = thread::spawn(move || {
            loop {
                let job = {
                    let (lock, wake) = &*scheduler_shared;
                    let mut queue = queue_lock(lock);
                    loop {
                        if let Some(job) = queue
                            .ticks
                            .pop_front()
                            .or_else(|| queue.windows.pop_front())
                        {
                            scheduler_pending.fetch_sub(1, Ordering::AcqRel);
                            break job;
                        }
                        if queue.stopped {
                            return;
                        }
                        queue = match wake.wait(queue) {
                            Ok(queue) => queue,
                            Err(_) => {
                                panic!(
                                    "execution scheduler queue mutex is never held across a wait panic"
                                )
                            }
                        };
                    }
                };

                if let Err(error) = job.control.start() {
                    if job.control.cancellation_won() {
                        job.control.acknowledge_cancellation();
                    }
                    send_event(job.events, JobEvent::Finished(Err(error)));
                    continue;
                }
                send_event(job.events.clone(), JobEvent::Started);

                match catch_unwind(AssertUnwindSafe(|| decoder.decode(job.features.view()))) {
                    Ok(result) => {
                        if job.control.cancellation_won() {
                            job.control.acknowledge_cancellation();
                            send_event(
                                job.events,
                                JobEvent::Finished(Err("execution cancelled".into())),
                            );
                        } else {
                            send_event(job.events, JobEvent::Finished(result));
                        }
                    }
                    Err(_) => {
                        job.control.fail();
                        send_event(
                            job.events,
                            JobEvent::Finished(Err("decoder panicked".into())),
                        );
                        stop_after_panic(&scheduler_shared, &scheduler_pending);
                        return;
                    }
                }
            }
        });
        Self {
            shared,
            frame_rate,
            pending,
            handle: Some(handle),
        }
    }

    /// Produces a file-window decoder bound to `control`.
    pub fn window_channel(&self, control: ExecutionControl) -> ScheduledRecognizer {
        ScheduledRecognizer::new(self.parts(Priority::Window, control))
    }

    /// Produces a streaming-tick decoder bound to `control`.
    pub fn tick_channel(&self, control: ExecutionControl) -> ScheduledRecognizer {
        ScheduledRecognizer::new(self.parts(Priority::Tick, control))
    }

    fn parts(&self, priority: Priority, control: ExecutionControl) -> ScheduledRecognizerParts {
        ScheduledRecognizerParts {
            shared: Arc::clone(&self.shared),
            frame_rate: self.frame_rate,
            priority,
            pending: Arc::clone(&self.pending),
            control,
        }
    }

    pub const fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    /// Current count of queued, not-yet-started recognition calls.
    pub fn pending(&self) -> u64 {
        self.pending.load(Ordering::Acquire)
    }

    /// Whether the dedicated decoder thread has stopped. The worker stops only after a decoder
    /// panic; ordinary decode errors are returned to the calling request and do not stop the
    /// worker. A stopped scheduler fails every later call; the process must be restarted.
    pub fn is_stopped(&self) -> bool {
        let (lock, _wake) = &*self.shared;
        match lock.lock() {
            Ok(queue) => queue.stopped,
            // A poisoned queue mutex means a panic escaped while the lock was held, so the
            // worker is reported stopped instead of propagating a panic into a readiness probe.
            Err(_) => true,
        }
    }
}

impl Drop for ExecutionScheduler {
    fn drop(&mut self) {
        {
            let (lock, wake) = &*self.shared;
            queue_lock(lock).stopped = true;
            wake.notify_all();
        }
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .expect("execution scheduler catches every decoder panic before thread exit");
        }
    }
}

fn stop_after_panic(shared: &Arc<(Mutex<Queue>, Condvar)>, pending: &AtomicU64) {
    let (lock, wake) = &**shared;
    let mut queue = queue_lock(lock);
    queue.stopped = true;
    let mut drained: Vec<Job> = queue.ticks.drain(..).collect();
    drained.extend(queue.windows.drain(..));
    pending.fetch_sub(queue_count(drained.len()), Ordering::AcqRel);
    drop(queue);
    for job in drained {
        if job.control.cancellation_won() {
            job.control.acknowledge_cancellation();
            send_event(
                job.events,
                JobEvent::Finished(Err("execution cancelled".into())),
            );
        } else {
            job.control.fail();
            send_event(
                job.events,
                JobEvent::Finished(Err(
                    "execution scheduler stopped after decoder failure".into()
                )),
            );
        }
    }
    wake.notify_all();
}

/// Opaque synchronous decoder facade for one explicit scheduling priority and execution control.
pub struct ScheduledRecognizer {
    shared: Arc<(Mutex<Queue>, Condvar)>,
    frame_rate: FrameRate,
    priority: Priority,
    pending: Arc<AtomicU64>,
    control: ExecutionControl,
}

impl ScheduledRecognizer {
    fn new(parts: ScheduledRecognizerParts) -> Self {
        Self {
            shared: parts.shared,
            frame_rate: parts.frame_rate,
            priority: parts.priority,
            pending: parts.pending,
            control: parts.control,
        }
    }

    fn enqueue(&self, job: Job) -> Result<(), String> {
        let (lock, wake) = &*self.shared;
        let mut queue = queue_lock(lock);
        if queue.stopped {
            self.control.fail();
            return Err("execution scheduler stopped".into());
        }
        match self.priority {
            Priority::Tick => queue.ticks.push_back(job),
            Priority::Window => queue.windows.push_back(job),
        }
        self.pending.fetch_add(1, Ordering::AcqRel);
        wake.notify_one();
        Ok(())
    }

    fn request_queue_wait_cancellation(&self) {
        self.control.request_cancellation();
    }

    fn await_terminal_after_queue_wait(
        control: &ExecutionControl,
        events: Receiver<JobEvent>,
    ) -> Result<(), String> {
        loop {
            match events.recv() {
                Ok(JobEvent::Started) => {}
                Ok(JobEvent::Finished(_)) => {
                    control.acknowledge_cancellation();
                    return Ok(());
                }
                Err(_) => {
                    return Err(
                        "execution scheduler stopped before queued work became terminal".into(),
                    );
                }
            }
        }
    }
}

impl WindowDecoder for ScheduledRecognizer {
    fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
        self.control.enqueue()?;
        let (events_tx, events_rx): (Sender<JobEvent>, Receiver<JobEvent>) = mpsc::channel();
        self.enqueue(Job {
            // Queued work outlives the caller's feature borrow, so the scheduler is the one
            // execution boundary that materializes an owned validated matrix.
            features: features.to_owned(),
            control: self.control.clone(),
            events: events_tx,
        })?;

        match events_rx.recv_timeout(QUEUE_WAIT_BOUND) {
            Ok(JobEvent::Started) => events_rx
                .recv()
                .map_err(|_| "execution scheduler stopped before decoder completion".to_owned())?
                .into_result(),
            Ok(JobEvent::Finished(result)) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.request_queue_wait_cancellation();
                Self::await_terminal_after_queue_wait(&self.control, events_rx)?;
                Err("execution scheduler queue wait exceeded 60 seconds".into())
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err("execution scheduler stopped before decoder start".into())
            }
        }
    }
}

impl JobEvent {
    fn into_result(self) -> Result<Decoded, String> {
        match self {
            Self::Started => Err("execution scheduler reported duplicate decoder start".into()),
            Self::Finished(result) => result,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutionScheduler, JobEvent, ScheduledRecognizer};
    use crate::contracts::{Decoded, FrameRate, WindowDecoder};
    use crate::execution::{ExecutionControl, ExecutionState};
    use gigaam_audio::{FeatureMatrix, FeatureMatrixView};
    use std::sync::mpsc::{self, Receiver, SyncSender};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    const TEST_DEADLOCK_GUARD: Duration = Duration::from_secs(10);

    fn features(marker: usize) -> FeatureMatrix {
        FeatureMatrix::from_values(1, marker, vec![0.0; marker])
            .expect("test feature dimensions and values are valid")
    }

    fn decoded(frames: usize) -> Result<Decoded, String> {
        Decoded::new(vec![], vec![false; frames], frames, 0.0)
    }

    fn wait_until(description: &str, condition: impl Fn() -> bool) {
        let deadline = Instant::now() + TEST_DEADLOCK_GUARD;
        while !condition() {
            assert!(
                Instant::now() < deadline,
                "test deadlock guard elapsed while waiting for {description}"
            );
            thread::yield_now();
        }
    }

    fn wait_for_state(control: &ExecutionControl, expected: ExecutionState) {
        wait_until("the expected execution state", || {
            control.state() == expected
        });
    }

    fn wait_for_pending(scheduler: &ExecutionScheduler, expected: u64) {
        wait_until("the expected pending queue depth", || {
            scheduler.pending() == expected
        });
    }

    #[test]
    fn finished_before_cancellation_is_terminally_acknowledged_after_consumption() {
        let control = ExecutionControl::for_request();
        control
            .enqueue()
            .expect("a ready request can enqueue the controlled scheduler event");
        assert_eq!(control.state(), ExecutionState::Queued);
        control
            .start()
            .expect("queued controlled scheduler work can start");
        assert_eq!(control.state(), ExecutionState::Running);
        let (events_tx, events_rx) = mpsc::channel();
        events_tx
            .send(JobEvent::Finished(Err(
                "controlled finished event before cancellation".into(),
            )))
            .expect("test event receiver remains available");
        assert_eq!(
            control.request_cancellation(),
            ExecutionState::CancelRequested
        );
        ScheduledRecognizer::await_terminal_after_queue_wait(&control, events_rx)
            .expect("a preavailable finished event must terminally acknowledge cancellation");
        assert_eq!(control.state(), ExecutionState::Cancelled);
    }

    struct BlockingFake {
        log: Arc<Mutex<Vec<usize>>>,
        first_started: SyncSender<()>,
        release_first: Receiver<()>,
    }

    impl WindowDecoder for BlockingFake {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(25.0).expect("test frame rate is valid")
        }

        fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            let marker = features.frames();
            self.log
                .lock()
                .expect("test log mutex is never held across a panic")
                .push(marker);
            if marker == 1 {
                self.first_started
                    .send(())
                    .expect("test must observe the first decoder call");
                self.release_first
                    .recv()
                    .expect("test must release the first decoder call");
            }
            decoded(marker)
        }
    }

    #[test]
    fn tick_precedes_queued_window_after_a_blocked_call() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (first_started_tx, first_started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let scheduler = Arc::new(ExecutionScheduler::spawn(BlockingFake {
            log: Arc::clone(&log),
            first_started: first_started_tx,
            release_first: release_rx,
        }));
        let first_control = ExecutionControl::without_deadline();
        let first_scheduler = Arc::clone(&scheduler);
        let first = thread::spawn(move || {
            let features = features(1);
            first_scheduler
                .window_channel(first_control)
                .decode(features.view())
        });
        first_started_rx
            .recv()
            .expect("the first call must block inside the fake decoder");

        let window_control = ExecutionControl::without_deadline();
        let observed_window_control = window_control.clone();
        let window_scheduler = Arc::clone(&scheduler);
        let window = thread::spawn(move || {
            let features = features(3);
            window_scheduler
                .window_channel(window_control)
                .decode(features.view())
        });
        wait_for_state(&observed_window_control, ExecutionState::Queued);
        wait_for_pending(&scheduler, 1);

        let tick_control = ExecutionControl::without_deadline();
        let observed_tick_control = tick_control.clone();
        let tick_scheduler = Arc::clone(&scheduler);
        let tick = thread::spawn(move || {
            let features = features(2);
            tick_scheduler
                .tick_channel(tick_control)
                .decode(features.view())
        });
        wait_for_state(&observed_tick_control, ExecutionState::Queued);
        wait_for_pending(&scheduler, 2);
        release_tx
            .send(())
            .expect("test must release the blocked first call");
        first
            .join()
            .expect("first caller must not panic")
            .expect("first call must succeed");
        tick.join()
            .expect("tick caller must not panic")
            .expect("tick call must succeed");
        window
            .join()
            .expect("window caller must not panic")
            .expect("window call must succeed");
        assert_eq!(
            *log.lock()
                .expect("test log mutex is never held across a panic"),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn queued_cancellation_never_invokes_the_decoder() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (first_started_tx, first_started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let scheduler = Arc::new(ExecutionScheduler::spawn(BlockingFake {
            log: Arc::clone(&log),
            first_started: first_started_tx,
            release_first: release_rx,
        }));
        let running = ExecutionControl::without_deadline();
        let first_scheduler = Arc::clone(&scheduler);
        let first = thread::spawn(move || {
            let features = features(1);
            first_scheduler
                .window_channel(running)
                .decode(features.view())
        });
        first_started_rx
            .recv()
            .expect("the first call must block inside the fake decoder");

        let cancelled = ExecutionControl::for_request();
        let queued_scheduler = Arc::clone(&scheduler);
        let queued_control = cancelled.clone();
        let queued = thread::spawn(move || {
            let features = features(2);
            queued_scheduler
                .window_channel(queued_control)
                .decode(features.view())
        });
        wait_for_state(&cancelled, ExecutionState::Queued);
        assert_eq!(
            cancelled.request_cancellation(),
            ExecutionState::CancelRequested
        );
        release_tx
            .send(())
            .expect("test must release the blocked first call");
        first
            .join()
            .expect("first caller must not panic")
            .expect("first call must succeed");
        assert!(
            queued
                .join()
                .expect("queued caller must not panic")
                .is_err()
        );
        wait_for_state(&cancelled, ExecutionState::Cancelled);
        assert_eq!(
            *log.lock()
                .expect("test log mutex is never held across a panic"),
            vec![1]
        );
        wait_for_pending(&scheduler, 0);
    }

    #[test]
    fn cancellation_during_a_call_discards_late_success_and_blocks_successor() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (first_started_tx, first_started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let scheduler = Arc::new(ExecutionScheduler::spawn(BlockingFake {
            log: Arc::clone(&log),
            first_started: first_started_tx,
            release_first: release_rx,
        }));
        let control = ExecutionControl::for_request();
        let active_scheduler = Arc::clone(&scheduler);
        let active_control = control.clone();
        let active = thread::spawn(move || {
            let features = features(1);
            active_scheduler
                .window_channel(active_control)
                .decode(features.view())
        });
        first_started_rx
            .recv()
            .expect("the first call must block inside the fake decoder");
        wait_for_state(&control, ExecutionState::Running);
        assert_eq!(
            control.request_cancellation(),
            ExecutionState::CancelRequested
        );
        let mut successor = scheduler.window_channel(control.clone());
        let features = features(2);
        assert!(successor.decode(features.view()).is_err());
        assert_eq!(control.state(), ExecutionState::CancelRequested);
        release_tx
            .send(())
            .expect("test must release the first call");
        assert!(
            active
                .join()
                .expect("active caller must not panic")
                .is_err()
        );
        wait_for_state(&control, ExecutionState::Cancelled);
        assert_eq!(
            *log.lock()
                .expect("test log mutex is never held across a panic"),
            vec![1]
        );
    }

    struct PanicAfterRelease {
        first_started: SyncSender<()>,
        release_first: Receiver<()>,
    }

    impl WindowDecoder for PanicAfterRelease {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(25.0).expect("test frame rate is valid")
        }

        fn decode(&mut self, _features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            self.first_started
                .send(())
                .expect("test must observe the panicking decoder call");
            self.release_first
                .recv()
                .expect("test must release the panicking decoder call");
            panic!("controlled test decoder panic");
        }
    }

    #[test]
    fn panic_stops_drains_and_resets_pending() {
        let (first_started_tx, first_started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let scheduler = Arc::new(ExecutionScheduler::spawn(PanicAfterRelease {
            first_started: first_started_tx,
            release_first: release_rx,
        }));
        let active_control = ExecutionControl::without_deadline();
        let first_scheduler = Arc::clone(&scheduler);
        let first_control = active_control.clone();
        let first = thread::spawn(move || {
            let features = features(1);
            first_scheduler
                .window_channel(first_control)
                .decode(features.view())
        });
        first_started_rx
            .recv()
            .expect("the first decoder call must be blocked before panic");
        let second_control = ExecutionControl::without_deadline();
        let second_scheduler = Arc::clone(&scheduler);
        let drained_control = second_control.clone();
        let second = thread::spawn(move || {
            let features = features(2);
            second_scheduler
                .window_channel(drained_control)
                .decode(features.view())
        });
        wait_for_state(&second_control, ExecutionState::Queued);
        wait_for_pending(&scheduler, 1);
        release_tx
            .send(())
            .expect("test must release the panicking decoder call");
        assert!(first.join().expect("first caller must not panic").is_err());
        assert!(
            second
                .join()
                .expect("second caller must not panic")
                .is_err()
        );
        assert_eq!(active_control.state(), ExecutionState::Failed);
        assert_eq!(second_control.state(), ExecutionState::Failed);
        wait_for_pending(&scheduler, 0);

        let fresh_control = ExecutionControl::for_request();
        let mut fresh = scheduler.window_channel(fresh_control.clone());
        let features = features(3);
        assert!(fresh.decode(features.view()).is_err());
        assert_eq!(fresh_control.state(), ExecutionState::Failed);
    }

    #[test]
    fn is_stopped_reflects_the_panic_stop_transition() {
        let (first_started_tx, first_started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let scheduler = Arc::new(ExecutionScheduler::spawn(PanicAfterRelease {
            first_started: first_started_tx,
            release_first: release_rx,
        }));
        assert!(!scheduler.is_stopped());

        let control = ExecutionControl::without_deadline();
        let call_scheduler = Arc::clone(&scheduler);
        let call = thread::spawn(move || {
            let features = features(1);
            call_scheduler
                .window_channel(control)
                .decode(features.view())
        });
        first_started_rx
            .recv()
            .expect("the panicking decoder call must be blocked before panic");
        assert!(!scheduler.is_stopped());
        release_tx
            .send(())
            .expect("test must release the panicking decoder call");
        assert!(
            call.join()
                .expect("panicking caller must not panic")
                .is_err()
        );
        wait_until(
            "the scheduler to report stopped after the decoder panic",
            || scheduler.is_stopped(),
        );
    }

    #[test]
    fn ordinary_routing_preserves_the_decoder_frame_rate_and_result() {
        struct Fake;
        impl WindowDecoder for Fake {
            fn frame_rate(&self) -> FrameRate {
                FrameRate::new(50.0).expect("test frame rate is valid")
            }

            fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
                decoded(features.frames())
            }
        }
        let scheduler = ExecutionScheduler::spawn(Fake);
        let control = ExecutionControl::without_deadline();
        let features = features(7);
        let decoded = scheduler
            .window_channel(control)
            .decode(features.view())
            .expect("ordinary scheduled decoding must return its result");
        assert_eq!(decoded.output_frames(), 7);
        assert_eq!(scheduler.frame_rate().get(), 50.0);
    }
}
