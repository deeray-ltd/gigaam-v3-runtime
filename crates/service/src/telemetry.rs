// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Detached, bounded best-effort delivery for runtime access and trace telemetry.

use crate::admission::{AdmissionState, ServiceAdmission};
use crate::log::AccessLine;
use crate::metrics::Metrics;
use gigaam_transcription::{ObservationMode, WindowTiming, WindowTimingObserver};
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Destination {
    StdoutAccess,
    StderrTrace,
}

/// One actual producer transition boundary used by the normal offer path.
///
/// Production supplies a no-op observer. Deterministic tests pause the same transition to prove
/// that a terminal writer close conserves records already inside the active submission window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SubmissionCut {
    InitialClosedRead,
    ActiveEntered,
    ClosedRechecked,
    SlotReserved,
    UndeliveredTransferred,
    BeforeSend,
}

/// The only record forms accepted by the detached destination writer.
struct TelemetryRecord {
    destination: Destination,
    bytes: String,
    slot: QueueSlot,
    owns_undelivered: bool,
}

impl TelemetryRecord {
    fn new(destination: Destination, bytes: String, slot: QueueSlot) -> Self {
        Self {
            destination,
            bytes,
            slot,
            owns_undelivered: true,
        }
    }

    fn release_slot_before_write(&mut self) {
        self.slot.release();
    }

    fn delivered(&mut self) {
        self.release_undelivered();
        increment(&self.slot.state.delivered, "telemetry delivered counter");
    }

    fn disconnected(mut self) {
        self.slot.release();
        self.release_undelivered();
        self.slot.state.sink_closed(1);
    }

    fn terminal_claimed(&mut self) {
        self.owns_undelivered = false;
    }

    fn release_undelivered(&mut self) {
        if self.owns_undelivered {
            decrement(
                &self.slot.state.undelivered,
                "telemetry undelivered ownership",
            );
            self.owns_undelivered = false;
        }
    }
}

impl Drop for TelemetryRecord {
    fn drop(&mut self) {
        if self.owns_undelivered {
            if self.slot.state.closed.load(Ordering::SeqCst) {
                // The sole writer has transferred all live U ownership through U.swap(0).
                self.owns_undelivered = false;
            } else {
                self.release_undelivered();
            }
        }
    }
}

/// A non-cloneable logical capacity reservation held through receiver removal.
struct QueueSlot {
    state: Arc<TelemetryState>,
    held: bool,
}

impl QueueSlot {
    fn release(&mut self) {
        if self.held {
            decrement(&self.state.queued, "telemetry queued slot");
            self.held = false;
        }
    }
}

impl Drop for QueueSlot {
    fn drop(&mut self) {
        self.release();
    }
}

/// The private accounting and coordination state shared by all producer clones and the writer.
struct TelemetryState {
    capacity: usize,
    closed: AtomicBool,
    active_submissions: AtomicUsize,
    queued: AtomicUsize,
    undelivered: AtomicUsize,
    entered: AtomicUsize,
    remaining: AtomicUsize,
    delivered: AtomicUsize,
    queue_full: AtomicUsize,
    sink_closed: AtomicUsize,
    terminal_claimed: AtomicUsize,
    metrics: Arc<Metrics>,
}

impl TelemetryState {
    fn new(capacity: usize, metrics: Arc<Metrics>) -> Self {
        Self {
            capacity,
            closed: AtomicBool::new(false),
            active_submissions: AtomicUsize::new(0),
            queued: AtomicUsize::new(0),
            undelivered: AtomicUsize::new(0),
            entered: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            delivered: AtomicUsize::new(0),
            queue_full: AtomicUsize::new(0),
            sink_closed: AtomicUsize::new(0),
            terminal_claimed: AtomicUsize::new(0),
            metrics,
        }
    }

    fn sink_closed(&self, count: usize) {
        if count != 0 {
            add(&self.sink_closed, count, "telemetry sink-closed counter");
            self.metrics.telemetry_sink_closed(count);
        }
    }

    fn queue_full(&self) {
        increment(&self.queue_full, "telemetry queue-full counter");
        self.metrics.telemetry_queue_full();
    }

    fn reserve_slot(self: &Arc<Self>) -> Option<QueueSlot> {
        let mut current = self.queued.load(Ordering::SeqCst);
        loop {
            if current >= self.capacity {
                return None;
            }
            let next = match current.checked_add(1) {
                Some(next) => next,
                None => panic!("telemetry queued capacity reservation must not overflow"),
            };
            match self
                .queued
                .compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => {
                    return Some(QueueSlot {
                        state: Arc::clone(self),
                        held: true,
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

/// Ensures every initially-open submission contributes exactly one active-window exit.
struct ActiveSubmission {
    state: Arc<TelemetryState>,
}

impl ActiveSubmission {
    fn enter(state: Arc<TelemetryState>) -> Self {
        increment(
            &state.active_submissions,
            "telemetry active submission counter",
        );
        Self { state }
    }
}

impl Drop for ActiveSubmission {
    fn drop(&mut self) {
        decrement(
            &self.state.active_submissions,
            "telemetry active submission counter",
        );
    }
}

fn increment(counter: &AtomicUsize, name: &str) {
    match counter.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
        current.checked_add(1)
    }) {
        Ok(_) => {}
        Err(_) => panic!("{name} must not overflow"),
    }
}

fn add(counter: &AtomicUsize, value: usize, name: &str) {
    match counter.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
        current.checked_add(value)
    }) {
        Ok(_) => {}
        Err(_) => panic!("{name} must not overflow"),
    }
}

fn decrement(counter: &AtomicUsize, name: &str) {
    match counter.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
        current.checked_sub(1)
    }) {
        Ok(_) => {}
        Err(_) => panic!("{name} must have prior ownership"),
    }
}

/// The only cloneable application-side sender capability.
#[derive(Clone)]
pub(crate) struct TelemetryProducer {
    sender: Sender<TelemetryRecord>,
    state: Arc<TelemetryState>,
}

impl TelemetryProducer {
    pub(crate) fn offer_access(&self, line: AccessLine) {
        self.offer(Destination::StdoutAccess, line.into_inner());
    }

    fn offer_trace(&self, line: String) {
        self.offer(Destination::StderrTrace, line);
    }

    fn offer(&self, destination: Destination, bytes: String) {
        self.offer_with(destination, bytes, |_| {});
    }

    fn offer_with(
        &self,
        destination: Destination,
        bytes: String,
        mut at_cut: impl FnMut(SubmissionCut),
    ) {
        assert!(
            bytes.ends_with('\n'),
            "telemetry records must be serialized as newline-terminated lines"
        );
        increment(&self.state.entered, "telemetry entered-offer counter");
        increment(&self.state.remaining, "telemetry remaining-offer counter");
        let initially_closed = self.state.closed.load(Ordering::SeqCst);
        at_cut(SubmissionCut::InitialClosedRead);
        if initially_closed {
            self.remaining_to_sink_closed();
            return;
        }

        let _active = ActiveSubmission::enter(Arc::clone(&self.state));
        at_cut(SubmissionCut::ActiveEntered);
        let closed_after_entry = self.state.closed.load(Ordering::SeqCst);
        at_cut(SubmissionCut::ClosedRechecked);
        if closed_after_entry {
            self.remaining_to_sink_closed();
            return;
        }

        let slot = match self.state.reserve_slot() {
            Some(slot) => slot,
            None => {
                self.remaining_to_queue_full();
                return;
            }
        };
        at_cut(SubmissionCut::SlotReserved);

        increment(&self.state.undelivered, "telemetry undelivered ownership");
        decrement(&self.state.remaining, "telemetry remaining offer transfer");
        at_cut(SubmissionCut::UndeliveredTransferred);
        let record = TelemetryRecord::new(destination, bytes, slot);
        at_cut(SubmissionCut::BeforeSend);
        if let Err(error) = self.sender.send(record) {
            error.0.disconnected();
        }
    }

    fn remaining_to_queue_full(&self) {
        decrement(&self.state.remaining, "telemetry remaining offer transfer");
        self.state.queue_full();
    }

    fn remaining_to_sink_closed(&self) {
        decrement(&self.state.remaining, "telemetry remaining offer transfer");
        self.state.sink_closed(1);
    }
}

/// A prepared, linear owner which allocates no capacity-sized transport storage and starts no I/O.
pub(crate) struct PreparedTelemetry {
    max_http: usize,
    max_ws: usize,
    capacity: usize,
    sender: Sender<TelemetryRecord>,
    receiver: Receiver<TelemetryRecord>,
    state: Arc<TelemetryState>,
}

impl PreparedTelemetry {
    pub(crate) fn prepare(
        admission: &ServiceAdmission,
        metrics: Arc<Metrics>,
    ) -> Result<Self, String> {
        let max_http = admission.max_http();
        let max_ws = admission.max_ws();
        let capacity = capacity_for(max_http, max_ws)?;
        let (sender, receiver) = mpsc::channel();
        Ok(Self {
            max_http,
            max_ws,
            capacity,
            sender,
            receiver,
            state: Arc::new(TelemetryState::new(capacity, metrics)),
        })
    }

    pub(crate) fn observations(&self, enabled: bool) -> ObservationMode {
        match enabled {
            true => ObservationMode::enabled(Arc::new(ServiceWindowObserver {
                producer: self.producer(),
            })),
            false => ObservationMode::disabled(),
        }
    }

    pub(crate) fn matches_admission(
        &self,
        admission: &ServiceAdmission,
        state: &AdmissionState,
    ) -> bool {
        match capacity_for(admission.max_http(), admission.max_ws()) {
            Ok(capacity) => {
                self.max_http == admission.max_http()
                    && self.max_ws == admission.max_ws()
                    && self.capacity == capacity
                    && state.max_http() == admission.max_http()
                    && state.max_ws() == admission.max_ws()
            }
            Err(_) => false,
        }
    }

    pub(crate) fn start(self) -> Result<TelemetryProducer, String> {
        self.start_with(launch_writer)
    }

    #[cfg(test)]
    pub(crate) fn refuse_start_for_test(self) -> Result<TelemetryProducer, String> {
        self.start_with(|_| Err("controlled telemetry launcher refusal".to_owned()))
    }

    fn producer(&self) -> TelemetryProducer {
        TelemetryProducer {
            sender: self.sender.clone(),
            state: Arc::clone(&self.state),
        }
    }

    fn start_with(
        self,
        launch: impl FnOnce(TelemetryWriter) -> Result<(), String>,
    ) -> Result<TelemetryProducer, String> {
        let Self {
            sender,
            receiver,
            state,
            ..
        } = self;
        let producer = TelemetryProducer {
            sender: sender.clone(),
            state: Arc::clone(&state),
        };
        drop(sender);
        launch(TelemetryWriter { receiver, state })?;
        Ok(producer)
    }
}

fn capacity_for(max_http: usize, max_ws: usize) -> Result<usize, String> {
    let ws_weight = max_ws
        .checked_mul(3)
        .ok_or_else(|| "telemetry capacity exceeds usize".to_owned())?;
    let capacity = max_http
        .checked_add(ws_weight)
        .ok_or_else(|| "telemetry capacity exceeds usize".to_owned())?;
    let upper = tokio::sync::Semaphore::MAX_PERMITS
        .checked_mul(4)
        .ok_or_else(|| "telemetry capacity exceeds usize".to_owned())?;
    if capacity < 4 || capacity > upper || upper == usize::MAX {
        return Err("telemetry capacity is outside the supported admission range".into());
    }
    Ok(capacity)
}

/// The only receiver owner after start.
struct TelemetryWriter {
    receiver: Receiver<TelemetryRecord>,
    state: Arc<TelemetryState>,
}

trait DestinationSink {
    fn write(&mut self, destination: Destination, bytes: &str) -> io::Result<()>;
}

struct StandardDestination;

impl DestinationSink for StandardDestination {
    fn write(&mut self, destination: Destination, bytes: &str) -> io::Result<()> {
        match destination {
            Destination::StdoutAccess => {
                let stdout = io::stdout();
                stdout.lock().write_all(bytes.as_bytes())
            }
            Destination::StderrTrace => {
                let stderr = io::stderr();
                stderr.lock().write_all(bytes.as_bytes())
            }
        }
    }
}

impl TelemetryWriter {
    fn run(self, sink: impl DestinationSink, after_claim: impl FnOnce(usize)) {
        let TelemetryWriter { receiver, state } = self;
        let mut sink = sink;
        let mut after_claim = Some(after_claim);
        while let Ok(mut record) = receiver.recv() {
            record.release_slot_before_write();
            if sink.write(record.destination, &record.bytes).is_ok() {
                record.delivered();
                continue;
            }

            match record.destination {
                Destination::StdoutAccess => state.metrics.telemetry_stdout_failure(),
                Destination::StderrTrace => state.metrics.telemetry_stderr_failure(),
            }
            state.closed.store(true, Ordering::SeqCst);
            while state.active_submissions.load(Ordering::SeqCst) != 0 {
                std::thread::yield_now();
            }
            let claimed = state.undelivered.swap(0, Ordering::SeqCst);
            state.terminal_claimed.store(claimed, Ordering::SeqCst);
            match after_claim.take() {
                Some(hook) => hook(claimed),
                None => panic!("telemetry writer terminal hook must run at most once"),
            }
            state.sink_closed(claimed);
            state.terminal_claimed.store(0, Ordering::SeqCst);
            record.terminal_claimed();
            drop(record);
            drop(receiver);
            return;
        }
    }
}

fn launch_writer(writer: TelemetryWriter) -> Result<(), String> {
    std::thread::Builder::new()
        .name("gigaam-service-telemetry".to_owned())
        .spawn(move || writer.run(StandardDestination, |_| {}))
        .map(|_| ())
        .map_err(|error| format!("start telemetry writer: {error}"))
}

/// Process-only trace observation routed through the same producer lineage as access records.
struct ServiceWindowObserver {
    producer: TelemetryProducer,
}

impl WindowTimingObserver for ServiceWindowObserver {
    fn observe(&self, observation: WindowTiming) {
        self.producer.offer_trace(format!(
            "#   window {:7.2}s frames {:5} encoder {:6.1} ms\n",
            observation.offset_sec(),
            observation.frames(),
            observation.encoder_seconds() * 1000.0
        ));
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{
        Destination, DestinationSink, PreparedTelemetry, TelemetryProducer, TelemetryState,
        TelemetryWriter,
    };
    use crate::admission::ServiceAdmission;
    use crate::metrics::Metrics;
    use std::io;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::sync::mpsc;
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    /// Private state facts used only to verify the telemetry conservation equations.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) struct Snapshot {
        pub(crate) entered: usize,
        pub(crate) remaining: usize,
        pub(crate) undelivered: usize,
        pub(crate) queued: usize,
        pub(crate) delivered: usize,
        pub(crate) queue_full: usize,
        pub(crate) sink_closed: usize,
        pub(crate) terminal_claimed: usize,
        pub(crate) active: usize,
    }

    /// Test-only owner of the real prepared telemetry state and its observable accounting facts.
    pub(crate) struct PreparedHarness {
        prepared: PreparedTelemetry,
        state: Arc<TelemetryState>,
        metrics: Arc<Metrics>,
    }

    /// Test-only started telemetry backed by the production writer transition and a failing stdout.
    pub(crate) struct FailingStdoutHarness {
        state: Arc<TelemetryState>,
        metrics: Arc<Metrics>,
        producer: TelemetryProducer,
        terminal: mpsc::Receiver<()>,
        writer: JoinHandle<()>,
    }

    /// Test-only owner after the real prepared receiver has been dropped.
    pub(crate) struct ReceiverDroppedHarness {
        state: Arc<TelemetryState>,
        producer: TelemetryProducer,
    }

    struct FailingStdout;

    impl DestinationSink for FailingStdout {
        fn write(&mut self, destination: Destination, _bytes: &str) -> io::Result<()> {
            match destination {
                Destination::StdoutAccess => Err(io::Error::other("controlled stdout failure")),
                Destination::StderrTrace => Ok(()),
            }
        }
    }

    /// Prepares the same capacity/channel ownership used by production without starting a writer.
    pub(crate) fn prepare(max_http: usize, max_ws: usize) -> Result<PreparedHarness, String> {
        let admission = ServiceAdmission::new(max_http, max_ws, Duration::from_secs(1))?;
        let metrics = Arc::new(Metrics::new(max_http, max_ws));
        let prepared = PreparedTelemetry::prepare(&admission, Arc::clone(&metrics))?;
        Ok(PreparedHarness {
            state: Arc::clone(&prepared.state),
            metrics,
            prepared,
        })
    }

    impl PreparedHarness {
        /// Returns one real producer while its receiver remains paused in prepared ownership.
        pub(crate) fn producer(&self) -> TelemetryProducer {
            self.prepared.producer()
        }

        pub(crate) fn snapshot(&self) -> Snapshot {
            snapshot(self.state.as_ref())
        }

        pub(crate) fn metrics_arc(&self) -> Arc<Metrics> {
            Arc::clone(&self.metrics)
        }

        /// Drops the real receiver before an offer, exercising standard-channel disconnection.
        pub(crate) fn into_receiver_dropped(self) -> ReceiverDroppedHarness {
            let producer = self.prepared.producer();
            drop(self.prepared);
            ReceiverDroppedHarness {
                state: self.state,
                producer,
            }
        }

        /// Starts the real writer transition with a deterministic stdout destination failure.
        pub(crate) fn into_failing_stdout(self) -> Result<FailingStdoutHarness, String> {
            let (handle_tx, handle_rx) = mpsc::channel();
            let (terminal_tx, terminal_rx) = mpsc::channel();
            let producer = self.prepared.start_with(move |writer: TelemetryWriter| {
                let handle = thread::Builder::new()
                    .name("telemetry-test-failing-stdout".to_owned())
                    .spawn(move || {
                        writer.run(FailingStdout, |_| {});
                        terminal_tx.send(()).expect(
                            "the test harness must retain its terminal completion receiver",
                        );
                    })
                    .map_err(|error| format!("spawn controlled telemetry writer: {error}"))?;
                handle_tx
                    .send(handle)
                    .map_err(|_| "return controlled telemetry writer handle".to_owned())
            })?;
            let writer = handle_rx
                .recv()
                .map_err(|_| "receive controlled telemetry writer handle".to_owned())?;
            Ok(FailingStdoutHarness {
                state: self.state,
                metrics: self.metrics,
                producer,
                terminal: terminal_rx,
                writer,
            })
        }
    }

    impl ReceiverDroppedHarness {
        pub(crate) fn producer(&self) -> TelemetryProducer {
            self.producer.clone()
        }

        pub(crate) fn snapshot(&self) -> Snapshot {
            snapshot(self.state.as_ref())
        }
    }

    impl FailingStdoutHarness {
        pub(crate) fn producer(&self) -> TelemetryProducer {
            self.producer.clone()
        }

        pub(crate) fn snapshot(&self) -> Snapshot {
            snapshot(self.state.as_ref())
        }

        pub(crate) fn metrics(&self) -> &Metrics {
            self.metrics.as_ref()
        }

        /// Waits for the real writer to complete terminal ownership transfer after its failure.
        pub(crate) fn wait_until_terminal(&self) {
            self.terminal
                .recv_timeout(Duration::from_secs(1))
                .expect("controlled telemetry writer must complete after its stdout failure");
        }

        /// Drops the final production producer and joins only the test-owned writer thread.
        pub(crate) fn finish(self) {
            drop(self.producer);
            if self.writer.join().is_err() {
                panic!("controlled telemetry writer must not panic");
            }
        }
    }

    fn snapshot(state: &TelemetryState) -> Snapshot {
        Snapshot {
            entered: state.entered.load(Ordering::SeqCst),
            remaining: state.remaining.load(Ordering::SeqCst),
            undelivered: state.undelivered.load(Ordering::SeqCst),
            queued: state.queued.load(Ordering::SeqCst),
            delivered: state.delivered.load(Ordering::SeqCst),
            queue_full: state.queue_full.load(Ordering::SeqCst),
            sink_closed: state.sink_closed.load(Ordering::SeqCst),
            terminal_claimed: state.terminal_claimed.load(Ordering::SeqCst),
            active: state.active_submissions.load(Ordering::SeqCst),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Destination, DestinationSink, PreparedTelemetry, SubmissionCut, TelemetryProducer,
        TelemetryState, TelemetryWriter,
    };
    use crate::admission::{AdmissionState, ServiceAdmission};
    use crate::log::{AccessEvent, WsErrorAccess, render};
    use crate::metrics::Metrics;
    use gigaam_audio::ChannelAudio;
    use gigaam_recognition::{Decoded, ExecutionControl, FrameRate, WindowDecoder};
    use gigaam_transcription::{BatchConfig, BatchSetup, BatchTranscriber, PadPolicy};
    use std::io;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Snapshot {
        entered: usize,
        remaining: usize,
        undelivered: usize,
        queued: usize,
        delivered: usize,
        queue_full: usize,
        sink_closed: usize,
        terminal_claimed: usize,
        active: usize,
    }

    fn snapshot(state: &TelemetryState) -> Snapshot {
        Snapshot {
            entered: state.entered.load(Ordering::SeqCst),
            remaining: state.remaining.load(Ordering::SeqCst),
            undelivered: state.undelivered.load(Ordering::SeqCst),
            queued: state.queued.load(Ordering::SeqCst),
            delivered: state.delivered.load(Ordering::SeqCst),
            queue_full: state.queue_full.load(Ordering::SeqCst),
            sink_closed: state.sink_closed.load(Ordering::SeqCst),
            terminal_claimed: state.terminal_claimed.load(Ordering::SeqCst),
            active: state.active_submissions.load(Ordering::SeqCst),
        }
    }

    fn assert_live_conservation(snapshot: Snapshot) {
        let accounted = snapshot
            .remaining
            .checked_add(snapshot.undelivered)
            .and_then(|value| value.checked_add(snapshot.terminal_claimed))
            .and_then(|value| value.checked_add(snapshot.delivered))
            .and_then(|value| value.checked_add(snapshot.queue_full))
            .and_then(|value| value.checked_add(snapshot.sink_closed));
        assert_eq!(
            accounted,
            Some(snapshot.entered),
            "the private telemetry ledger must conserve every entered record"
        );
    }

    fn prepared(max_http: usize, max_ws: usize) -> (PreparedTelemetry, Arc<Metrics>) {
        let admission = ServiceAdmission::new(max_http, max_ws, Duration::from_secs(1))
            .expect("controlled telemetry admission is valid");
        let metrics = Arc::new(Metrics::new(max_http, max_ws));
        let prepared = PreparedTelemetry::prepare(&admission, Arc::clone(&metrics))
            .expect("validated telemetry admission prepares");
        (prepared, metrics)
    }

    fn access(message: &str) -> crate::log::AccessLine {
        render(AccessEvent::WsError(WsErrorAccess {
            phase: "test",
            message: message.to_owned(),
        }))
    }

    fn join(handle: JoinHandle<()>) {
        if handle.join().is_err() {
            panic!("controlled telemetry writer must not panic");
        }
    }

    struct RecordingSink {
        records: Arc<std::sync::Mutex<Vec<(Destination, String)>>>,
        fail_destination: Option<Destination>,
        began: Option<Sender<()>>,
        release: Option<Receiver<()>>,
    }

    impl RecordingSink {
        fn collecting(records: Arc<std::sync::Mutex<Vec<(Destination, String)>>>) -> Self {
            Self {
                records,
                fail_destination: None,
                began: None,
                release: None,
            }
        }

        fn failing(destination: Destination) -> Self {
            Self {
                records: Arc::new(std::sync::Mutex::new(Vec::new())),
                fail_destination: Some(destination),
                began: None,
                release: None,
            }
        }

        fn blocked(
            records: Arc<std::sync::Mutex<Vec<(Destination, String)>>>,
            began: Sender<()>,
            release: Receiver<()>,
        ) -> Self {
            Self {
                records,
                fail_destination: None,
                began: Some(began),
                release: Some(release),
            }
        }

        fn blocked_failing(
            destination: Destination,
            began: Sender<()>,
            release: Receiver<()>,
        ) -> Self {
            Self {
                records: Arc::new(std::sync::Mutex::new(Vec::new())),
                fail_destination: Some(destination),
                began: Some(began),
                release: Some(release),
            }
        }
    }

    impl DestinationSink for RecordingSink {
        fn write(&mut self, destination: Destination, bytes: &str) -> io::Result<()> {
            if let Some(began) = self.began.take() {
                began
                    .send(())
                    .map_err(|_| io::Error::other("test receiver left"))?;
            }
            if let Some(release) = self.release.take() {
                release
                    .recv()
                    .map_err(|_| io::Error::other("test writer release missing"))?;
            }
            if self.fail_destination == Some(destination) {
                return Err(io::Error::other("controlled destination failure"));
            }
            let mut records = match self.records.lock() {
                Ok(records) => records,
                Err(_) => return Err(io::Error::other("test record sink is poisoned")),
            };
            records.push((destination, bytes.to_owned()));
            Ok(())
        }
    }

    fn start_with_sink(
        prepared: PreparedTelemetry,
        sink: RecordingSink,
    ) -> Result<(TelemetryProducer, JoinHandle<()>), String> {
        start_with_sink_after_claim(prepared, sink, |_| {})
    }

    fn start_with_sink_after_claim(
        prepared: PreparedTelemetry,
        sink: RecordingSink,
        after_claim: impl FnOnce(usize) + Send + 'static,
    ) -> Result<(TelemetryProducer, JoinHandle<()>), String> {
        let (handle_tx, handle_rx) = mpsc::channel();
        let producer = prepared.start_with(move |writer| {
            let handle = thread::Builder::new()
                .name("telemetry-test-writer".to_owned())
                .spawn(move || writer.run(sink, after_claim))
                .map_err(|error| format!("spawn controlled telemetry writer: {error}"))?;
            handle_tx
                .send(handle)
                .map_err(|_| "return controlled telemetry writer handle".to_owned())
        })?;
        let handle = handle_rx
            .recv()
            .map_err(|_| "receive controlled telemetry writer handle".to_owned())?;
        Ok((producer, handle))
    }

    fn wait_until_closed(state: &TelemetryState) {
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !state.closed.load(Ordering::SeqCst) {
            if std::time::Instant::now() >= deadline {
                panic!("controlled terminal writer must publish its closed state");
            }
            thread::yield_now();
        }
    }

    struct ObservationDecoder;

    impl WindowDecoder for ObservationDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(25.0).expect("the controlled observation frame rate is positive")
        }

        fn decode(
            &mut self,
            features: gigaam_audio::FeatureMatrixView<'_>,
        ) -> Result<Decoded, String> {
            Decoded::new(
                Vec::new(),
                vec![false; features.frames()],
                features.frames(),
                0.0,
            )
        }
    }

    #[test]
    fn access_and_trace_records_have_exact_destinations_newlines_and_fifo_membership() {
        let (prepared, _metrics) = prepared(1, 1);
        let records = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (producer, handle) =
            start_with_sink(prepared, RecordingSink::collecting(Arc::clone(&records)))
                .expect("controlled telemetry writer starts");

        producer.offer_access(access("first"));
        producer.offer_trace("#   window    1.00s frames     3 encoder    4.0 ms\n".to_owned());
        producer.offer_access(access("last"));
        drop(producer);
        join(handle);

        let records = match records.lock() {
            Ok(records) => records,
            Err(_) => panic!("controlled record list must not be poisoned"),
        };
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].0, Destination::StdoutAccess);
        assert_eq!(records[1].0, Destination::StderrTrace);
        assert_eq!(records[2].0, Destination::StdoutAccess);
        assert!(records.iter().all(|(_, bytes)| bytes.ends_with('\n')));
        assert!(records[0].1.contains("\"message\":\"first\""));
        assert_eq!(
            records[1].1,
            "#   window    1.00s frames     3 encoder    4.0 ms\n"
        );
        assert!(records[2].1.contains("\"message\":\"last\""));
    }

    #[test]
    fn process_observation_mode_routes_real_successful_batch_windows_only_when_enabled()
    -> Result<(), String> {
        let fixture = crate::tests::assembly_fixture()?;
        let config = BatchConfig::new(fixture.frontend.sample_rate(), 1.0, 0.0, PadPolicy::Exact)?;
        let input = ChannelAudio::new(vec![0.0; 640])?;

        let (enabled_prepared, _enabled_metrics) = prepared(1, 1);
        let enabled_observations = enabled_prepared.observations(true);
        let enabled_records = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (enabled_producer, enabled_writer) = start_with_sink(
            enabled_prepared,
            RecordingSink::collecting(Arc::clone(&enabled_records)),
        )?;
        let mut enabled = BatchTranscriber::new(BatchSetup {
            frontend: Arc::clone(&fixture.frontend),
            decoder: ObservationDecoder,
            config,
            control: ExecutionControl::without_deadline(),
            observations: enabled_observations,
        })?;
        let enabled_transcript = enabled.transcribe_channel(&input)?;
        drop(enabled);
        drop(enabled_producer);
        join(enabled_writer);

        let enabled_records = match enabled_records.lock() {
            Ok(records) => records,
            Err(_) => panic!("enabled process trace records must not be poisoned"),
        };
        assert_eq!(enabled_records.len(), 1);
        assert_eq!(enabled_records[0].0, Destination::StderrTrace);
        assert_eq!(
            enabled_records[0].1,
            "#   window    0.00s frames     3 encoder    0.0 ms\n"
        );

        let (disabled_prepared, _disabled_metrics) = prepared(1, 1);
        let disabled_observations = disabled_prepared.observations(false);
        let disabled_records = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (disabled_producer, disabled_writer) = start_with_sink(
            disabled_prepared,
            RecordingSink::collecting(Arc::clone(&disabled_records)),
        )?;
        let mut disabled = BatchTranscriber::new(BatchSetup {
            frontend: Arc::clone(&fixture.frontend),
            decoder: ObservationDecoder,
            config,
            control: ExecutionControl::without_deadline(),
            observations: disabled_observations,
        })?;
        let disabled_transcript = disabled.transcribe_channel(&input)?;
        drop(disabled);
        drop(disabled_producer);
        join(disabled_writer);

        assert_eq!(disabled_transcript, enabled_transcript);
        let disabled_records = match disabled_records.lock() {
            Ok(records) => records,
            Err(_) => panic!("disabled process trace records must not be poisoned"),
        };
        assert!(
            disabled_records.is_empty(),
            "disabled process observations must not create a destination record"
        );
        Ok(())
    }

    #[test]
    fn logical_capacity_accepts_c_nonwaiting_offers_and_drops_the_next_as_queue_full() {
        let (prepared, metrics) = prepared(1, 1);
        let state = Arc::clone(&prepared.state);
        let producer = prepared.producer();

        for number in 0..4 {
            producer.offer_trace(format!("trace-{number}\n"));
        }
        producer.offer_trace("full\n".to_owned());

        let current = snapshot(&state);
        assert_eq!(current.entered, 5);
        assert_eq!(current.remaining, 0);
        assert_eq!(current.undelivered, 4);
        assert_eq!(current.queued, 4);
        assert_eq!(current.queue_full, 1);
        assert_eq!(current.sink_closed, 0);
        assert_live_conservation(current);
        assert!(
            metrics
                .render(1, 1, 0, None)
                .contains("asr_telemetry_dropped_total{reason=\"queue_full\"} 1\n")
        );

        drop(producer);
        drop(prepared);
        let final_state = snapshot(&state);
        assert_eq!(final_state.queued, 0);
        assert_eq!(final_state.undelivered, 0);
    }

    #[test]
    fn receiver_removal_releases_q_before_a_blocked_destination_write() {
        let (prepared, _metrics) = prepared(1, 1);
        let state = Arc::clone(&prepared.state);
        let records = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (began_tx, began_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (producer, handle) = start_with_sink(
            prepared,
            RecordingSink::blocked(Arc::clone(&records), began_tx, release_rx),
        )
        .expect("controlled telemetry writer starts");

        producer.offer_trace("first\n".to_owned());
        began_rx
            .recv()
            .expect("writer must remove the first record before its destination write");
        assert_eq!(snapshot(&state).queued, 0);
        for number in 0..4 {
            producer.offer_trace(format!("later-{number}\n"));
        }
        producer.offer_trace("overflow\n".to_owned());
        let paused = snapshot(&state);
        assert_eq!(paused.queued, 4);
        assert_eq!(paused.undelivered, 5);
        assert_eq!(paused.queue_full, 1);
        assert_live_conservation(paused);

        release_tx
            .send(())
            .expect("controlled writer release receiver remains available");
        drop(producer);
        join(handle);
        let final_state = snapshot(&state);
        assert_eq!(final_state.queued, 0);
        assert_eq!(final_state.undelivered, 0);
        assert_eq!(final_state.delivered, 5);
        assert_live_conservation(final_state);
    }

    #[test]
    fn receiver_disconnection_returns_the_record_without_retry_or_destination_write() {
        let (prepared, metrics) = prepared(1, 1);
        let state = Arc::clone(&prepared.state);
        let producer = prepared.producer();
        drop(prepared);

        producer.offer_access(access("disconnected"));
        let current = snapshot(&state);
        assert_eq!(current.entered, 1);
        assert_eq!(current.remaining, 0);
        assert_eq!(current.undelivered, 0);
        assert_eq!(current.queued, 0);
        assert_eq!(current.sink_closed, 1);
        assert_eq!(current.delivered, 0);
        assert_live_conservation(current);
        let rendered = metrics.render(1, 1, 0, None);
        assert!(rendered.contains("asr_telemetry_dropped_total{reason=\"sink_closed\"} 1\n"));
    }

    #[test]
    fn first_destination_failure_claims_u_once_then_counts_current_queued_and_later_loss() {
        let (prepared, metrics) = prepared(1, 1);
        let state = Arc::clone(&prepared.state);
        let queued_producer = prepared.producer();
        queued_producer.offer_access(access("failed-current"));
        queued_producer.offer_access(access("queued"));
        let (claim_tx, claim_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let (handle_tx, handle_rx) = mpsc::channel();
        let started = prepared
            .start_with(move |writer: TelemetryWriter| {
                let handle = thread::Builder::new()
                    .name("telemetry-terminal-test".to_owned())
                    .spawn(move || {
                        writer.run(
                            RecordingSink::failing(Destination::StdoutAccess),
                            move |count| {
                                claim_tx
                                    .send(count)
                                    .expect("terminal claim observer remains available");
                                resume_rx
                                    .recv()
                                    .expect("terminal claim test must resume the writer");
                            },
                        )
                    })
                    .map_err(|error| format!("spawn terminal test writer: {error}"))?;
                handle_tx
                    .send(handle)
                    .map_err(|_| "return terminal test writer handle".to_owned())
            })
            .expect("controlled terminal writer starts");
        let handle = handle_rx
            .recv()
            .expect("controlled terminal writer handle is available");
        let claimed = claim_rx
            .recv()
            .expect("terminal writer must expose its unique claim");
        assert_eq!(claimed, 2);
        let at_claim = snapshot(&state);
        assert_eq!(at_claim.undelivered, 0);
        assert_eq!(at_claim.terminal_claimed, 2);
        assert_eq!(at_claim.sink_closed, 0);
        assert_live_conservation(at_claim);

        queued_producer.offer_access(access("later-closed"));
        let after_later_offer = snapshot(&state);
        assert_eq!(after_later_offer.sink_closed, 1);
        assert_live_conservation(after_later_offer);

        resume_tx
            .send(())
            .expect("terminal writer waits for deterministic test release");
        drop(started);
        drop(queued_producer);
        join(handle);
        let final_state = snapshot(&state);
        assert_eq!(final_state.terminal_claimed, 0);
        assert_eq!(final_state.queued, 0);
        assert_eq!(final_state.undelivered, 0);
        assert_eq!(final_state.sink_closed, 3);
        assert_live_conservation(final_state);
        let rendered = metrics.render(1, 1, 0, None);
        assert!(
            rendered.contains("asr_telemetry_write_failures_total{destination=\"stdout\"} 1\n")
        );
        assert!(rendered.contains("asr_telemetry_dropped_total{reason=\"sink_closed\"} 3\n"));
    }

    #[test]
    fn terminal_writer_conserves_every_actual_active_submission_cut() {
        for cut in [
            SubmissionCut::ActiveEntered,
            SubmissionCut::ClosedRechecked,
            SubmissionCut::SlotReserved,
            SubmissionCut::UndeliveredTransferred,
            SubmissionCut::BeforeSend,
        ] {
            let (
                expected_claimed,
                expected_sink_closed_at_claim,
                expected_sink_closed_after_later,
                expected_queued_after_later,
            ) = match cut {
                SubmissionCut::ActiveEntered => (1, 1, 2, 0),
                SubmissionCut::ClosedRechecked
                | SubmissionCut::SlotReserved
                | SubmissionCut::UndeliveredTransferred
                | SubmissionCut::BeforeSend => (2, 0, 1, 1),
                SubmissionCut::InitialClosedRead => {
                    panic!("the active-cut witness must not select the initial closed read")
                }
            };
            let (prepared, metrics) = prepared(1, 1);
            let state = Arc::clone(&prepared.state);
            let (write_began_tx, write_began_rx) = mpsc::channel();
            let (fail_tx, fail_rx) = mpsc::channel();
            let (claim_tx, claim_rx) = mpsc::channel();
            let (claim_resume_tx, claim_resume_rx) = mpsc::channel();
            let (producer, writer) = start_with_sink_after_claim(
                prepared,
                RecordingSink::blocked_failing(Destination::StdoutAccess, write_began_tx, fail_rx),
                move |claimed| {
                    claim_tx
                        .send(claimed)
                        .expect("terminal claim observer remains available");
                    claim_resume_rx
                        .recv()
                        .expect("terminal claim test must resume the writer");
                },
            )
            .expect("controlled telemetry writer starts");
            producer.offer_access(access("trigger"));
            write_began_rx
                .recv()
                .expect("the trigger record must reach its destination attempt");

            let active_producer = producer.clone();
            let (cut_reached_tx, cut_reached_rx) = mpsc::channel();
            let (active_resume_tx, active_resume_rx) = mpsc::channel();
            let active = thread::spawn(move || {
                active_producer.offer_with(
                    Destination::StdoutAccess,
                    access("active").into_inner(),
                    |actual_cut| {
                        if actual_cut == cut {
                            cut_reached_tx
                                .send(())
                                .expect("the active offer test must observe its selected cut");
                            active_resume_rx
                                .recv()
                                .expect("the active offer test must resume after writer closure");
                        }
                    },
                );
            });
            cut_reached_rx
                .recv()
                .expect("the real active offer must reach its selected transition cut");
            fail_tx
                .send(())
                .expect("the controlled destination remains blocked until the active cut");
            wait_until_closed(state.as_ref());
            active_resume_tx
                .send(())
                .expect("the active producer remains paused at the selected transition cut");
            if active.join().is_err() {
                panic!("the active producer transition must not panic");
            }

            let claimed = claim_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("the terminal writer must claim every accepted active record");
            assert_eq!(claimed, expected_claimed, "{cut:?}");
            let at_claim = snapshot(state.as_ref());
            assert_eq!(at_claim.undelivered, 0, "{cut:?}");
            assert_eq!(at_claim.terminal_claimed, expected_claimed, "{cut:?}");
            assert_eq!(
                at_claim.sink_closed, expected_sink_closed_at_claim,
                "{cut:?}"
            );
            assert_live_conservation(at_claim);

            // This offer observes the writer's actual closed state before it can enter A, Q, or U.
            producer.offer_access(access("after-terminal-close"));
            let after_closed_offer = snapshot(state.as_ref());
            assert_eq!(after_closed_offer.active, 0, "{cut:?}");
            assert_eq!(
                after_closed_offer.queued, expected_queued_after_later,
                "{cut:?}"
            );
            assert_eq!(after_closed_offer.undelivered, 0, "{cut:?}");
            assert_eq!(
                after_closed_offer.sink_closed, expected_sink_closed_after_later,
                "{cut:?}"
            );
            assert_live_conservation(after_closed_offer);

            claim_resume_tx
                .send(())
                .expect("the terminal writer must remain paused at its observable claim");
            drop(producer);
            join(writer);
            let final_state = snapshot(state.as_ref());
            assert_eq!(final_state.active, 0, "{cut:?}");
            assert_eq!(final_state.queued, 0, "{cut:?}");
            assert_eq!(final_state.undelivered, 0, "{cut:?}");
            assert_eq!(final_state.terminal_claimed, 0, "{cut:?}");
            assert_eq!(final_state.sink_closed, 3, "{cut:?}");
            assert_live_conservation(final_state);
            assert!(
                metrics
                    .render(1, 1, 0, None)
                    .contains("asr_telemetry_dropped_total{reason=\"sink_closed\"} 3\n"),
                "{cut:?} must classify the trigger, active record, and later offer once"
            );
        }
    }

    #[test]
    fn stderr_failure_uses_only_the_stderr_failure_label() {
        let (prepared, metrics) = prepared(1, 1);
        let producer = prepared.producer();
        producer.offer_trace("trace\n".to_owned());
        let (started, handle) =
            start_with_sink(prepared, RecordingSink::failing(Destination::StderrTrace))
                .expect("controlled stderr writer starts");
        drop(started);
        drop(producer);
        join(handle);
        let rendered = metrics.render(1, 1, 0, None);
        assert!(
            rendered.contains("asr_telemetry_write_failures_total{destination=\"stderr\"} 1\n")
        );
        assert!(
            rendered.contains("asr_telemetry_write_failures_total{destination=\"stdout\"} 0\n")
        );
    }

    #[test]
    fn final_application_producer_drop_does_not_join_a_blocked_writer() {
        let (prepared, _metrics) = prepared(1, 1);
        let records = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (began_tx, began_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (producer, handle) = start_with_sink(
            prepared,
            RecordingSink::blocked(records, began_tx, release_rx),
        )
        .expect("controlled telemetry writer starts");
        producer.offer_trace("blocked\n".to_owned());
        began_rx
            .recv()
            .expect("writer must enter the controlled blocked destination");
        let (dropped_tx, dropped_rx) = mpsc::channel();
        thread::spawn(move || {
            drop(producer);
            dropped_tx
                .send(())
                .expect("drop observer receiver remains available");
        });
        dropped_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("application producer drop must not join the writer");
        release_tx
            .send(())
            .expect("controlled writer release receiver remains available");
        join(handle);
    }

    #[test]
    fn preparation_checks_capacity_domain_and_common_assembly_tuple() {
        let (minimum, _) = prepared(1, 1);
        assert_eq!(minimum.capacity, 4);
        let maximum = tokio::sync::Semaphore::MAX_PERMITS;
        let (maximum_prepared, _) = prepared(maximum, maximum);
        let maximum_capacity = match maximum.checked_mul(4) {
            Some(capacity) => capacity,
            None => panic!("the supported telemetry capacity must fit usize"),
        };
        assert_eq!(maximum_prepared.capacity, maximum_capacity);
        let admission = ServiceAdmission::new(1, 1, Duration::from_secs(1))
            .expect("controlled admission is valid");
        let state = AdmissionState::new(&admission).expect("controlled admission state is valid");
        assert!(minimum.matches_admission(&admission, &state));
        assert!(!maximum_prepared.matches_admission(&admission, &state));
        assert!(ServiceAdmission::new(0, 1, Duration::from_secs(1)).is_err());
        let above_maximum = match maximum.checked_add(1) {
            Some(value) => value,
            None => return,
        };
        assert!(ServiceAdmission::new(above_maximum, 1, Duration::from_secs(1)).is_err());
    }

    #[test]
    fn launcher_refusal_keeps_prepared_state_unstarted() {
        let (refused_preparation, _metrics) = prepared(1, 1);
        let state = Arc::clone(&refused_preparation.state);
        let refusal =
            refused_preparation.start_with(|_| Err("controlled launcher refusal".to_owned()));
        match refusal {
            Ok(_) => panic!("the controlled launcher must refuse before writer start"),
            Err(error) => assert_eq!(error, "controlled launcher refusal"),
        }
        assert_eq!(snapshot(&state).entered, 0);
    }
}
