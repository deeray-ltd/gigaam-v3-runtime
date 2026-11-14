# Architecture

GigaAM v3 Runtime is one modular monolith, not a set of network services. It has two
executables:

- asr-serve is the HTTP and WebSocket process.
- asr is an offline command-line process.

Both use the same in-process capability graph. A model package and ONNX Runtime shared
libraries are external inputs. CPU, CUDA, and TensorRT are selected Recognition providers,
not separate application topologies.

## Logical capability graph

The runtime has six product capabilities, implemented by six product Cargo packages, plus
the dependency-free Primitives support package. Primitives does not create a seventh
product capability or a generic Core facade.

~~~text
Primitives (support) -> Model Package -> Audio -> Recognition -> Transcription -> Service
                                                                  \-----------> CLI
~~~

The compact diagram expresses the ownership direction. The Cargo manifests are the exact
dependency authority: packages may take the downward dependencies they need, but no
capability owns a reverse dependency.

| Product capability | Cargo package | Owns | Does not own |
| --- | --- | --- | --- |
| Model Package | gigaam-model-package | Versioned config.kv schema, compatibility, selected-asset containment, and typed model definitions. | ONNX Runtime sessions, provider choice, or process environment. |
| Audio | gigaam-audio | Validated file and frame ingestion, codecs, resampling, channel analysis, and log-mel frontend construction. | Process environment, provider selection, or transcript policy. |
| Recognition | gigaam-recognition | Decoder contracts and algorithms, native-output validation, the only ONNX Runtime/provider assembly, execution control, and scheduler. | HTTP, WebSocket, dialogs, or process output. |
| Transcription | gigaam-transcription | Batch and stream workflows, channel policy, window stitching, endpointing, revisions, dialog reconstruction, and observations. | Process environment, direct sockets, or ONNX Runtime ownership. |
| Service | gigaam-service | HTTP/WebSocket adapters, process configuration, admission, lifecycle, wire projection, metrics, and bounded runtime telemetry. | Audio/recognition business policy or a second model/runtime path. |
| CLI | gigaam-cli | Offline grammar, configuration, direct composition, and terminal projection. | A service listener, scheduler wrapper, or generic application facade. |

Primitives support (`gigaam-primitives`) provides checked numeric conversions at typed
boundaries. It owns no product policy, model, protocol, or process configuration and is
not a separately deployable capability.

This split stays in one process because the model session, audio-to-window transitions, and
streaming frontier state require low-latency in-process ownership. It does not imply
independent deployment, independent scaling, or independent failure domains for the
logical capabilities.

## C1 system context

~~~text
HTTP client ───────┐
WebSocket client ──┼──> asr-serve ──> GigaAM v3 model package
Operator ──────────┘          │
                               └──> dynamically loaded ONNX Runtime
                                      └── CPU / CUDA / TensorRT provider

Prometheus scraper / monitoring system ── GET /metrics ──> asr-serve

Offline user ───────────────> asr ──> the same package and provider contract
~~~

The Prometheus scraper or monitoring system is an external C1 participant. The runtime
serves its metrics exposition but neither deploys nor configures a monitoring component.

The service itself provides no authentication, authorization, TLS, persistent transcript
storage, or training path. A deployment that needs those controls places a separately
configured ingress and storage policy around the process. The runtime still enforces its
own audio validation, bounded admission, provider refusal, and lifecycle behavior.

## Data and ownership flow

1. Model Package validates a closed versioned configuration before selected assets or native
   sessions are used. Service selects one V3 variant and family; the library and CLI also retain
   V1/V2 packages. Typed frontend, tokenizer, family, VAD and artifact definitions feed Recognition.
2. Audio decodes complete input or validated incremental frames, rejects invalid terminal
   frame state, resamples to the model rate, and creates log-mel features from package
   tables.
3. Recognition owns the selected ONNX Runtime provider and validates native outputs before
   CTC, RNN-T, or VAD state consumes them.
4. Transcription owns the application workflow: batch windows and stitching; streaming
   revisions and frontiers; channel selection; and ordered dialog turn patches.
5. Service or CLI project those typed results to a transport or terminal without recreating
   transcription policy.

### Library batch speech segmentation

`BatchTranscriber::transcribe_channel_with_vad` and
`MultiChannelBatchTranscriber::transcribe_with_vad` borrow a caller-owned
`SpeechProbabilityDetector` for one operation. The existing `BatchSetup` and methods without
VAD retain their behavior. The VAD path requires a 16 kHz model/frontend rate before detector
execution; Audio adapters still own source-to-model resampling. Each detector call covers a
complete independent waveform with one probability per 512 samples, including its partial
last frame. Segmentation refuses incomplete/excess coverage and values outside finite `[0, 1]`.

Multi-channel batch selects original physical channels before detection and reuses the one
detector and decoder sequentially. A selected silent channel remains in the result with no
words. All-silence input performs no frontend or decoder work; a separately requested warmup
is not part of that guarantee. Detector ownership, native-session creation, admission and
worker allocation remain process-composition decisions.

Accepted padded detector spans are greedily grouped while the envelope from the first start to
the next end fits the configured recognition window. An oversized individual span remains alone
and uses the existing internal windowing. Each group retains the contiguous original samples,
including gaps and any detector-rejected speech within them. The caller's padding policy applies
to each group. Stitching is local to a group; separate groups are concatenated without deleting
repeated words. Words are clipped to the real group, then shifted once onto the original channel
clock. Planned window counts sum across groups even when a window has no usable feature frames.
`PadPolicy::Exact` retains the existing limit that groups no longer than the frontend FFT length
produce no decoder call; `PadToWindow` explicitly supplies the configured window shape. Minimum
speech duration remains a strict detector boundary: equality is discarded; minimum silence closes
a detector span and does not itself force a recognition group boundary.

Window timings and evidence temporal fields use the original channel clock, including any
synthetic window padding. Evidence retains local feature/token coordinates and the exact integer
`frontend_origin_sample` of the crop. The source offset of a local token is that origin divided
by the sample rate, plus `first_feature_frame * hop_length / sample_rate`, plus the token frame
divided by the decoder frame rate. A crop may begin between source hop boundaries, so its sample
origin must not be rounded into a feature-frame index. The original complete-channel path has
zero origin and preserves its observation values and existing CLI evidence bytes.

Cancellation is checked before/after detector and segmentation, spans, frontend and recognition
work. An executing native call may finish; cancellation prevents successor work and a completed
batch result. Successful detector/segmentation time is a separate `StageTimings::vad_seconds`
component, not frontend time. HTTP/CLI batch VAD switches are not provided by this library
capability: their integration must choose native detector lifecycle and propagate nonzero crop
origins through the CLI evidence format before enabling segmented evidence output.

The package graph deliberately has one ONNX Runtime owner: Recognition. CPU and TensorRT
either initialize as selected or refuse; TensorRT requires the CUDA feature but registers
TensorRT rather than silently using CUDA. CUDA registers exactly CUDA as its accelerator
provider. CUDA encoders leave built-in CPU placement available while ORT records graph
assignment, and refuse any observed provider other than CPU or CUDA. CUDA prediction and
joint sessions disable CPU fallback and report observed assignment. TensorRT loads
prepared engines through EPContext wrappers with CPU fallback disabled; ordinary ONNX
packages are refused for that serving path. It does not register CUDA as a TensorRT fallback.

For CUDA encoders, Recognition derives the CTC/RNN-T role set before package or native work and parses
the corresponding assignment policy there. Default `verified` mode requires a distinct
SHA-256 fingerprint for every required role; the selected role is admitted again before
`Session::builder` for a direct native caller. After the session is created, but before it can be
returned or used for warmup/inference, one private API-24 adapter copies ORT-owned
graph-assignment records into owned values. It accepts a nonempty CPU/CUDA-only assignment with
at least one CUDA node, sorts the full multiset, and binds a versioned, length-prefixed
`(provider, node, domain, operator_type)` representation. The resulting evidence must equal the
role fingerprint. Explicit `allow-unverified` skips equality only and projects that owned
evidence at the Service/CLI process boundary before work starts. No Model Package or HTTP schema
contains this deployment-specific policy. The guard detects a model or ONNX Runtime change only
through a changed observed assignment; it does not independently hash model or runtime bytes.

## Process configuration and execution

Service and CLI parse their applicable command-line and environment values before package,
frontend, scheduler, or native-runtime construction. The model rate is then bound to
time-based transcription configuration, preventing adapters from maintaining a parallel
unvalidated sample-rate value. Recognition returns an owned `NativeRuntime` from explicit
initialization; native constructors receive that capability and their sessions retain the
underlying environment. CLI retains it through command completion. Service shares it with
lazy VAD construction and returns from post-initialization errors only after owned resources
unwind; scheduler workers join after physically executing work finishes. Synthetic Service
facades may omit the runtime, but a request selecting native VAD then refuses.

HTTP batch work receives a shared execution control. When its deadline publishes a 504,
queued or successor windows cannot start after cancellation is observed; the admission
permit remains owned until the work terminalizes; and a late result cannot replace the
timeout response. Transcription passes that same control through normal, scored, warmup and
flush window decoding. Scheduled contextual calls refuse a control with different shared identity;
legacy scheduled calls use the bound control, while direct native legacy calls create an operation
control. Boxed decoders preserve all entry points. Decoder algorithms check cancellation between
supported steps without terminally completing the request. An already executing native call may
finish and retains worker capacity until return; whole VAD calls, validation and reconstruction do
not promise interruption within the call. Streaming and offline non-deadline paths use explicit
control without a deadline.

The scheduler currently gives Tick work strict priority before Window work. That is a
selected implementation baseline, not a compatibility promise or a fairness guarantee.
Fairness and multi-request batching require workload evidence and remain separate work.

## Service boundary

Service owns the sole network and process lifecycle boundary:

- POST /v1/transcribe adapts raw batch audio to Transcription.
- GET /v1/stream adapts WebSocket frames to streaming Transcription.
- GET /health, GET /livez, GET /readyz, and GET /metrics project capability and lifecycle
  facts.

One admission state moves monotonically from Running to Draining. HTTP admission and
complete pre-upgrade WebSocket admission share it; readiness observes the same admission
state and additionally fails once a recognition worker has stopped after a panic (draining
takes precedence). No new work or WebSocket warmup begins after draining linearizes, while
already admitted work retains its terminal behavior.

Service emits access and optional trace records through one bounded logical FIFO telemetry
writer. Its capacity is derived from admission limits. Queue-full records are dropped;
sink closure and destination write failure are counted. Lifecycle, startup, and fatal
diagnostics use direct stderr attempts so they do not depend on detached telemetry
delivery.

## Streaming and dialog semantics

Streaming sends tail-replacement word events and may send stable and final frontiers.
Committed output never changes. Stable output has settled order but may still receive a
text adjustment. For multichannel input, Transcription emits each channel's events in
source-channel order and then the optional dialog patch. A turn patch replaces the local
dialogue tail from revise_from; a turn is identified by its channel and per-channel index.

The Service WebSocket adapter does not own a second session, resampler, endpoint, dialog,
or channel-selection implementation. It validates the handshake, holds all requested
channel permits, transports frames, and projects the result.

## Component map

| Package | Principal components |
| --- | --- |
| gigaam-model-package | schema parser, compatibility validator, contained regular-asset resolver, typed package definition |
| gigaam-audio | container/raw/interleaved decoders, G.711 conversion, resampler, frontend, channel analysis |
| gigaam-recognition | CTC/RNN-T/VAD contracts, native-output validation, provider plan, ORT adapters, execution control, scheduler |
| gigaam-transcription | batch and multichannel batch flows, stream and multichannel stream flows, stitching, endpointing, turns, dialog, observations |
| gigaam-service | startup/configuration, capabilities/policy context, admission, HTTP, WebSocket, router, health, lifecycle, JSON, metrics, log, telemetry |
| gigaam-cli | grammar, typed configuration, command composition, stdout/stderr projection |

## Boundaries not claimed by this architecture

The current implementation does not establish model-package hashes or provenance,
compatibility ranges, offline pack verification, schema retirement, native in-flight ONNX
Runtime interruption, fairness, batching, RNN-T streaming, confidence, diarization,
real-provider acceptance, a published image, or a stable pre-1.0 protocol. Those boundaries
are not established by the current implementation and are not part of its public contract.
Deferred product work and what unlocks it is recorded in [ROADMAP.md](ROADMAP.md);
real-provider acceptance and image publication are release steps outside that document.

Recognition decoders declare an immutable silence-mask capability. Transcription admits the
legacy blank heuristic only for decoders declaring it; non-legacy VAD sessions accumulate
newly observed source samples across bounded windows. Core manual sessions commit on flush.
The process adapters expose only the endpoint selectors listed in the API documentation.


The optional streaming silence gate observes the complete retained real-audio window before an
Advance decode. A channel owns one detector for both the gate and VAD endpointing, reusing the
observation when both apply. Only all-subthreshold windows with no pending transcript work skip;
they do not produce a synthetic decoder result. A processed-sample cursor is distinct from the
feature cache and decoded position, and rollover retains real overlap and source time. Nonempty
flush decodes normally. The native CTC WebSocket validates enabled policy and selected VAD
capability before upgrade; non-deadline work still uses the existing execution ownership.
