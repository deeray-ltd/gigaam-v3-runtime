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

1. Model Package validates the closed version-1 configuration before a selected asset or
   native session is used. It exposes typed frontend, CTC, RNN-T, VAD, tensor-name, and
   artifact definitions.
2. Audio decodes complete input or validated incremental frames, rejects invalid terminal
   frame state, resamples to the model rate, and creates log-mel features from package
   tables.
3. Recognition owns the selected ONNX Runtime provider and validates native outputs before
   CTC, RNN-T, or VAD state consumes them.
4. Transcription owns the application workflow: batch windows and stitching; streaming
   revisions and frontiers; channel selection; and ordered dialog turn patches.
5. Service or CLI project those typed results to a transport or terminal without recreating
   transcription policy.

The package graph deliberately has one ONNX Runtime owner: Recognition. CPU and TensorRT
either initialize as selected or refuse; TensorRT requires the CUDA feature but registers
TensorRT rather than silently using CUDA. CUDA registers exactly CUDA as its accelerator
provider, intentionally leaves built-in CPU placement available while ORT records graph
assignment, and refuses any observed provider other than CPU or CUDA. It does not register
CUDA as a TensorRT fallback.

For CUDA, Recognition derives the CTC/RNN-T role set before package or native work and parses
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
unvalidated sample-rate value.

HTTP batch work receives a shared execution control. When its deadline publishes a 504,
queued or successor windows cannot start after cancellation is observed; the admission
permit remains owned until the work terminalizes; and a late result cannot replace the
timeout response. An already executing native call may finish. Streaming and offline
non-deadline paths use explicit non-cancelling execution control.

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
