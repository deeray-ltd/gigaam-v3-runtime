# Design decisions

These are the current durable design choices for GigaAM v3 Runtime. They explain the
present seven-package modular-monolith structure; they do not claim that every earlier
revision used the same structure.

## 1. One modular monolith, not distributed services

**Context.** Audio preparation, model sessions, scheduler ownership, streaming frontiers,
and dialog merging have tight latency and state dependencies. Splitting them into network
services would add serialization, distributed failure modes, deployment coordination, and
accelerator ownership without an established independent-scaling or isolation need.

**Decision.** Keep one Service process and one offline CLI executable. Divide source
ownership into Model Package, Audio, Recognition, Transcription, Service, CLI, and the
dependency-free Primitives support package.

**Cost.** Components are independently owned in the Cargo graph but are not independently
deployed or fault-isolated. Process supervision remains the recovery boundary.

## 2. Closed, versioned model packages

**Context.** A permissive anonymous key/value file makes a misspelled key, changed meaning,
or package built by a different exporter indistinguishable from a valid runtime input.

**Decision.** Require format_version=1 and a closed V1 inventory: 35 typed runtime keys,
the required discriminator, and eight opaque retained keys. Parse and compatibility-check
the configuration before selecting assets or creating a native session.

**Cost.** Unversioned packages refuse and a future key or changed meaning needs an explicit
schema branch and migration. Hashes, provenance, compatibility ranges, offline verification,
and schema retirement are deliberately not implied by the V1 parser.

## 3. ONNX Runtime is dynamic and Recognition is its only owner

**Context.** Static linking would tightly bind the executable to a native CUDA/runtime
stack, while allowing provider construction in several layers would weaken selected-provider
truthfulness.

**Decision.** Recognition alone constructs ONNX Runtime plans, provider registration,
schedulers, and native-output adapters. Service and CLI pass one explicit library path and
one selected provider through typed configuration. TensorRT is a CUDA-dependent feature but
registers TensorRT only when selected.

**Cost.** Native library paths and provider libraries are deployment inputs. A missing,
conflicting, invalid, or unavailable selected provider refuses rather than falling back.
Building or type-checking the source does not demonstrate successful execution with a real
model and provider.

## 4. Process adapters own configuration; capabilities receive typed values

**Context.** Environment reads and duration conversion scattered through workflow code make
failure order, reuse, and testability ambiguous.

**Decision.** Service and CLI parse their own arguments and environment once. They bind
time-based application configuration to Audio's validated model sample rate before
initializing the runtime. Audio owns frontend modes; Transcription owns observation ports;
neither reads process environment or writes process streams.

**Cost.** Startup has a strict validation order and refuses invalid inputs early. A caller
cannot rely on a hidden default after providing an invalid value.

## 5. Validate audio at the boundary

**Context.** Truncating partial frames or passing non-finite/mismatched input forward turns
corrupt data into a plausible transcript.

**Decision.** Audio validates rate, samples, dimensions, channel relationships, complete WAV
frames, and terminal incremental frame state before resampling or frontend work.

**Cost.** Invalid media is refused rather than repaired. Incremental inputs may span a frame
boundary across pushes, but the final remainder must be empty.

## 6. Transcription owns temporal and dialog policy

**Context.** Batch windows, streaming revisions, channel selection, endpointing, and dialog
merge are one transcription domain. Recreating them in HTTP, WebSocket, or CLI adapters
would produce divergent policy and reverse dependencies.

**Decision.** Transcription owns batch and streaming workflows, typed sample-bound
configuration, revisions/frontiers, turns, dialog reconstruction, and observations. Service
and CLI only choose a supported workflow and project its result.

**Cost.** Transport adapters are intentionally narrow. Changes to frontier or dialog
semantics are protocol changes, not local socket changes.

## 7. HTTP cancellation is cooperative and provider-independent

**Context.** Returning a timeout while queued successor work can still start wastes scarce
capacity and permits a late result to contradict the visible timeout.

**Decision.** An HTTP deadline or scheduler wait expiry requests one shared execution
cancellation state. Transcription and Recognition observe it before later work begins; the
Service holds capacity until terminal acknowledgement and keeps the published 504 terminal.

**Cost.** A native call already executing may finish. The decision does not claim native
ONNX Runtime interruption. Streaming and non-deadline CLI paths explicitly select a
non-cancelling execution policy.

## 8. Admission drains monotonically

**Context.** Readiness, HTTP permits, and WebSocket upgrade permits must not race into
inconsistent shutdown behavior.

**Decision.** Service has one Running-to-Draining admission state. HTTP and complete
pre-upgrade WebSocket admission use that state; readiness observes the same admission state
and additionally fails once a recognition worker has stopped after a panic (draining takes
precedence); no transition returns to Running.

**Cost.** New work is rejected immediately after draining linearizes, while work admitted
before it keeps its own terminal behavior. A WebSocket reserves all requested channel
permits atomically.

## 9. Runtime telemetry is bounded best effort

**Context.** Synchronous access/trace writes can delay request and lifecycle paths, while
unbounded buffering turns a failed destination into unbounded memory growth.

**Decision.** Service sends runtime access and trace records to one bounded logical FIFO
writer. Queue-full records are dropped newest-first; sink closure and destination failures
are observable through metrics. Startup, lifecycle, and fatal diagnostics attempt direct
stderr output outside that writer.

**Cost.** Runtime telemetry is not a delivery guarantee, particularly after a destination
failure. Shutdown does not join a potentially blocking writer.

## 10. Images are source-built provider variants

**Context.** A native runtime artifact is truthful only when its compiled features, selected
libraries, notices, entrypoint, and refusal behavior agree.

**Decision.** Maintain three independent source-build Dockerfiles: CPU, CUDA, and TensorRT.
Each uses pinned selected native inputs, builds gigaam-service with its matching feature
set, ships a non-root asr-serve entrypoint, and leaves the model external.

**Cost.** Image definitions and local construction do not prove real-model/provider behavior
or publication. CPU and accelerator runtime acceptance, ingress placement, and publication
remain separate, outstanding steps before any image is described as released.

## 11. CUDA encoder placement is a verified composite

**Context.** Registering a CUDA execution provider proves neither that the intended encoder
graph contains CUDA work nor that an ONNX Runtime upgrade, export change, or provider change
kept the same graph placement. Disabling CPU fallback before inspecting the graph removes the
evidence required to distinguish a mixed CPU/CUDA placement from a failed composition.

**Decision.** Recognition derives the required CTC/RNN-T roles before package/native work. For
selected CUDA, default `verified` startup requires one role-specific SHA-256 fingerprint; the
explicit `allow-unverified` policy skips equality only. The private API-24 adapter records and
copies graph assignments after session construction, permits only CPU/CUDA records with at least
one CUDA node, canonicalizes their sorted multiset of provider, node, domain, and operator type,
and gates return of the session on the result. CUDA registers its requested provider and retains
built-in CPU placement for observation. TensorRT remains a TensorRT-only selection with disabled
CPU fallback. Service and CLI project unverified evidence to stderr before warmup/inference.

**Cost.** CUDA operators must provide current per-role fingerprints or explicitly request
unverified evidence. Real model/GPU acceptance remains external because the fingerprint binds an
actual ORT placement, not a source-only claim; a model or runtime change refuses only if it
changes that placement. This adds no Model Package fields and no HTTP/WS schema.
