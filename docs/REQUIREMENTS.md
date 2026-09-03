<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
<!-- Copyright (C) 2026 Yuriy Krasilnikov -->
<!-- Copyright (C) 2026 Deeray Ltd. -->

# GigaAM v3 Runtime Requirements

## Scope

This document states the behavior the public runtime commits to. README, API, operations,
and release material describe the same behavior in more detail and do not contradict it.

## Requirement records

### ID-001 · GigaAM v3 identity

- Requirement: The exact public title is `GigaAM v3 Runtime`; the repository URL is
  `https://github.com/deeray-ltd/gigaam-v3-runtime`; and Yuriy Krasilnikov is the first
  listed author and copyright holder in every applicable public projection. Public identity,
  package descriptions, runtime behavior, and release artifacts describe this runtime only.
- Scope: Tracked public text and shipped metadata, including manifests, public
  documentation, copyright headers, image metadata, and release automation.

### RUST-001 · Supported Rust contract

- Requirement: The seven-package workspace uses resolver 3, Edition 2024, and Rust 1.98.0.
  gigaam-primitives, gigaam-model-package, gigaam-audio, gigaam-recognition,
  gigaam-transcription, gigaam-service, and gigaam-cli inherit the workspace edition and
  rust-version contract.
- Scope: The Cargo workspace and all crate targets, enforced by `rust-toolchain.toml` and
  Cargo metadata across local, CI, and release builds.

### BACKEND-001 · No fallback or legacy compatibility path

- Requirement: A selected backend, provider, artifact, configuration, or public interface either
  works as selected or refuses. It must not silently fall back, accept legacy aliases, or retain a
  compatibility path that changes the selected behavior.
- Scope: Provider selection, runtime initialization, public configuration, release
  artifacts, and documentation. Consumers include Recognition, Service, CLI, container
  images, and operators.

### CONFIG-001 · Typed configuration and strict refusal

- Requirement: Startup and public configuration parse typed values strictly; missing documented
  defaults remain valid, while present empty, invalid, duplicate, or conflicting values refuse.
  An unknown value in a public enumeration is refused rather than silently ignored, and invalid
  input is rejected when state is constructed rather than when it is later used.
- Scope: CLI, environment, HTTP/WS, provider initialization, signal handling, and public
  state. Consumers include clients, operators, service handlers, and runtime initialization.

### PACK-STRICT-001 · Strict typed model-package configuration

- Requirement: config.kv is parsed once by Model Package into typed definitions. Malformed
  lines, duplicate keys, empty values, invalid typed values, unknown keys, and incompatible
  declared roles refuse before selected-asset or native-session construction. Valid migrated
  package behavior remains the compatibility oracle; no fallback parser accepts a different
  interpretation.
- Scope: Model Package configuration parsing and every Service or CLI package-open path;
  refines CONFIG-001 and BACKEND-001. Consumers include Audio, Recognition, Transcription,
  Service, CLI, and model-package users.

### PACK-SCHEMA-001 · Explicit closed package schema

- Requirement: format_version=1 is a required version discriminator. Version 1 admits exactly
  36 required entries, including 35 typed runtime keys, and eight optional opaque retained keys.
  Missing, malformed, duplicate, unknown, or unsupported configuration refuses before asset
  resolution or native-session construction. No unversioned legacy path exists; a new key or
  changed meaning needs an explicitly supported schema version and migration rule.
- Scope: Model Package schema and the public model-pack contract; refines PACK-STRICT-001.
  Consumers include Model Package, Service, CLI, and model-pack producers.

### PROCESS-CONFIG-001 · Typed process configuration ownership

- Requirement: Service and CLI read applicable command-line and process-environment values once,
  validate them before runtime/model construction, and pass typed choices to capability
  constructors. For `asr bench`, the selected Model Package supplies the validated SampleRate
  used to derive the --window and optional --alt-window plan; Audio resamples decoded WAV input
  to that rate before native runtime initialization. Audio defines frontend modes but never reads
  ASR_FRONTEND; Transcription defines the observation port but never reads ASR_TRACE or writes
  process streams. ASR_TRACE is absent when disabled, enabled by a nonempty value, and refused
  when present but empty. This boundary does not create a stable terminal-output promise.
- Scope: Service and CLI startup/composition, provider configuration, frontend selection,
  and trace selection; refines CONFIG-001. Consumers include operators, Service, CLI,
  Audio, Transcription, and Recognition.

### CUDA-ASSIGNMENT-001 · Verified CUDA encoder composition

- Requirement: Before Model Package or ONNX Runtime work, selected CUDA composition derives the
  exact CTC/RNN-T roles it will construct and validates a typed assignment policy. Default
  `verified` mode requires one valid SHA-256 fingerprint per required role. The only bypass is
  explicit `allow-unverified`, which conflicts with every expected fingerprint and skips equality
  only. CPU and TensorRT refuse CUDA assignment settings; extra valid fingerprints for roles not
  required by the composition are inert. A CUDA encoder session records API-24 graph assignment,
  copies its entries into owned evidence, accepts only CPU/CUDA assignments with at least one CUDA
  node, and verifies the deterministic sorted-multiset SHA-256 over versioned length-prefixed
  provider, node, domain, and operator-type fields before the session can be returned or used.
  CUDA deliberately retains built-in CPU placement for this observation; TensorRT retains disabled
  CPU fallback and never registers CUDA as its fallback provider. In unverified mode Service and
  CLI project the role, observed fingerprint, and CPU/CUDA counts to stderr after session creation
  and before warmup or inference. Model Package and HTTP/WS schemas do not carry this policy. The
  fingerprint binds observed assignment only, so byte-only model or runtime changes are outside
  this requirement unless they change that assignment.
- Scope: Recognition provider/session construction, Service/CLI process configuration and
  startup projection, provider-feature artifacts, and operations guidance; refines
  BACKEND-001 and CONFIG-001. A real GPU and model are required to accept a
  particular placement fingerprint. Consumers include Service, CLI, operators, and image
  definitions.

### AUDIO-VALID-001 · Validated audio boundary

- Requirement: Audio rejects invalid sample rates, non-finite decoded samples, zero or
  mismatched channel and matrix dimensions, and terminal partial sample frames before
  resampling, frontend, or recognition. A WAV data chunk with a partial sample frame refuses.
  Incremental interleaved input may retain a partial group between pushes, but finish refuses a
  remaining partial group. Valid complete finite input preserves existing decoding and
  resampling behavior.
- Scope: Audio ingestion, interleaved streaming input, resampling, and frontend inputs.
  Applies to HTTP, WebSocket, and CLI audio paths before Transcription or Recognition
  consumes audio; refines CONFIG-001.

### EXEC-CANCEL-001 · Cooperative HTTP batch cancellation

- Requirement: An HTTP deadline or scheduler-wait expiry requests terminal cancellation through
  one shared execution control. Queued work and successor windows cannot begin after
  cancellation is observed; a late result cannot replace a published 504; and the admission
  permit remains held until terminal acknowledgement. A native call already executing may
  finish. Streaming and non-deadline CLI paths explicitly use non-cancelling execution control.
- Scope: HTTP batch work, Transcription checkpoints, Recognition scheduler, and Service
  admission ownership. It does not establish native ONNX Runtime interruption,
  stable error codes, or new public budgets.

### OBS-001 · Bounded best-effort runtime telemetry

- Requirement: Service uses one bounded logical FIFO writer for runtime access and trace records.
  Its checked capacity is max_http + 3 times max_ws; a full queue drops the newest record.
  Sink closure and stdout/stderr destination write failures are counted. Lifecycle, startup, and
  fatal diagnostics attempt synchronous stderr output directly rather than entering this queue.
  Shutdown never waits for a potentially blocking telemetry join, and delivery through a failed
  sink is not guaranteed.
- Scope: Service access logs, trace observations, metrics, and process diagnostics.
  Consumers include operators, metrics scrapers, and HTTP/WebSocket clients.

### MODEL-OUTPUT-FINITE-001 · Validated recognition-output domain

- Requirement: Every model output crossing a Recognition native-adapter or pure-algorithm
  boundary satisfies every shape and storage-cardinality constraint declared by that boundary and
  contains only finite numeric values before token, recurrent, endpoint, or result state consumes
  it. Invalid output refuses locally; it is never rewritten, retried through another provider, or
  converted into a plausible result. Valid finite output preserves its selected provider and
  observable recognition behavior.
- Scope: Recognition CTC, RNN-T, encoder, prediction, joint, and VAD outputs plus direct
  callers. A supported intentional infinite output domain requires a role-specific algebra
  and oracle before this requirement is changed.

### SHUTDOWN-ADMISSION-001 · Monotonic draining admission

- Requirement: Service owns one monotonic `Running -> Draining` admission state shared by
  readiness, HTTP permit acquisition, and complete pre-upgrade WebSocket permit acquisition.
  After the transition linearizes, no new work is admitted or warmed; work admitted before it
  retains the bounded graceful-drain behavior. Readiness and admission cannot disagree about
  draining or return to `Running`. Readiness additionally answers 503 once a recognition
  worker has stopped after a decoder panic while admission remains `Running`, with draining
  taking precedence.
- Scope: Service lifecycle, readiness, HTTP/WebSocket admission, permit ownership,
  recognition worker state, and graceful shutdown.

### REL-001 · Three truthful release artifacts

- Requirement: The public release has distinct CPU, CUDA, and TensorRT artifacts with truthful
  tags `deeray/asr:0.1.0-cpu`, `deeray/asr:0.1.0-cuda`, and
  `deeray/asr:0.1.0-tensorrt`.
- Scope: Image definitions, tags, runtime contents, and publication records.

### REL-002 · Hermetic pinned provenance

- Requirement: A release build obtains all native runtime inputs from declared, hermetic build
  inputs with exact versions, integrity pins, and recorded provenance; it does not depend on
  host-provisioned payloads or undeclared network state.
- Scope: Docker build stages, native payload assembly, manifests, and verification for CPU,
  CUDA, and TensorRT payloads.

### REL-003 · License and notice payload

- Requirement: Every public runtime image carries the project license texts and authoritative
  notices for copied native runtime components in an identifiable runtime location.
- Scope: Final image filesystems, payload assembly, third-party notices, and release
  verification for CPU, CUDA, and TensorRT images.

### REL-004 · Final-rootfs closure

- Requirement: A final image contains only required runtime files and dependencies: no Python,
  model pack, source tree, build cache, or undeclared native payload. The model remains external.
- Scope: Final CPU, CUDA, and TensorRT image layers.

### REL-005 · Provider and runtime truthfulness

- Requirement: Each image's compiled features, selected runtime provider, required libraries, and
  refusal behavior agree with its declared CPU, CUDA, or TensorRT artifact identity.
- Scope: Rust features, runtime initialization, image payloads, health state, and artifact
  verification.
