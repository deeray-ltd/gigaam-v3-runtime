# Changelog

All notable changes are recorded here. Pre-1.0 releases do not promise a stable protocol.

## Unreleased

### Added

- Optional OpenAI-compatible batch transcription and model-list routes with JSON, text,
  and verbose output, local model identifiers, Russian language declaration, and the
  existing batch VAD controls. `ASR_API_MODE` selects the native or OpenAI API.
- Separate upload reader capacity and timeout settings bound body ingestion independently
  from batch processing capacity and its deadline.

- Optional native token emission scores and source-window provenance for scored batch
  segments, while ordinary transcript words and timing remain unchanged.

- Complete existing WebSocket stream-policy controls, bound before upgrade at the model
  rate. Optional `applied_config` adds a versioned bound-policy event before transcript
  events while preserving default output, active shutdown precedence, and session ownership.
- Complete existing stream-policy controls in CLI `stream` and `dialog`, including the
  detector threshold, both silence rules, window cadence, horizon/locking, seam delay,
  and retained silence. Optional `--applied-config` reports the bound policy after successful
  execution. Existing defaults, channel behavior, and output without the flag are preserved.
- Opt-in HTTP batch VAD through explicit response version 2, with the same validated
  threshold, duration, and padding controls as the CLI, original-clock speech spans, and
  an applied-policy response. Version 1 bodies and VAD-off asset independence are preserved.
  Version 2 provides typed post-binding refusal codes; malformed queries retain legacy
  errors. Request-owned VAD construction/detection phases and completed VAD timing expose
  work while cancellation retains admission until native work and session cleanup finish.

- Opt-in `transcribe --vad` for CTC and RNN-T batch speech segmentation, with explicit
  threshold, minimum speech/silence, and padding controls. It requires a 16 kHz model,
  preserves batch channel selection and source times, and reports applied policy and VAD
  duration separately. Segmented evidence requires explicit version 2.
- Standalone `vad --channels all|0,2` selects physical channels without deduplication,
  including labelled silent channels. Omitting selection preserves channel-0 output.
- Explicit `transcribe --evidence-version 2` adds physical channel identity, integer feature
  crop origin, applied VAD policy and successful selected-channel coverage, including channels
  with zero emitted windows. The existing observer port is preserved; direct single-channel
  library evidence has no physical-channel identity. Default version-1 evidence bytes remain
  unchanged, and RNN-T evidence remains unsupported.
- Correlated batch HTTP phase records for actual audio decoding, resampling, aggregate
  transcription, response projection and work termination through the existing bounded
  telemetry writer. HTTP deadline notification remains distinct from actual cancellation
  completion; transcript responses and terminal access records are unchanged.
- Library batch VAD methods for one channel or selected physical channels, borrowing a
  caller-owned detector at the 16 kHz model rate. Accepted speech spans are grouped within
  the recognition window while retaining original gaps, source time, crop origins in window
  evidence, and repeated words across groups; silent channels
  remain present without frontend or decoder work. VAD duration is reported separately
  from frontend time. HTTP and CLI adapters opt into this library path explicitly.
- OCI image labels (title, description, source, url, documentation, vendor, licenses, version)
  on the CPU, CUDA, and TensorRT image definitions.
- `transcribe --evidence PATH`, an opt-in per-window decode-evidence file (raw CTC labels,
  non-emission mask, and collapsed tokens alongside each window's geometry) for CTC
  transcription; the transcript is unchanged whether or not it is used.
- A `words` line in `transcribe --evidence` output, recording the token-index span the decoder's
  own word rule assigned to each decoded word, so an external reader can check its word grouping
  against the decoder's without re-implementing the rule.

### Changed

- Recognition `init_runtime` returns an owned `NativeRuntime`. Pass a reference to direct
  recognizer, encoder, and VAD constructors; retain an owner while further sessions may be
  constructed. Service facade callers provide `Some(runtime)` for lazy native VAD, or `None`
  when their synthetic capabilities do not construct VAD. Post-initialization Service errors
  unwind owned work before terminal exit.

- Standalone VAD minimum speech/silence and padding accept finite nonnegative fractional
  milliseconds, including zero, through the same checked policy as batch VAD.
- Speech segmentation now refuses incomplete or excess detector probability coverage,
  including silent output, instead of interpreting it as a complete waveform. Each
  independent 16 kHz input requires one finite probability in `[0, 1]` per 512 samples,
  including a partial final frame.

## 0.1.0 - 2026-09-03

First public release: source on GitHub and the CPU, CUDA, and TensorRT images on Docker Hub
(`deerayltd/asr:0.1.0-cpu`, `-cuda`, `-tensorrt`).

### Added

- Seven-package GigaAM v3 Runtime modular-monolith structure: gigaam-primitives,
  gigaam-model-package, gigaam-audio, gigaam-recognition, gigaam-transcription,
  gigaam-service, and gigaam-cli.
- Typed version-1 model-package parsing with required format_version, a closed key
  inventory, selected contained regular assets, and no unversioned legacy path.
- Validated audio ingestion, model-rate-bound transcription policy, finite native-output
  validation, and one Recognition-owned ONNX Runtime/provider boundary.
- Batch CTC/RNN-T workflows; CTC WebSocket streaming; revisions, stable/final frontiers,
  multichannel dialog patches, endpointing, and offline CLI commands.
- Monotonic Service admission/draining, cooperative HTTP batch cancellation, Prometheus
  metrics, structured access records, and bounded best-effort runtime trace telemetry.
- Source Dockerfile definitions for CPU, CUDA, and TensorRT asr-serve images, with an
  external model mount, selected native notices, and the exact static Rust dependency
  license/source payload.
- `ASR_ORT_THREADS`, an intra-op thread count applied to every ONNX Runtime session,
  accepting a positive 32-bit integer.
- Readiness returns 503 `{"error":"worker stopped"}` once a recognition worker has stopped
  after a decoder panic.
- An explicit 16 MiB WebSocket message and frame limit, refused with close code 1009.
- [docs/ROADMAP.md](docs/ROADMAP.md) as the public record of deferred work.

### Changed

- CUDA now verifies the exact role-specific composite CPU/CUDA assignment by default;
  `allow-unverified` explicitly collects initial assignment evidence only and does not accept
  CUDA results.
- Benchmark windows now derive their cyclic buffer lengths from the selected package sample
  rate, and decoded WAV input is resampled to that rate before direct CTC measurement.
- Raw WebSocket f32 samples use a closed normalized `[-1, 1]` wire domain; finite float
  container samples retain their decoded amplitude, including finite overshoot.
- The unconsumed legacy `model/config.json` example was removed; `model/config.kv` is the only
  tracked version-1 package configuration example.
- Corrected the public-source header set to identify GigaAM v3 Runtime.
- Public documentation now names current package, binary, API, model-package, image, and
  release boundaries.
- Container and source checks distinguish construction evidence from real-model/provider
  acceptance and publication.
- `gigaam-recognition` `Vad::from_pack` and `OrtEnvironment` now carry the optional intra-op
  thread count.

### Release status

- No GigaAM v3 Runtime repository release or Docker Hub tag is asserted as published.
- CPU, CUDA, and TensorRT images still require validation against a real GigaAM v3 model
  and, for CUDA/TensorRT, a real GPU, built from this exact source, before any tag is
  described as published.
- Pre-1.0 HTTP, WebSocket, CLI, and model-package interfaces may change.
