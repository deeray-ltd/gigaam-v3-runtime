# Changelog

All notable changes are recorded here. Pre-1.0 releases do not promise a stable protocol.

## Unreleased

### Added

- Opt-in `transcribe --vad` for CTC and RNN-T batch speech segmentation, with explicit
  threshold, minimum speech/silence, and padding controls. It requires a 16 kHz model,
  preserves batch channel selection and source times, and reports applied policy and VAD
  duration separately. Existing `evidence v1` is refused with VAD until it can represent
  segmented channel origins.
- Standalone `vad --channels all|0,2` selects physical channels without deduplication,
  including labelled silent channels. Omitting selection preserves channel-0 output.
- Correlated batch HTTP phase records for actual audio decoding, resampling, aggregate
  transcription, response projection and work termination through the existing bounded
  telemetry writer. HTTP deadline notification remains distinct from actual cancellation
  completion; transcript responses and terminal access records are unchanged.
- Library batch VAD methods for one channel or selected physical channels, borrowing a
  caller-owned detector at the 16 kHz model rate. Accepted speech spans preserve source
  time, crop origins in window evidence, and independent repeated words; silent channels
  remain present without frontend or decoder work. VAD duration is reported separately
  from frontend time. HTTP batch activation and its native detector lifecycle remain pending.
- OCI image labels (title, description, source, url, documentation, vendor, licenses, version)
  on the CPU, CUDA, and TensorRT image definitions.
- `transcribe --evidence PATH`, an opt-in per-window decode-evidence file (raw CTC labels,
  non-emission mask, and collapsed tokens alongside each window's geometry) for CTC
  transcription; the transcript is unchanged whether or not it is used.
- A `words` line in `transcribe --evidence` output, recording the token-index span the decoder's
  own word rule assigned to each decoded word, so an external reader can check its word grouping
  against the decoder's without re-implementing the rule.

### Changed

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
