# Changelog

All notable changes are recorded here. Pre-1.0 releases do not promise a stable protocol.

## Unreleased

### Added

- OCI image labels (title, description, source, url, documentation, vendor, licenses, version)
  on the CPU, CUDA, and TensorRT image definitions.

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
