# GigaAM v3 Runtime

[![Latest release](https://img.shields.io/github/v/release/deeray-ltd/gigaam-v3-runtime)](https://github.com/deeray-ltd/gigaam-v3-runtime/releases/latest)
[![CI](https://github.com/deeray-ltd/gigaam-v3-runtime/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/deeray-ltd/gigaam-v3-runtime/actions/workflows/ci.yml?query=branch%3Amain)

GigaAM v3 Runtime is a Rust speech-recognition runtime for GigaAM v3 model packages.
It prepares audio, runs a selected ONNX Runtime provider, decodes CTC or RNN-T batch
audio, and serves revisable streaming word and dialog events. Python and a deep-learning
framework are absent from the serving path; ONNX Runtime is loaded dynamically through an
explicit shared-library path.

See the [latest release](https://github.com/deeray-ltd/gigaam-v3-runtime/releases/latest)
for its version, image digests and validation boundaries. This README describes the current
branch; use the documentation at a release tag when deploying that release. Pre-1.0 interfaces
may change.

## What it provides

- HTTP batch transcription through asr-serve, including an OpenAI-compatible transcription/model-list subset.
- WebSocket CTC streaming with tail revisions, stable/final frontiers, and multichannel
  dialog turn patches.
- Offline commands through asr: transcribe, bench, stream, vad, and dialog.
- CPU and CUDA execution, plus TensorRT loading of engines prepared offline.
- RNN-T encoder, prediction and joint use the selected provider, with explicit FP32/FP16 selection.
- Typed V1/V2 model-package validation before selected asset or native-session use.
- Bounded Service admission, cooperative HTTP batch cancellation, Prometheus metrics, and
  bounded best-effort runtime access/trace telemetry.

## Package structure

The repository is one modular monolith with seven workspace packages:

| Package | Responsibility |
| --- | --- |
| gigaam-primitives | Dependency-free checked numeric conversions. |
| gigaam-model-package | Versioned package schema, compatibility, and selected asset validation. |
| gigaam-audio | Audio decoding, resampling, channel analysis, and log-mel frontend. |
| gigaam-recognition | Decoders, native-output validation, ONNX Runtime/provider assembly, and scheduling. |
| gigaam-transcription | Batch/stream workflows, revisions, endpointing, channel policy, and dialog reconstruction. |
| gigaam-service | HTTP/WebSocket process, admission, lifecycle, metrics, and telemetry; builds as asr-serve. |
| gigaam-cli | Offline adapter; builds as asr. |

The exact dependency graph and logical ownership are described in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Prerequisites

- Rust 1.98.0 with a C toolchain.
- A valid external GigaAM v3 model package; see [docs/MODELS.md](docs/MODELS.md).
- A regular ONNX Runtime library passed through --ort-dylib or ORT_DYLIB_PATH.
- CUDA/cuDNN and the selected provider libraries for GPU operation.

Weights, vocabularies, frontend tables, and VAD artifacts are not bundled. Their exact
terms remain external to this repository.

## Quick start

~~~sh
cargo build --release --package gigaam-cli --no-default-features --features decoders
cargo build --release --package gigaam-service --no-default-features

export ORT_DYLIB_PATH=/path/to/libonnxruntime.so.1.29.0
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib

./target/release/asr transcribe audio.wav --model model --words
./target/release/asr-serve --model model --port 8080

curl --data-binary @audio.wav \
  'http://127.0.0.1:8080/v1/transcribe?model=ctc&words=true'
~~~

The default `ASR_API_MODE=native` serves native HTTP and WebSocket routes. Set
`ASR_API_MODE=openai` to accept OpenAI Python SDK transcription requests at
`/v1/audio/transcriptions` instead of the native API (supported model IDs and fields are
documented in [docs/API.md](docs/API.md)). The SDK bearer token is not authenticated; bind
locally or place the service behind a trusted ingress. This mode supports file transcription
and model listing; OpenAI-compatible SSE and Realtime APIs are not implemented.

This quick start is CPU-only. Default-feature Service and CLI builds select CUDA and require
role-specific verified CTC/RNN-T assignment fingerprints. Explicit
`ASR_CUDA_ASSIGNMENT_POLICY=allow-unverified` is only for collecting initial assignment evidence;
it does not accept CUDA results. See [docs/OPERATIONS.md](docs/OPERATIONS.md) for the exact
role, environment, and evidence rules.

Provider selection is strict: the provider must be compiled into the executable, agree
between command line and environment when both are supplied, and initialize successfully.
CUDA encoder placement follows the existing assignment policy and may include CPU graph
nodes; RNN-T GPU heads disable CPU fallback. TensorRT uses prepared EPContext artifacts,
not serving-time compilation of ordinary ONNX models.

| Precision choice | Behavior |
| --- | --- |
| Default | FP32 encoder and RNN-T heads; CPU execution remains available. |
| Service `--ctc-fp16` or CTC CLI `--fp16` | FP16Io32 encoder: FP16 model with FP32 tensor I/O. |
| Service `--rnnt-fp16` or `asr transcribe --rnnt --fp16` | Encoder-only FP16Io32; RNN-T heads stay FP32. |
| `--rnnt-precision fp16` or `ASR_RNNT_PRECISION=fp16` | FP16 RNN-T heads on GPU; V2 uses native FP16 encoder I/O, V1 uses the explicit FP16Io32 boundary conversion. |

The all-role RNN-T choice also accepts `fp32`, requires compatible assets and conflicts
with the legacy encoder-only flag. CLI/environment choices must agree. It applies to
service RNN-T and offline `asr transcribe --rnnt`, not CTC-only commands. Missing FP16
assets refuse rather than silently substituting FP32. Native-FP16 CTC artifacts are
supported by the library but are not selectable through the current service/CLI flags.
See [API](docs/API.md) and [operations](docs/OPERATIONS.md) for exact configuration rules.

Streaming on the CPU provider at the default 30-second window can be slower than real time
for one stream, depending on the host and `ASR_ORT_THREADS`; measure on the target host
before relying on it, and prefer a CUDA or TensorRT provider for streaming.

## Docker images

Published images are on Docker Hub at [deerayltd/asr](https://hub.docker.com/r/deerayltd/asr).
They contain no model. Choose a released version and pin its provider tag or digest from
[the release notes](https://github.com/deeray-ltd/gigaam-v3-runtime/releases/latest).
Tags follow `deerayltd/asr:<version>-cpu`, `-cuda` and `-tensorrt`; the dynamic release link
is for discovery, not an instruction to automatically upgrade a running service.
Mount a compatible model package read-only at `/app/model`:

Replace `<released-version>` below with the selected version (without the `v` prefix).

~~~sh
ASR_VERSION='<released-version>'
docker pull "deerayltd/asr:${ASR_VERSION}-cpu"

docker run --rm -p 127.0.0.1:8080:8080 \
  --mount type=bind,src="$PWD/model",dst=/app/model,readonly \
  "deerayltd/asr:${ASR_VERSION}-cpu"
~~~

The CUDA and TensorRT images additionally need `--gpus all`; the CUDA image also needs the
verified encoder placement fingerprints described in [docs/OPERATIONS.md](docs/OPERATIONS.md).
CPU/CUDA accept V1 ONNX packages and compatible V2 ONNX packages. TensorRT requires V2
`artifact_format=trt_ep_context` with engines prepared for the selected runtime, hardware
and shapes. Prepare the artifacts offline with `scripts/prepare_gpu_model.py`; see
[model preparation](docs/MODELS.md#version-2-gpu-artifacts).

### Building from source

Dockerfile.cpu, Dockerfile.cuda, and Dockerfile.tensorrt define standalone source builds
for CPU, CUDA, and TensorRT. Each builds gigaam-service from the repository root, carries
only selected runtime libraries and notices into a non-root final image, and expects a
read-only model mount at /app/model.

~~~sh
docker buildx build --load --platform linux/amd64 \
  -f Dockerfile.cpu -t gigaam-v3-runtime:cpu .

docker run --rm -p 127.0.0.1:8080:8080 \
  --mount type=bind,src="$PWD/model",dst=/app/model,readonly \
  gigaam-v3-runtime:cpu
~~~

Building an image from source is not the same as validating it: an image built locally must
still be run against a real model package, and a real GPU for the accelerator variants, before
it is described as accepted.

## Validation

~~~sh
cargo test --workspace
cargo check --workspace --all-targets --no-default-features
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets --all-features --no-run
~~~

The normal workspace suite is fixture-free. Golden tests require externally supplied model
and binary artifacts; compile their targets without executing them when those artifacts
are unavailable. Build or source checks alone do not prove successful inference for a
particular model, native library, GPU, or deployment.

## Documentation

- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — canonical product and engineering requirements
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — capability ownership and data flow
- [docs/DECISIONS.md](docs/DECISIONS.md) — design choices and costs
- [docs/MODELS.md](docs/MODELS.md) — V1/V2 package contracts and offline engine preparation
- [docs/API.md](docs/API.md) — HTTP, WebSocket, and CLI contracts
- [docs/OPERATIONS.md](docs/OPERATIONS.md) — configuration, telemetry, lifecycle, and images
- [docs/ROADMAP.md](docs/ROADMAP.md) — deferred work, why it is deferred, and what unlocks it
- [THIRD_PARTY.md](THIRD_PARTY.md) — dependency and native-payload notices
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor guidance

## License

Copyright (C) 2026 [Yuriy Krasilnikov](https://github.com/YuriyKrasilnikov) and Deeray Ltd.

GigaAM v3 Runtime is free software under the GNU Lesser General Public License, version 3
or later. Read [COPYING.LESSER](COPYING.LESSER) and [COPYING](COPYING) for the complete
terms. Third-party terms are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
