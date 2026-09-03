# GigaAM v3 Runtime

GigaAM v3 Runtime is a Rust speech-recognition runtime for GigaAM v3 model packages.
It prepares audio, runs a selected ONNX Runtime provider, decodes CTC or RNN-T batch
audio, and serves revisable streaming word and dialog events. Python and a deep-learning
framework are absent from the serving path; ONNX Runtime is loaded dynamically through an
explicit shared-library path.

> Status: 0.1.0-dev. Source architecture and image definitions are under release
> acceptance. No repository release or Docker Hub tag is asserted as published, and
> pre-1.0 interfaces may change.

## What it provides

- HTTP batch transcription through asr-serve.
- WebSocket CTC streaming with tail revisions, stable/final frontiers, and multichannel
  dialog turn patches.
- Offline commands through asr: transcribe, bench, stream, vad, and dialog.
- Explicit CPU, CUDA, and TensorRT provider selection with no provider fallback.
- Version-1 typed model-package validation before selected asset or native-session use.
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

This quick start is CPU-only. Default-feature Service and CLI builds select CUDA and require
role-specific verified CTC/RNN-T assignment fingerprints. Explicit
`ASR_CUDA_ASSIGNMENT_POLICY=allow-unverified` is only for collecting initial assignment evidence;
it does not accept CUDA results. See [docs/OPERATIONS.md](docs/OPERATIONS.md) for the exact
role, environment, and evidence rules.

Provider selection is strict. An explicit CPU, CUDA, or TensorRT choice must be compiled
into the executable, agree between command line and environment when both are supplied,
and initialize successfully. CTC fp32 is the default precision; fp16io32 is an explicit
choice and may change model output. Streaming on the CPU provider at the default 30-second
window can be slower than real time for one stream, depending on the host and
ASR_ORT_THREADS; measure on the target host before relying on it, and prefer a CUDA or
TensorRT provider for streaming.

## Docker source images

Dockerfile.cpu, Dockerfile.cuda, and Dockerfile.tensorrt define standalone source builds
for CPU, CUDA, and TensorRT. Each builds gigaam-service from the repository root, carries
only selected runtime libraries and notices into a non-root final image, and expects a
read-only model mount at /app/model.

~~~sh
docker buildx build --load --platform linux/amd64 \
  -f Dockerfile.cpu -t gigaam-v3-runtime:cpu .

docker run --rm -p 8080:8080 \
  --mount type=bind,src="$PWD/model",dst=/app/model,readonly \
  gigaam-v3-runtime:cpu
~~~

The intended public names are deeray/asr:0.1.0-cpu, deeray/asr:0.1.0-cuda, and
deeray/asr:0.1.0-tensorrt. They are release targets, not published-image claims. Building an
image from source is not the same as validating it: CPU and accelerator images still require
running each image against a real GigaAM v3 model and, for CUDA/TensorRT, a real GPU, before
either is described as accepted or published.

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
- [docs/MODELS.md](docs/MODELS.md) — version-1 package contract
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
