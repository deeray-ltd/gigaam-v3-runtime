# Contributing

Thank you for your interest in GigaAM v3 Runtime.

## Orientation

- [README.md](README.md) — what the runtime is, how to build it, and how to run it
- [docs/API.md](docs/API.md) — HTTP and WebSocket request, response, and error contracts
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — package ownership and data flow
- [docs/DECISIONS.md](docs/DECISIONS.md) — recorded design decisions and their rationale
- [docs/MODELS.md](docs/MODELS.md) — the version-1 model-package contract
- [docs/OPERATIONS.md](docs/OPERATIONS.md) — images, configuration, and deployment
- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — the behavior the runtime commits to
- [docs/ROADMAP.md](docs/ROADMAP.md) — deferred work, why it is deferred, and what unlocks it

## Building and checking a change

Service and CLI default to the CUDA feature. A CPU-only closure uses
`--no-default-features`, and TensorRT is selected explicitly through the `tensorrt` feature,
which requires CUDA support.

Run the same checks continuous integration runs:

~~~sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check --workspace --all-targets --no-default-features
~~~

The default workspace suite is fixture-free, so a fresh checkout can run it without any
external asset. Golden targets need an external model package and native binaries; compile
them without executing when those artifacts are unavailable:

~~~sh
cargo test -p gigaam-service --features golden-tests --test http_golden --no-run
~~~

Running a golden target additionally needs `ASR_GOLDEN_ARTIFACT_ROOT` set to an absolute
existing directory holding the model package and audio fixtures. Any check that touches a
model needs `ORT_DYLIB_PATH` and a compatible native ONNX Runtime library; see
[docs/MODELS.md](docs/MODELS.md). Compiling from source does not demonstrate
successful CPU, CUDA, or TensorRT inference — that is a separate acceptance boundary.

## Pull requests

Describe the observable behavior your change alters and why. Update the public documents
affected by a durable behavior, API, model, image, or requirement change in the same pull
request. Keep all tracked public content in English.

## Licensing

Contributions are accepted under the GNU LGPL version 3 or later. By contributing you
confirm that you have the right to submit the work under those terms. New Rust source files
carry the project SPDX and copyright header; copy it from a nearby source file and add the
applicable holder.

## Conduct and security

Participation is covered by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Report a
security-sensitive issue through [SECURITY.md](SECURITY.md) rather than a public issue.
