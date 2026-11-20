# Third-party components

GigaAM v3 Runtime is licensed under the GNU Lesser General Public License, version 3 or
later; see [COPYING.LESSER](COPYING.LESSER). The components below remain under their own
terms. This inventory describes the current source and container inputs; it is not legal
advice, and each component's distributed notice and license remain authoritative.

## Native runtime payloads

| Component | Current role and acquisition | Notice or terms |
| --- | --- | --- |
| ONNX Runtime 1.32.0 | Loaded dynamically by Recognition through C API 24. The CPU image extracts the native core and shared provider from the official Linux x86-64 wheel. The CUDA and TensorRT images extract their libraries from `Microsoft.ML.OnnxRuntime.Gpu.Linux`. The TensorRT image includes the CUDA provider DSO for native linking and registers TensorRT for inference. | MIT. Each image includes the upstream LICENSE and ThirdPartyNotices.txt for its distribution. |
| NVIDIA CUDA and cuDNN | The CUDA and TensorRT images inherit the pinned NVIDIA CUDA 13.5.0 runtime with cuDNN 9.28.0.42 for Ubuntu 24.04. | NVIDIA-distributed terms and notices in the base image. |
| NVIDIA TensorRT 10.16.1.11 | The TensorRT image extracts the engine runtime, plugin library, and ONNX parser from the matching NVIDIA Ubuntu 24.04 packages. | NVIDIA-distributed terms and per-package documentation under the TensorRT license directory in the image. |
| GigaAM model artifacts, vocabularies, and VAD graph | External model-package inputs selected at runtime. | Upstream terms for the exact artifact; this repository grants no rights to them. |

The three Dockerfiles are source builds. They pin their selected base images, Rust bootstrap,
ONNX Runtime archives, and the selected TensorRT packages with checksum verification.
They copy the runtime binary, selected native libraries, this file, project licenses, and
required native notices into their final images. They do not use a host-prepared native
payload, bundle a model package, or include Python, source, Cargo state, or build state in
the final runtime filesystem.

This file is itself copied into every final image. A local image built from an earlier
revision is evidence only for the inputs it actually used; it is not a release candidate
and does not establish a published image tag.

## Rust dependency inventory

The locked Cargo metadata currently resolves seven workspace packages and 120 registry
packages when all features are considered. Direct registry dependencies are:

| Crate | Workspace owner | Role | License expression in metadata |
| --- | --- | --- | --- |
| ort | gigaam-recognition | Dynamic ONNX Runtime binding with the API-24 graph-assignment surface; its transitive ort-sys binding follows the same expression. | MIT OR Apache-2.0 |
| sha2 | gigaam-recognition | Canonical CUDA graph-assignment SHA-256 binding. | MIT OR Apache-2.0 |
| opus-decoder | gigaam-audio | Optional Opus decoding. | MIT OR Apache-2.0 |
| symphonia | gigaam-audio | Optional MP3, FLAC, Vorbis, AAC, ISO-BMFF, Ogg, WAV, PCM, and Matroska decoding. | MPL-2.0 |
| axum | gigaam-service | HTTP and WebSocket routing. | MIT |
| tokio | gigaam-service | Async runtime, networking, signals, synchronization, and timers. | MIT |
| tungstenite | gigaam-service | WebSocket close-frame and capacity-error types shared with axum's WebSocket support. | MIT OR Apache-2.0 |

The metadata license expressions for the 120 registry packages are:

| License expression | Packages |
| --- | ---: |
| MIT OR Apache-2.0 | 61 |
| MIT | 21 |
| MPL-2.0 | 13 |
| Apache-2.0 OR MIT | 6 |
| MIT/Apache-2.0 | 5 |
| Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | 3 |
| BSD-2-Clause OR Apache-2.0 OR MIT | 2 |
| (Apache-2.0 OR MIT) AND BSD-3-Clause | 1 |
| (MIT OR Apache-2.0) AND Unicode-3.0 | 1 |
| Apache-2.0 | 1 |
| Apache-2.0 OR BSL-1.0 | 1 |
| ISC | 1 |
| MIT AND BSD-3-Clause | 1 |
| MIT OR Apache-2.0 OR LGPL-2.1-or-later | 1 |
| Unlicense OR MIT | 1 |
| Zlib OR Apache-2.0 OR MIT | 1 |

License expressions are package metadata, not a claim that all alternatives have the same
redistribution conditions. In particular, the 13 Symphonia packages are MPL-2.0, and the
r-efi package offers an LGPL-2.1-or-later alternative alongside MIT and Apache-2.0.

## Static Rust dependency payload

Every final CPU, CUDA, and TensorRT image copies the generated payload at
`/usr/share/licenses/gigaam-v3-runtime/rust`. It describes the exact non-dev Rust registry
closure statically linked into `gigaam-service` for `x86_64-unknown-linux-gnu`. The CPU,
CUDA, and TensorRT service configurations currently resolve to the same 101-package static
closure; packages whose only library target is a procedural macro are excluded because they
run during compilation and are not linked into `asr-serve`. Project workspace code remains
covered by the parent `COPYING` and `COPYING.LESSER` files.

`scripts/generate_rust_notice_payload.py` derives
`crates/service/licenses/rust` from the locked Cargo metadata, Cargo.lock checksums, and the
matching cached crates.io archives. It refuses a changed provider closure, a missing archive,
or an archive whose SHA-256 differs from Cargo.lock. The generated `NOTICE.md` records each
crate's locked version, metadata license expression, declared authors and repository when
present, checksum-bound crates.io source archive, and every top-level packaged
`LICENSE*`, `COPYING*`, `NOTICE*`, or `COPYRIGHT*` file. Notice texts are copied byte-for-byte
into content-addressed files beneath `texts/`; the manifest maps each package notice to its
SHA-256 text.

The Symphonia MPL-2.0 crates and `opus-decoder` 0.1.1 contain no standalone license or notice
file in their locked crate archives. Their generated records retain the metadata expression and
the exact checksum-bound crates.io source archive. Every MPL-2.0 record also links the canonical
[Mozilla Public License 2.0 terms](https://www.mozilla.org/MPL/2.0/) and identifies the archive
as its source-acquisition location.
CI compares the checked-in payload with those inputs and verifies that each final image carries
the payload; image construction remains the release-artifact witness.

## Attribution

The GPLv3 and LGPLv3 texts in [COPYING](COPYING) and
[COPYING.LESSER](COPYING.LESSER) are published by the Free Software Foundation and are
included verbatim.

The G.711 A-law and mu-law conversion is maintained in
[crates/audio/src/g711.rs](crates/audio/src/g711.rs). It is a port of the Sun
Microsystems g711.c reference implementation; see the associated
[Oracle notice](https://docs.oracle.com/cd/E50381_01/doc/sr_scx640_license.pdf).
