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

## SentencePiece model schema

The Model Package reader uses Rust types generated from Google's SentencePiece
v0.2.3 protobuf schema. The schema and the generated types are covered by the
[Apache License 2.0](crates/model-package/licenses/sentencepiece/LICENSE).
The license and copyright notice ship in the SentencePiece directory of all
service images.

## Rust dependency inventory

Selected direct registry dependencies are listed below. The generated service notices carry
the locked package versions, license expressions and redistributed texts.

| Crate | Workspace owner | Role | License expression in metadata |
| --- | --- | --- | --- |
| ort | gigaam-recognition | Dynamic ONNX Runtime binding with the API-24 graph-assignment surface; its transitive ort-sys binding follows the same expression. | MIT OR Apache-2.0 |
| sha2 | gigaam-recognition | Canonical CUDA graph-assignment SHA-256 binding. | MIT OR Apache-2.0 |
| prost | gigaam-model-package | Decodes the existing SentencePiece model protobuf format. | Apache-2.0 |
| opus-decoder | gigaam-format-codecs | Optional Opus decoding. | MIT OR Apache-2.0 |
| symphonia | gigaam-format-codecs | Optional MP3, FLAC, Vorbis, AAC, ISO-BMFF, Ogg, WAV, PCM, and Matroska decoding. | MPL-2.0 |
| axum | gigaam-service | HTTP and WebSocket routing. | MIT |
| tokio | gigaam-service | Async runtime, networking, signals, synchronization, and timers. | MIT |
| tungstenite | gigaam-service | WebSocket close-frame and capacity-error types shared with axum's WebSocket support. | MIT OR Apache-2.0 |

## Static Rust dependency payload

Every final CPU, CUDA, and TensorRT image copies the generated payload at
`/usr/share/licenses/gigaam-v3-runtime/rust`. It lists the non-dev Rust registry dependencies discovered from the locked Cargo
metadata for `gigaam-service` on `x86_64-unknown-linux-gnu`. The generator checks that the
CPU, CUDA and TensorRT inventories agree; packages whose only library target is a procedural macro are excluded because they
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

Where a locked registry crate has no standalone license or notice, its generated record
retains the metadata expression and exact checksum-bound crates.io source archive. Every MPL-2.0 record also links the canonical
[Mozilla Public License 2.0 terms](https://www.mozilla.org/MPL/2.0/) and identifies the archive
as its source-acquisition location.
CI compares the checked-in payload with those inputs. The Dockerfiles copy the payload to each
final image. Inspect the built images to verify the shipped files.

## Modified Symphonia demuxers

The ISO-BMFF and Matroska format crates use local copies of version 0.6.1 with a
sequential batch read path. Their MPL-2.0 license and modified source accompany
the corresponding GigaAM v3 Runtime checkout. See the exact upstream revision and
archive in the [ISO-BMFF source record](crates/third-party/symphonia-format-isomp4/SOURCE.md)
and [Matroska source record](crates/third-party/symphonia-format-mkv/SOURCE.md).
Each service image includes the license and source record for both demuxers under
`/usr/share/licenses/gigaam-v3-runtime`. The modified source is available under
`crates/third-party` in the repository revision used to build the image.

## Attribution

The GPLv3 and LGPLv3 texts in [COPYING](COPYING) and
[COPYING.LESSER](COPYING.LESSER) are published by the Free Software Foundation and are
included verbatim.

The G.711 A-law and mu-law conversion is maintained in
[crates/audio/src/g711.rs](crates/audio/src/g711.rs). It is a port of the Sun
Microsystems g711.c reference implementation; see the associated
[Oracle notice](https://docs.oracle.com/cd/E50381_01/doc/sr_scx640_license.pdf).
