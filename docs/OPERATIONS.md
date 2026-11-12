# Operations

## Runtime prerequisites

asr-serve loads ONNX Runtime dynamically. Supply one regular ONNX Runtime library through
--ort-dylib or ORT_DYLIB_PATH; at least one is required, and both must name the same path
when present. The process validates the selected file before model/session construction.
LD_LIBRARY_PATH must make its selected sibling provider libraries visible. The service has no built-in authentication; the OpenAI-compatible SDK bearer token is ignored. Prefer binding Docker to `127.0.0.1:8080:8080` and do not expose the service publicly without an authenticated, TLS-protected ingress covering every route.

The model package is external. Its directory must contain a valid version-1 config.kv and
the artifacts selected by the process configuration. A missing or invalid package refuses
startup; the runtime does not discover a substitute package or provider.

## Service command line

~~~text
asr-serve [--model DIR] [--port 1..65535] [--ep cpu|cuda|tensorrt]
          [--ort-dylib PATH] [--ctc-fp16] [--no-rnnt] [--rnnt-fp16] [--rnnt-precision fp32|fp16]
~~~

model defaults to model and port defaults to 8080. Each option can occur once; positional
arguments and unknown options refuse. --no-rnnt conflicts with --rnnt-fp16. CTC uses fp32
unless --ctc-fp16 selects fp16io32; RNN-T is loaded unless --no-rnnt, and --rnnt-fp16
selects its fp16io32 encoder only. `--rnnt-precision fp32|fp16` or
`ASR_RNNT_PRECISION=fp32|fp16` selects all RNN-T roles; fp16 requires compatible
package assets and does not fall back to fp32. Both settings may be supplied only when
they agree. They conflict with `--rnnt-fp16` and `--no-rnnt`. When absent, the legacy
encoder choice and fp32 heads remain in effect. Full fp16 is intended for GPU providers.
Precision selection may change model output.

## Service limits and policy environment

All process environment values must be UTF-8. An absent setting receives only its documented
default; a present empty, malformed, non-finite, out-of-range, duplicate, conflicting, or
unsupported value refuses rather than falling back.

| Variable | Default | Accepted value and effect |
| --- | --- | --- |
| ASR_API_MODE | native | `native` or `openai`; selects exclusive application routes. |
| ASR_LANGUAGE | ru | Recognition language declaration; currently only `ru` is supported. |
| ASR_MAX_CONCURRENCY | 8 | Positive integer maximum for admitted HTTP batch processing. |
| ASR_MAX_UPLOADS | Resolved ASR_MAX_CONCURRENCY | Positive integer maximum for concurrent upload readers. |
| ASR_UPLOAD_TIMEOUT_SEC | Resolved ASR_REQ_TIMEOUT_SEC | Positive integer deadline for reading and validating an upload. |
| ASR_MAX_STREAMS | 32 | Positive integer maximum for admitted WebSocket source channels. |
| ASR_BODY_LIMIT_MB | 64 | Positive integer MiB limit for the complete encoded request body, including multipart framing. |
| ASR_REQ_TIMEOUT_SEC | 120 | Positive integer HTTP processing deadline in seconds. |
| ASR_DEDUP | true | 1, 0, true, false, on, or off; default streaming channel-dedup choice. |
| ASR_DEDUP_WINDOW_SEC | 4 | Positive decimal seconds, with one through nine fractional digits when fractional. |
| ASR_DEDUP_THRESHOLD | 0.99 | Finite float in 0 through 1. |
| ASR_BACKCHANNEL_MAX_MS | 0 | Nonnegative decimal milliseconds, with one through nine fractional digits when fractional. |

The service converts validated temporal policy to model samples only after opening the
typed package and validating its sample rate. Decimal conversion uses integer arithmetic
with nonnegative half-sample rounding upward; it does not rely on a floating-point text
round trip.

A recognition call waits at most 60 seconds to start once its decoder worker queue accepts
it, independent of ASR_REQ_TIMEOUT_SEC; that queue-wait expiry returns 503. The separate
processing deadline configured by ASR_REQ_TIMEOUT_SEC returns 504 instead.

## Provider, frontend, and trace environment

| Variable | Default | Accepted value and effect |
| --- | --- | --- |
| ASR_RNNT_PRECISION | Absent | `fp32` or `fp16`; selects precision for all RNN-T roles. |
| ASR_ENCODER_EP | Compiled default | cpu, cuda, or tensorrt. It must agree with --ep when both are set. |
| ORT_DYLIB_PATH | None | Required regular ONNX Runtime library unless --ort-dylib supplies the same path. |
| ASR_ORT_MEMPATTERN | 1 | 1 preserves ORT memory-pattern optimization; 0 disables it. |
| ASR_ORT_THREADS | Library default | Positive integer up to 2147483647; intra-op thread count applied to every ONNX Runtime session, including the CPU-only RNN-T prediction/joint and VAD sessions of accelerator images. Has no effect on an ONNX Runtime library built with OpenMP, which takes its thread count from OMP_NUM_THREADS. |
| ASR_ORT_ARENA | default | default or same; same is a CUDA arena policy and requires selected CUDA. |
| ASR_TRT_CACHE | None | Existing directory for selected TensorRT engine cache. |
| ASR_TRT_PROFILE_MIN | None | TensorRT shape profile minimum; all three profile variables must be supplied together or omitted together. |
| ASR_TRT_PROFILE_OPT | None | TensorRT shape profile optimum. |
| ASR_TRT_PROFILE_MAX | None | TensorRT shape profile maximum. |
| ASR_FRONTEND | scalar | scalar or batched log-mel frontend. |
| ASR_TRACE | Disabled | Absent disables; any nonempty value enables runtime window observations; an empty present value refuses. |
| ASR_CUDA_ASSIGNMENT_POLICY | verified for CUDA | Absent or `verified` verifies declared encoder-role fingerprints; `allow-unverified` is the explicit evidence-only mode. It is invalid for CPU or TensorRT. |
| ASR_CUDA_CTC_ASSIGNMENT_SHA256 | None | Required for a verified CUDA CTC role: exactly 64 hexadecimal SHA-256 characters. Invalid for CPU/TensorRT and conflicts with `allow-unverified`. |
| ASR_CUDA_RNNT_ASSIGNMENT_SHA256 | None | Required for a verified CUDA RNN-T role: exactly 64 hexadecimal SHA-256 characters. Invalid for CPU/TensorRT and conflicts with `allow-unverified`. |

ASR_FRONTEND selects the log-mel implementation; byte-for-byte reproducible output is
guaranteed within one frontend mode, not across scalar and batched. Comparing runs across
the two modes may show small numeric differences.

TensorRT profile entries name tensor shapes. Profile names/ranks/order must agree, dimensions
must be positive, and each minimum through optimum through maximum dimension must be ordered.
TensorRT cache and profiles require selected TensorRT; they are not ignored under another
provider.

| Artifact feature selection | Compiled providers | Default when no provider value is supplied |
| --- | --- | --- |
| Service with --no-default-features | CPU | cpu |
| Service with default features | CPU and CUDA | cuda |
| Service with --features tensorrt | CPU, CUDA, TensorRT | cuda |

The tensorrt feature includes CUDA support. Selecting tensorrt registers TensorRT only; it
does not silently register or select CUDA as a substitute.

Streaming on the CPU provider at the default 30-second window can be slower than real time
for one stream, depending on the host and ASR_ORT_THREADS. Measure on the target host before
relying on it, and prefer a CUDA or TensorRT provider for streaming.

## CUDA assignment verification

The Service derives CTC and, unless `--no-rnnt` is present, RNN-T as required roles. CLI
derives CTC for ordinary transcribe, bench, stream, and dialog; RNN-T for RNN-T transcribe;
and no encoder role for VAD. Process configuration requires one matching fingerprint for each
of those roles before Model Package or native-runtime work. A direct CUDA encoder repeats role
admission before its session builder. A syntactically valid fingerprint for a role that will not
be constructed is inert; no additional native session is created to consume it.

CUDA starts one CUDA execution provider with ORT graph-assignment recording enabled and leaves
built-in CPU placement available for that observation. Before a session is returned, the runtime
accepts only CPU/CUDA assignments and requires at least one CUDA node; it then compares the
canonical assignment fingerprint in verified mode. TensorRT retains disabled CPU fallback and
does not use this CUDA assignment configuration. `allow-unverified` retains every placement
validity check but skips the comparison, then writes its role, observed fingerprint, and CPU/CUDA
node counts to stderr before any warmup or inference. The fingerprint covers observed assignment,
not raw model or ONNX Runtime bytes, so a byte change alone does not independently cause refusal.

## Admission, cancellation, and lifecycle

Service admission is one monotonic state:

~~~text
Running -> Draining
~~~

On SIGTERM or SIGINT, Service enters Draining, closes HTTP and WebSocket admission, makes
GET /readyz return 503, and waits up to ten seconds for reserved WebSocket channels to
return before completing Axum graceful shutdown. Existing admitted work retains its own
terminal behavior; no new HTTP work or WebSocket warmup starts after the transition. There
is no transition back to Running.

GET /readyz also returns 503 {"error":"worker stopped"} once the dedicated CTC or, when
loaded, RNN-T recognition worker has stopped after a decoder panic, while the admission
phase remains Running. Ordinary decode errors fail only the request that hit them and leave
readiness unchanged. A stopped worker fails every later call; restart the process rather
than waiting for recovery. Draining takes precedence: a draining process with a stopped
worker still returns 503 {"error":"draining"}, not a worker-stopped reason.

HTTP capacity is held through terminal acknowledgement even when the client has already
received 504. WebSocket capacity is acquired atomically for all requested source channels.
The service has no built-in TLS, authentication, authorization, CORS policy, client rate
limit, or persistent transcript store; operate it behind an independently configured
ingress when those controls are required.

## Docker source builds

Dockerfile.cpu, Dockerfile.cuda, and Dockerfile.tensorrt are independent Linux/amd64
source builds from the repository root. Their limited context contains the workspace
manifest/lock, all crate sources, project licenses, and THIRD_PARTY.md; it excludes
documentation, model packages, and build output.

| File | Service build | Final native payload |
| --- | --- | --- |
| Dockerfile.cpu | gigaam-service with --no-default-features | Ubuntu 24.04 and selected ONNX Runtime 1.29.0 CPU libraries. |
| Dockerfile.cuda | gigaam-service with --no-default-features --features cuda | NVIDIA CUDA 13.0.2 cuDNN runtime and selected ONNX Runtime CUDA provider. |
| Dockerfile.tensorrt | gigaam-service with --no-default-features --features tensorrt | The same NVIDIA CUDA/cuDNN runtime, selected ONNX Runtime TensorRT provider, and selected TensorRT 10.13.3.9 wheel libraries. |

Each source build uses pinned selected inputs, strips the final asr-serve binary, copies
the relevant ONNX Runtime and TensorRT notices, runs as uid/gid 10001 named gigaam, exposes
8080/tcp, and enters:

~~~text
/app/asr-serve --model /app/model --port 8080
~~~

The final images contain no model directory, repository source, Cargo state, Rust toolchain,
or Python runtime. Mount a valid model read-only:

~~~sh
docker buildx build --load --platform linux/amd64 -f Dockerfile.cpu -t gigaam-v3-runtime:cpu .
docker run --rm -p 8080:8080 \
  --mount type=bind,src="$PWD/model",dst=/app/model,readonly gigaam-v3-runtime:cpu
~~~

CUDA and TensorRT additionally need the host NVIDIA runtime and a supported GPU platform.
TensorRT creates a writable /app/trt-cache owned by the non-root image user. No Dockerfile
performs its own test suite: CI owns Rust checks, external-golden target compilation, the
Docker build-context allowlist, and Dockerfile construction policy; the images themselves are
built and accepted outside CI. Those checks do not prove real model or accelerator inference.

The published release images are deerayltd/asr:0.1.0-cpu, deerayltd/asr:0.1.0-cuda, and
deerayltd/asr:0.1.0-tensorrt on Docker Hub; their digests are listed in the README. Pull by
digest when an exact artifact matters.

## Metrics

GET /metrics renders Prometheus text directly. The fixed counters are:

- asr_transcribe_requests_total
- asr_transcribe_errors_total
- asr_overload_total
- asr_timeout_total
- asr_ws_sessions_total
- asr_ws_rejected_total
- asr_ws_frames_total
- asr_ws_channels_total
- asr_ws_turn_patches_total
- asr_dedup_collapsed_total
- asr_ws_errors_total

Telemetry adds:

- asr_telemetry_dropped_total with reason="queue_full" or reason="sink_closed"
- asr_telemetry_write_failures_total with destination="stdout" or destination="stderr"

The gauges are asr_active_transcribe, asr_active_streams, asr_max_transcribe,
asr_max_channels, asr_worker_pending with worker="ctc" and, when RNN-T is loaded,
worker="rnnt", and asr_build_info with version. asr_active_streams measures reserved
WebSocket channel capacity, not an exact socket count.

The histograms are asr_transcribe_latency_seconds, asr_frontend_seconds,
asr_encoder_seconds, asr_decode_seconds, and asr_vad_seconds. Each has cumulative buckets at 0.01, 0.025,
0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, and +Inf; plus _sum, _count, and
_rejected_total with reason="non_finite" or reason="negative". Histograms accept only
finite nonnegative observations. A finite aggregate that overflows is exposed as +Inf,
not rewritten into a finite value.

## Access and trace telemetry

Admitted terminal HTTP histories emit JSON Lines access records to stdout. WebSocket
histories emit ws_open, ws_close, and ws_error records. Access records carry ts_ms when the
system clock can be represented as a JSON integer; otherwise they carry timestamp_error
instead of a synthetic timestamp. ASR_TRACE window observations are text records sent to
stderr through the same runtime telemetry lineage.

Admitted batch HTTP requests also emit `transcribe_phase` JSON Lines through the same
bounded writer. These records carry the existing `req_id`, `elapsed_ms` since handler
entry, and typed `phase` and `state` values:

| Phase | States | Observed boundary |
| --- | --- | --- |
| `admission` | `admitted` | The request owns HTTP admission capacity. |
| `work` | `started`, `terminal` | The blocking work starts, then actually finishes or unwinds. |
| `audio_decode` | `started`, `completed`, `failed`, `interrupted` | Encoded-audio validation and file decoding. |
| `resample` | `started`, `completed`, `failed`, `interrupted` | Conversion of decoded audio to the model sample rate. |
| `vad_initialization` | `started`, `completed`, `failed`, `interrupted` | Enabled request-owned native VAD session construction and the following cancellation checkpoint. |
| `vad_detection` | `started`, `completed`, `failed`, `interrupted` | One actual detector call for a selected physical channel and the following cancellation checkpoint. |
| `batch_transcription` | `started`, `completed`, `failed`, `interrupted` | The aggregate batch workflow, including frontend work, scheduler waiting, native recognition and stitching. |
| `response_projection` | `started`, `completed`, `failed`, `interrupted` | Transcript response construction and serialization. |
| `deadline` | `elapsed` | The HTTP deadline requests cancellation; this is not work completion. |

Finished stage records include `phase_ms`. `interrupted` means the stage unwound before
returning a result. Work-terminal and deadline records include the actually observed
`execution_state`; a deadline racing with completion may observe a terminal state already.
The terminal work record is emitted after the request owner acknowledges completion,
failure or cancellation and releases admission. The established `transcribe` access record
and version 1 transcript response remain unchanged. A 504 access record can precede the actual
work-terminal record while blocking/native work still owns admission.

These phases start after the body has been fully buffered and admission has succeeded.
They do not measure upload/body waiting, pre-admission refusals, or an individual native
queue wait. `batch_transcription` does not distinguish queue waiting from native execution;
the worker-pending gauge and aggregate timing histograms remain additional observations.
Enabled batch VAD adds initialization before `batch_transcription`, and detection phases
inside it. These events measure actual calls, not an inferred model-worker queue. The
`asr_vad_seconds` histogram records completed HTTP batch detection and segmentation time,
excluding detector construction; failed/cancelled batches publish no aggregate VAD sample.
No VAD phases or histogram samples are added for disabled requests. Initialization time is
visible in its phase record and included in enabled request RTF. One session is constructed
inside each admitted blocking request, reused sequentially, and dropped before admission
is released, including after the caller has received a timeout.
An accepted stage-start record can identify unfinished work while no later record is
available, but a missing finish is not proof of a hang: telemetry is best-effort and records
can be dropped. Check the telemetry drop and sink-failure counters when interpreting gaps.

The runtime writer has checked logical capacity:

~~~text
ASR_MAX_CONCURRENCY + 3 * ASR_MAX_STREAMS
~~~

It preserves FIFO queue order for accepted records, drops the newest record when full, and
records sink-closed or destination write failures in the metric families above. It is
best-effort: no delivery is promised after a sink failure. Startup, lifecycle, and fatal
diagnostics use direct stderr attempts outside the writer, and shutdown does not wait for a
potentially blocking telemetry join.
