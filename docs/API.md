# API reference

asr-serve exposes one HTTP batch endpoint, one WebSocket streaming endpoint, three probes,
and Prometheus metrics:

| Method | Route | Contract |
| --- | --- | --- |
| POST | /v1/transcribe | Complete-audio transcription |
| GET | /v1/stream | WebSocket upgrade for live audio |
| GET | /health | Loaded model families and selected provider |
| GET | /livez | Process liveness |
| GET | /readyz | Admission readiness |
| GET | /metrics | Prometheus text exposition |

Application-owned HTTP responses use application/json; charset=utf-8. Errors use the
shape {"error":"message"}. Messages are English diagnostics, not stable machine-readable
error codes. The API has no formal schema version beyond its /v1 route prefix.

For both request types, a nonempty query string is parsed as unique nonempty key=value
pairs after plus/percent decoding to UTF-8. Unknown names, duplicate names, empty values,
malformed percent escapes, non-UTF-8 decoded text, and values outside the stated domain
refuse rather than selecting a default.

## HTTP batch transcription

### POST /v1/transcribe

Send the audio file as raw request bytes. The handler does not use a request Content-Type
to select a decoder. Its configured body boundary is described in
[OPERATIONS.md](OPERATIONS.md).

| Query key | Accepted values | Default |
| --- | --- | --- |
| model | ctc; rnnt only when the service loaded RNN-T | ctc |
| ext | wav, flac, ogg, opus, mp3, aac, m4a, mp4, mkv, vorbis, pcm | source detection |
| words | 1, 0, true, false | false |
| segments | 1, 0, true, false | false |
| turns | 1, 0, true, false | false |
| turn_gap | finite seconds greater than zero | 1.0 |
| channels | split | omitted |

The body must be nonempty. Decoded source audio must have a rate in 4000 through 192000 Hz
and at least one channel. Within that range, a rate is accepted only when its ratio to the
model rate, reduced to lowest terms, stays within 1000 on both sides; other rates refuse
with an explicit message. The service resamples accepted input to the model rate. The
channel policy is selected from the output request: ordinary text is a single-output
workflow; split channels or requested turns selects the separate-channel workflow.

Every usable request gets an x-request-id response header. A supplied value must be
nonempty visible ASCII text; otherwise the service returns 400. If absent, the service
generates a header-safe identifier.

### Successful response

Every successful response contains:

| Field | Meaning |
| --- | --- |
| duration_sec | Input duration, projected to two decimal places. |
| sample_rate_in | Decoded input sample rate in Hz. |
| channels | Number of source channels. |
| dual_mono | Whether duplicate-channel classification collapsed a dual-mono source. |
| model | ctc or rnnt. |
| rtf | Processing duration divided by input duration, projected to four decimal places. |

The response has exactly one text projection: text normally, or channels_text as an array
when channels=split produces more than one channel result. words is present only when
requested; each item has start, end, text, and channel only for multichannel output.
segments is present only when requested; each item has start, end, channel, and text.
turns is present only when requested and multichannel output exists; each item has
channel, start, end, and text.

### Refusal and timeout behavior

| Status | Meaning |
| --- | --- |
| 400 | Invalid header/query/audio input, unavailable requested RNN-T, or another request-domain refusal. |
| 413 | Request body exceeds the route body limit. This is an Axum layer response; its body is not an application JSON contract. |
| 503 | Admission overload or draining, worker failure, a worker queue wait exceeding 60 seconds, or an operational transcription failure. |
| 504 | The processing deadline expired. |

A recognition call waits at most 60 seconds to start once its decoder worker queue accepts
it; that queue-wait expiry returns 503. The separate whole-request processing deadline,
ASR_REQ_TIMEOUT_SEC, returns 504 instead; see [OPERATIONS.md](OPERATIONS.md).

An HTTP deadline returning 504 requests terminal cancellation. Queued or successor windows
cannot begin after cancellation is observed; a native call already executing may finish;
its late result cannot replace the 504; and the request capacity remains reserved until
the work terminalizes. This does not claim that ONNX Runtime interrupts an in-flight native
call.

## Health, liveness, readiness, and metrics

GET /health returns a JSON object with status set to ok, a models array containing ctc and,
when loaded, rnnt, and the selected provider string cpu, cuda, or tensorrt. It is a
capability projection, not a real-model inference probe.

GET /livez returns {"status":"alive"}. GET /readyz returns {"status":"ready"} while
admission is Running and 503 {"error":"draining"} after draining begins. The Service keeps
the same admission state for readiness, HTTP acquisition, and complete pre-upgrade
WebSocket acquisition. GET /readyz also returns 503 {"error":"worker stopped"} once a
recognition worker has stopped after a decoder panic, while admission remains Running.
Ordinary decode errors fail only the request that hit them and leave readiness unchanged.
Draining takes precedence once it begins, so a draining process with a stopped worker still
reports 503 {"error":"draining"}. See [OPERATIONS.md](OPERATIONS.md).

GET /metrics returns text/plain; version=0.0.4; charset=utf-8. Its series and operational
meaning are listed in [OPERATIONS.md](OPERATIONS.md).

## WebSocket streaming

### GET /v1/stream

Open a WebSocket with a GET upgrade request. Before upgrade, the endpoint parses and
validates this query:

| Query key | Accepted values | Default |
| --- | --- | --- |
| model | ctc | ctc |
| rate | integer 8000 through 192000 Hz | required |
| fmt | pcm16, f32, alaw, ulaw | pcm16 |
| channels | integer 1 through 8 | 1 |
| emit | turns, words, both | turns |
| endpoint | blank, vad | blank |
| horizon | finite seconds greater than zero | 5.0 |
| lock | 1, 0, true, false | false |
| turn_gap | finite seconds greater than zero | 0.8 |
| dedup | 1, 0, true, false | ASR_DEDUP for multichannel input; disabled for mono input |
| backchannel_max_ms | finite milliseconds greater than or equal to zero | ASR_BACKCHANNEL_MAX_MS |

Within the accepted range, rate is admitted only when its ratio to the model rate, reduced
to lowest terms, stays within 1000 on both sides; other rates refuse with an explicit
message.

The connection reserves one streaming admission permit per requested source channel, not
per socket. It cannot reserve more than the configured streaming limit even though the
protocol maximum is eight channels. Any explicitly supplied dedup value, including false,
refuses for mono input. With multichannel input, an omitted value uses ASR_DEDUP and an
explicit accepted value selects the deduplication setting; with mono input and no dedup key,
deduplication is disabled.

Invalid query/setup input returns 400 JSON before upgrade. Overload or draining returns
503 JSON before upgrade. No model work or active session begins for either pre-upgrade
refusal.

### Frames and controls

Binary frames carry raw interleaved samples in the declared format and channel count:

~~~text
channel-0 sample, channel-1 sample, ..., channel-N sample, channel-0 sample, ...
~~~

For `fmt=f32`, every little-endian IEEE-754 sample must be finite and lie in the closed
normalized interval `[-1, 1]`.

Frame boundaries may split an interleaved sample group; the decoder retains the incomplete
group for the next binary frame. A terminal incomplete group refuses instead of being
discarded. After deinterleaving, audio is resampled while preserving stream state.

A single WebSocket message or frame larger than 16 MiB is refused with WebSocket close code
1009 (message too big), ending the session without a transcript. Send audio as a sequence of
small binary frames, as the examples above do, rather than one large frame.

The only accepted text control is {"type":"end"}, with whitespace allowed between its JSON
tokens. Ping and Pong frames are ignored by the application. A client close or transport EOF
ends the server's local session without an attempt to send an end event to an already closed
peer.

### Server events

The server sends JSON text frames:

| Type | Fields | Meaning |
| --- | --- | --- |
| words | type, at, revise_from, words | Replace the channel word tail at revise_from. Each word has start, end, text, stable. |
| stable | type, at, upto | Word order is settled through upto. |
| final | type, at, upto, endpoint | Output is immutable through upto; endpoint indicates endpoint-caused finality. |
| turns | type, revise_from, frontier, turns | Replace the dialogue tail from revise_from. Each turn has channel, k, start, end, text, stable, final, backchannel. |
| error | type, message | Terminal diagnostic before server close for an in-band application failure. |
| end | type | Successful terminal event. |

Channel is omitted from words, stable, and final in mono sessions and present in
multichannel sessions. In every emission group, channel events appear in ascending original
channel order before the optional turns patch. A final word or turn is immutable; stable
means order is settled but wording may still be adjusted. The turns frontier is the
immutable dialogue frontier.

After a valid client end or shutdown of an active session, the server flushes terminal domain
events, sends end, and closes. If shutdown reaches a connection after HTTP upgrade but before
the stream session opens, the server sends only a close frame: no transcript, end, error, or
open-session access record exists yet. A post-upgrade application error normally sends error
and then a close frame without a close code or reason.

## Offline CLI

The asr executable has exactly five commands. Each command places its required input file
immediately after the command. Unknown options, duplicate options, missing values, extra
positionals, invalid UTF-8 arguments, and invalid scalar values refuse. It has no generic
help or version command grammar.

~~~text
asr transcribe FILE [--model DIR] [--ep cpu|cuda|tensorrt] [--ort-dylib PATH]
                    [--fp16] [--rnnt] [--window S] [--overlap S] [--pad]
                    [--words] [--split-channels] [--turns] [--turn-gap S]

asr bench [--audio FILE] [--model DIR] [--ep cpu|cuda|tensorrt] [--ort-dylib PATH]
          [--fp16] [--window S] [--iters N] [--gap-ms N] [--alt-window S]

asr stream FILE [--model DIR] [--ep cpu|cuda|tensorrt] [--ort-dylib PATH]
              [--fp16] [--chunk-ms N] [--step-ms N] [--horizon S]
              [--window S] [--overlap S] [--endpoint blank|vad] [--silence-ms N]
              [--ref FILE] [--lock] [--events]

asr vad FILE [--model DIR] [--ort-dylib PATH] [--threshold F] [--neg-threshold F]
           [--min-speech-ms N] [--min-silence-ms N] [--speech-pad-ms N]

asr dialog FILE [--model DIR] [--ep cpu|cuda|tensorrt] [--ort-dylib PATH]
              [--turn-gap S] [--endpoint blank|vad] [--no-dedup]
              [--backchannel-max-ms N]
~~~

| Command | Defaults and constraints |
| --- | --- |
| transcribe | model is model; CTC is selected unless --rnnt; --window is 30 seconds; --overlap is 6 seconds and must be shorter than window; --turn-gap is 1 second; --words conflicts with --turns; --pad selects fixed-window padding. |
| bench | audio defaults to fixtures/long_example.wav, a path not bundled by the repository; window is 30 seconds; iters is 20 and positive; gap-ms is 0 and nonnegative; alt-window is optional and positive. |
| stream | chunk-ms is 100 and positive; step-ms is 500 and positive; horizon is 4 seconds and positive; window/overlap default to 30/6 seconds with overlap shorter than window; endpoint defaults to blank; silence-ms is optional and nonnegative; --lock selects committed-stable locking; --events prints event projections. |
| vad | threshold is 0.5; neg-threshold is 0.35; both are finite in 0 through 1 and neg-threshold cannot exceed threshold; min-speech-ms and min-silence-ms default to 250 and 100 and are positive; speech-pad-ms defaults to 30 and is nonnegative. |
| dialog | turn-gap is 0.8 seconds and positive; endpoint defaults to blank; --no-dedup disables offline dialog deduplication; backchannel-max-ms defaults to 0 and is nonnegative. |

`bench` opens the selected package before native runtime initialization. Its --window and
--alt-window values are durations at that package's validated sample rate: each becomes a
model-rate buffer length, and the decoded WAV input is resampled to the same rate before
warmup or measurement. The default therefore remains a 30-second duration for every valid
package rate rather than a fixed sample count.

For applicable commands, --model defaults to model. --ort-dylib or ORT_DYLIB_PATH is
required; when both are present, they must be identical. Provider selection and runtime
environment follow [OPERATIONS.md](OPERATIONS.md).

### CUDA graph-assignment verification

This is process configuration, not an HTTP or WebSocket field. With selected CUDA,
`ASR_CUDA_ASSIGNMENT_POLICY` is absent or `verified` by default and requires the exact
64-hex-character SHA-256 value for every encoder role the command or Service will construct:
`ASR_CUDA_CTC_ASSIGNMENT_SHA256` and, when RNN-T is required,
`ASR_CUDA_RNNT_ASSIGNMENT_SHA256`. Service requires CTC and also RNN-T unless started with
`--no-rnnt`; ordinary transcribe, bench, stream, and dialog require CTC; RNN-T transcribe
requires RNN-T; VAD requires neither.

`ASR_CUDA_ASSIGNMENT_POLICY=allow-unverified` is the only bypass. It conflicts with either
fingerprint setting and skips fingerprint equality only. The session still refuses an empty
assignment, an assignment without CUDA work, or any provider other than the built-in CPU and
CUDA providers. In this explicit mode Service and CLI write the role, observed SHA-256, and
CPU/CUDA node counts to stderr after session construction and before warmup or inference.
The variables are invalid for CPU and TensorRT selections; empty, unknown, malformed, and
conflicting values refuse before package/native runtime work. CUDA role admission also occurs
before the session builder; placement extraction and verified equality occur after session
construction but before it is returned. The fingerprint binds observed assignment only and does
not independently reject a model or ONNX Runtime byte change.

### Current terminal projection

The CLI writes functional results to stdout and diagnostics to stderr. Its current
human-oriented projections are:

| Command | stdout | stderr |
| --- | --- | --- |
| transcribe | Transcript text; in the text path, --split-channels adds a [channel N] heading per channel and --words adds indented start/end/text rows. --turns with multichannel output instead emits [channel N start-end] turn rows. | Stage and audio/runtime timing summaries; ASR_TRACE adds per-window timing rows. |
| bench | One window summary containing the optional alternate duration, precision, selected device, pause, frontend/encoder median timing, encoder minimum/maximum, and iteration count. | Command diagnostics on refusal. |
| stream | The final client transcript as one joined text line. | Stream statistics; --ref FILE adds a reference-comparison row with batch and stream WER, which is absent without --ref; --events prints each applied stream event; ASR_TRACE adds per-window timing rows. |
| vad | One start/end/duration row for each detected speech segment. | Aggregate speech, VAD timing, and RTF summary. |
| dialog | One [channel N start-end] row per reconstructed turn, with [bc] on marked backchannels. | Channel, active-channel, and turn-count summary. |

When a CTC vocabulary has no standalone `▁`, CTC construction emits a blank-only
silence-mask warning to stderr. `transcribe` and `stream` emit it once per constructed
decoder; `dialog` emits it once per constructed channel-local decoder. `bench` and `vad`
do not emit this warning.

Grammar refusal exits with status 2. Non-UTF-8 operating-system arguments and command or
runtime failures exit with status 1 and a textual diagnostic. These lines, status values,
and diagnostic strings are current process projections, not stable pre-1.0 byte-format or
machine-readable error-code promises.
