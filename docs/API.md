# API reference

The `ASR_API_MODE` setting selects one batch API. Native mode (the default) exposes the
following routes. OpenAI mode replaces the native batch and WebSocket routes with the
OpenAI-compatible batch and model-list routes described below. Probes and metrics remain
available in both modes:

| Method | Route | Contract |
| --- | --- | --- |
| POST | /v1/transcribe | Complete-audio transcription |
| GET | /v1/stream | WebSocket upgrade for live audio |
| GET | /health | Loaded model families and selected provider |
| GET | /livez | Process liveness |
| GET | /readyz | Admission readiness |
| GET | /metrics | Prometheus text exposition |

Application-owned HTTP responses use application/json; charset=utf-8. Legacy errors use
the shape {"error":"message"}; these English diagnostic messages are not stable error
codes. HTTP batch requests can explicitly select response version 2, including typed
post-binding errors, while the route prefix and other endpoint contracts remain unchanged.

For both request types, a nonempty query string is parsed as unique nonempty key=value
pairs after plus/percent decoding to UTF-8. Unknown names, duplicate names, empty values,
malformed percent escapes, non-UTF-8 decoded text, and values outside the stated domain
refuse rather than selecting a default.

## OpenAI-compatible batch transcription

Set `ASR_API_MODE=openai` to expose `POST /v1/audio/transcriptions` and `GET /v1/models`.
Native `/v1/transcribe` and `/v1/stream` are unavailable in this mode. The model list
contains the exact loaded variant and its matching family aliases. CTC exposes `gigaam-v3`
and `gigaam-v3-ctc`; RNN-T exposes `gigaam-v3-rnnt`. `gigaam-v3` always means CTC and
refuses on an RNN-T instance. An explicit other variant or family refuses.

Send multipart form data with `file` and `model`. The optional `response_format` is
`json` (default), `text`, or `verbose_json`; `language` must be `ru` when supplied and
`temperature` must be zero. `timestamp_granularities[]` accepts `word` and `segment`
only with `verbose_json`; absent granularities default to segments. Streaming and nonempty
prompts are not supported; an empty `prompt` is accepted. Requests may contain at most
16 multipart parts, with each non-file field limited to 16 KiB. The configured request
body limit also applies to the complete encoded multipart body. Unknown, repeated,
unsupported, or conflicting fields refuse. The
six VAD fields in the native table below are also accepted as multipart fields with the
same defaults and constraints; no `response_version` field is used in this API.
Multichannel files are averaged to mono before recognition.

Verbose language is `russian`, the declared model language rather than a detection.
Segment tokens and average log probability come from the retained native greedy decoder
emissions. `seek` identifies the owning recognition window in centiseconds; segment
compression ratio uses UTF-8 bytes and zlib DEFLATE level 6. `no_speech_prob` is one
minus the maximum VAD speech observation over the owning window's real source coverage.
These scores are native estimates and are not calibrated Whisper confidence values.
Segment output requires a compatible VAD asset; ordinary formats and word-only verbose
output do not require the diagnostic detector unless VAD is enabled.

## HTTP batch transcription

### POST /v1/transcribe

Send the audio file as raw request bytes. The handler does not use a request Content-Type
to select a decoder. Its configured body boundary is described in
[OPERATIONS.md](OPERATIONS.md).

| Query key | Accepted values | Default |
| --- | --- | --- |
| model | selected exact V3 identity or matching `ctc` / `rnnt` family | loaded variant |
| ext | wav, flac, ogg, opus, mp3, aac, m4a, mp4, mkv, vorbis, pcm | source detection |
| words | 1, 0, true, false | false |
| segments | 1, 0, true, false | false |
| turns | 1, 0, true, false | false |
| turn_gap | finite seconds greater than zero | 1.0 |
| channels | split | omitted |
| response_version | 1 or 2 | 1 |
| vad | true or false; requires explicit response_version=2 | false |
| vad_threshold | Finite probability in [0, 1]; requires vad=true | 0.5 |
| vad_neg_threshold | Finite probability in [0, vad_threshold]; requires vad=true | 0.35 |
| vad_min_speech_ms | Finite nonnegative milliseconds; requires vad=true | 250 |
| vad_min_silence_ms | Finite nonnegative milliseconds; requires vad=true | 100 |
| vad_speech_pad_ms | Finite nonnegative one-sided padding milliseconds; requires vad=true | 30 |

Every VAD query key, including `vad=false`, requires explicit `response_version=2`.
Tuning keys require `vad=true`; disabled tuning is refused rather than ignored. Durations
accept zero and fractional milliseconds when their sample counts, including doubled padding,
are representable. Thresholds and durations bind to the same validated policy as CLI batch
VAD. The enabled detector requires a 16000 Hz model/frontend; other supported source rates
are resampled before detection. VAD-off requests do not select or construct a VAD asset.

Enabled VAD processes each selected physical channel independently, after the existing
channel-selection and dual-mono policy. Silent selected channels remain present. Accepted
speech spans are grouped greedily when the envelope from the first span's start through the
next span's end fits the configured recognition window. Oversized individual spans remain
separate and use the existing windowing. Recognition uses original samples within each group,
including gaps, so speech rejected by the detector in a gap can still be recognized. Windows
are stitched only within groups, and words retain the original recording clock. The existing
minimum frontend window geometry still applies to very short groups. One native detector belongs to the admitted request and is
reused sequentially across its selected channels; no shared VAD worker or session pool is
introduced.

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

Version 1 preserves the established response body. Explicit version 2 adds
`schema_version: 2`, the exact loaded `variant`, and `applied.vad`. The existing `model`
field remains the family (`ctc` or `rnnt`). With VAD disabled that object is exactly
`{"enabled":false}`. With VAD enabled it contains `enabled:true`, `sample_rate:16000`,
`threshold`, `neg_threshold`, `min_speech_ms`, `min_silence_ms`, and `speech_pad_ms` from
the validated policy actually used by the batch call. Policy values preserve the bound
32-bit float value; they do not use word-timestamp rounding. They can therefore display
more decimal digits than the query text.

The `rtf` interval excludes audio ingest, resampling, and response projection. With VAD
enabled it starts immediately before request-owned detector construction and includes
construction, detection, segmentation, recognition, and stitching. The disabled interval
is unchanged.

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

Recognition library callers can pass an `ExecutionControl` to
`WindowDecoder::decode_in_context` to request cooperative cancellation. Controls cloned from
one request share identity; a scheduled decoder refuses a different control before enqueueing.
Legacy `WindowDecoder` implementations remain usable through before/after checks, and boxed
decoders forward legacy and contextual calls. Direct native legacy decode calls create their
own control, while scheduled legacy calls use the control bound when their channel was created.
The request owner retains terminal completion responsibility. Cancellation does not interrupt an
in-flight native or whole VAD call or establish a bounded response time.

After a query has completely bound to version 2, application errors use
`{"schema_version":2,"error":{"code":"...","message":"..."}}` and retain the
request-ID header:

| Status | Code | Boundary |
| --- | --- | --- |
| 400 | invalid_request | Input/audio/resampling/channel-selection refusal. This broad category is not proof of a permanently invalid file. |
| 400 | unsupported_capability | Requested model not loaded, enabled VAD model rate unsupported, or selected VAD asset missing/invalid at package selection. |
| 503 | unavailable | Native initialization/execution or worker failure. A regular file that fails native graph construction is an operational failure. |
| 503 | overloaded | Admission capacity unavailable. |
| 503 | draining | Admission closed for shutdown. |
| 504 | timeout | Whole-request processing deadline elapsed. |

Request-ID validation, empty-body rejection, and all query-binding failures retain the
legacy `{"error":"..."}` envelope, even when query text contains `response_version=2`.
This includes invalid versions, duplicate keys, and invalid VAD policy. Version 1 errors
and the layer-level 413 boundary remain unchanged. An executing VAD constructor or detector
is cooperatively cancelled before/after its native call; its session and HTTP capacity stay
owned until that call returns and the request actually terminalizes.

## Health, liveness, readiness, and metrics

GET /health returns a JSON object with status set to ok, a models array containing the
single loaded V3 variant identity, and the selected provider string cpu, cuda, or tensorrt. It is a
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
| vad_threshold | Finite probability in [0, 1]; requires endpoint=vad | 0.5 |
| silence_skip | true or false; optional acoustic predecode gate | false |
| skip_vad_threshold | Finite probability in [0, 1]; requires silence_skip=true | 0.5 |
| silence_ms | Finite nonnegative after-speech silence, milliseconds | 1200 |
| no_speech_ms | Finite nonnegative silence without recognized speech, milliseconds | 2400 |
| window | Finite positive seconds | 30 |
| overlap | Finite nonnegative seconds, shorter than window | 6 |
| step_ms | Finite positive milliseconds | 500 |
| horizon | finite seconds greater than zero | 5.0 |
| lock | 1, 0, true, false | false |
| seam_after | Finite nonnegative seam delay, seconds | 2 |
| keep_silence | Finite nonnegative retained silence, seconds | 0.5 |
| applied_config | 1, 0, true, false | false |
| turn_gap | finite seconds greater than zero | 0.8 |
| dedup | 1, 0, true, false | ASR_DEDUP for multichannel input; disabled for mono input |
| backchannel_max_ms | finite milliseconds greater than or equal to zero | ASR_BACKCHANNEL_MAX_MS |

Within the accepted range, rate is admitted only when its ratio to the model rate, reduced
to lowest terms, stays within 1000 on both sides; other rates refuse with an explicit
message.

Streaming timing controls bind together at the selected model rate before upgrade or native
session construction. Fractions and zero are accepted where stated; window and step must
truncate to at least one model-rate sample, and all derived counts must be representable.
Overlap and retained silence may truncate to zero. Both blank and VAD endpointing use the
silence rules, seam delay, and retained silence. Only an explicit `vad_threshold` requires
`endpoint=vad`. Enabled VAD requires a 16000 Hz model/frontend; `rate` still describes the
source audio, which is resampled normally. These controls preserve the existing endpoint
algorithm and do not enable batch segmentation. Optional `silence_skip=true` separately
checks the entire retained real-audio window before an Advance decode. It skips frontend and
recognition work only when every probability is below `skip_vad_threshold` and no words,
seam or un-emitted update are pending. Equality counts as speech. Enabled skipping requires
a 16000 Hz model and the selected package's VAD asset; it shares one detector per channel
with VAD endpointing. Off requires no detector for skipping. Retained overlap and source time
stay intact; a nonempty flush still decodes. Detection errors and cancellation refuse the
tick rather than treating it as silence. This does not promise interruption of a VAD or
recognition call, acoustic timestamp accuracy or a speedup.

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
| applied_stream_config | type, schema_version, model_sample_rate, endpoint, window_sec, overlap_sec, step_sec, horizon_sec, lock, seam_after_sec, keep_silence_sec, endpoint_rules, vad, optional silence_skip | Opt-in confirmation of the immutable policy bound to the active session. |
| words | type, at, revise_from, words | Replace the channel word tail at revise_from. Each word has start, end, text, stable. |
| stable | type, at, upto | Word order is settled through upto. |
| final | type, at, upto, endpoint | Output is immutable through upto; endpoint indicates endpoint-caused finality. |
| turns | type, revise_from, frontier, turns | Replace the dialogue tail from revise_from. Each turn has channel, k, start, end, text, stable, final, backchannel. |
| error | type, message | Terminal diagnostic before server close for an in-band application failure. |
| end | type | Successful terminal event. |

`applied_config=1` or `true` requests one `applied_stream_config` event after successful
active-session construction and before transcript events. Its version-1 object has the same
fields and numeric semantics as the CLI applied-policy record below; the WebSocket horizon
default remains 5 seconds. When skipping is enabled, the event adds `silence_skip` with
`enabled: true` and the effective `threshold`; when disabled that field is omitted and the
existing event is unchanged. Values come from the setup bound to the actual session. The event
does not assert completed inference or selected physical-channel identities: deduplication
may defer channel activation until enough input arrives. No record is sent after failed
construction or shutdown before opening. An already-ready active shutdown takes precedence
over pending metadata. A metadata delivery failure terminates through the existing event
delivery failure path without recursive error delivery. Omitted, `0`, or `false` preserves
the existing event stream; new control keys do not require opting into this event.

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
asr transcribe FILE [--model DIR] [--variant ID] [--ep cpu|cuda|tensorrt] [--ort-dylib PATH]
                    [--fp16] [--ctc-precision fp32|fp16io32|fp16] [--rnnt] [--rnnt-precision fp32|fp16] [--window S] [--overlap S] [--pad]
                    [--words] [--split-channels] [--turns] [--turn-gap S]
                    [--evidence PATH] [--evidence-version 1|2]
                    [--vad] [--vad-threshold F] [--vad-neg-threshold F]
                    [--vad-min-speech-ms N] [--vad-min-silence-ms N] [--vad-speech-pad-ms N]

asr bench [--audio FILE] [--model DIR] [--variant ID] [--ep cpu|cuda|tensorrt] [--ort-dylib PATH]
          [--fp16] [--ctc-precision fp32|fp16io32|fp16] [--window S] [--iters N] [--gap-ms N] [--alt-window S]

asr stream FILE [--model DIR] [--variant ID] [--ep cpu|cuda|tensorrt] [--ort-dylib PATH]
              [--fp16] [--ctc-precision fp32|fp16io32|fp16] [--chunk-ms N] [--step-ms N] [--horizon S]
              [--window S] [--overlap S] [--endpoint blank|vad] [--silence-ms N]
              [--no-speech-ms N] [--vad-threshold F] [--seam-after S] [--keep-silence S]
              [--ref FILE] [--lock] [--events] [--applied-config]

asr vad FILE [--model DIR] [--variant ID] [--ort-dylib PATH] [--threshold F] [--neg-threshold F]
           [--min-speech-ms N] [--min-silence-ms N] [--speech-pad-ms N] [--channels all|0,2]

asr dialog FILE [--model DIR] [--variant ID] [--ep cpu|cuda|tensorrt] [--ort-dylib PATH]
              [--ctc-precision fp32|fp16io32|fp16]
              [--turn-gap S] [--endpoint blank|vad] [--no-dedup]
              [--backchannel-max-ms N] [--window S] [--overlap S] [--step-ms N]
              [--horizon S] [--silence-ms N] [--no-speech-ms N] [--vad-threshold F]
              [--seam-after S] [--keep-silence S] [--lock] [--applied-config]
~~~

| Command | Defaults and constraints |
| --- | --- |
| transcribe | model is model; CTC is selected unless --rnnt; --window is 30 seconds; --overlap is 6 seconds and must be shorter than window; --turn-gap is 1 second; --words conflicts with --turns; --pad selects fixed-window padding; --evidence is optional and requires CTC. --vad enables batch speech segmentation; VAD tuning flags require it. |
| bench | audio defaults to fixtures/long_example.wav, a path not bundled by the repository; window is 30 seconds; iters is 20 and positive; gap-ms is 0 and nonnegative; alt-window is optional and positive. |
| stream | chunk-ms is 100 and positive; horizon is 4 seconds; --events prints event projections. All streaming policy controls below are supported. |
| vad | threshold is 0.5; neg-threshold is 0.35; both are finite in 0 through 1 and neg-threshold cannot exceed threshold; min-speech-ms and min-silence-ms default to 250 and 100; speech-pad-ms defaults to 30. All durations accept finite nonnegative fractional milliseconds, including zero. Without --channels only physical channel 0 is processed; explicit selection accepts all or unique comma-separated zero-based physical indices. |
| dialog | turn-gap is 0.8 seconds and positive; horizon is 5 seconds; --no-dedup disables offline dialog deduplication; backchannel-max-ms defaults to 0 and is nonnegative. All streaming policy controls below are supported. |

### Streaming CLI policy

Both `stream` and `dialog` bind these controls to the selected package's model sample rate
before initializing the native runtime. `stream` still processes only the first physical
channel and runs its batch reference pass; `dialog` retains its multichannel selection and
deduplication behavior.

| Option | Accepted values and units | Default |
| --- | --- | --- |
| --endpoint | blank or vad | blank |
| --vad-threshold | Finite probability in [0, 1]; requires explicit --endpoint vad | 0.5 |
| --silence-ms | Finite nonnegative after-speech silence, milliseconds | 1200 |
| --no-speech-ms | Finite nonnegative silence without recognized speech, milliseconds | 2400 |
| --window | Finite positive seconds | 30 |
| --overlap | Finite nonnegative seconds, shorter than window | 6 |
| --step-ms | Finite positive milliseconds | 500 |
| --horizon | Finite positive seconds | 4 for stream; 5 for dialog |
| --lock | Flag selecting committed-stable locking | Advisory locking |
| --seam-after | Finite nonnegative seam delay, seconds | 2 |
| --keep-silence | Finite nonnegative retained silence, seconds | 0.5 |
| --applied-config | Flag adding a versioned applied-policy record to stderr | Omitted |

Silence rules, seam delay, and retained silence apply to both blank and VAD endpointing.
Only the detector threshold requires VAD; explicitly supplying it with blank endpointing
refuses. These are existing streaming endpoint controls, not the batch segmentation
hysteresis, minimum-speech, or padding controls. They do not skip ASR on silent input.

Durations accept fractions and zero where stated. Window and step must truncate to at least
one model-rate sample; overlap and retained silence may truncate to zero. Unrepresentable
sample counts refuse. Coupled window/overlap changes bind together. VAD endpointing requires
a 16000 Hz model/frontend before native work; accepted source audio is resampled normally.

After successful transcription and its existing terminal projection, `--applied-config`
writes one JSON line to stderr. Existing stdout remains unchanged. The record describes the
validated configuration used by the command, not the original argument text:

~~~json
{"type":"applied_stream_config","schema_version":1,"model_sample_rate":16000,"endpoint":"vad","window_sec":30,"overlap_sec":6,"step_sec":0.5,"horizon_sec":4,"lock":"advisory","seam_after_sec":2,"keep_silence_sec":0.5,"endpoint_rules":{"after_speech_sec":1.2000000476837158,"without_speech_sec":2.4000000953674316},"vad":{"enabled":true,"threshold":0.5}}
~~~

With blank endpointing, `endpoint` is `blank` and `vad` is exactly `{"enabled":false}`.
Lock values are `advisory` and `commit_stable`. Values preserve the bound 32-bit floats,
without word-timestamp rounding. A prior command failure emits no applied-policy record;
the record does not promise atomic or durable terminal output. Omitting the flag preserves
the existing terminal projection.

`bench` opens the selected package before native runtime initialization. Its --window and
--alt-window values are durations at that package's validated sample rate: each becomes a
model-rate buffer length, and the decoded WAV input is resampled to the same rate before
warmup or measurement. The default therefore remains a 30-second duration for every valid
package rate rather than a fixed sample count.

`transcribe --vad` uses the same segmentation policy and defaults as standalone `vad`, with
the five tuning options prefixed by `--vad-`. Speech and silence thresholds use hysteresis;
accepted speech must be strictly longer than the minimum speech duration, while the minimum
silence comparison is inclusive and closes detector spans without necessarily separating
recognition groups. Duration conversion uses the library's checked truncation
of `milliseconds / 1000 * 16000` to samples; unrepresentable counts or doubled padding refuse.
Tiny positive durations may therefore become zero samples. Padding extends accepted speech
within the input and divides shared gaps between adjacent spans. It does not change the
separate ASR `--pad` policy, which pads each recognition group to the configured ASR window.

Batch VAD requires a 16 kHz model/frontend and refuses another rate before native initialization
or evidence-file creation. Input audio is resampled to the model rate as usual. One local
detector is reused for independent whole-channel calls over the existing batch selection:
combined output selects the first channel, split/turn output selects separate channels, and
existing dual-mono detection still selects the first channel. Silent selected channels remain
in the result. Recognition groups retain intervening original audio and original time coordinates;
windows are not stitched across groups. Exact groups no longer than the frontend FFT window retain the existing no-decode
behavior. Normal startup warmup is separate from transcription; a silent transcription adds
no frontend or ASR decoder work. The enabled command reports its applied VAD policy and VAD
duration separately from frontend time. Without `--vad`, its existing path and output remain
unchanged and no VAD session or asset is required.

Standalone `vad --channels` processes the requested physical channels in ascending original
index order, including identical and silent channels, without correlation deduplication.
Invalid or out-of-range selection refuses before any detector call. Each selected channel
gets a labelled stdout section even when it has no speech; stderr identifies the selection
and applied policy. Without this option, the existing channel-0 segment rows and summary are
preserved. Standalone VAD resamples input to its fixed 16 kHz rate independently of the ASR
frontend rate. Neither CLI path introduces a shared detector pool or service worker policy.

`transcribe --evidence PATH` writes one per-window decode-evidence file alongside the ordinary
transcript: a plain ASCII text file with a header line, one `model` line and one `audio` line
naming the package and input, and one block per decoded window carrying that window's geometry,
raw per-frame CTC labels, non-emission mask, collapsed tokens, and the token-index span behind
each decoded word, so an external reader can check its own word grouping against the decoder's
own word rule without re-implementing it. PATH is validated once, before any window is decoded:
its parent directory must already exist, and PATH itself must not already be a directory. The
transcript is unaffected by --evidence; only CTC transcription supports it, so it conflicts with
--rnnt. Version 1 remains the default and retains its existing bytes. `--evidence-version`
requires a destination and accepts only `1` or `2`. With VAD enabled, evidence requires an
explicit `--evidence-version 2`; version 1 cannot represent segmented origins and refuses
before file creation. Version 2 also works with VAD disabled.

`evidence v2` retains each window's existing local feature/token coordinates and adds
`channel` (the zero-based physical source channel) and `frontend_origin_sample` (the integer
model-rate sample offset of the feature crop in that channel). Temporal fields and
`first_sample` are already on the source-channel timeline; never add the crop origin to
them again. The crop offset need not be divisible by the frontend hop length. Each channel
has its own planned window indices; gaps in indices need not be missing records because
short no-decode windows can be planned without emitting evidence.

After the audio header, version 2 records `vad disabled` or `vad enabled` with the applied
rate, thresholds and duration parameters. After successful transcription it adds:

~~~text
coverage source_channels 3 selected_channels 0 1 2 duplicate not-dual-mono
channel_windows 0 emitted 0
channel_windows 1 emitted 2
channel_windows 2 emitted 1
complete
~~~

Coverage comes from the successful multi-channel result, including every selected quiet or
short channel. `emitted` counts actual written window records, not planned windows.
Duplicate classification is `dual-mono` or `not-dual-mono`; selected identities are never
renumbered. Failed transcription, invalid coverage, or a previously observed write failure
does not receive a successful closing record. Earlier records are flushed before attempting
the footer. A late final write/flush failure still fails the command and may leave footer
bytes, so success requires both zero CLI exit status and a valid complete file. The format
does not promise atomic publication or durable storage.

For applicable commands, `--variant` or `ASR_MODEL_VARIANT` selects an exact V3 variant;
when both are supplied they must agree. Omission retains the legacy V1/V2 CLI path. A V3
RNN-T transcribe needs no `--rnnt`; that flag is an optional RNN-T assertion. `--model` defaults to model. --ort-dylib or ORT_DYLIB_PATH is
required; when both are present, they must be identical. Provider selection and runtime
environment follow [OPERATIONS.md](OPERATIONS.md).

### CUDA graph-assignment verification

This is process configuration, not an HTTP or WebSocket field. With selected CUDA,
`ASR_CUDA_ASSIGNMENT_POLICY` is absent or `verified` by default and requires the exact
64-hex-character SHA-256 value for every encoder role the command or Service will construct:
`ASR_CUDA_CTC_ASSIGNMENT_SHA256` and, when RNN-T is required,
`ASR_CUDA_RNNT_ASSIGNMENT_SHA256`. Service requires only its selected variant's family role;
CLI transcribe requires its selected family, bench, stream and dialog require CTC, and VAD
requires neither.

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

For CTC transcribe, bench, stream, and dialog, `--ctc-precision` or
`ASR_CTC_PRECISION` selects `fp32`, `fp16io32`, or native-I/O `fp16`. Equal
CLI and environment choices are accepted; different choices or either choice combined
with the applicable legacy `--fp16` flag refuse. Dialog does not accept `--fp16`.
RNN-T transcription and VAD ignore the CTC environment setting and reject the CTC option.

For offline RNN-T transcription, `--rnnt-precision fp32|fp16` or
`ASR_RNNT_PRECISION=fp32|fp16` selects all RNN-T roles. Both may be supplied
when they agree; either conflicts with the legacy `--fp16` encoder-only flag.
The CLI option requires `--rnnt`; CTC-only commands ignore the environment
setting. Without the new setting, the legacy encoder choice and fp32 heads
remain in effect. Full fp16 requires compatible assets and a GPU provider;
it does not fall back to fp32.


When a CTC vocabulary has no standalone `▁`, CTC construction emits a blank-only
silence-mask warning to stderr. `transcribe` and `stream` emit it once per constructed
decoder; `dialog` emits it once per constructed channel-local decoder. `bench` and `vad`
do not emit this warning.

Grammar refusal exits with status 2. Non-UTF-8 operating-system arguments and command or
runtime failures exit with status 1 and a textual diagnostic. These lines, status values,
and diagnostic strings are current process projections, not stable pre-1.0 byte-format or
machine-readable error-code promises.
