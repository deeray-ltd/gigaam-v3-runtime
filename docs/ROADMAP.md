# Roadmap

This document records the work that GigaAM v3 Runtime deliberately leaves outside its
first release, why each item is deferred, and what evidence would unlock it. It carries no
dates. Items move out of this file only when the release that delivers them lands in
[CHANGELOG.md](../CHANGELOG.md).

The first release delivers batch CTC and RNN-T transcription over HTTP, CTC streaming over
WebSocket with revisable word and dialogue events, multichannel dialogue reconstruction for
separate-channel audio, strict CPU, CUDA, and TensorRT provider selection, and three source
image definitions. The boundaries it does not establish are listed in
[ARCHITECTURE.md](ARCHITECTURE.md) and expanded below.

## Runtime lifetime validation

The owned native runtime capability keeps the ONNX Runtime environment alive through session
and worker lifetimes and lets normal process cleanup occur before terminal exit. Validate the
CUDA teardown regression with the retained failing fixture and native provider library; a
successful build or transcript alone does not establish that process cleanup succeeds.

## Deferred work

### Model package integrity and compatibility

Packages are parsed strictly but are not bound to their assets by any checksum, carry no
provenance or license metadata, and declare no runtime compatibility range. Offline
verification and a retirement rule for the version-1 schema are also absent.

Unlocked by: an inventory of supported package producers and a migration policy that says
how a new schema version replaces version 1.

### Wire protocol maturity

- Legacy version 1 errors and failures before version 2 query binding use English messages
  under one `error` key without stable machine-readable codes. After an HTTP batch request
  binds to version 2, application errors use typed codes, including distinct overload,
  draining, and unavailable responses. Broader protocol harmonization remains deferred,
  including RFC 9457 Problem Details, a consistent code registry, and a `Retry-After` hint
  on overload and draining responses.
- A recognition call waits at most 60 seconds to start; that bound is fixed in the source
  and independent of `ASR_REQ_TIMEOUT_SEC`. The two limits should be one consistent
  contract, either derived from the request deadline or configured explicitly.
- A native ONNX Runtime call that is already executing cannot be interrupted; cancellation
  is cooperative between calls.
- Streaming has no explicit per-session resource budget beyond the message and frame size
  limit and the channel permits.
- Pre-1.0 interfaces carry no version negotiation or deprecation policy.
- Readiness currently reports only that a recognition worker stopped; naming the stopped
  worker in the readiness body and in a metric is deferred.

Unlocked by: evidence of how each provider behaves under interruption, and compatibility
evidence from external clients.

### Quality, latency, throughput, and capacity

No accuracy, latency, or capacity claim is made, and this repository tracks no measurement
that would support one. Missing measurements include:

- word and character error rates on a representative corpus, by domain and channel
  condition, including 8 kHz telephone audio;
- the error rate at streaming window seams, where the sliding-window design relies on
  heuristic word matching rather than acoustic alignment;
- calibration of word timestamps, which are CTC emission times and start later than the
  acoustic onset; the offset has not been measured;
- real-time factor and latency per provider and thread configuration, on CPU in
  particular, where streaming at the default window can be slower than real time;
- concurrent-load behavior at the admission limits.

Unlocked by: a licensed representative corpus with speaker-turn and timing references, a
slice taxonomy, and agreed regression thresholds.

### TensorRT engine cache lifecycle and default provider

TensorRT engines are built on first use into a cache directory whose lifetime, invalidation,
and sharing rules are not defined. No provider is declared the production default.

Unlocked by: package identity from the integrity work above and the provider quality and
capacity measurements.

### Fair scheduling and GPU batching

Each recognition decoder runs on one dedicated thread that serves streaming ticks before
batch windows without fairness; every request is decoded alone, so a GPU never batches
across requests.

Unlocked by: the cancellation and resource contract above plus latency, memory, and
throughput measurements.

### RNN-T streaming

Streaming is CTC only. RNN-T runs in batch mode with its prediction and joint networks on
the CPU.

Unlocked by: the protocol work above and quality and latency evidence for RNN-T.

### Confidence and domain adaptation

Words carry no confidence score, and there is no mechanism for vocabulary or domain
adaptation.

Unlocked by: a calibrated confidence metric and representative evaluation data.

### Speaker attribution for single-channel audio

Speakers are identified only by audio channel. Mixed single-channel recordings receive no
speaker labels, so dialogue turns are unavailable for them. The intended design is
speech-activity segmentation, speaker embeddings over short windows, clustering constrained
to a small number of speakers, and attribution of words to speaker segments, with streaming
labels that can be revised until the corresponding words become final.

Privacy boundary: speaker embeddings are voice biometric data. Within one session they exist
only in memory and yield local labels; no embedding is persisted and no identity is matched
across sessions.

Unlocked by: agreed product semantics, the privacy boundary above, an evaluation set with
speaker-turn references, and the timestamp calibration from the quality work.

### API selection and OpenAI-compatible transcription

Planned: select one API at startup with `ASR_API_MODE=native|openai`, defaulting to
`native`. Empty, unknown, and non-Unicode values must refuse startup before model or
listener initialization. Native mode exposes the native transcription and WebSocket
routes; OpenAI mode exposes compatible transcription, model listing, and, when delivered,
Realtime routes, and disables the native WebSocket. Health and metrics remain available
in both modes; unselected routes return 404.

The compatible file API is planned to support multipart transcription, model listing,
JSON, text, and complete `verbose_json` responses for CTC and RNN-T. Define supported
language, prompt, temperature, timestamp, and model options and refuse unsupported
combinations. Verbose output must use truthful language, duration, text, word and requested
segment fields, including token, timing, compression, and score data with documented
provenance. Do not substitute fabricated scores or imply Whisper calibration. Token and
score provenance must survive filtering, source-time shifts, window stitching, and segment
projection; no-speech semantics require a supported measurement, not CTC blank alone.

Both APIs should use a shared batch execution result while retaining native response
versions, VAD policy, channel semantics, cancellation, and shutdown behavior. Define upload
admission, body and processing deadlines, decoded-memory bounds, and ownership through
physical work and response projection before exposing the new interface. Validate both
models, formats, errors, and supported client behavior against the selected contract.

### Incremental file output and Realtime

Planned file SSE and Realtime WebSocket transcription must provide genuine partial output
before all audio has arrived or been processed for every supported CTC and RNN-T model.
The final file result must agree with the corresponding batch result, including source
clock, VAD grouping, window seams, text and metadata. Define the supported combinations of
streaming, output format, and timestamp granularity; handle failures after streaming begins
as terminal stream errors.

Realtime needs an explicit revision and finalization contract: provisional words can change,
so publication cannot blindly append revisions as irreversible text. Define item identity,
commit, clear, completion, cancellation, and ordering, and bound media and event queues
with backpressure. Validate disconnect, slow consumers, shutdown, model-specific quality,
latency and CPU capacity. Native WebSocket currently supports CTC; RNN-T support requires
an actual model-specific streaming path and acceptance. WebRTC is deferred.

### VAD and configuration

Batch VAD groups consecutive padded detector spans within the configured recognition window;
it retains intervening original audio and source time. CLI and native WebSocket expose the
existing streaming endpoint controls, including optional confirmation of applied policy.
Preserve independent detector thresholds, speech and silence durations, and padding, together
with validated window, overlap, step, horizon, seam, and endpoint settings. Make supported
settings consistently available and explicit across applicable file and streaming interfaces;
reject invalid or inapplicable combinations and bind model-rate-dependent values before work.
Evaluate short and quiet speech, multiple channels, endpoint behavior and quality across
representative audio rather than treating a particular crop as a general quality guarantee.

### CPU pipeline

Improve CPU throughput and latency across audio decoding, resampling, feature extraction,
VAD, encoder and decoder execution, stitching, and response projection. Measure actual
stage costs and memory under representative batch and streaming workloads before selecting
optimizations; preserve recognition quality, timestamps, cancellation, bounded admission,
and provider behavior. Review window reuse, avoidable copying and allocation, threading,
queue fairness and optional silence skipping against their quality and resource tradeoffs.

### CTC precision selection

Service and CTC CLI commands select the existing FP32, FP16Io32, or native FP16
asset through typed process configuration. Native CUDA and prepared TensorRT execution
requires validation with compatible artifacts and a known speech fixture; CPU native FP16
execution remains unverified. No performance claim follows from selection alone.

### Model variants and decoding

V3 metadata admission now represents one explicit variant and family; the existing Service
and CLI still use legacy packages. Recognition now exposes selected-family V3 native construction and tokenizer projection; Service and CLI composition and all four real-package recognition witnesses remain pending. Bind advertised model variants to their actual package assets, tokenizer, frontend,
precision, and decoder capabilities. Plan explicit base and end-to-end CTC and RNN-T
variants with compatible package evolution; support each variant's word boundaries and
streaming contract. Evaluate beam decoding against greedy decoding using comparable quality
and cost measurements before offering a supported selectable decoder. SSL output remains
separate future work.

### CTC acoustic timestamps

Dedicated correction of CTC acoustic timing is deferred. Token emission times do not establish
word onset; a correction requires representative aligned references and must preserve source
clock and existing model/tokenizer semantics. Decoder or tokenizer changes alone do not
establish timing accuracy.
