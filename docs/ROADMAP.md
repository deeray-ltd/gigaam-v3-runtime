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

## Deferred work

### Model package integrity and compatibility

Packages are parsed strictly but are not bound to their assets by any checksum, carry no
provenance or license metadata, and declare no runtime compatibility range. Offline
verification and a retirement rule for the version-1 schema are also absent.

Unlocked by: an inventory of supported package producers and a migration policy that says
how a new schema version replaces version 1.

### Wire protocol maturity

- Errors are English messages under one `error` key; there are no stable machine-readable
  codes, so a client can distinguish causes only by HTTP status, and 503 currently covers
  overload, draining, a stopped worker, queue-wait expiry, and transcription failure. The
  intended direction is RFC 9457 Problem Details with a registry of stable codes and a
  `Retry-After` hint on overload and draining responses.
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
