# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately to **yury.krasilnikov@gmail.com**
rather than in a public issue. Include what you observed, the steps to reproduce it and,
if you have one, a proof of concept. You will get an acknowledgement, and we will tell you
whether the report is accepted and when a fix is expected.

Please do not run intrusive tests against systems you do not own.

## Scope

This repository ships a speech-recognition runtime and a service that exposes it over
HTTP and WebSocket. Relevant to a security review:

- **Untrusted audio input.** The service parses arbitrary uploaded bytes: containers,
  codecs, sample rates and channel layouts. Parser crashes, unbounded allocations and
  decompression amplification are in scope. Uploads are size-limited and sample rates are
  range-checked, but audio parsing remains the largest attack surface.
- **Resource exhaustion.** Concurrency and channel limits, request timeouts and the
  streaming buffers are meant to bound work per caller. A way around them is in scope.
- **The service assumes a trusted network position.** It has no authentication,
  authorization or rate limiting of its own and is designed to sit behind an ingress
  layer that provides them. "The endpoint is unauthenticated" is therefore not a
  vulnerability report; a way to bypass a documented limit is.

Out of scope: the security of ONNX Runtime, CUDA, TensorRT or model weights themselves —
report those upstream — and issues in third-party components as shipped by their vendors.

## Handling of data

The service processes audio and returns text. It does not persist audio or transcripts.
Access logs contain request metadata (identifier, status, duration, payload size, model),
not payloads.
