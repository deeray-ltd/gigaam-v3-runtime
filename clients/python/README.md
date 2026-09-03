# Python ASR service client

Uses only the standard library; it has no dependencies.

```python
from asr_client import AsrClient
cli = AsrClient("127.0.0.1", 8080)

# Batch transcription (HTTP POST /v1/transcribe)
res = cli.transcribe_file("audio.wav", model="ctc", words=True, segments=True, turns=True)
print(res["text"])

# Streaming (WS /v1/stream): generator of words/stable/final events
for ev in cli.stream(pcm16_chunks, rate=16000, fmt="pcm16", emit="words", horizon=5.0):
    if ev["type"] == "words":
        # ev = {type, at, revise_from, words:[{start,end,text,stable}]}
        words[ev["revise_from"]:] = [w["text"] for w in ev["words"]]
```

- `transcribe(data, *, model, words, segments, turns, channels, ext, turn_gap)` -> parsed JSON.
- `transcribe_file(path, **kw)` -> the same, with `ext` inferred from the filename extension.
- `stream(chunks, *, rate, fmt="pcm16", model="ctc", channels=1, emit=None, endpoint=None, dedup=None, turn_gap=None, horizon=None, lock=False)` -> event generator. `channels` declares interleaved input channels; `endpoint` selects blank or VAD endpointing; `dedup` controls dual-mono collapse; `turn_gap` sets the turn boundary in seconds.
- `health()` -> {status, models, provider}.

Demo: `python3 example.py <audio.wav>` while the service listens on port 8080. The
repository bundles no audio; supply your own file. Without an argument the script looks
for `fixtures/example.wav`.
