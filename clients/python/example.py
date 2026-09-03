#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Yuriy Krasilnikov
# Copyright (C) 2026 Deeray Ltd.
#
# This file is part of GigaAM v3 Runtime, free software distributed under
# the terms of the GNU Lesser General Public License, either version 3 of the
# License, or (at your option) any later version. See COPYING.LESSER and COPYING
# for the full terms. There is NO WARRANTY, to the extent permitted by law.

"""Example asr_client use: health, batch transcription (HTTP), and streaming (WS).
Run while the service listens on port 8080: python3 example.py [fixtures/example.wav]"""
import sys, time, wave
from asr_client import AsrClient

path = sys.argv[1] if len(sys.argv) > 1 else "fixtures/example.wav"
cli = AsrClient("127.0.0.1", 8080)

print("health:", cli.health())

# --- HTTP: batch ---
res = cli.transcribe_file(path, model="ctc", words=True)
print("\n[HTTP] text:", res["text"])
print(f"[HTTP] rtf={res['rtf']} words={len(res.get('words', []))} duration={res['duration_sec']:.1f}s")

# --- WS: stream ---
w = wave.open(path, "rb")
rate, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
assert sw == 2, "example expects pcm16 WAV"
pcm = w.readframes(w.getnframes())
if nch > 1:  # downmix to one channel because this example opens a one-channel session
    mono = bytearray()
    for i in range(0, len(pcm), 2 * nch):
        mono += pcm[i:i + 2]
    pcm = bytes(mono)

def chunks(step_ms=100):
    step = int(rate * step_ms / 1000) * 2
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]
        time.sleep(step_ms / 1000)  # simulate real time

from asr_client import apply_turns
print("\n[WS] stream (100 ms/frame, emit=turns by default)…")
dialog = []
patches = 0
for ev in cli.stream(chunks(), rate=rate, fmt="pcm16", horizon=5.0):
    if ev["type"] == "turns":
        patches += 1
        apply_turns(dialog, ev)
    elif ev["type"] == "error":
        print("[WS] server error:", ev.get("message")); break
print(f"[WS] dialogue patches={patches}")
for t in dialog:
    print(f"[WS] ch{t['channel']} {t['start']:.1f}-{t['end']:.1f} {t['text']}")
