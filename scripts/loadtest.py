#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Yuriy Krasilnikov
# Copyright (C) 2026 Deeray Ltd.
#
# This file is part of GigaAM v3 Runtime, free software distributed under
# the terms of the GNU Lesser General Public License, either version 3 of the
# License, or (at your option) any later version. See COPYING.LESSER and COPYING
# for the full terms. There is NO WARRANTY, to the extent permitted by law.

# /v1/transcribe load test using only the standard library; sweeps concurrency.
# Usage: loadtest.py <host:port> <file> <model> <dur_sec> <conc1,conc2,...>
import sys, time, threading, http.client, statistics, os, math

host_port, path_file, model, dur, conc_list = sys.argv[1:6]
dur = float(dur)
concs = [int(x) for x in conc_list.split(",")]
host, port = host_port.split(":")
port = int(port)
body = open(path_file, "rb").read()
audio_name = os.path.basename(path_file)

def worker(deadline, lats, counts, lock):
    ok = c503 = err = 0
    ls = []
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        try:
            conn = http.client.HTTPConnection(host, port, timeout=60)
            conn.request("POST", f"/v1/transcribe?model={model}",
                         body=body, headers={"Content-Type": "application/octet-stream"})
            r = conn.getresponse()
            _ = r.read()
            dt = (time.perf_counter() - t0) * 1000
            if r.status == 200: ok += 1; ls.append(dt)
            elif r.status == 503: c503 += 1
            else: err += 1
            conn.close()
        except Exception:
            err += 1
    with lock:
        counts[0] += ok; counts[1] += c503; counts[2] += err
        lats.extend(ls)

def pct(v, p):
    if not v: return float("nan")
    v = sorted(v)
    i = min(len(v) - 1, max(0, math.ceil(p / 100 * len(v)) - 1))  # nearest-rank
    return v[i]

print(f"# file={audio_name} ({len(body)} bytes) model={model} duration/level={dur}s")
print(f"{'conc':>5} {'req/s':>8} {'ok':>6} {'503':>5} {'err':>4} {'p50ms':>8} {'p90ms':>8} {'p99ms':>8} {'max':>8}")
for conc in concs:
    lats = []; counts = [0,0,0]; lock = threading.Lock()
    deadline = time.perf_counter() + dur
    ts = [threading.Thread(target=worker, args=(deadline, lats, counts, lock)) for _ in range(conc)]
    t0 = time.perf_counter()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.perf_counter() - t0
    ok, c503, err = counts
    rps = ok / wall
    print(f"{conc:>5} {rps:>8.1f} {ok:>6} {c503:>5} {err:>4} "
          f"{pct(lats,50):>8.0f} {pct(lats,90):>8.0f} {pct(lats,99):>8.0f} {(max(lats) if lats else 0):>8.0f}")
    time.sleep(1)
