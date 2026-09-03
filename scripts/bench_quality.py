#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Yuriy Krasilnikov
# Copyright (C) 2026 Deeray Ltd.
#
# This file is part of GigaAM v3 Runtime, free software distributed under
# the terms of the GNU Lesser General Public License, either version 3 of the
# License, or (at your option) any later version. See COPYING.LESSER and COPYING
# for the full terms. There is NO WARRANTY, to the extent permitted by law.

"""Quality benchmark harness: runs the Rust `asr` runtime over a manifest and calculates metrics
(scripts/metrics.py provides WER/CER/punctuation/case without Torch). Compare a configuration matrix
(ctc/rnnt, fp16/fp32, windows, --ep cuda|tensorrt, ASR_FRONTEND=batched) on one corpus to select
production defaults. CUDA evaluation inherits model-specific verified CTC/RNN-T fingerprints from
its environment. Use the per-row `env` object in the JSON configuration form for environment
switches; explicit allow-unverified rows collect initial assignment evidence only. The corpus
manifest is TSV `audio<TAB>reference`.

Configurations:
  --configs "label=flags;label2=flags2"        (without environment variables)
  --config-file cfg.json  -> [{"label","flags","env":{...}}]  (with environment switches)
Example:
  python3 scripts/bench_quality.py --manifest val.tsv \
      --configs "ctc-fp32=;ctc-fp16=--fp16;rnnt=--rnnt"
"""
import sys, os, subprocess, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics


def read_manifest(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) < 2:
            continue
        rows.append((p[0], p[1]))
    return rows


def load_configs(args):
    if args.config_file:
        return json.load(open(args.config_file, encoding="utf-8"))
    out = []
    for item in args.configs.split(";"):
        item = item.strip()
        if not item:
            continue
        label, _, flags = item.partition("=")
        out.append({"label": label.strip(), "flags": flags.strip(), "env": {}})
    return out


def run_asr(asr_bin, model, audio, flags, env):
    cmd = [asr_bin, "transcribe", audio, "--model", model] + (flags.split() if flags else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1200)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"!! {audio} [{' '.join(flags.split())}] timeout 1200s -> skipped\n")
        return ""
    if r.returncode != 0:
        sys.stderr.write(f"!! {audio} [{' '.join(flags.split())}] rc={r.returncode}: {r.stderr[-300:]}\n")
        return ""
    return r.stdout.strip()  # stdout is transcript; trace goes to stderr


def fmt(x):
    return f"{x*100:.2f}%" if isinstance(x, (int, float)) else "  n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="TSV: audio_path<TAB>reference_text")
    ap.add_argument("--asr", default="./target/release/asr")
    ap.add_argument("--model", default="model")
    ap.add_argument("--configs", default="ctc-fp16=--fp16")
    ap.add_argument("--config-file")
    ap.add_argument("--json")
    a = ap.parse_args()

    rows = read_manifest(a.manifest)
    refs = {audio: ref for audio, ref in rows}
    configs = load_configs(a)
    print(f"# manifest: {len(rows)} files, {sum(len(r.split()) for _,r in rows)} reference words")
    hdr = f"{'config':<22}{'WER':>9}{'CER':>9}{'punctF1':>9}{'case':>9}{'wall_s':>8}"
    print(hdr); print("-" * len(hdr))
    out = {}
    for cfg in configs:
        env = dict(os.environ); env.update(cfg.get("env", {}))
        hyps = {}
        t0 = time.perf_counter()
        for audio, _ in rows:
            hyps[audio] = run_asr(a.asr, a.model, audio, cfg.get("flags", ""), env)
        wall = time.perf_counter() - t0
        m = metrics.evaluate(refs, hyps)
        line = (f"{cfg['label']:<22}{fmt(m['WER']):>9}{fmt(m['CER']):>9}"
                f"{m['punct_F1_micro']*100:>8.2f}%{fmt(m['case_accuracy']):>9}{wall:>8.1f}")
        print(line)
        out[cfg["label"]] = {k: v for k, v in m.items() if k != "per_file"}
        out[cfg["label"]]["wall_s"] = round(wall, 2)
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"# JSON -> {a.json}")


if __name__ == "__main__":
    main()
