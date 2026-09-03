#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Yuriy Krasilnikov
# Copyright (C) 2026 Deeray Ltd.
#
# This file is part of GigaAM v3 Runtime, free software distributed under
# the terms of the GNU Lesser General Public License, either version 3 of the
# License, or (at your option) any later version. See COPYING.LESSER and COPYING
# for the full terms. There is NO WARRANTY, to the extent permitted by law.

"""Quality-evaluation harness with no external dependencies.

Metrics are deliberately separate:
  * WER / CER use normalized text without punctuation or case to measure recognition;
  * punctuation uses P/R/F1 of marks on aligned words;
  * case measures uppercase accuracy on aligned words.

Input is a TSV manifest `path<TAB>reference` and a matching hypothesis file
(`path<TAB>recognized`), or two columns in one file.
"""
import json
import re
import sys
import unicodedata
from collections import Counter

PUNCT_CLASSES = {",": ",", ".": ".", "?": "?", "!": "!", "…": ".", ";": ",", ":": ","}


def normalize(text, fold_yo=True):
    yo = "\u0451"
    t = unicodedata.normalize("NFC", text).lower().replace(yo, "\u0435" if fold_yo else yo)
    t = re.sub(r"[^\w\s'-]", " ", t)          # strip punctuation
    t = re.sub(r"[\s_]+", " ", t).strip()
    return t


def edit_ops(ref, hyp):
    """Levenshtein distance with backtracking. Returns (distance, alignment),
    a list of (i_ref|None, j_hyp|None)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    i, j, align = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            align.append((i-1, j-1)); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i-1][j] + 1:
            align.append((i-1, None)); i -= 1
        else:
            align.append((None, j-1)); j -= 1
    return d[n][m], align[::-1]


def wer(ref, hyp):
    r, h = normalize(ref).split(), normalize(hyp).split()
    return (edit_ops(r, h)[0], len(r))


def cer(ref, hyp):
    r, h = normalize(ref).replace(" ", ""), normalize(hyp).replace(" ", "")
    return (edit_ops(list(r), list(h))[0], len(r))


def tokens_with_marks(text):
    """Word -> (normalized word, terminal-punctuation class, is-uppercase?)."""
    out = []
    for raw in unicodedata.normalize("NFC", text).split():
        core = raw.strip("«»\"'()[]")
        mark = ""
        while core and core[-1] in PUNCT_CLASSES:
            mark = PUNCT_CLASSES[core[-1]] if not mark else mark
            core = core[:-1]
        if not core:
            continue
        out.append((normalize(core), mark, core[0].isupper()))
    return [x for x in out if x[0]]


def punct_and_case(ref, hyp):
    """Compare punctuation and case only on aligned matching words."""
    R, H = tokens_with_marks(ref), tokens_with_marks(hyp)
    _, align = edit_ops([x[0] for x in R], [x[0] for x in H])
    tp, fp, fn = Counter(), Counter(), Counter()
    case_ok = case_n = 0
    for i, j in align:
        if i is None or j is None or R[i][0] != H[j][0]:
            continue
        rm, hm = R[i][1], H[j][1]
        if rm and hm == rm: tp[rm] += 1
        elif rm and hm != rm: fn[rm] += 1; fp[hm] += (1 if hm else 0)
        elif hm: fp[hm] += 1
        case_n += 1; case_ok += int(R[i][2] == H[j][2])
    return tp, fp, fn, case_ok, case_n


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def read_tsv(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        rows[parts[0]] = parts[1] if len(parts) > 1 else ""
    return rows


def evaluate(refs, hyps):
    we = wn = ce = cn = 0
    tp, fp, fn = Counter(), Counter(), Counter()
    case_ok = case_n = 0
    per_file = []
    for k, ref in refs.items():
        hyp = hyps.get(k, "")
        e, n = wer(ref, hyp); we += e; wn += n
        e2, n2 = cer(ref, hyp); ce += e2; cn += n2
        t, f_, n_, co, cn_ = punct_and_case(ref, hyp)
        tp += t; fp += f_; fn += n_; case_ok += co; case_n += cn_
        per_file.append({"file": k, "wer": round(e / n, 4) if n else None, "ref_words": n})
    marks = {m: prf(tp[m], fp[m], fn[m]) for m in sorted(set(tp) | set(fp) | set(fn))}
    return {
        "files": len(refs),
        "WER": round(we / wn, 4) if wn else None,
        "CER": round(ce / cn, 4) if cn else None,
        "punct_PRF_by_mark": marks,
        "punct_F1_micro": prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))[2],
        "case_accuracy": round(case_ok / case_n, 4) if case_n else None,
        "per_file": per_file,
    }


def _selftest():
    ref = "Hello world."
    hyp = "hello world,"
    r = evaluate({"a": ref}, {"a": hyp})
    assert r["WER"] == 0.0, r          # words match; WER must ignore punctuation and case
    assert r["case_accuracy"] < 1.0    # lowercase versus uppercase word
    assert r["punct_PRF_by_mark"]["."][1] < 1.0  # one period is replaced by a comma
    assert normalize("\u0451") == "\u0435"
    assert normalize("\u0451", fold_yo=False) == "\u0451"
    print("selftest OK:", json.dumps({k: v for k, v in r.items() if k != "per_file"}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selftest()
    else:
        refs, hyps = read_tsv(sys.argv[1]), read_tsv(sys.argv[2])
        out = evaluate(refs, hyps)
        print(json.dumps({k: v for k, v in out.items() if k != "per_file"}, ensure_ascii=False, indent=2))
        if len(sys.argv) > 3:
            json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False, indent=2)
