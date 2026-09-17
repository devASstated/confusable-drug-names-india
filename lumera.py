#!/usr/bin/env python3
"""
lumera.py — LUMERA scorer (inference only).

Loads the character kernel learned by lumera_train.py and exposes score(a, b).
No training code, no pandas, no refset needed at runtime — this is the module
other scripts import.

    from lumera import score
    score("dopamine", "dobutamine")   # -> float in [0,1], higher = more alike

LUMERA is a QUARANTINED RESEARCH PROTOTYPE. It does NOT feed the deliverable
confusability score C, which uses established measures only (Jaro-Winkler,
edit distance, bigram-Dice). Benchmark it with bench_measures.py; do not wire
it into scoring without saying so explicitly.

If the model file is missing, train it first:
    python lumera_train.py
"""
import os

import numpy as np

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "lumera_model.npz")
_M = None   # lazy-loaded (M, cidx, gap_open, gap_ext)

# kept in step with lumera_train.ALPHABET; the artifact carries its own copy
# and that copy wins, so an older model still scores with the alphabet it was
# trained on rather than being silently reinterpreted.
_FALLBACK_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789- "


def _load(path=_MODEL_PATH):
    global _M
    if _M is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"model not found: {path}\nRun:  python lumera_train.py")
        d = np.load(path, allow_pickle=True)
        alpha = "".join(list(d["alphabet"])) if "alphabet" in d \
            else _FALLBACK_ALPHABET
        _M = (d["M"], {c: i for i, c in enumerate(alpha)},
              float(d["gap_open"]) if "gap_open" in d else 0.6,
              float(d["gap_ext"]) if "gap_ext" in d else 0.15)
    return _M


def _cseq(s, cidx):
    import re
    s = re.sub(r"[^a-z0-9\- ]", "", str(s).lower())
    s = re.sub(r"\s+", " ", s).strip()
    return [cidx[c] for c in s if c in cidx]


def _align(sa, sb, M, gap_open, gap_ext):
    n, m = len(sa), len(sb)
    if not n or not m:
        return 0.0
    Ma, Mb = M[sa], M[sb]
    Ma = Ma / (np.linalg.norm(Ma, axis=1, keepdims=True) + 1e-9)
    Mb = Mb / (np.linalg.norm(Mb, axis=1, keepdims=True) + 1e-9)
    S = np.clip(Ma @ Mb.T, 0.0, 1.0)
    H = np.zeros((n + 1, m + 1))
    E = np.zeros((n + 1, m + 1))
    F = np.zeros((n + 1, m + 1))
    best = 0.0
    for i in range(1, n + 1):
        row = S[i - 1]
        for j in range(1, m + 1):
            E[i, j] = max(H[i, j - 1] - gap_open, E[i, j - 1] - gap_ext)
            F[i, j] = max(H[i - 1, j] - gap_open, F[i - 1, j] - gap_ext)
            v = max(0.0, H[i - 1, j - 1] + row[j - 1], E[i, j], F[i, j])
            H[i, j] = v
            if v > best:
                best = v
    return best / max(n, m)


def score(name_a, name_b):
    """LUMERA resemblance in [0,1] between two drug names."""
    M, cidx, go, ge = _load()
    return float(_align(_cseq(name_a, cidx), _cseq(name_b, cidx), M, go, ge))


def model_info():
    """Metadata of the loaded artifact — which model produced a given column."""
    d = np.load(_MODEL_PATH, allow_pickle=True)
    return {k: (float(d[k]) if d[k].ndim == 0 and d[k].dtype.kind == "f"
                else int(d[k]) if d[k].ndim == 0 and d[k].dtype.kind in "iu"
                else None)
            for k in ("heldout_auc", "train_auc", "rank", "gap_open",
                      "gap_ext", "split_seed", "init_seed", "epochs")
            if k in d}


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        print(f"{score(sys.argv[1], sys.argv[2]):.4f}")
    else:
        print(__doc__)
        try:
            print("loaded model:", model_info())
        except FileNotFoundError as e:
            print(e)
