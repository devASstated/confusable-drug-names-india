#!/usr/bin/env python3
"""
kbigram.py — K-Bigram SCORER (inference only).

Loads the kernel learned by kbigram_train.py (kbigram_model.npz) and exposes
score(a, b). No training code, no pandas, no refset needed at runtime — this is
the lightweight module that rank.py and evaluate scripts import.

    from kbigram import score
    score("pantopride", "pantoride")   # -> float in [0,1], higher = more confusable

If the model file is missing, train it first:
    python kbigram_train.py
"""
import os
import numpy as np

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "kbigram_model.npz")
_M = None   # lazy-loaded (amap, F, gap)


def _load(path=_MODEL_PATH):
    global _M
    if _M is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"model not found: {path}\nRun:  python kbigram_train.py")
        d = np.load(path, allow_pickle=True)
        active = list(d["active"])
        amap = {g: i for i, g in enumerate(active)}
        gap = float(d["gap"]) if "gap" in d else 0.5
        _M = (amap, d["F"], gap)
    return _M


def _bseq(s, vocab):
    s = str(s).lower()
    return [vocab[s[i:i+2]] for i in range(len(s) - 1) if s[i:i+2] in vocab]


def _align(sa, sb, F, gap):
    if not sa or not sb:
        return 0.0
    Fa = F[sa]; Fb = F[sb]
    Fa = Fa / (np.linalg.norm(Fa, axis=1, keepdims=True) + 1e-9)
    Fb = Fb / (np.linalg.norm(Fb, axis=1, keepdims=True) + 1e-9)
    S = np.clip(Fa @ Fb.T, 0.0, 1.0)
    n, m = len(sa), len(sb)
    prev = np.zeros(m + 1); best = 0.0
    for i in range(1, n + 1):
        cur = np.zeros(m + 1); row = S[i - 1]
        for j in range(1, m + 1):
            v = max(0.0, prev[j-1] + row[j-1], prev[j] - gap, cur[j-1] - gap)
            cur[j] = v
            if v > best:
                best = v
        prev = cur
    return best / max(n, m)


def score(name_a, name_b):
    """K-Bigram similarity in [0,1] between two drug names."""
    amap, F, gap = _load()
    return float(_align(_bseq(name_a, amap), _bseq(name_b, amap), F, gap))


def model_info():
    """Return metadata about the loaded model (held-out AUC, rank, vocab size)."""
    d = np.load(_MODEL_PATH, allow_pickle=True)
    return {"heldout_auc": float(d["heldout_auc"]) if "heldout_auc" in d else None,
            "rank": int(d["rank"]) if "rank" in d else None,
            "vocab": len(d["active"])}


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
