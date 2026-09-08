#!/usr/bin/env python3
"""
kbigram_train.py — LEARN the K-Bigram confusability kernel.

Trains the low-rank bigram kernel F on a fixed 60% split of the ISMP/IMSN
refset and writes the model artifact (kbigram_model.npz), which the scorer
(kbigram.py) loads at inference time. Run this only when you want to (re)train;
day-to-day scoring never touches this file.

    K-Bigram: local sequence alignment over a name's bigrams, where matching
    bigram g to h pays a LEARNED, cosine-bounded similarity <f_g,f_h> in [0,1].
    Alignment preserves order (what makes confusability measures work); the
    learned kernel adds "similar bigrams attract".

USAGE
    python kbigram_train.py                       # train with defaults
    python kbigram_train.py --rank 16 --epochs 25 # tune
"""
import argparse
import numpy as np
import pandas as pd

# shared bigram vocabulary (must match the scorer's alphabet)
CH = "abcdefghijklmnopqrstuvwxyz0123456789-"
BG = {}
for a in CH:
    for b in CH:
        BG[a + b] = len(BG)


def bseq(s, vocab):
    s = str(s).lower()
    return [vocab[s[i:i+2]] for i in range(len(s) - 1) if s[i:i+2] in vocab]


def align(sa, sb, F, gap=0.5):
    """Local alignment; each match cosine-bounded to [0,1]; normalised by longer len."""
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


def auc(y, sc):
    order = np.argsort(sc); ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc)+1)
    s = np.asarray(sc)
    for v in np.unique(s):
        ii = np.where(s == v)[0]; ranks[ii] = ranks[ii].mean()
    pos = y == 1; npos, nneg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - npos*(npos+1)/2) / (npos*nneg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refset", default="results/lasa_refset.csv")
    ap.add_argument("--out", default="kbigram_model.npz")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.refset).dropna(subset=["name_a", "name_b"])
    y = df.label.values.astype(int)
    sa_full = [bseq(x, BG) for x in df.name_a]
    sb_full = [bseq(x, BG) for x in df.name_b]

    # compact vocab of bigrams that actually occur
    active = sorted({g for s in sa_full + sb_full for g in s})
    amap = {g: i for i, g in enumerate(active)}
    inv = {v: k for k, v in BG.items()}
    A = [[amap[g] for g in s] for s in sa_full]
    B = [[amap[g] for g in s] for s in sb_full]
    NB = len(active)

    idx = np.random.default_rng(args.split_seed).permutation(len(df))
    cut = int(0.6 * len(df)); tr, te = idx[:cut], idx[cut:]
    F = np.random.default_rng(args.init_seed).normal(size=(NB, args.rank)) * 0.3

    def score_all(ix):
        return np.array([align(A[i], B[i], F) for i in ix])

    print(f"training: {NB} active bigrams, rank {args.rank}, {len(tr)} train pairs")
    for ep in range(args.epochs):
        s = score_all(tr); err = s - y[tr]; gF = np.zeros_like(F)
        for k, i in enumerate(tr):
            a, b = A[i], B[i]
            if not a or not b:
                continue
            Smat = F[a] @ F[b].T
            for ai, g in enumerate(a):
                h = b[int(np.argmax(Smat[ai]))]
                gF[g] -= err[k] * F[h] / len(tr)
                gF[h] -= err[k] * F[g] / len(tr)
        F -= args.lr * (gF + args.l2 * F)

    tra, tea = auc(y[tr], score_all(tr)), auc(y[te], score_all(te))
    print(f"  train AUC {tra:.4f}   held-out AUC {tea:.4f}")

    # write the model artifact: F, vocab (as strings), metadata
    np.savez(args.out,
             F=F,
             active=np.array([inv[a] for a in active]),
             rank=args.rank, gap=0.5,
             train_idx=tr, test_idx=te,
             heldout_auc=tea)
    print(f"  model -> {args.out}")


if __name__ == "__main__":
    main()
