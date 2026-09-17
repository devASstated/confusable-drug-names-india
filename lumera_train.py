#!/usr/bin/env python3
"""
lumera_train.py — learn the LUMERA character-confusability kernel.

LUMERA — Learned Unified Metric for Resemblance Alignment.

WHAT IT IS
    Local sequence alignment (Smith-Waterman with affine gaps) over the two
    names' CHARACTERS, where the reward for aligning character x with
    character y is a LEARNED similarity in [0,1] instead of a hard 0/1 match.
    The alignment supplies sequence order — the property that makes
    confusability measures work at all — and the learned kernel supplies
    "these two characters look alike" (o/0, b/d, the vowels).

WHY CHARACTER LEVEL
    The earlier bigram-level prototype (kbigram.py) generalised fine but
    correlated 0.939 with plain bigram-Dice and beat it by 0.0085 AUC: a
    rank-16 kernel over 452 bigrams was largely reconstructing a measure that
    already exists and is simpler. A character kernel is not a reimplementation
    of any existing measure, and it is ~20x smaller, which is why it is worth
    trying instead of tuning the bigram version further.

WHAT IS NOT NOVEL HERE
    Learning edit/alignment costs from examples is established work (Ristad &
    Yianilos 1998; Bilenko & Mooney 2003 added affine gaps; Neural String Edit
    Distance 2022; differentiable Smith-Waterman exists). The narrow claim is
    the application: a compact, interpretable character-confusability kernel
    learned from LASA-labelled drug-name pairs. Do not describe "learned
    similarity inside alignment" as new.

SPECIFICATION (locked here so results are reproducible, not recovered)
    alphabet     a-z, 0-9, hyphen, space. Characters outside it are DROPPED.
                 Space is kept deliberately: 'vent sf' and 'ventsf' are
                 different strings and the word boundary is information.
    kernel       M is (alphabet x rank). Character similarity is the cosine of
                 two rows, clipped to [0,1], so a match can never score above
                 an exact match and the DP stays bounded.
    alignment    Smith-Waterman, local, affine gaps (open != extend), so a
                 shared RUN costs one gap-open rather than one per character.
    score        best cell in the DP / length of the LONGER bigram-free
                 character sequence, giving [0,1].
    training     surrogate gradient, NOT backpropagation — see below.
    class weight OFF by default. The refset is 63% positive; weighting is
                 available via --class-weight but changes the objective, so it
                 is not silently on.
    seeds        --split-seed controls the train/test split, --init-seed the
                 kernel initialisation. Both are recorded in the artifact.

ON THE TRAINING RULE (be honest about this in any write-up)
    The DP contains max() operations, so the alignment path is not a smooth
    function of M. This implementation uses a SURROGATE update: score each
    training pair, take err = score - label, and for every character of A pull
    its best-matching character in B closer (if the pair should score higher)
    or push it away (if lower), with weight decay. It ignores the gaps and the
    chosen path.
    This is a crude approximation, and it is NOT evidence that proper
    differentiable training is infeasible. Smooth Smith-Waterman via
    log-sum-exp is established and works, including with affine gaps. An
    earlier exploratory soft-DP attempt here was numerically unstable; that is
    a statement about that implementation, not about the method. Implement the
    surrogate faithfully first so that architecture and optimiser are not
    changed at the same time.

USAGE
    python lumera_train.py
    python lumera_train.py --rank 8 --epochs 30
    python lumera_train.py --show-kernel      # print what it learned
"""
import argparse
import re

import numpy as np
import pandas as pd

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789- "
CIDX = {c: i for i, c in enumerate(ALPHABET)}
NC = len(ALPHABET)
_KEEP = re.compile(r"[^a-z0-9\- ]")


def cseq(s):
    """Name -> list of character indices. Unsupported characters are dropped."""
    s = _KEEP.sub("", str(s).lower())
    s = re.sub(r"\s+", " ", s).strip()
    return [CIDX[c] for c in s if c in CIDX]


def align(sa, sb, M, gap_open=0.6, gap_ext=0.15):
    """Smith-Waterman with affine gaps. Match reward = clipped cosine in [0,1].

    H = best score ending here; E = best ending in a gap in b (horizontal);
    F = best ending in a gap in a (vertical). The max(0, ...) makes it LOCAL,
    so an alignment can start anywhere rather than being dragged down by a
    poor prefix.
    """
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


def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    s = np.asarray(s)
    for v in np.unique(s):
        i = np.where(s == v)[0]
        ranks[i] = ranks[i].mean()
    pos = np.asarray(y) == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def train_kernel(A, B, y, train_idx, rank=8, epochs=30, lr=0.4, l2=1e-2,
                 gap_open=0.6, gap_ext=0.15, init_seed=1, class_weight=False,
                 verbose=False):
    """Surrogate-gradient training. Returns the kernel M."""
    M = np.random.default_rng(init_seed).normal(size=(NC, rank)) * 0.3
    w = np.ones(len(y))
    if class_weight:                       # balance if asked; off by default
        npos, nneg = (y == 1).sum(), (y == 0).sum()
        w = np.where(y == 1, len(y) / (2 * max(npos, 1)),
                     len(y) / (2 * max(nneg, 1)))
    for ep in range(epochs):
        sc = np.array([align(A[i], B[i], M, gap_open, gap_ext)
                       for i in train_idx])
        err = (sc - y[train_idx]) * w[train_idx]
        gM = np.zeros_like(M)
        for k, i in enumerate(train_idx):
            a, b = A[i], B[i]
            if not a or not b:
                continue
            S = M[a] @ M[b].T
            for ai, ca in enumerate(a):
                cb = b[int(np.argmax(S[ai]))]
                gM[ca] -= err[k] * M[cb] / len(train_idx)
                gM[cb] -= err[k] * M[ca] / len(train_idx)
        M -= lr * (gM + l2 * M)
        if verbose and (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{epochs}")
    return M


def kernel_table(M, top=15):
    """Most-similar character pairs the kernel learned (excluding self-match)."""
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = np.clip(Mn @ Mn.T, 0, 1)
    out = []
    for i in range(NC):
        for j in range(i + 1, NC):
            out.append((S[i, j], ALPHABET[i], ALPHABET[j]))
    return sorted(out, reverse=True)[:top]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refset", default="results/lasa_refset.csv")
    ap.add_argument("--out", default="lumera_model.npz")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--gap-open", type=float, default=0.6)
    ap.add_argument("--gap-ext", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=1)
    ap.add_argument("--class-weight", action="store_true",
                    help="balance positive/negative weight (refset is 63%% pos)")
    ap.add_argument("--show-kernel", action="store_true",
                    help="print the character pairs the kernel learned")
    args = ap.parse_args()

    df = pd.read_csv(args.refset).dropna(subset=["name_a", "name_b"])
    y = df.label.values.astype(int)
    A = [cseq(x) for x in df.name_a]
    B = [cseq(z) for z in df.name_b]

    idx = np.random.default_rng(args.split_seed).permutation(len(df))
    cut = int(0.6 * len(df))
    tr, te = idx[:cut], idx[cut:]

    print(f"LUMERA training")
    print(f"  alphabet {NC} chars, rank {args.rank}, {len(tr):,} train pairs")
    print(f"  gaps: open {args.gap_open}, extend {args.gap_ext}   "
          f"class_weight={args.class_weight}")
    M = train_kernel(A, B, y, tr, rank=args.rank, epochs=args.epochs,
                     lr=args.lr, l2=args.l2, gap_open=args.gap_open,
                     gap_ext=args.gap_ext, init_seed=args.init_seed,
                     class_weight=args.class_weight, verbose=True)

    s_tr = np.array([align(A[i], B[i], M, args.gap_open, args.gap_ext) for i in tr])
    s_te = np.array([align(A[i], B[i], M, args.gap_open, args.gap_ext) for i in te])
    a_tr, a_te = auc(y[tr], s_tr), auc(y[te], s_te)
    print(f"\n  train AUC {a_tr:.4f}   held-out AUC {a_te:.4f}")
    print(f"  NOTE this is ONE 60/40 split. For the number to report, run")
    print(f"  bench_measures.py, which cross-validates and also tests whether")
    print(f"  the measure survives on names unseen in training.")

    if args.show_kernel:
        print(f"\n  top learned character similarities:")
        for s, c1, c2 in kernel_table(M):
            n1 = "' '" if c1 == " " else c1
            n2 = "' '" if c2 == " " else c2
            print(f"    {n1} ~ {n2}   {s:.3f}")
        print(f"  Characters that appear rarely in the refset are barely")
        print(f"  constrained by training; treat their values as noise.")

    np.savez(args.out, M=M, alphabet=np.array(list(ALPHABET)),
             rank=args.rank, gap_open=args.gap_open, gap_ext=args.gap_ext,
             train_idx=tr, test_idx=te, heldout_auc=a_te, train_auc=a_tr,
             split_seed=args.split_seed, init_seed=args.init_seed,
             epochs=args.epochs, lr=args.lr, l2=args.l2,
             class_weight=args.class_weight)
    print(f"\n  model -> {args.out}")


if __name__ == "__main__":
    main()
