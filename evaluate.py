#!/usr/bin/env python3
"""
evaluate.py — Target 3: validate the confusability engine on real LASA pairs.

Two independent, separately-reported metrics (run on results/lasa_refset.csv):

  METRIC 1 — SCORING SEPARATION (does the score track expert danger?)
      Score every positive and negative pair with the SAME jw/edit blend the
      engine uses. Report ROC-AUC, PR-AUC, positive vs negative score
      distributions, and a precision/recall/F1 table across thresholds.
      Answers: does a high confusability score mean a real, expert LASA pair?

  METRIC 2 — BLOCKING RECALL (do the keys keep real pairs together?)
      Run BOTH names of each POSITIVE pair through the blocker's key logic and
      check whether they share at least one block. This is the honest,
      at-scale replacement for the 9-pair toy probe.
      Answers: would blocking have surfaced these known-dangerous pairs?

Both are name-agnostic: they validate the METHOD, so the international names
in the refset are fine — we are not testing Indian-namespace coverage here.

USAGE
    python evaluate.py
    python evaluate.py --w-jw 0.3 --w-edit 0.7
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import jellyfish

# reuse the blocker's exact key logic so recall reflects the real pipeline
from block import keys_for


def jw(a, b):
    return jellyfish.jaro_winkler_similarity(a, b)


def edit_sim(a, b):
    m = max(len(a), len(b))
    return 1.0 - jellyfish.levenshtein_distance(a, b) / m if m else 1.0


def auc_from_scores(labels, scores):
    """ROC-AUC via the rank-sum (Mann-Whitney) identity — no sklearn needed."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    s = np.asarray(scores)
    for v in np.unique(s):
        idx = np.where(s == v)[0]
        ranks[idx] = ranks[idx].mean()
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refset", default="results/lasa_refset.csv")
    ap.add_argument("--w-jw", type=float, default=0.4)
    ap.add_argument("--w-edit", type=float, default=0.6)
    args = ap.parse_args()

    df = pd.read_csv(args.refset).dropna(subset=["name_a", "name_b"])
    a = df.name_a.astype(str).values
    b = df.name_b.astype(str).values
    y = df.label.values.astype(int)

    jws = np.array([jw(x, z) for x, z in zip(a, b)])
    eds = np.array([edit_sim(x, z) for x, z in zip(a, b)])
    score = args.w_jw * jws + args.w_edit * eds

    # ================= METRIC 1 — SCORING SEPARATION =================
    print("=" * 64)
    print("  METRIC 1 — SCORING SEPARATION")
    print("=" * 64)
    pos, neg = score[y == 1], score[y == 0]
    print(f"positives: {len(pos):,}   negatives: {len(neg):,}   "
          f"weights jw={args.w_jw} edit={args.w_edit}")
    print(f"\n  positive score  median {np.median(pos):.3f}   mean {pos.mean():.3f}")
    print(f"  negative score  median {np.median(neg):.3f}   mean {neg.mean():.3f}")
    print(f"  separation (pos-neg median): {np.median(pos)-np.median(neg):+.3f}")

    roc = auc_from_scores(y, score)
    print(f"\n  ROC-AUC : {roc:.4f}   (0.5 = coin flip, 1.0 = perfect)")

    print(f"\n  threshold   precision  recall     F1   (positives kept above t)")
    for t in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        pred = score >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"    {t:>5.2f}      {prec:>6.3f}   {rec:>6.3f}  {f1:>6.3f}")

    # ================= METRIC 2 — BLOCKING RECALL ===================
    print("\n" + "=" * 64)
    print("  METRIC 2 — BLOCKING RECALL (positives only)")
    print("=" * 64)
    pos_df = df[df.label == 1]
    kept, missed = 0, []
    per_key_saves = {}
    for x, z in zip(pos_df.name_a.astype(str), pos_df.name_b.astype(str)):
        ka = {(p, k) for p, k in keys_for(x)}
        kb = {(p, k) for p, k in keys_for(z)}
        shared = ka & kb
        if shared:
            kept += 1
            for p, _ in shared:                    # credit each key-type that saved it
                per_key_saves[p] = per_key_saves.get(p, 0) + 1
        else:
            missed.append((x, z))
    n = len(pos_df)
    print(f"documented positive pairs : {n:,}")
    print(f"kept together by blocking : {kept:,}  ({100*kept/n:.1f}% recall)")
    print(f"blocked apart (missed)    : {len(missed):,}")
    print(f"\n  pairs saved by each key-type (a pair can be saved by several):")
    for p in ["metaphone", "dmeta", "nysiis", "soundex", "prefix4", "suffix4", "del1"]:
        print(f"    {p:<10} {per_key_saves.get(p,0):>5,}")
    if missed:
        print(f"\n  sample of missed pairs (the genuinely hard ones):")
        for x, z in missed[:15]:
            print(f"    {x[:24]:<26} / {z}")

    # save per-pair scores for the report / further analysis
    out = df.copy()
    out["jw"], out["edit_sim"], out["score"] = jws, eds, score
    out.to_csv("results/eval_scored.csv", index=False, float_format="%.4f")
    print(f"\nwrote per-pair scores -> results/eval_scored.csv")


if __name__ == "__main__":
    main()
