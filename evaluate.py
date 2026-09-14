#!/usr/bin/env python3
"""
evaluate.py — Target 3: validate the confusability engine on real LASA pairs.

Two independent, separately-reported metrics (run on results/lasa_refset.csv):

  METRIC 1 — SCORING SEPARATION (does the score track documented confusability?)
      Score every positive and negative pair with the SAME jw/edit blend the
      engine uses. Report ROC-AUC, PR-AUC (average precision), positive vs
      negative score distributions, and a precision/recall/F1 table across
      thresholds.
      Answers: does a high confusability score mean a pair that expert lists
      actually record as confusable?

      NOTE this metric evaluates CONFUSABILITY (C) only. It says nothing about
      clinical danger — that is the divergence term (D), computed downstream.
      A high C pair may be clinically harmless (same drug, different brand).

  METRIC 2 — BLOCKING RECALL (do the keys keep real pairs together?)
      Run BOTH names of each POSITIVE pair through the blocker's key logic and
      check whether they share at least one block. This is the honest,
      at-scale replacement for the 9-pair toy probe.
      Answers: would blocking have surfaced these known-dangerous pairs?

Both are name-agnostic: they validate the METHOD, so the international names
in the refset are fine — we are not testing Indian-namespace coverage here.

CAVEAT (applies to every number below): the negatives in the refset are
SYNTHETIC RANDOM pairs, not expert-confirmed non-confusable pairs — no safety
body publishes those. Separating documented pairs from RANDOM pairs is an
easy task, so all AUC/AP figures here are RELATIVE (valid for comparing
measures under identical conditions) but INFLATED as absolute numbers. Do not
quote them as real-world discrimination.

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


def average_precision(labels, scores):
    """PR-AUC as average precision: sum over positives of P(k) * dRecall(k).

    Computed by walking the pairs in descending score order and accumulating
    precision at each rank where a positive is retrieved. Ties are handled by
    grouping equal scores into one rank block, so a block of tied pairs cannot
    be credited as if it were perfectly ordered inside the block.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s)                 # descending score
    y, s = y[order], s[order]

    ap_sum = 0.0
    tp = 0
    seen = 0
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:  # one block of tied scores
            j += 1
        block_pos = int((y[i:j] == 1).sum())
        seen += (j - i)
        tp += block_pos
        if block_pos:
            precision_here = tp / seen      # precision after consuming the block
            ap_sum += precision_here * block_pos
        i = j
    return ap_sum / n_pos


def baseline_precision(labels):
    """Precision of a random ranker = positive prevalence. PR-AUC floor."""
    y = np.asarray(labels)
    return float((y == 1).sum()) / len(y) if len(y) else float("nan")


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
    prauc = average_precision(y, score)
    base = baseline_precision(y)
    print(f"\n  ROC-AUC : {roc:.4f}   (0.5 = coin flip, 1.0 = perfect)")
    print(f"  PR-AUC  : {prauc:.4f}   (average precision; "
          f"random baseline = {base:.4f} = positive prevalence)")
    print(f"            PR-AUC is the more honest headline here: the refset is")
    print(f"            imbalanced ({int((y==1).sum()):,} pos / {int((y==0).sum()):,} neg), and ROC-AUC")
    print(f"            flatters imbalanced, easy-negative tasks.")

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
    per_key_saves = {}      # pairs this pass kept together (deduped per pair)
    per_key_unique = {}     # pairs ONLY this pass kept — its irreplaceable value
    for x, z in zip(pos_df.name_a.astype(str), pos_df.name_b.astype(str)):
        ka = {(p, k) for p, k in keys_for(x)}
        kb = {(p, k) for p, k in keys_for(z)}
        shared = ka & kb
        if shared:
            kept += 1
            passes = {p for p, _ in shared}        # DEDUPE: one credit per pass
            for p in passes:                       # (a pair can share several
                per_key_saves[p] = per_key_saves.get(p, 0) + 1   # keys of one pass)
            if len(passes) == 1:                   # this pass ALONE saved the pair
                only = next(iter(passes))
                per_key_unique[only] = per_key_unique.get(only, 0) + 1
        else:
            missed.append((x, z))
    n = len(pos_df)
    print(f"documented positive pairs : {n:,}")
    print(f"kept together by blocking : {kept:,}  ({100*kept/n:.1f}% recall)")
    print(f"blocked apart (missed)    : {len(missed):,}")
    print(f"\n  {'pass':<10}{'saved':>8}{'ONLY this pass':>16}")
    print(f"  {'-'*10}{'-'*8}{'-'*16}")
    for p in ["metaphone", "dmeta", "nysiis", "soundex", "prefix4", "suffix4", "del1"]:
        print(f"  {p:<10}{per_key_saves.get(p,0):>8,}{per_key_unique.get(p,0):>16,}")
    print(f"\n  'saved'          = pairs this pass kept together (counted ONCE per")
    print(f"                     pair, even if the pair shares several of its keys)")
    print(f"  'ONLY this pass' = pairs NO other pass caught. A pass with a high")
    print(f"                     'saved' but ~0 unique is redundant; a pass with")
    print(f"                     unique saves is irreplaceable for recall.")
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
