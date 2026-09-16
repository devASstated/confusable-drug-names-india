#!/usr/bin/env python3
"""
bench_measures.py — benchmark similarity measures against the LASA refset.

Evaluates every measure on EQUAL TERMS so their numbers are comparable, and
so any learned measure has to earn its place against the established ones
rather than being assumed useful.

WHY THIS EXISTS
    The k-bigram prototype (kbigram.py / kbigram_train.py) was developed and
    benchmarked interactively; its results lived only in notes, not in the
    repository. That makes them anecdotes, not measurements. This script is the
    reproducible instrument: anyone can run it and get the numbers themselves.
    Add a new measure to MEASURES below and it is held to the same standard.

WHAT IT REPORTS
  1. STANDALONE AUC, cross-validated. Fixed measures (edit, JW, bigram-Dice)
     are simply scored per fold. LEARNED measures are RETRAINED INSIDE EACH
     FOLD on that fold's training pairs only — no leakage. Mean +/- std over
     repeated stratified folds, because a single split on ~1,400 pairs is a
     noisy estimate and quoting one number from it overstates precision.

  2. TWO EVALUATION VIEWS, which answer different questions:
       pair-level    — folds split PAIRS. A drug name may appear in both train
                       and test. Comparable to the historical prototype runs.
       name-disjoint — folds split NAMES. A test pair is kept only if BOTH its
                       names are unseen in training. This is the stronger
                       question: does the measure generalise to names it has
                       never encountered? A learned measure can look good on
                       the first view and poor on the second.

  3. CORRELATION between measures. A new measure that correlates ~0.95 with
     edit distance carries no new information however good its AUC looks.

  4. BLEND CONTRIBUTION. Held-out AUC of a logistic combination WITH and
     WITHOUT each measure. This is the real test of whether a measure earns
     its place: does adding it improve the combination?

CAVEAT THAT GOVERNS EVERY NUMBER BELOW
    The refset's negatives are SYNTHETIC RANDOM pairs, not expert-confirmed
    non-confusable pairs — no safety body publishes those. Separating
    documented pairs from RANDOM pairs is an easy task, so all AUC figures
    here are RELATIVE (valid for ranking measures against each other under
    identical conditions) and INFLATED as absolute numbers. They do not
    estimate real-world discrimination.

USAGE
    python bench_measures.py
    python bench_measures.py --folds 5 --repeats 3
    python bench_measures.py --skip-learned      # fixed measures only, fast
"""
import argparse
import numpy as np
import pandas as pd
import jellyfish


# ----------------------------------------------------------------------
# FIXED MEASURES
# ----------------------------------------------------------------------
def m_edit(a, b):
    m = max(len(a), len(b))
    return 1.0 - jellyfish.levenshtein_distance(a, b) / m if m else 1.0


def m_jw(a, b):
    return jellyfish.jaro_winkler_similarity(a, b)


def m_dice(a, b):
    A = {a[i:i+2] for i in range(len(a)-1)} or {a}
    B = {b[i:i+2] for i in range(len(b)-1)} or {b}
    return 2*len(A & B)/(len(A)+len(B)) if (A or B) else 0.0


FIXED = {"edit": m_edit, "jaro_winkler": m_jw, "bigram_dice": m_dice}


# ----------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------
def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    s = np.asarray(s)
    for v in np.unique(s):
        i = np.where(s == v)[0]
        ranks[i] = ranks[i].mean()
    pos = np.asarray(y) == 1
    np_, nn = pos.sum(), (~pos).sum()
    return float("nan") if np_ == 0 or nn == 0 else \
        (ranks[pos].sum() - np_*(np_+1)/2) / (np_*nn)


def logistic_heldout_auc(X, y, tr, te, iters=3000, lr=0.1, l2=1e-3):
    """Small logistic regression, no sklearn dependency."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1/(1 + np.exp(-(Xs[tr] @ w + b)))
        g = p - y[tr]
        w -= lr*(Xs[tr].T @ g/len(tr) + l2*w)
        b -= lr*g.mean()
    return auc(y[te], Xs[te] @ w + b)


# ----------------------------------------------------------------------
# SPLITS
# ----------------------------------------------------------------------
def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % k].append(ix)
    return [np.array(sorted(f)) for f in folds]


def name_disjoint_folds(a, b, y, k, seed):
    """Split by NAME. A pair joins fold f only if BOTH its names were assigned
    to f. Pairs straddling folds are dropped from the test set — that is the
    price of the stronger claim, and it is paid honestly rather than hidden."""
    rng = np.random.default_rng(seed)
    names = sorted(set(a) | set(b))
    rng.shuffle(names)
    home = {n: i % k for i, n in enumerate(names)}
    folds = [[] for _ in range(k)]
    for i, (x, z) in enumerate(zip(a, b)):
        if home[x] == home[z]:
            folds[home[x]].append(i)
    return [np.array(sorted(f)) for f in folds]


# ----------------------------------------------------------------------
# LEARNED MEASURES
# ----------------------------------------------------------------------
def make_kbigram(train_idx, a, b, y, rank=16, epochs=25, lr=0.3, l2=1e-2, seed=1):
    """Train the k-bigram kernel on train_idx ONLY; return a scorer."""
    from kbigram_train import bseq, align, BG
    sa = [bseq(x, BG) for x in a]
    sb = [bseq(z, BG) for z in b]
    active = sorted({g for i in train_idx for g in (sa[i] + sb[i])})
    amap = {g: j for j, g in enumerate(active)}
    if not active:
        return lambda i: 0.0
    A = [[amap[g] for g in s if g in amap] for s in sa]
    B = [[amap[g] for g in s if g in amap] for s in sb]
    F = np.random.default_rng(seed).normal(size=(len(active), rank)) * 0.3
    for _ in range(epochs):
        sc = np.array([align(A[i], B[i], F) for i in train_idx])
        err = sc - y[train_idx]
        gF = np.zeros_like(F)
        for k_, i in enumerate(train_idx):
            if not A[i] or not B[i]:
                continue
            S = F[A[i]] @ F[B[i]].T
            for ai, g in enumerate(A[i]):
                h = B[i][int(np.argmax(S[ai]))]
                gF[g] -= err[k_]*F[h]/len(train_idx)
                gF[h] -= err[k_]*F[g]/len(train_idx)
        F -= lr*(gF + l2*F)
    return lambda i: align(A[i], B[i], F)


LEARNED = {"kbigram": make_kbigram}
# To add LUMERA later: give it a make_lumera(train_idx, a, b, y) -> scorer(i)
# and register it here. It is then held to exactly this standard.


# ----------------------------------------------------------------------
def run_view(view, a, b, y, folds_fn, args, fixed_scores):
    print(f"\n{'='*70}\n  {view}\n{'='*70}")
    res = {m: [] for m in list(FIXED) + ([] if args.skip_learned else list(LEARNED))}
    n_test = []
    for rep in range(args.repeats):
        folds = folds_fn(rep)
        for f in range(args.folds):
            te = folds[f]
            tr = np.concatenate([folds[j] for j in range(args.folds) if j != f])
            if len(te) < 10 or len(set(y[te])) < 2:
                continue
            n_test.append(len(te))
            for m, v in fixed_scores.items():
                res[m].append(auc(y[te], v[te]))
            if not args.skip_learned:
                for m, mk in LEARNED.items():
                    sc = mk(tr, a, b, y)
                    res[m].append(auc(y[te], np.array([sc(i) for i in te])))
        print(f"  repeat {rep+1}/{args.repeats} done")
    print(f"\n  mean test-fold size: {int(np.mean(n_test)) if n_test else 0:,}")
    print(f"\n  {'measure':<16}{'AUC mean':>10}{'std':>9}")
    print(f"  {'-'*35}")
    for m, arr in res.items():
        if arr:
            arr = np.array(arr)
            print(f"  {m:<16}{arr.mean():>10.4f}{arr.std():>9.4f}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refset", default="results/lasa_refset.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-learned", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.refset).dropna(subset=["name_a", "name_b"])
    a = df.name_a.astype(str).values
    b = df.name_b.astype(str).values
    y = df.label.values.astype(int)
    print("="*70)
    print("  MEASURE BENCHMARK — LASA reference set")
    print("="*70)
    print(f"\npairs: {len(df):,}   positives: {(y==1).sum():,}   "
          f"negatives: {(y==0).sum():,}")
    print("\nCAVEAT: negatives are SYNTHETIC RANDOM pairs. Every AUC below is")
    print("RELATIVE (comparable between measures) but INFLATED as an absolute")
    print("number. Do not quote these as real-world discrimination.")

    fixed_scores = {m: np.array([f(x, z) for x, z in zip(a, b)])
                    for m, f in FIXED.items()}

    run_view("VIEW 1 — PAIR-LEVEL folds (a name may appear in train AND test)",
             a, b, y, lambda r: stratified_folds(y, args.folds, r),
             args, fixed_scores)
    run_view("VIEW 2 — NAME-DISJOINT folds (test names UNSEEN in training)",
             a, b, y, lambda r: name_disjoint_folds(a, b, y, args.folds, r),
             args, fixed_scores)

    # ---- correlation + blend, on a single honest split ----------------
    print(f"\n{'='*70}\n  CORRELATION BETWEEN MEASURES\n{'='*70}")
    all_scores = dict(fixed_scores)
    if not args.skip_learned:
        f0 = stratified_folds(y, args.folds, 0)
        tr0 = np.concatenate(f0[1:])
        for m, mk in LEARNED.items():
            sc = mk(tr0, a, b, y)
            all_scores[m] = np.array([sc(i) for i in range(len(y))])
    names = list(all_scores)
    print(f"\n  {'':<16}" + "".join(f"{n[:11]:>12}" for n in names))
    for n1 in names:
        row = "".join(f"{np.corrcoef(all_scores[n1], all_scores[n2])[0,1]:>12.3f}"
                      for n2 in names)
        print(f"  {n1:<16}{row}")
    print("\n  A measure correlating ~0.95 with an existing one carries little")
    print("  new information, however good its standalone AUC looks.")

    if not args.skip_learned:
        print(f"\n{'='*70}\n  BLEND CONTRIBUTION (held-out)\n{'='*70}")
        f0 = stratified_folds(y, args.folds, 0)
        te, tr = f0[0], np.concatenate(f0[1:])
        base = list(FIXED)
        Xb = np.column_stack([all_scores[m] for m in base])
        a0 = logistic_heldout_auc(Xb, y, tr, te)
        print(f"\n  {'+'.join(base)}")
        print(f"    without : {a0:.4f}")
        for m in LEARNED:
            Xw = np.column_stack([all_scores[k] for k in base + [m]])
            aw = logistic_heldout_auc(Xw, y, tr, te)
            print(f"    with {m:<10}: {aw:.4f}   ({aw-a0:+.4f})")
        print("\n  A positive delta is the only evidence that a measure earns")
        print("  its place. A near-zero delta means it is redundant.")
    print()


if __name__ == "__main__":
    main()
