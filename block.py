#!/usr/bin/env python3
"""
block.py — recall-first blocking for the confusability engine (Target 1).

THE PROBLEM
    Comparing every brand root to every other is N*(N-1)/2 comparisons.
    For ~199,000 roots that is ~20 billion pairs — far too many to score.

THE IDEA
    "Blocking" stamps each name with a few short KEYS. Only names that
    share a key are ever compared. Everything else is assumed too
    different to bother scoring.

WHY MULTI-PASS (this is the recall-first part)
    A single key is brittle: a real look-alike pair can fall into two
    different buckets and be missed. So we run SEVERAL independent passes
    and take the UNION of the pairs they produce. A pair survives if it
    collides in ANY pass — so a name that hides from one key can still be
    caught by another. This keeps as many real pairs as possible, at the
    cost of a larger candidate set (which the scoring stage then filters).

THE PASSES
    metaphone  — a phonetic code               (sound-alike)
    dmeta      — double-metaphone primary code (sound-alike, alt spellings)
    nysiis     — a second phonetic algorithm   (sound-alike, more recall)
    prefix4    — first 4 letters               (look-alike, shared start)
    suffix4    — last 4 letters                (look-alike, shared end)

OUTPUT
    candidate_pairs.csv  — the pairs the scoring stage will consume
    block_report.txt     — block sizes, candidate count, reduction ratio

USAGE
    python block.py brand_roots.csv
    python block.py brand_roots.csv --check known_lasa_pairs.csv
    python block.py brand_roots.csv --max-block 4000   # optional skew guard
"""

import argparse
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
import jellyfish
from metaphone import doublemetaphone


# --------------------------------------------------------------------------
# KEY FUNCTIONS  — each turns a name into one short blocking key
# --------------------------------------------------------------------------

def letters_only(root: str) -> str:
    """Strip to bare lowercase letters: 'olez-d' -> 'olezd'. Keys are built
    from this so punctuation/spacing never splits a bucket by accident."""
    return re.sub(r"[^a-z]", "", str(root).lower())


def keys_for(root: str):
    """Return the set of blocking keys for one name, tagged by pass name.
    Empty/degenerate keys are dropped so we never bucket on ''. """
    s = letters_only(root)
    if len(s) < 2:                       # 1-letter roots have no useful key
        return []
    keys = []
    mp = jellyfish.metaphone(s)
    if mp:
        keys.append(("metaphone", mp))
    dm_primary, _ = doublemetaphone(s)
    if dm_primary:
        keys.append(("dmeta", dm_primary))
    ny = jellyfish.nysiis(s)
    if ny:
        keys.append(("nysiis", ny))
    sx = jellyfish.soundex(s)            # crude phonetic — over-collides on
    if sx:                               # purpose, to catch distant sound-alikes
        keys.append(("soundex", sx))
    keys.append(("prefix4", s[:4]))      # shared start — the strongest LASA signal
    keys.append(("suffix4", s[-4:]))     # shared end — real once dosage debris is
                                         # cleaned out (catches dopamine/dobutamine).
                                         # Run clean_roots.py FIRST or this bloats.
    # deletion neighbourhood: name with each single char removed. Two names
    # within edit-distance 1 share at least one of these -> catches olez/olex,
    # the single most common look-alike shape. Capped by length to stay cheap.
    if 3 <= len(s) <= 14:
        for k in range(len(s)):
            keys.append(("del1", s[:k] + s[k + 1:]))
    return keys


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots_csv", help="CSV with a 'brand_root' column")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--max-block", type=int, default=0,
                    help="skip blocks larger than this (0 = keep all; "
                         "recall-first default). Use only if memory is tight.")
    ap.add_argument("--check", default="",
                    help="CSV of known pairs (cols root_a,root_b) to verify "
                         "they survive blocking — a quick recall probe.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- load the distinct brand roots -------------------------------
    df = pd.read_csv(args.roots_csv)
    if "brand_root" not in df.columns:
        sys.exit("ERROR: input needs a 'brand_root' column")
    roots = (df["brand_root"].dropna().astype(str).drop_duplicates()
             .reset_index(drop=True))
    N = len(roots)
    id_of = {r: i for i, r in enumerate(roots)}      # name -> integer id
    print(f"distinct brand roots : {N:,}")

    # ---- PASS 1: assign every root to its blocks ---------------------
    # blocks[(pass, key)] = list of root-ids sharing that key
    blocks = defaultdict(list)
    for i, r in enumerate(roots):
        for pass_name, key in keys_for(r):
            blocks[(pass_name, key)].append(i)

    # ---- PASS 2: within each block, emit all pairs -------------------
    # We pack an unordered pair (i<j) into a single int  i*N + j  and keep
    # a set of them, so a pair found in several passes is stored only once.
    pairset = set()
    skipped_big = 0
    per_pass = defaultdict(int)
    big_blocks = []

    for (pass_name, key), ids in blocks.items():
        b = len(ids)
        if b < 2:
            continue
        if args.max_block and b > args.max_block:
            skipped_big += 1
            big_blocks.append((b, pass_name, key))
            continue
        if b > 1500:                      # remember the skewed ones to report
            big_blocks.append((b, pass_name, key))
        ids.sort()
        for i, jx in combinations(ids, 2):
            pairset.add(i * N + jx)
            per_pass[pass_name] += 1      # counts pre-dedup contribution

    # ---- stats -------------------------------------------------------
    all_pairs = N * (N - 1) // 2
    n_cand = len(pairset)
    reduction = 100 * (1 - n_cand / all_pairs) if all_pairs else 0
    block_sizes = sorted((len(v) for v in blocks.values() if len(v) > 1),
                         reverse=True)

    report = []
    report.append("=" * 64)
    report.append("  BLOCKING REPORT — recall-first multi-pass")
    report.append("=" * 64)
    report.append(f"brand roots            : {N:,}")
    report.append(f"all-pairs (no blocking): {all_pairs:,}")
    report.append(f"candidate pairs (union): {n_cand:,}")
    report.append(f"reduction vs all-pairs : {reduction:.4f}%")
    report.append(f"avg comparisons / root : {2*n_cand/max(N,1):.1f}")
    report.append("")
    report.append("candidate contribution by pass (before de-dup):")
    for p in ["metaphone", "dmeta", "nysiis", "soundex", "prefix4", "suffix4", "del1"]:
        report.append(f"    {p:<10} {per_pass.get(p,0):>14,}")
    report.append("")
    report.append(f"non-singleton blocks   : {len(block_sizes):,}")
    if block_sizes:
        report.append(f"largest block          : {block_sizes[0]:,}")
        report.append(f"median block size      : {block_sizes[len(block_sizes)//2]:,}")
    if args.max_block:
        report.append(f"blocks skipped (>{args.max_block}) : {skipped_big:,}  "
                      f"(recall lost here — tune if needed)")
    if big_blocks:
        report.append("")
        report.append("biggest blocks (watch these for skew):")
        for b, p, k in sorted(big_blocks, reverse=True)[:8]:
            report.append(f"    {b:>7,}  {p}:{k}")
    report_txt = "\n".join(report)
    print("\n" + report_txt)
    (outdir / "block_report.txt").write_text(report_txt + "\n")

    # ---- optional recall probe against known pairs -------------------
    if args.check:
        chk = pd.read_csv(args.check)
        found, missed = 0, []
        for a, bb in zip(chk.root_a.astype(str), chk.root_b.astype(str)):
            ia, ib = id_of.get(a), id_of.get(bb)
            if ia is None or ib is None:
                missed.append((a, bb, "not in roots"))
                continue
            i, jx = sorted((ia, ib))
            if i * N + jx in pairset:
                found += 1
            else:
                missed.append((a, bb, "blocked apart"))
        print(f"\n[RECALL PROBE] known pairs kept: {found}/{len(chk)}")
        for a, bb, why in missed:
            print(f"    MISSED  {a} / {bb}  ({why})")

    # ---- write candidate pairs for the scoring stage -----------------
    # (unpack the packed ints back into name pairs)
    out = outdir / "candidate_pairs.csv"
    with out.open("w") as f:
        f.write("root_a,root_b\n")
        for packed in pairset:
            i, jx = divmod(packed, N)
            f.write(f"{roots[i]},{roots[jx]}\n")
    print(f"\nwrote {n_cand:,} candidate pairs -> {out}")


if __name__ == "__main__":
    main()
