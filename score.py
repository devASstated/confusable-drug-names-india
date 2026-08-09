#!/usr/bin/env python3
"""
score.py — orthographic scoring, streamed (Target 2, part 1 of 2).

Reads candidate_pairs.csv in CHUNKS, scores each chunk, writes it straight
to disk, and frees it — so peak memory is ONE chunk, not the whole 20.6M.
Does NOT rank. Ranking is rank.py's job (a cheap top-N pass you re-run when
tuning weights), so the expensive scoring happens once.

Two metrics, kept as separate raw columns (they are NOT on the same scale —
jw runs ~0.1-0.3 higher — so never collapse them here):
    jw        Jaro-Winkler similarity
    edit_sim  1 - levenshtein/max(len)

Output: results/scored_pairs.csv  (root_a, root_b, jw, edit_sim)

USAGE
    python score.py results/candidate_pairs.csv
    python score.py results/candidate_pairs.csv --chunk 1000000
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import jellyfish


def score_chunk(df: pd.DataFrame) -> pd.DataFrame:
    a = df["root_a"].astype(str)
    b = df["root_b"].astype(str)
    keep = a != b                                   # drop self-pairs (useless 1.0)
    a, b = a[keep], b[keep]
    jw, es = [], []
    for x, y in zip(a, b):
        jw.append(jellyfish.jaro_winkler_similarity(x, y))
        d = jellyfish.levenshtein_distance(x, y)
        m = max(len(x), len(y))
        es.append(1.0 - d / m if m else 1.0)
    return pd.DataFrame({"root_a": a.values, "root_b": b.values,
                         "jw": jw, "edit_sim": es})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs_csv", help="candidate_pairs.csv from block.py")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--chunk", type=int, default=1_000_000,
                    help="rows scored per chunk (default 1,000,000)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "scored_pairs.csv"

    total, wrote_header = 0, False
    with open(out, "w", newline="") as fout:
        for chunk in pd.read_csv(args.pairs_csv, chunksize=args.chunk,
                                 dtype=str, keep_default_na=False):
            scored = score_chunk(chunk)
            scored.to_csv(fout, header=not wrote_header, index=False,
                          float_format="%.4f")
            wrote_header = True
            total += len(scored)
            print(f"  scored {total:,} pairs ...", file=sys.stderr)

    print(f"\ndone: {total:,} scored pairs -> {out}", file=sys.stderr)
    print(f"next: python rank.py {out} --top 50", file=sys.stderr)


if __name__ == "__main__":
    main()
