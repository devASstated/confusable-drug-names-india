#!/usr/bin/env python3
"""
score.py — orthographic scoring, streamed (Target 2, part 1 of 2).

Reads candidate_pairs.csv in CHUNKS, scores each chunk, writes it straight
to disk, frees it — peak memory is one chunk, not the whole ~20M. Ranking is
rank.py's job. Cheap per-pair measures only (the expensive k-bigram alignment
is computed at rank time on the top-N, not here).

Columns written: root_a, root_b, jw, edit_sim, bigram_dice
"""
import argparse, sys
from pathlib import Path
import pandas as pd
import jellyfish


def _bigram_dice(x, y):
    A = {x[i:i+2] for i in range(len(x)-1)} or {x}
    B = {y[i:i+2] for i in range(len(y)-1)} or {y}
    return 2*len(A & B)/(len(A)+len(B)) if (A or B) else 0.0


def score_chunk(df: pd.DataFrame) -> pd.DataFrame:
    a = df["root_a"].astype(str); b = df["root_b"].astype(str)
    keep = a != b                                   # drop self-pairs
    a, b = a[keep], b[keep]
    jw, es, dc = [], [], []
    for x, y in zip(a, b):
        jw.append(jellyfish.jaro_winkler_similarity(x, y))
        m = max(len(x), len(y))
        es.append(1.0 - jellyfish.levenshtein_distance(x, y)/m if m else 1.0)
        dc.append(_bigram_dice(x, y))
    return pd.DataFrame({"root_a": a.values, "root_b": b.values,
                         "jw": jw, "edit_sim": es, "bigram_dice": dc})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs_csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "scored_pairs.csv"
    total, hdr = 0, False
    with open(out, "w", newline="") as fout:
        for chunk in pd.read_csv(args.pairs_csv, chunksize=args.chunk,
                                 dtype=str, keep_default_na=False):
            sc = score_chunk(chunk)
            sc.to_csv(fout, header=not hdr, index=False, float_format="%.4f")
            hdr = True; total += len(sc)
            print(f"  scored {total:,} pairs ...", file=sys.stderr)
    print(f"\ndone: {total:,} scored pairs -> {out}", file=sys.stderr)
    print(f"next: python rank.py {out} --top 50", file=sys.stderr)


if __name__ == "__main__":
    main()
