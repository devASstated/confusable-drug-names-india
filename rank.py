#!/usr/bin/env python3
"""
rank.py — top-N ranking with tunable weights (Target 2, part 2 of 2).

Streams scored_pairs.csv in chunks and keeps only the best N pairs in a
running heap — so memory stays flat (holds N rows, never the full 20.6M),
and re-ranking with new weights is a cheap re-read, not a re-score.

    score = w_jw * jw  +  w_edit * edit_sim

USAGE
    python rank.py results/scored_pairs.csv --top 50
    python rank.py results/scored_pairs.csv --top 5000 --w-jw 0.3 --w-edit 0.7
    python rank.py results/scored_pairs.csv --top 2000 --out results/top2000.csv

Score once with score.py; tune weights here as often as you like.
"""

import argparse
import heapq
import itertools
import re
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scored_csv", help="scored_pairs.csv from score.py")
    ap.add_argument("--top", type=int, default=50,
                    help="how many top-ranked pairs to keep (default 50)")
    ap.add_argument("--w-jw", type=float, default=0.4,
                    help="weight on Jaro-Winkler (default 0.4 — JW runs high)")
    ap.add_argument("--w-edit", type=float, default=0.6,
                    help="weight on edit-distance similarity (default 0.6)")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--out", default="", help="optional CSV to write the top-N to")
    ap.add_argument("--print", dest="show", type=int, default=0,
                    help="how many of the top-N to print (default: all of --top)")
    ap.add_argument("--no-digits", action="store_true",
                    help="skip pairs where either root contains a digit — hides "
                         "same-brand dose/variant leftovers so real look-alikes show")
    args = ap.parse_args()

    # default: print exactly as many as we kept (fixes the --top/--print mismatch)
    show = args.show if args.show > 0 else args.top

    wj, we = args.w_jw, args.w_edit
    heap = []                          # min-heap of (score, tiebreak, row-tuple)
    counter = itertools.count()        # stable tiebreak so tuples never compare
    seen = 0
    has_digit = re.compile(r"\d").search

    for chunk in pd.read_csv(args.scored_csv, chunksize=args.chunk):
        chunk["score"] = wj * chunk["jw"] + we * chunk["edit_sim"]
        for a, b, jw, es, sc in zip(chunk.root_a, chunk.root_b,
                                    chunk.jw, chunk.edit_sim, chunk.score):
            seen += 1
            if args.no_digits and (has_digit(str(a)) or has_digit(str(b))):
                continue               # hide dose/variant leftovers from the view
            item = (sc, next(counter), (a, b, jw, es, sc))
            if len(heap) < args.top:
                heapq.heappush(heap, item)
            elif sc > heap[0][0]:      # beats the current worst kept
                heapq.heapreplace(heap, item)

    top = [h[2] for h in sorted(heap, key=lambda h: h[0], reverse=True)]
    flt = "  (no-digits filter)" if args.no_digits else ""
    print(f"scanned {seen:,} pairs   weights: jw={wj} edit={we}   top {len(top)}{flt}")
    print(f"\n  {'root_a':<20}{'root_b':<20}{'jw':>7}{'edit':>7}{'score':>8}")
    for a, b, jw, es, sc in top[:show]:
        print(f"  {str(a)[:19]:<20}{str(b)[:19]:<20}{jw:>7.3f}{es:>7.3f}{sc:>8.3f}")

    if args.out:
        pd.DataFrame(top, columns=["root_a", "root_b", "jw", "edit_sim", "score"]) \
          .to_csv(args.out, index=False, float_format="%.4f")
        print(f"\nwrote top {len(top)} -> {args.out}")


if __name__ == "__main__":
    main()
