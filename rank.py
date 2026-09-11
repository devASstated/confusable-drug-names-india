#!/usr/bin/env python3
"""
rank.py — top-N ranking with tunable weights (Target 2, part 2 of 2).

Streams scored_pairs.csv, keeps the best N by the combined jw/edit score in a
heap (flat memory), then — for those N survivors only — computes the expensive
K-Bigram alignment score, so you can compare it against the standard measures
on the pairs your engine actually surfaced.

    combined = w_jw*jw + w_edit*edit_sim        (the ranking score)
    columns shown: jw  edit  bigram_dice  combined  kbigram

USAGE
    python rank.py results/scored_pairs.csv --top 50
    python rank.py results/scored_pairs.csv --top 50 --no-digits --w-jw 0.3 --w-edit 0.7
"""
import argparse, heapq, itertools, re
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scored_csv")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--w-jw", type=float, default=0.4)
    ap.add_argument("--w-edit", type=float, default=0.6)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--out", default="")
    ap.add_argument("--print", dest="show", type=int, default=0,
                    help="how many to print (default: all of --top)")
    ap.add_argument("--no-digits", action="store_true",
                    help="skip pairs where either root has a digit (hides dose variants)")
    ap.add_argument("--no-kbigram", action="store_true",
                    help="skip the k-bigram column (faster)")
    args = ap.parse_args()
    show = args.show if args.show > 0 else args.top
    wj, we = args.w_jw, args.w_edit
    has_digit = re.compile(r"\d").search

    heap = []; counter = itertools.count(); seen = 0
    have_dice = None
    for chunk in pd.read_csv(args.scored_csv, chunksize=args.chunk):
        if have_dice is None:
            have_dice = "bigram_dice" in chunk.columns
        chunk["combined"] = wj*chunk["jw"] + we*chunk["edit_sim"]
        dice_col = chunk["bigram_dice"] if have_dice else [float("nan")]*len(chunk)
        for a, b, jw, es, dc, sc in zip(chunk.root_a, chunk.root_b, chunk.jw,
                                        chunk.edit_sim, dice_col, chunk.combined):
            seen += 1
            if args.no_digits and (has_digit(str(a)) or has_digit(str(b))):
                continue
            item = (sc, next(counter), (a, b, jw, es, dc, sc))
            if len(heap) < args.top:
                heapq.heappush(heap, item)
            elif sc > heap[0][0]:
                heapq.heapreplace(heap, item)

    top = [h[2] for h in sorted(heap, key=lambda h: h[0], reverse=True)]

    # k-bigram only on the survivors (expensive DP alignment)
    kb = [float("nan")] * len(top)
    if not args.no_kbigram:
        try:
            from kbigram import score as kbscore
            kb = [kbscore(a, b) for (a, b, *_ ) in top]
        except Exception as e:
            print(f"[k-bigram unavailable: {e}]")

    flt = "  (no-digits)" if args.no_digits else ""
    print(f"scanned {seen:,} pairs   weights jw={wj} edit={we}   top {len(top)}{flt}")
    print(f"\n  {'root_a':<51}{'root_b':<51}{'jw':>6}{'edit':>6}{'dice':>6}{'comb':>7}{'kbig':>7}")
    for (a, b, jw, es, dc, sc), k in list(zip(top, kb))[:show]:
        dcs = f"{dc:>6.2f}" if dc == dc else "   -  "
        kbs = f"{k:>7.2f}" if k == k else "   -   "
        print(f"  {str(a)[:50]:<51}{str(b)[:50]:<51}{jw:>6.2f}{es:>6.2f}{dcs}{sc:>7.3f}{kbs}")
    print("\n  note: jw/edit/dice/comb are 0-1 look-alike scores; kbig (k-bigram)")
    print("  is on its own compressed scale - compare its RANK order, not its level.")

    if args.out:
        pd.DataFrame([(a, b, jw, es, dc, sc, k) for (a, b, jw, es, dc, sc), k in zip(top, kb)],
                     columns=["root_a","root_b","jw","edit_sim","bigram_dice","combined","kbigram"]
                     ).to_csv(args.out, index=False, float_format="%.4f")
        print(f"\nwrote top {len(top)} -> {args.out}")


if __name__ == "__main__":
    main()
