#!/usr/bin/env python3
"""
transcribe_en_us.py — transcribe brand roots to General American phonemes.

Stage 1 of the phonetic path. Reads the cleaned brand roots and writes one
pronunciation per root; transcribe_compare.py consumes this file next.

Transcription uses g2p-en: CMUdict lookup, with a neural seq2seq fallback for
words not in the dictionary. Every invented brand name takes the fallback
path. The output is one hypothesis — General American — not a claim about how
the names are actually pronounced in India. The file is named after its
hypothesis so a second provider can sit beside it rather than overwrite it.

Output is ARPABET (space-separated), which is already segmented, so there is
no risk of splitting a phoneme the way naive indexing of IPA would split a
two-code-point sound. Stress digits are dropped: AH0, AH1 and AH2 all become
AH, since stress is emphasis rather than phoneme identity and keeping it would
split one sound across three key symbols.

Columns: brand_root, phones, n_phones
A root that transcribes to nothing is written with an empty phones field
rather than dropped, so the file remains a complete image of the namespace.

Usage:
    pip install g2p-en
    python transcribe_en_us.py results/brand_roots_clean.csv
    python transcribe_en_us.py results/brand_roots_clean.csv --resume
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots_csv")
    ap.add_argument("--out", default="results/phones_en_us.csv")
    ap.add_argument("--col", default="brand_root")
    ap.add_argument("--limit", type=int, default=0,
                    help="transcribe only the first N roots")
    ap.add_argument("--resume", action="store_true",
                    help="keep rows already in --out and transcribe only the "
                         "roots missing from it")
    ap.add_argument("--progress-secs", type=int, default=120,
                    help="seconds between progress lines")
    args = ap.parse_args()

    try:
        from g2p_en import G2p
    except ImportError:
        sys.exit("needs g2p-en:  pip install g2p-en\n"
                 "if NLTK complains on first run:\n"
                 "  python -c \"import nltk; "
                 "nltk.download('averaged_perceptron_tagger_eng')\"")

    df = pd.read_csv(args.roots_csv)
    if args.col not in df.columns:
        sys.exit(f"input needs a '{args.col}' column; found {list(df.columns)}")
    roots = df[args.col].dropna().astype(str).drop_duplicates().tolist()
    if args.limit:
        roots = roots[:args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = {}
    if args.resume and out.exists():
        prev = pd.read_csv(out, dtype=str, keep_default_na=False)
        done = dict(zip(prev["brand_root"], prev["phones"]))
        print(f"resuming: {len(done):,} roots already transcribed",
              file=sys.stderr)

    todo = [r for r in roots if r not in done]
    print(f"roots to transcribe : {len(todo):,} of {len(roots):,}",
          file=sys.stderr)
    if not todo:
        print("nothing to do", file=sys.stderr)
        return

    g2p = G2p()
    t0 = t_last = time.time()
    empty = 0
    results = dict(done)

    for i, root in enumerate(todo, 1):
        try:
            ph = [p.strip().rstrip("012") for p in g2p(root)
                  if p.strip() and p.strip()[0].isalpha()]
        except Exception:
            ph = []
        if not ph:
            empty += 1
        results[root] = " ".join(ph)

        now = time.time()
        if now - t_last >= args.progress_secs or i == len(todo):
            rate = i / max(now - t0, 1e-9)
            left = (len(todo) - i) / max(rate, 1e-9)
            print(f"  {i:,}/{len(todo):,}  ({100*i/len(todo):.1f}%)  "
                  f"{rate:.0f} roots/s  ~{left/60:.1f} min left",
                  file=sys.stderr)
            t_last = now

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["brand_root", "phones", "n_phones"])
        for r in roots:
            ph = results.get(r, "")
            w.writerow([r, ph, len(ph.split()) if ph else 0])

    lens = [len(results[r].split()) for r in roots if results.get(r)]
    print("\n" + "=" * 60)
    print("  TRANSCRIPTION — General American (g2p-en, ARPABET)")
    print("=" * 60)
    print(f"roots written        : {len(roots):,}")
    print(f"empty transcription  : {empty:,}")
    if lens:
        lens.sort()
        print(f"phones per root      : min {lens[0]}  median "
              f"{lens[len(lens)//2]}  max {lens[-1]}")
    print(f"elapsed              : {(time.time()-t0)/60:.1f} min")
    print(f"\nwrote -> {out}")
    print(f"next: python transcribe_compare.py {out}")


if __name__ == "__main__":
    main()
