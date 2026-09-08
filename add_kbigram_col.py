#!/usr/bin/env python3
"""
add_kbigram_col.py — add (or refresh) a K-Bigram/LAK score column on an
already-scored pairs file, IN PLACE, so you can eyeball how the learned
measure rates the same real drug-name pairs as the standard columns — and
compare across model improvements by using a different column name each time.

It imports the CURRENT model via kbigram.score(), so after you retrain
(kbigram_train.py) this script automatically uses the latest kernel.

    python add_kbigram_col.py results/scored_pairs.csv --col kbigram
    python add_kbigram_col.py results/scored_pairs.csv --col lak_blockmatch
    python add_kbigram_col.py results/scored_pairs.csv --col kbigram --limit 5000
    python add_kbigram_col.py results/scored_pairs.csv --col kbigram --force

NOTES
  * --col   name of the column to add. If it already exists the script stops
            unless --force is given (so you don't silently overwrite a run).
  * --limit score only the first N rows (rest left blank) — for a quick look
            before committing to the full, slow pass over ~19.7M pairs.
  * The K-Bigram score is a per-pair DP alignment, so a full pass is SLOW by
            nature. Use --limit first; run the full pass when you mean it.
"""
import argparse, os, sys, shutil, tempfile, csv
from kbigram import score as kbscore   # imports the CURRENT cached model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scored_csv")
    ap.add_argument("--col", default="kbigram", help="name of the column to add")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N rows (0 = all)")
    ap.add_argument("--resume", action="store_true",
                    help="continue a partial run: skip rows that already have a "
                         "value in --col and score only the empty ones. Use only "
                         "if the SAME model produced the existing values.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the column if it already exists")
    ap.add_argument("--a", default="root_a")
    ap.add_argument("--b", default="root_b")
    args = ap.parse_args()

    path = args.scored_csv
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")

    # peek header
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    if args.col in header and not args.force and not args.resume:
        sys.exit(f"column '{args.col}' already exists — use --resume to fill only "
                 f"the empty rows, --force to recompute all, or a new --col name.")
    if args.a not in header or args.b not in header:
        sys.exit(f"need '{args.a}' and '{args.b}' columns; found {header}")

    if args.resume and args.col not in header:
        sys.exit(f"--resume needs an existing '{args.col}' column to continue from; "
                 f"none found. Run without --resume to start it.")

    ia, ib = header.index(args.a), header.index(args.b)
    col_exists = args.col in header
    icol = header.index(args.col) if col_exists else None

    # stream row-by-row -> temp file (flat memory, safe for 19.7M rows)
    tmpfd, tmppath = tempfile.mkstemp(suffix=".csv", dir=os.path.dirname(path) or ".")
    os.close(tmpfd)
    n, scored, skipped = 0, 0, 0
    with open(path, newline="") as fin, open(tmppath, "w", newline="") as fout:
        r = csv.reader(fin); w = csv.writer(fout)
        _ = next(r)                               # old header
        out_header = header if col_exists else header + [args.col]
        w.writerow(out_header)
        for row in r:
            n += 1
            # resume: keep rows that already have a value; score only the empties
            if args.resume and col_exists and icol < len(row) and row[icol] != "":
                skipped += 1
                w.writerow(row)
                continue
            do = (args.limit == 0 or scored < args.limit)
            val = f"{kbscore(row[ia], row[ib]):.4f}" if do else ""
            if do:
                scored += 1
            if col_exists:
                while len(row) <= icol:
                    row.append("")
                row[icol] = val
                w.writerow(row)
            else:
                w.writerow(row + [val])
            if n % 200000 == 0:
                print(f"  processed {n:,} rows ({scored:,} newly scored, "
                      f"{skipped:,} kept) ...", file=sys.stderr)

    shutil.move(tmppath, path)
    print(f"done: {n:,} rows, {scored:,} newly scored, {skipped:,} already filled "
          f"-> column '{args.col}' in {path}", file=sys.stderr)
    if args.limit:
        print(f"(stopped after {args.limit:,} new scores; rerun --resume for more)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
