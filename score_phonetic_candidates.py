#!/usr/bin/env python3
"""
score_phonetic_candidates.py — token-aware scoring of LASA candidate pairs.

Scores the SAME candidate-pair rows as score_candidates.py, but over one of the
phonetic representations rather than over brand_root spelling:

  --mode us     -> phones_g2p_arpa
  --mode hindi  -> predicted_normalized_phonemes

Output has the same five columns/order used everywhere else:
    jw, edit, kbigram, lumera, feline

For phonetic modes, JW and edit operate on PHONEME TOKENS, not characters.
FELINE uses the same affine-local-alignment recurrence over phoneme tokens.
kbigram and lumera are deliberately written as 0.0 because their trained
models are orthographic and have no defensible phoneme-space interpretation.

Example:
  python score_phonetic_candidates.py \
      --mode us \
      --phoneme-csv normalized_inference_output.csv \
      --pairs-csv /mnt/nexus/candidate_pairs.csv \
      --outdir /mnt/nexus/candidate_scores/phonetic_us \
      --dtype float16 --resume

  python score_phonetic_candidates.py \
      --mode hindi \
      --phoneme-csv normalized_inference_output.csv \
      --pairs-csv /mnt/nexus/candidate_pairs.csv \
      --outdir /mnt/nexus/candidate_scores/phonetic_hindi \
      --dtype float16 --resume
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import score_candidates as S

MEASURES = S.MEASURES
MODE_COLUMN = {
    "us": "phones_g2p_arpa",
    "hindi": "predicted_normalized_phonemes",
}


def load_phone_sequences(name_index: Path, phoneme_csv: Path, mode: str):
    ni = pd.read_csv(name_index, dtype={"name_id": np.int32, "brand_root": str},
                     keep_default_na=False)
    if not (ni["name_id"].to_numpy() == np.arange(len(ni))).all():
        sys.exit("name_index.csv name_id must be exactly 0..n-1 in row order")

    col = MODE_COLUMN[mode]
    ph = pd.read_csv(phoneme_csv, usecols=["brand_root", col],
                     dtype={"brand_root": str, col: str}, keep_default_na=False)
    if len(ph) != len(ni):
        sys.exit(f"{phoneme_csv}: {len(ph):,} rows, name index has {len(ni):,}")
    if not np.array_equal(ph["brand_root"].astype(str).to_numpy(),
                          ni["brand_root"].astype(str).to_numpy()):
        sys.exit("phoneme CSV brand_root order does not exactly match name_index.csv")

    tokens = [str(x).strip().split() if str(x).strip() else [] for x in ph[col]]
    vocab = sorted({t for seq in tokens for t in seq})
    tok = {t: i for i, t in enumerate(vocab)}
    seqs = [[tok[t] for t in seq] for seq in tokens]
    return ni["brand_root"].astype(str).tolist(), tokens, seqs, vocab


def make_encoding(seqs, n_tokens):
    over = max((len(x) for x in seqs), default=0)
    if over > S.MAXLEN:
        sys.exit(f"phoneme sequence length {over} exceeds MAXLEN={S.MAXLEN}")

    # JW/edit and FELINE all consume the same token sequence. kbigram/lumera
    # are intentionally empty because their learned models are orthographic.
    raw_off, raw_flat = S._pack(seqs, np.int16)
    fe_off, fe_flat = S._pack(seqs, np.int16)
    empty = [[] for _ in seqs]
    lu_off, lu_flat = S._pack(empty, np.int16)
    kb_off, kb_flat = S._pack(empty, np.int16)
    enc = (raw_off, raw_flat, fe_off, fe_flat,
           lu_off, lu_flat, kb_off, kb_flat, n_tokens)

    S_id = np.eye(max(n_tokens, 1), dtype=np.float32)
    dummy = np.zeros((1, 1), np.float32)
    models = (S_id, dummy, 0.6, 0.15, dummy, 0.5)
    return enc, models


def py_jw(a, b):
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0
    window = max(n, m) // 2 - 1
    window = max(window, 0)
    fa, fb = [False] * n, [False] * m
    matches = 0
    for i in range(n):
        for j in range(max(0, i-window), min(m-1, i+window)+1):
            if not fb[j] and a[i] == b[j]:
                fa[i] = True; fb[j] = True; matches += 1; break
    if matches == 0:
        return 0.0
    mb = [b[j] for j in range(m) if fb[j]]
    ma = [a[i] for i in range(n) if fa[i]]
    trans = sum(x != y for x, y in zip(ma, mb)) // 2
    mf = float(matches)
    jaro = (mf/n + mf/m + (mf-trans)/mf) / 3.0
    if jaro > S.JW_BOOST_THRESHOLD:
        p = 0
        for x, y in zip(a[:S.JW_PREFIX_MAX], b[:S.JW_PREFIX_MAX]):
            if x != y: break
            p += 1
        jaro += p * S.JW_PREFIX_SCALE * (1.0-jaro)
    return jaro


def py_edit(a, b):
    if not a and not b:
        return 1.0
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1] + (x != y)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b), 1)


def py_feline(a, b):
    n, m = len(a), len(b)
    if not n or not m:
        return 0.0
    H = [0.0] * (m+1)
    F = [0.0] * (m+1)
    best = 0.0
    for i in range(1, n+1):
        hdiag = 0.0
        ecol = 0.0
        H[0] = 0.0
        for j in range(1, m+1):
            e = max(H[j-1]-S.FELINE_GAP_OPEN, ecol-S.FELINE_GAP_EXT)
            ecol = e
            f = max(H[j]-S.FELINE_GAP_OPEN, F[j]-S.FELINE_GAP_EXT)
            F[j] = f
            v = max(0.0, hdiag + (1.0 if a[i-1] == b[j-1] else 0.0), e, f)
            hdiag, H[j] = H[j], v
            best = max(best, v)
    raw = best / max(n, m)
    p = 0
    for x, y in zip(a[:S.FELINE_PREFIX_MAX], b[:S.FELINE_PREFIX_MAX]):
        if x != y: break
        p += 1
    return min(1.0, max(0.0, raw + p*S.FELINE_PREFIX_SCALE*(1.0-raw)))


def verify(n, seqs, enc, models, use_numba, tol, seed):
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(seqs), n, dtype=np.int32)
    ib = rng.integers(0, len(seqs), n, dtype=np.int32)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    want = [True, True, False, False, True]
    got = S.score_ids(ia, ib, enc, models, want, use_numba)
    refs = {
        "jw": (0, py_jw),
        "edit": (1, py_edit),
        "feline": (4, py_feline),
    }
    bad = False
    print(f"checked on {len(ia):,} random phonetic pairs")
    for name, (col, fn) in refs.items():
        ref = np.array([fn(seqs[x], seqs[y]) for x, y in zip(ia, ib)], np.float64)
        d = np.abs(ref - got[:, col].astype(np.float64))
        print(f"  {name:<8} max={d.max():.3e} mean={d.mean():.3e}")
        bad |= bool(d.max() > tol)
    if bad:
        sys.exit("VERIFY FAILED")
    print("verify OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=sorted(MODE_COLUMN), required=True)
    ap.add_argument("--phoneme-csv", default="normalized_inference_output.csv")
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--name-index", default="results/name_index.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dtype", choices=("float32", "float16"), default="float16")
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--limit-parts", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-numba", action="store_true")
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress-secs", type=int, default=30)
    args = ap.parse_args()

    use_numba = S.HAVE_NUMBA and not args.no_numba
    print("="*78)
    print(f"  PHONETIC CANDIDATE SCORING — {args.mode.upper()}")
    print("="*78)
    print(f"representation  : {MODE_COLUMN[args.mode]}")
    print("measures        : jw, edit, feline (kbigram/lumera intentionally zero)")
    print(f"kernel          : {'numba (jit, all cores)' if use_numba else 'numpy fallback'}")

    names, token_text, seqs, vocab = load_phone_sequences(
        Path(args.name_index), Path(args.phoneme_csv), args.mode)
    print(f"names           : {len(names):,}")
    print(f"phoneme vocab   : {len(vocab):,}")
    enc, models = make_encoding(seqs, len(vocab))

    if args.verify:
        verify(args.verify, seqs, enc, models, use_numba, args.tol, args.seed)
        return

    path = Path(args.pairs_csv)
    if not path.exists():
        sys.exit(f"missing {path}")
    ranges = S.byte_chunks(path, args.chunks)
    jobs = [("range", i, (path, lo, hi)) for i, (lo, hi) in enumerate(ranges)]
    if args.limit_parts:
        jobs = jobs[:args.limit_parts]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dt = np.float16 if args.dtype == "float16" else np.float32
    want = [True, True, False, False, True]

    def load(job):
        _, _, spec = job
        a, b = S.read_range(*spec)
        return job, a, b

    manifest = []
    total_rows = 0
    t0 = tlast = time.time()
    depth = max(1, args.prefetch)
    with ThreadPoolExecutor(max_workers=depth) as pre:
        queue = [pre.submit(load, j) for j in jobs[:depth]]
        nxt = depth
        for n, job in enumerate(jobs):
            _, i, _ = job
            stem = f"chunk_{i:05d}"
            dest = outdir / f"{stem}.npy"
            _, ia, ib = queue.pop(0).result()
            if nxt < len(jobs):
                queue.append(pre.submit(load, jobs[nxt])); nxt += 1

            if args.resume and dest.exists():
                try:
                    if len(np.load(dest, mmap_mode="r")) == len(ia):
                        manifest.append((stem, total_rows, len(ia), str(dest)))
                        total_rows += len(ia)
                        continue
                except Exception:
                    pass

            sc = S.score_ids(ia, ib, enc, models, want, use_numba)
            np.save(dest, sc.astype(dt))
            manifest.append((stem, total_rows, len(ia), str(dest)))
            total_rows += len(ia)

            now = time.time()
            if now-tlast >= args.progress_secs or n+1 == len(jobs):
                print(f"  {n+1:,}/{len(jobs):,} | rows={total_rows:,} | "
                      f"{total_rows/max(now-t0,1e-9):,.0f} pairs/s | "
                      f"elapsed={(now-t0)/60:.1f}m")
                tlast = now

    mpath = outdir / "manifest.csv"
    with mpath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "first_row_index", "n_rows", "path"])
        w.writerows(manifest)

    (outdir / "representation.txt").write_text(
        f"mode={args.mode}\ncolumn={MODE_COLUMN[args.mode]}\n"
        "measures=jw,edit,feline\nkbigram=0\nlumera=0\n")
    el = time.time()-t0
    print("\nDONE")
    print(f"pairs scored    : {total_rows:,}")
    print(f"elapsed         : {el/60:.1f} min")
    print(f"throughput      : {total_rows/el:,.0f} pairs/s" if el else "")
    print(f"manifest        : {mpath}")


if __name__ == "__main__":
    main()
