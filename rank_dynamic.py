#!/usr/bin/env python3
"""
rank_dynamic.py — interactive re-weighting of the 2.24-billion-pair candidate set.

score_candidates.py wrote five similarity columns per pair into float16 sidecars
(20.91 GiB over 400 chunks). This module turns that into something a slider can
drive, and is the single implementation of the ranking maths that the Streamlit
app imports — so the app and the CLI cannot drift apart.

THE TWO PROBLEMS, AND WHY THE SECOND ONE HAS AN EXACT ANSWER
    1. The sidecars hold scores but NOT ids. name_id_a/name_id_b exist only in
       candidate_pairs.csv (97.76 GiB), so nothing can name the pairs it ranks
       until the ids are extracted once.

    2. A full re-rank means reading all 20.91 GiB: 6 s on fast NVMe, 45 s on a
       slow disk. Acceptable for a batch job, useless behind a slider.

    The prune for (2) is exact rather than heuristic. For weights w with every
    w_c >= 0 and sum(w) = 1,

        s . w = SUM_c w_c s_c  <=  SUM_c w_c max_c s_c  =  max_c s_c

    so a row whose LARGEST column is below T cannot reach T under ANY such
    weighting. Keeping the rows with max >= T therefore loses nothing that any
    convex re-weighting could have surfaced. That is a guarantee about the whole
    weight simplex, not about one setting, and it is why the shortlist can be
    built once and reused for every slider position.

    The guarantee is also CHECKABLE at query time: if the Nth-best score in the
    shortlist is itself >= T, the top-N is provably the true top-N of all
    2.24e9 rows. rank() returns that certificate, and says so when it fails
    instead of quietly serving an approximate list.

    NEGATIVE WEIGHTS BREAK THE BOUND. A measure given a negative weight can
    pull a high-max row down and push a low-max row up, so the pruned rows are
    no longer provably irrelevant. rank() refuses negative weights rather than
    returning a wrong answer; use --full for that case.

STAGES
    survey  read the score sidecars only (no CSV) and report the distribution
            of the row max, so T is chosen from the data instead of guessed.
    build   write the shortlist: one pass over the sidecars to decide which
            rows survive, one pass over the CSV to recover their ids and the
            blocker's own provenance columns.
    rank    load the shortlist and re-rank in RAM, with the exactness check.
    full    exact top-N over every row for one weight vector, including
            negative weights. Needs the CSV (or a prior build) to name the
            winners. Slow, and the way to confirm a shortlist answer.

USAGE
    python rank_dynamic.py survey  --scores /mnt/nexus/candidate_scores
    python rank_dynamic.py build   --scores /mnt/nexus/candidate_scores \
                                   --pairs-csv /mnt/nexus/candidate_pairs.csv \
                                   --min-max 0.55
    python rank_dynamic.py rank    --w-feline 0.4 --w-jw 0.3 --w-edit 0.3 --top 50
    python rank_dynamic.py rank    --preset feline-led --top 200 --out top200.csv
    python rank_dynamic.py full    --w-feline 1.0 --top 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MEASURES = ("jw", "edit", "kbigram", "lumera", "feline")
N_MEAS = len(MEASURES)

# Starting points, not recommendations. The TTAB validation found jw and feline
# the only two measures that separated adjudicated-similar from
# adjudicated-dissimilar better than chance by a clear margin, and found lumera
# below plain edit distance there; 'equal' is included because it is the
# obvious thing to try first, not because it is defensible.
PRESETS = {
    "equal":        dict(jw=.2, edit=.2, kbigram=.2, lumera=.2, feline=.2),
    "feline-led":   dict(jw=.3, edit=.2, kbigram=.0, lumera=.0, feline=.5),
    "established":  dict(jw=.5, edit=.5, kbigram=.0, lumera=.0, feline=.0),
    "ttab-informed": dict(jw=.45, edit=.1, kbigram=.0, lumera=.0, feline=.45),
}

SHORTLIST_NAME = "shortlist.npz"
STORE_NAME = "shortlist_store"

# row(uint32) 4 + ia/ib(int32) 8 + scores(f16 x5) 10 + flags 1 + blocker(f16 x3) 6
BYTES_PER_ROW = 29
STORE_COLS = ("row", "ia", "ib", "scores", "flags", "blocker")


# ----------------------------------------------------------------------
# sidecar plumbing
# ----------------------------------------------------------------------
def load_manifest(scores_dir):
    """Global row order comes from manifest.csv, not from sorting filenames.

    The manifest is what score_candidates.py wrote to record each unit's first
    row index; trusting a glob instead would silently reorder everything if a
    chunk were ever renamed or renumbered.
    """
    d = Path(scores_dir)
    mpath = d / "manifest.csv"
    if not mpath.exists():
        sys.exit(f"missing {mpath} — rerun score_candidates.py, or point "
                 f"--scores at the directory that holds it")
    m = pd.read_csv(mpath)
    need = {"unit", "first_row_index", "n_rows", "path"}
    if not need.issubset(m.columns):
        sys.exit(f"{mpath} lacks columns {need - set(m.columns)}")
    m = m.sort_values("first_row_index").reset_index(drop=True)

    # the recorded offsets must tile the row space with no gap or overlap
    expect = np.r_[0, np.cumsum(m.n_rows.to_numpy()[:-1])]
    if not (m.first_row_index.to_numpy() == expect).all():
        sys.exit(f"{mpath}: first_row_index does not match the cumulative row "
                 f"counts; the sidecars cannot be concatenated safely")

    missing = [p for p in m.path if not Path(p).exists()]
    if missing:
        sys.exit(f"{len(missing)} sidecar(s) in the manifest are missing, "
                 f"first: {missing[0]}")
    return m


def open_chunk(path):
    return np.load(path, mmap_mode="r")


# ----------------------------------------------------------------------
def survey(args):
    """Distribution of the per-row max, which is what --min-max thresholds."""
    m = load_manifest(args.scores)
    total = int(m.n_rows.sum())
    print("=" * 78)
    print("  SURVEY — what the five score columns actually look like")
    print("=" * 78)
    print(f"chunks          : {len(m):,}")
    print(f"rows            : {total:,}")

    ladder = np.array([0.0, .30, .40, .45, .50, .55, .60, .65, .70, .75,
                       .80, .85, .90, .95], np.float32)
    above = np.zeros(len(ladder), np.int64)
    bins = np.zeros(101, np.int64)
    per_col_sum = np.zeros(N_MEAS, np.float64)
    per_col_max = np.zeros(N_MEAS, np.float64)
    seen = 0
    t0 = tlast = time.time()

    for i, r in enumerate(m.itertuples()):
        s = np.asarray(open_chunk(r.path), np.float16)
        if s.ndim != 2 or s.shape[1] != N_MEAS:
            sys.exit(f"{r.path}: expected (n,{N_MEAS}), got {s.shape}")
        f = s.astype(np.float32)
        rowmax = f.max(axis=1)
        above += (rowmax[None, :] >= ladder[:, None]).sum(axis=1)
        bins += np.bincount(np.clip((rowmax * 100).astype(np.int32), 0, 100),
                            minlength=101)
        per_col_sum += f.sum(axis=0)
        per_col_max = np.maximum(per_col_max, f.max(axis=0))
        seen += len(f)
        now = time.time()
        if now - tlast >= args.progress_secs or i + 1 == len(m):
            print(f"  {i+1:,}/{len(m):,} chunks | {seen:,} rows | "
                  f"{seen/(now-t0)/1e6:.1f}M rows/s | "
                  f"{(now-t0)/60:.1f} min")
            tlast = now

    print(f"\n  mean score per measure")
    for c, name in enumerate(MEASURES):
        print(f"    {name:<10}{per_col_sum[c]/seen:>8.4f}   "
              f"max observed {per_col_max[c]:.4f}")

    print(f"\n  rows whose LARGEST column is >= T. Under any weighting with")
    print(f"  non-negative weights summing to 1, no row below T can score T,")
    print(f"  so this column is exactly the shortlist size --min-max T gives.")
    print(f"\n  {'T':>6}{'rows >= T':>18}{'fraction':>11}"
          f"{'store GiB':>15}{'scores GiB':>12}{'slider':>13}")
    print(f"  {'-'*75}")
    for t, n in zip(ladder, above):
        # 4 row(uint32) + 8 ids + 10 scores(f16 x5) + 1 flag + 6 blocker(f16 x3)
        gib = n * BYTES_PER_ROW / 1024 ** 3
        sc = n * 10 / 1024 ** 3
        verdict = ("instant" if sc < 0.6 else "ok" if sc < 2.0
                   else "sluggish" if sc < 6.0 else "batch only")
        print(f"  {t:>6.2f}{n:>18,}{n/seen:>11.5f}{gib:>15.2f}{sc:>12.2f}"
              f"{verdict:>13}")

    nz = np.flatnonzero(bins)
    print(f"\n  row-max histogram (1% buckets, non-empty only)")
    peak = bins.max()
    for b in nz:
        bar = "#" * max(1, int(60 * bins[b] / peak))
        print(f"    {b/100:>4.2f} {bins[b]:>14,}  {bar}")
    print(f"\n  'store GiB' is the whole shortlist; 'scores GiB' is all a")
    print(f"  re-rank actually reads, which is what sets slider latency once")
    print(f"  the file is in page cache (~12 GiB/s). Build at the LOW T you")
    print(f"  want the exactness guarantee to cover — `compact --tiers` then")
    print(f"  cuts the fast views out of it without re-reading the CSV.")


# ----------------------------------------------------------------------
def build(args):
    """Write the shortlist: ids + scores + blocker provenance for surviving rows.

    Two passes, because the cheap data and the expensive data live apart. Pass 1
    touches only the 20.91 GiB of sidecars and decides which rows survive. Pass 2
    streams the 97.76 GiB CSV once to recover ids for exactly those rows. Doing
    it the other way round would mean parsing the CSV to discover what to keep.
    """
    m = load_manifest(args.scores)
    total = int(m.n_rows.sum())
    T = float(args.min_max)
    print("=" * 78)
    print("  BUILD SHORTLIST")
    print("=" * 78)
    print(f"rows            : {total:,}")
    print(f"--min-max T     : {T}")
    print(f"guarantee       : every pair reachable by non-negative weights")
    print(f"                  summing to 1 with a score >= {T} is kept")

    # ---- pass 1: which rows survive, and their scores -------------------
    t0 = tlast = time.time()
    keep_scores = []
    keep_global = []
    seen = 0
    for i, r in enumerate(m.itertuples()):
        f = np.asarray(open_chunk(r.path), np.float16).astype(np.float32)
        loc = np.flatnonzero(f.max(axis=1) >= T)
        if len(loc):
            keep_global.append(loc.astype(np.int64) + int(r.first_row_index))
            keep_scores.append(f[loc].astype(np.float16))
        seen += len(f)
        now = time.time()
        if now - tlast >= args.progress_secs or i + 1 == len(m):
            kept = sum(len(x) for x in keep_global)
            print(f"  pass1 {i+1:,}/{len(m):,} | scanned {seen:,} | "
                  f"kept {kept:,} ({100*kept/max(seen,1):.3f}%) | "
                  f"{(now-t0)/60:.1f} min")
            tlast = now

    if not keep_global:
        sys.exit(f"no row has a column >= {T}; lower --min-max "
                 f"(run `survey` to see the distribution)")
    rows = np.concatenate(keep_global)
    scores = np.concatenate(keep_scores)
    del keep_global, keep_scores
    K = len(rows)
    print(f"\n  shortlist rows  : {K:,} of {total:,} "
          f"({100*K/total:.4f}%)")
    print(f"  pass 1 took     : {(time.time()-t0)/60:.1f} min")

    # ---- pass 2: ids (and optionally blocker columns) from the CSV -------
    csv_path = Path(args.pairs_csv)
    if not csv_path.exists():
        sys.exit(f"missing {csv_path} — the ids live only there")
    usecols = [0, 1] if args.no_blocker else [0, 1, 2, 3, 4, 5, 6, 7]
    ia = np.zeros(K, np.int32)
    ib = np.zeros(K, np.int32)
    flags = np.zeros(K, np.uint8) if not args.no_blocker else None
    bsc = np.zeros((K, 3), np.float16) if not args.no_blocker else None

    print(f"\n  pass 2: streaming {csv_path} for those {K:,} rows")
    t1 = tlast = time.time()
    off = 0          # global row index at the head of the current block
    wptr = 0         # how many shortlist rows filled
    nblocks = 0
    reader = pd.read_csv(csv_path, usecols=usecols, header=0,
                         chunksize=args.csv_rows, engine="c")
    for blk in reader:
        n = len(blk)
        lo = np.searchsorted(rows, off, "left")
        hi = np.searchsorted(rows, off + n, "left")
        if hi > lo:
            local = (rows[lo:hi] - off).astype(np.int64)
            v = blk.to_numpy()
            ia[lo:hi] = v[local, 0].astype(np.int32)
            ib[lo:hi] = v[local, 1].astype(np.int32)
            if not args.no_blocker:
                o = v[local, 2].astype(np.uint8)
                u = v[local, 3].astype(np.uint8)
                d = v[local, 4].astype(np.uint8)
                flags[lo:hi] = o | (u << 1) | (d << 2)
                bsc[lo:hi, 0] = v[local, 5]
                bsc[lo:hi, 1] = v[local, 6]
                bsc[lo:hi, 2] = v[local, 7]
            wptr = hi
        off += n
        nblocks += 1
        now = time.time()
        if now - tlast >= args.progress_secs:
            print(f"  pass2 {off:,}/{total:,} rows | filled {wptr:,}/{K:,} | "
                  f"{off/(now-t1)/1e6:.1f}M rows/s | {(now-t1)/60:.1f} min")
            tlast = now

    if off != total:
        sys.exit(f"CSV has {off:,} data rows but the sidecars cover "
                 f"{total:,}; they are not the same run")
    if wptr != K:
        sys.exit(f"only resolved {wptr:,} of {K:,} shortlist ids")

    out = Path(args.out or Path(args.scores) / SHORTLIST_NAME)
    payload = dict(row=rows, ia=ia, ib=ib, scores=scores,
                   measures=np.array(MEASURES), min_max=np.float32(T),
                   total_rows=np.int64(total))
    if not args.no_blocker:
        payload.update(flags=flags, blocker=bsc)
    np.savez(out, **payload)
    print(f"\n  pass 2 took     : {(time.time()-t1)/60:.1f} min")
    print(f"  wrote           : {out}  "
          f"({out.stat().st_size/1024**3:.2f} GiB)")
    print(f"\n  self-check: ids resolved for every shortlist row, and the CSV")
    print(f"  row count matches the sidecar row count exactly.")
    print(f"\n  Next: python rank_dynamic.py rank --preset feline-led --top 50")


# ----------------------------------------------------------------------
# the ranking maths — imported by the Streamlit app, not duplicated there
# ----------------------------------------------------------------------
class Shortlist:
    """Uniform read access to a shortlist, whether .npz or a memmap store.

    The npz that `build` writes is a ZIP container: every array has to be read
    whole, so at 804M rows it is ~25 GiB resident and cannot be memory-mapped
    at all. That is fine for a one-shot CLI on a small shortlist and hopeless
    behind a slider. A store is the same data as one .npy per column, so a
    re-rank touches only scores.npy and the OS page cache does the rest.

    Both forms expose the same attributes, so nothing downstream — the CLI, the
    app — needs to know which it was handed.
    """

    def __init__(self, path):
        p = Path(path)
        self.path = p
        if p.is_dir():
            self._init_store(p)
        else:
            self._init_npz(p)
        if list(self.measures) != list(MEASURES):
            sys.exit(f"shortlist was built for measures {list(self.measures)}, "
                     f"this code expects {list(MEASURES)}")

    def _init_store(self, p):
        import json
        meta = p / "meta.json"
        if not meta.exists():
            sys.exit(f"{p} is a directory but has no meta.json; pass the "
                     f"shortlist.npz, or run `compact` to create a store")
        m = json.loads(meta.read_text())
        self.kind = "store"
        self.measures = m["measures"]
        self.min_max = float(m["min_max"])
        self.total_rows = int(m["total_rows"])
        self.n = int(m["n_rows"])
        self._a = {}
        for c in STORE_COLS:
            f = p / f"{c}.npy"
            if f.exists():
                self._a[c] = np.load(f, mmap_mode="r")
        for c in ("row", "ia", "ib", "scores"):
            if c not in self._a:
                sys.exit(f"{p} is missing {c}.npy")

    def _init_npz(self, p):
        if not p.exists():
            sys.exit(f"missing {p}")
        d = np.load(p, allow_pickle=False)
        self.kind = "npz"
        self.measures = [str(x) for x in d["measures"]]
        self.min_max = float(d["min_max"])
        self.total_rows = int(d["total_rows"])
        self._d = d
        self.n = int(d["scores"].shape[0]) if "scores" in d.files else 0
        self._a = {c: d[c] for c in STORE_COLS if c in d.files}

    def __contains__(self, k):
        return k in self._a

    def __getitem__(self, k):
        return self._a[k]

    def col(self, name, lo=0, hi=None):
        """A slice as a real array — the only place a memmap is materialised."""
        a = self._a[name]
        hi = self.n if hi is None else hi
        return np.asarray(a[lo:hi])

    def describe(self):
        gib = sum(np.prod(a.shape) * a.dtype.itemsize
                  for a in self._a.values()) / 1024 ** 3
        return (f"{self.kind} {self.path.name}: {self.n:,} rows, "
                f"max measure >= {self.min_max:.2f}, {gib:.2f} GiB, "
                f"from {self.total_rows:,} scored pairs")


def load_shortlist(path):
    return Shortlist(path)


def default_shortlist(scores_dir):
    """Prefer a store over the npz when both are present — it is strictly the
    better artifact, and after `compact` the npz is only an archive."""
    d = Path(scores_dir)
    store = d / STORE_NAME
    if (store / "meta.json").exists():
        return store
    return d / SHORTLIST_NAME


def compact(args):
    """npz -> memmap column store, plus optional higher-T tiers.

    Everything here comes out of the shortlist that already exists. The 97.76
    GiB CSV is NOT touched: the ids for a tier are a subset of ids already
    resolved, so raising the threshold costs a sequential pass over local
    files rather than another 14-minute parse.

    Stage 1 converts column by column, so peak memory is one array (the scores,
    ~7.5 GiB at 804M rows) rather than the whole 25 GiB. `row` is rewritten as
    uint32 because the largest global row index is 2,244,989,056 < 2^32.
    """
    import json
    src = Path(args.shortlist or Path(args.scores) / SHORTLIST_NAME)
    if src.is_dir():
        sys.exit(f"{src} is already a store")
    if not src.exists():
        sys.exit(f"missing {src} — run `build` first")
    out = Path(args.out or Path(args.scores) / STORE_NAME)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  COMPACT — npz to memmap store")
    print("=" * 78)
    print(f"source          : {src}  ({src.stat().st_size/1024**3:.2f} GiB)")
    print(f"store           : {out}")
    print(f"the candidate CSV is not read; tiers come from this shortlist")

    d = np.load(src, allow_pickle=False)
    meas = [str(x) for x in d["measures"]]
    T = float(d["min_max"])
    total = int(d["total_rows"])

    t0 = time.time()
    n = None
    for c in STORE_COLS:
        if c not in d.files:
            continue
        t = time.time()
        a = d[c]
        if c == "row":
            if a.max() >= 2 ** 32:
                sys.exit("global row index exceeds uint32; keep int64")
            a = a.astype(np.uint32)
        n = len(a) if n is None else n
        if len(a) != n:
            sys.exit(f"{c} has {len(a):,} rows but earlier columns had {n:,}")
        np.save(out / f"{c}.npy", a)
        print(f"  {c:<9}{str(a.dtype):<9}{a.nbytes/1024**3:>8.2f} GiB  "
              f"{time.time()-t:>6.1f} s")
        del a
    (out / "meta.json").write_text(json.dumps(
        dict(measures=meas, min_max=T, total_rows=total, n_rows=int(n),
             source=str(src)), indent=2))
    print(f"  base store      : {n:,} rows, T={T:.2f}, "
          f"{(time.time()-t0)/60:.1f} min")

    # ---- tiers ------------------------------------------------------
    tiers = [float(x) for x in (args.tiers or "").split(",") if x.strip()]
    if not tiers:
        print(f"\n  no --tiers requested; the base store is usable as is")
        print(f"\n  Next: python rank_dynamic.py rank --scores {args.scores} "
              f"--preset ttab-informed --top 100")
        return

    base = Shortlist(out)
    for Tt in sorted(tiers):
        if Tt <= T:
            print(f"\n  tier {Tt} skipped: not above the base T={T:.2f}, so it "
                  f"would not be a subset")
            continue
        tdir = out.parent / f"{out.name}_t{int(round(Tt*100)):03d}"
        tdir.mkdir(parents=True, exist_ok=True)
        t = time.time()

        # which base rows survive — streamed, so memory stays bounded
        keep = []
        B = args.block
        for lo in range(0, base.n, B):
            hi = min(lo + B, base.n)
            f = base.col("scores", lo, hi).astype(np.float32)
            loc = np.flatnonzero(f.max(axis=1) >= Tt)
            if len(loc):
                keep.append((loc + lo).astype(np.uint32))
        if not keep:
            print(f"\n  tier {Tt}: no rows survive; skipped")
            continue
        sel = np.concatenate(keep)
        del keep

        for c in STORE_COLS:
            if c not in base:
                continue
            a = base[c]
            sub = np.empty((len(sel),) + a.shape[1:], a.dtype)
            for lo in range(0, len(sel), B):
                hi = min(lo + B, len(sel))
                sub[lo:hi] = a[sel[lo:hi]]
            np.save(tdir / f"{c}.npy", sub)
            del sub
        (tdir / "meta.json").write_text(json.dumps(
            dict(measures=meas, min_max=Tt, total_rows=total,
                 n_rows=int(len(sel)), source=str(out)), indent=2))
        gib = sum(p.stat().st_size for p in tdir.glob("*.npy")) / 1024 ** 3
        print(f"\n  tier T={Tt:.2f}  {len(sel):,} rows  {gib:.2f} GiB  "
              f"{time.time()-t:.0f} s  -> {tdir.name}")
        print(f"    re-rank reads {len(sel)*10/1024**3:.2f} GiB; exactness "
              f"guaranteed down to a {Tt:.2f} Nth score")

    print(f"\n  The base store keeps the wider guarantee (T={T:.2f}); a tier")
    print(f"  answers faster but only certifies results above its own T.")
    print(f"  rank() reports which, so a fast answer is never mistaken for a")
    print(f"  certified one.")


def top_n(combined, rows, n):
    """Indices of the n best, ties broken by ascending global row.

    float16 scores collide constantly — thousands of pairs share a value — so
    "the top 50" is only well defined once the tie rule is fixed. Without one,
    argpartition and argsort disagree about which of a tied group to show, the
    table reshuffles between identical runs, and comparing a shortlist answer
    against a full scan reports a mismatch that is nothing but tie order. Both
    code paths call this, so both answer the same question.
    """
    n = min(n, len(combined))
    if n <= 0:
        return np.empty(0, np.int64)
    part = np.argpartition(-combined, n - 1)[:n]
    cut = combined[part].min()
    cand = np.flatnonzero(combined >= cut)     # every row tied at the boundary
    order = np.lexsort((rows[cand], -combined[cand]))
    return cand[order][:n]


def weight_vector(spec):
    """Normalise a {measure: weight} dict to a vector summing to 1."""
    w = np.array([float(spec.get(k, 0.0)) for k in MEASURES], np.float32)
    if (w < 0).any():
        raise ValueError(
            "negative weights break the max-column bound the shortlist relies "
            "on: a negative weight can lift a pruned row above a kept one. "
            "Use the 'full' command for that.")
    tot = w.sum()
    if tot <= 0:
        raise ValueError("all weights are zero")
    return w / tot


def rank(short, spec, top=50, block=16_000_000, exclude_name_mask=None):
    """Top-N under one weighting, with an exactness certificate.

    Returns (DataFrame, info). info['exact'] is True only when the Nth score is
    at or above the shortlist threshold — in that case no pruned row could have
    displaced any result, so this IS the true top-N of all rows. When it is
    False the list is still the best available but the tail is not guaranteed,
    and the caller is told rather than left to assume.
    """
    w = weight_vector(spec)
    short = short if isinstance(short, Shortlist) else Shortlist(short)

    # Streamed in blocks: at 804M rows the score column alone is 7.5 GiB, and
    # materialising it to rank 100 results would be the one avoidable reason
    # this tool needs a big machine. Only the running top-N is held.
    best_s = np.empty(0, np.float32)
    best_r = np.empty(0, np.int64)
    best_i = np.empty(0, np.int64)
    for lo in range(0, short.n, block):
        hi = min(lo + block, short.n)
        c = short.col("scores", lo, hi).astype(np.float32) @ w
        r = short.col("row", lo, hi).astype(np.int64)
        if exclude_name_mask is not None:
            a = short.col("ia", lo, hi).astype(np.int64)
            b = short.col("ib", lo, hi).astype(np.int64)
            eligible = ~(exclude_name_mask[a] | exclude_name_mask[b])
            if not eligible.any():
                continue
            loc0 = np.flatnonzero(eligible)
            sub = top_n(c[loc0], r[loc0], top)
            loc = loc0[sub]
        else:
            loc = top_n(c, r, top)
        cs = np.r_[best_s, c[loc]]
        cr = np.r_[best_r, r[loc]]
        ci = np.r_[best_i, loc.astype(np.int64) + lo]
        keep = top_n(cs, cr, top)
        best_s, best_r, best_i = cs[keep], cr[keep], ci[keep]

    n = len(best_i)
    T = short.min_max
    nth = float(best_s[-1]) if n else 0.0
    info = {
        "weights": dict(zip(MEASURES, w.round(4).tolist())),
        "shortlist_rows": short.n,
        "total_rows": short.total_rows,
        "min_max": T,
        "nth_score": nth,
        "exact": bool(nth >= T),
    }

    idx = best_i
    s_sel = np.asarray(short["scores"][idx], np.float16).astype(np.float32)
    df = pd.DataFrame({
        "global_row": best_r,
        "name_id_a": np.asarray(short["ia"][idx]),
        "name_id_b": np.asarray(short["ib"][idx]),
        "combined": best_s,
    })
    for c, name in enumerate(MEASURES):
        df[name] = s_sel[:, c]
    if "flags" in short:
        fl = np.asarray(short["flags"][idx])
        df["ortho"] = (fl & 1) > 0
        df["phon_US"] = (fl & 2) > 0
        df["phon_IN"] = (fl & 4) > 0
        b = np.asarray(short["blocker"][idx], np.float16).astype(np.float32)
        df["hmlb"] = b[:, 0]
        df["ph_US_score"] = b[:, 1]
        df["ph_IN_score"] = b[:, 2]
    return df, info


def rank_best_available(shortlists, spec, top=50, exclude_name_mask=None):
    """Try narrow/high-T stores first and fall back until top-N is certified."""
    objs = [x if isinstance(x, Shortlist) else Shortlist(x) for x in shortlists]
    if not objs:
        raise ValueError("no shortlist stores supplied")
    objs.sort(key=lambda x: x.min_max, reverse=True)
    fallback = None
    for sh in objs:
        df, info = rank(sh, spec, top, exclude_name_mask=exclude_name_mask)
        info["shortlist_path"] = str(sh.path)
        fallback = (df, info)
        if info["exact"]:
            return df, info
    return fallback



def rank_page(shortlists, spec, page=1, page_size=100, exclude_name_mask=None):
    """Return one globally ranked page.

    Page 1 contains ranks 1..page_size, page 2 contains the next page, etc.
    The implementation asks rank_best_available() for the global top
    page*page_size rows and slices only after ranking, so page boundaries never
    depend on storage chunks or shortlist layout.

    Exactness refers to the end of the requested page: if the score at global
    rank page*page_size is certified by the chosen shortlist threshold, every
    row shown on the page is also certified.
    """
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    start = (page - 1) * page_size
    stop = page * page_size

    df, info = rank_best_available(
        shortlists, spec, stop, exclude_name_mask=exclude_name_mask
    )
    shown = df.iloc[start:stop].copy().reset_index(drop=True)
    if len(shown):
        shown.insert(0, "rank", np.arange(start + 1, start + 1 + len(shown)))
    else:
        shown.insert(0, "rank", np.array([], dtype=np.int64))

    info = dict(info)
    info.update({
        "page": page,
        "page_size": page_size,
        "page_start_rank": start + 1,
        "page_end_rank": start + len(shown),
        "requested_end_rank": stop,
        "returned_rows": len(shown),
    })
    return shown, info


def attach_names(df, name_index):
    ni = pd.read_csv(name_index, keep_default_na=False)
    names = ni["brand_root"].astype(str).to_numpy()
    out = df.copy()
    out.insert(0, "root_a", names[df.name_id_a.to_numpy()])
    out.insert(1, "root_b", names[df.name_id_b.to_numpy()])
    return out


def find_pairs(short, name_index, query, top=50):
    """Every shortlisted pair involving a name, for the lookup view."""
    ni = pd.read_csv(name_index, keep_default_na=False)
    names = ni["brand_root"].astype(str)
    q = str(query).strip().lower()
    hit = np.flatnonzero(names.str.lower().str.contains(q, regex=False).to_numpy())
    if not len(hit):
        return pd.DataFrame(), []
    hs = set(hit.tolist())
    ia = np.asarray(short["ia"])
    ib = np.asarray(short["ib"])
    mask = np.isin(ia, hit) | np.isin(ib, hit)
    rows = np.flatnonzero(mask)[:max(top, 1) * 20]
    s = np.asarray(short["scores"], np.float16)[rows].astype(np.float32)
    df = pd.DataFrame({"name_id_a": ia[rows], "name_id_b": ib[rows]})
    for c, nm in enumerate(MEASURES):
        df[nm] = s[:, c]
    del hs
    return df, names.to_numpy()[hit][:25].tolist()


# ----------------------------------------------------------------------
def cmd_rank(args):
    spec = dict(PRESETS[args.preset]) if args.preset else {}
    for k in MEASURES:
        v = getattr(args, f"w_{k}")
        if v is not None:
            spec[k] = v
    if not spec:
        spec = dict(PRESETS["established"])

    path = Path(args.shortlist) if args.shortlist \
        else default_shortlist(args.scores)
    if not (path.exists() or (path / "meta.json").exists()):
        sys.exit(f"missing {path} — run `build` first")
    short = Shortlist(path)
    print("=" * 78)
    print("  DYNAMIC WEIGHTED RANKING")
    print("=" * 78)
    print(f"shortlist       : {short.describe()}")
    try:
        df, info = rank(short, spec, args.top)
    except ValueError as e:
        sys.exit(f"{e}")

    print("weights         : " + "  ".join(
        f"{k}={v:.3f}" for k, v in info["weights"].items()))
    print(f"Nth score       : {info['nth_score']:.4f}")
    if info["exact"]:
        print(f"EXACT           : yes — the Nth score clears T, so no pruned")
        print(f"                  row could have displaced any result. This is")
        print(f"                  the true top-{args.top} of all "
              f"{info['total_rows']:,} pairs.")
    else:
        print(f"EXACT           : NO — the Nth score ({info['nth_score']:.4f}) "
              f"is below T ({info['min_max']}).")
        print(f"                  Rows pruned at build time could belong in")
        print(f"                  this list. Rebuild with a lower --min-max,")
        print(f"                  ask for fewer results, or use `full`.")

    if args.name_index and Path(args.name_index).exists():
        df = attach_names(df, args.name_index)
    print()
    cols = [c for c in ("root_a", "root_b", "combined", *MEASURES,
                        "ortho", "phon_US", "phon_IN") if c in df.columns]
    with pd.option_context("display.width", 200,
                           "display.max_columns", 40):
        print(df[cols].head(args.top).to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote -> {args.out}")


def cmd_full(args):
    """Exact top-N over every row, for one weighting. No shortlist, no bound.

    This is the escape hatch: it handles negative weights, and it is how a
    shortlist answer gets confirmed independently. It reads all 20.91 GiB.
    """
    m = load_manifest(args.scores)
    spec = dict(PRESETS[args.preset]) if args.preset else {}
    for k in MEASURES:
        v = getattr(args, f"w_{k}")
        if v is not None:
            spec[k] = v
    if not spec:
        spec = dict(PRESETS["established"])
    w = np.array([float(spec.get(k, 0.0)) for k in MEASURES], np.float32)
    tot = np.abs(w).sum()
    if tot <= 0:
        sys.exit("all weights are zero")
    w = w / w.sum() if w.sum() > 0 else w / tot
    print("=" * 78)
    print("  FULL EXACT SCAN")
    print("=" * 78)
    print("weights         : " + "  ".join(
        f"{k}={v:.3f}" for k, v in zip(MEASURES, w)))
    if (w < 0).any():
        print("                  (negative weight present — this is exactly")
        print("                   the case the shortlist cannot serve)")

    N = args.top
    best_s = np.full(N, -np.inf, np.float32)
    best_r = np.zeros(N, np.int64)
    t0 = tlast = time.time()
    seen = 0
    for i, r in enumerate(m.itertuples()):
        f = np.asarray(open_chunk(r.path), np.float16).astype(np.float32)
        c = f @ w
        gr = np.arange(len(c), dtype=np.int64) + int(r.first_row_index)
        loc = top_n(c, gr, N)
        cand_s = np.r_[best_s, c[loc]]
        cand_r = np.r_[best_r, gr[loc]]
        keep = top_n(cand_s, cand_r, N)
        best_s, best_r = cand_s[keep], cand_r[keep]
        seen += len(f)
        now = time.time()
        if now - tlast >= args.progress_secs or i + 1 == len(m):
            print(f"  {i+1:,}/{len(m):,} chunks | {seen:,} rows | "
                  f"{seen/(now-t0)/1e6:.1f}M rows/s | best "
                  f"{best_s[0]:.4f} | {(now-t0)/60:.1f} min")
            tlast = now

    df = pd.DataFrame({"global_row": best_r, "combined": best_s})
    path = Path(args.shortlist) if args.shortlist \
        else default_shortlist(args.scores)
    if path.exists() or (path / "meta.json").exists():
        short = Shortlist(path)
        srow = np.asarray(short["row"], np.int64)
        pos = np.searchsorted(srow, best_r)
        ok = (pos < len(srow)) & (srow[np.clip(pos, 0, len(srow) - 1)]
                                  == best_r)
        df["name_id_a"] = -1
        df["name_id_b"] = -1
        df.loc[ok, "name_id_a"] = np.asarray(short["ia"])[pos[ok]]
        df.loc[ok, "name_id_b"] = np.asarray(short["ib"])[pos[ok]]
        n_un = int((~ok).sum())
        if n_un:
            print(f"\n  {n_un} of {N} winners are not in the shortlist, so")
            print(f"  their ids are unknown here (name_id -1). That is itself")
            print(f"  the finding: this weighting reaches rows the shortlist")
            print(f"  pruned — expected only with negative weights.")
        if args.name_index and Path(args.name_index).exists() and ok.any():
            df = attach_names(df.assign(
                name_id_a=df.name_id_a.clip(lower=0),
                name_id_b=df.name_id_b.clip(lower=0)), args.name_index)
    print()
    with pd.option_context("display.width", 200):
        print(df.head(N).to_string(index=False,
                                   float_format=lambda x: f"{x:.4f}"))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote -> {args.out}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--scores", default="results/candidate_scores",
                       help="directory holding the sidecars and manifest.csv")
        p.add_argument("--progress-secs", type=int, default=20)

    def weights(p):
        p.add_argument("--preset", choices=sorted(PRESETS))
        for k in MEASURES:
            p.add_argument(f"--w-{k}", type=float, default=None,
                           dest=f"w_{k}")
        p.add_argument("--top", type=int, default=50)
        p.add_argument("--name-index", default="results/name_index.csv")
        p.add_argument("--shortlist")
        p.add_argument("--out")

    p = sub.add_parser("survey", help="score distribution; pick --min-max")
    common(p)

    p = sub.add_parser("build", help="write the reusable shortlist")
    common(p)
    p.add_argument("--pairs-csv", required=True)
    p.add_argument("--min-max", type=float, required=True,
                   help="keep rows whose largest column is >= this")
    p.add_argument("--csv-rows", type=int, default=4_000_000,
                   help="CSV rows parsed per block in pass 2")
    p.add_argument("--no-blocker", action="store_true",
                   help="store ids and scores only, dropping the blocker's "
                        "own flags and scores (saves 7 bytes/row)")
    p.add_argument("--out")

    p = sub.add_parser("compact",
                       help="npz -> memmap store (+ faster higher-T tiers)")
    common(p)
    p.add_argument("--shortlist")
    p.add_argument("--out")
    p.add_argument("--tiers", default="",
                   help="comma-separated thresholds above the base T, e.g. "
                        "0.70,0.80 — each becomes its own fast store")
    p.add_argument("--block", type=int, default=16_000_000,
                   help="rows per streaming block")

    p = sub.add_parser("rank", help="re-rank the shortlist")
    common(p)
    weights(p)

    p = sub.add_parser("full", help="exact scan of every row, one weighting")
    common(p)
    weights(p)

    args = ap.parse_args()
    {"survey": survey, "build": build, "compact": compact,
     "rank": cmd_rank, "full": cmd_full}[args.cmd](args)


if __name__ == "__main__":
    main()
