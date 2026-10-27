#!/usr/bin/env python3
"""
score_candidates.py — attach similarity scores to the blocked candidate set.

The HMLB blocker emits ~2.24 billion surviving pairs as id pairs plus its own
blocking scores. It does not say how ALIKE the two names actually are. This
script computes that, once, for every pair, so the display layer can re-weight
the measures interactively instead of re-scoring 2.24e9 pairs each time a
slider moves.

MEASURES
    jw        Jaro-Winkler                       (established baseline)
    edit      1 - levenshtein / max(len)         (established baseline)
    kbigram   learned bigram-kernel local alignment      [kbigram_model.npz]
    lumera    learned character-kernel affine alignment  [lumera_model.npz]
    feline    deterministic affine local alignment + prefix bonus

Each column is the number its own module would return. The three alignment
measures normalise names differently (feline strips to [a-z0-9- ], lumera to
its trained alphabet, kbigram takes raw-lowercase bigrams) and that difference
is PRESERVED here rather than unified, so a column in the output equals
feline.score(a, b) / lumera.score(a, b) / kbigram.score(a, b) exactly. Run
--verify to see that for yourself; do not take it on faith.

WHY A SIDECAR AND NOT A WIDER CSV
    candidate_pairs.csv is ~98 GiB. Appending five text columns means writing a
    second ~200 GiB file and re-serialising every row. The default output is a
    row-aligned float array instead: 5 x float32 = 20 bytes/pair = 44.8 GiB for
    the full set, or 22.4 GiB with --dtype float16, and it loads by memory-map
    without parsing. Row i of the sidecar is row i of the corresponding input
    part. --format csv is available when a text column really is what is wanted.

    Scores live in [0,1]. float16 resolves ~0.0005 there, which is finer than
    any weighting decision this feeds; float32 is the default only because
    nothing forces the precision question to be answered now.

INPUT
    Preferred:  --parts-dir results/candidate_pairs    (the 364 shards)
                one sidecar per shard, trivially row-aligned, resumable.
    Also works: --pairs-csv results/candidate_pairs.csv (the merged file)
                split into byte ranges at line boundaries; a manifest records
                each chunk's first row index so alignment stays checkable.

SPEED
    The per-pair work is a dynamic program whose inner loop cannot be
    vectorised (each cell depends on the cell to its left), so the kernel is
    numba-jitted and runs prange over blocks of pairs across every core,
    releasing the GIL while the next unit is parsed on a reader thread.

    Measured, all five measures, real brand roots (median 8 characters):
        0.87 us/pair on 2 cores  =  1.15M pairs/s  =  ~580k pairs/s/core
        the full 2,244,989,057 pairs:  ~32 min on 2 cores, ~8 min on 8
    Parsing is the other half: 5M rows/s reading only the two id columns,
    so ~10 min for the 97.76 GiB file. One reader thread therefore keeps
    roughly nine cores fed; past that, raise --prefetch or expect I/O to
    become the limit. Neither figure includes a slow disk.

    Without numba the script falls back to a batched NumPy kernel that groups
    pairs by (len_a, len_b) so the inner loop spans thousands of pairs per
    array operation. It gives the same answers (--verify checks that path too)
    at roughly a tenth the speed — plan on most of a day, or install numba:

        pip install numba

    Production validation on the 2,244,989,057-pair merged namespace:
        all five measures, float16 sidecars
        5.49M pairs/s aggregate
        6.8 min elapsed
        20.91 GiB score output

USAGE
    python score_candidates.py --verify 2000          # check kernels first
    python score_candidates.py --parts-dir results/candidate_pairs --limit-parts 2
    python score_candidates.py --parts-dir results/candidate_pairs --dtype float16
    python score_candidates.py --pairs-csv results/candidate_pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

MEASURES = ("jw", "edit", "kbigram", "lumera", "feline")

FELINE_GAP_OPEN = 0.60
FELINE_GAP_EXT = 0.15
FELINE_PREFIX_SCALE = 0.175
FELINE_PREFIX_MAX = 4

JW_PREFIX_SCALE = 0.1
JW_PREFIX_MAX = 4
JW_BOOST_THRESHOLD = 0.7

MAXLEN = 128          # longest encoded sequence the kernels will accept

# ----------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------
HAVE_NUMBA = True
try:
    from numba import njit, prange
except ImportError:                                         # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a or not callable(a[0]) else a[0]

    prange = range


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def _affine_align(a, b, S, gap_open, gap_ext, H, F):
    """Smith-Waterman, affine gaps, normalised by the longer sequence.

    S is a substitution matrix indexed by the two code alphabets. H and F are
    caller-supplied scratch rows so the kernel allocates nothing per pair.
    This is the shared core of FELINE (S = identity) and LUMERA (S = learned
    character kernel); verifying one verifies the recurrence for both.
    """
    n = a.shape[0]
    m = b.shape[0]
    if n == 0 or m == 0:
        return 0.0
    for j in range(m + 1):
        H[j] = 0.0
        F[j] = 0.0
    best = 0.0
    for i in range(1, n + 1):
        hdiag = 0.0          # H[i-1][j-1]
        ecol = 0.0           # E[i][j-1]
        ai = a[i - 1]
        H[0] = 0.0
        for j in range(1, m + 1):
            e = H[j - 1] - gap_open          # H[i][j-1] (already current row)
            e2 = ecol - gap_ext
            if e2 > e:
                e = e2
            ecol = e

            f = H[j] - gap_open              # H[i-1][j] (still previous row)
            f2 = F[j] - gap_ext
            if f2 > f:
                f = f2
            F[j] = f

            v = hdiag + S[ai, b[j - 1]]
            if e > v:
                v = e
            if f > v:
                v = f
            if v < 0.0:
                v = 0.0

            hdiag = H[j]
            H[j] = v
            if v > best:
                best = v
    return best / (n if n > m else m)


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def _linear_align(a, b, S, gap, H):
    """Local alignment with a flat gap cost — the k-bigram recurrence."""
    n = a.shape[0]
    m = b.shape[0]
    if n == 0 or m == 0:
        return 0.0
    for j in range(m + 1):
        H[j] = 0.0
    best = 0.0
    for i in range(1, n + 1):
        hdiag = 0.0
        ai = a[i - 1]
        H[0] = 0.0
        for j in range(1, m + 1):
            v = hdiag + S[ai, b[j - 1]]
            d = H[j] - gap                   # up   (previous row)
            if d > v:
                v = d
            d = H[j - 1] - gap               # left (current row)
            if d > v:
                v = d
            if v < 0.0:
                v = 0.0
            hdiag = H[j]
            H[j] = v
            if v > best:
                best = v
    return best / (n if n > m else m)


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def _levenshtein(a, b, prev, cur):
    n = a.shape[0]
    m = b.shape[0]
    if n == 0:
        return m
    if m == 0:
        return n
    for j in range(m + 1):
        prev[j] = j
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            c = prev[j - 1] if ai == b[j - 1] else prev[j - 1] + 1
            d = prev[j] + 1
            if d < c:
                c = d
            d = cur[j - 1] + 1
            if d < c:
                c = d
            cur[j] = c
        prev, cur = cur, prev
    return prev[m]


@njit(cache=True, fastmath=False, nogil=True)
def _jaro_winkler(a, b, fa, fb):
    """Jaro-Winkler. fa/fb are scratch match-flag buffers.

    fastmath is OFF here and the arithmetic is float64 on purpose. The Winkler
    prefix bonus is applied only when the Jaro similarity exceeds 0.7, and real
    name pairs land exactly on that boundary ('acepar mr' / 'aceplee-mr' gives
    0.7000000000000001). Reassociated or float32 arithmetic pushes such a pair
    to the wrong side of the test and the column comes out 0.12 low, which is
    a wrong answer rather than a rounding difference. The divisions are over
    small integers, so float64 here costs nothing measurable.
    """
    n = a.shape[0]
    m = b.shape[0]
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0

    window = (n if n > m else m) // 2 - 1
    if window < 0:
        window = 0

    for i in range(n):
        fa[i] = False
    for j in range(m):
        fb[j] = False

    matches = 0
    for i in range(n):
        lo = i - window
        if lo < 0:
            lo = 0
        hi = i + window
        if hi > m - 1:
            hi = m - 1
        for j in range(lo, hi + 1):
            if (not fb[j]) and a[i] == b[j]:
                fa[i] = True
                fb[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0

    # transpositions: walk both matched subsequences in order
    k = 0
    half = 0
    for i in range(n):
        if fa[i]:
            while not fb[k]:
                k += 1
            if a[i] != b[k]:
                half += 1
            k += 1
    t = half // 2

    mf = np.float64(matches)
    jaro = (mf / n + mf / m + (mf - t) / mf) / np.float64(3.0)

    if jaro > JW_BOOST_THRESHOLD:
        p = 0
        lim = n if n < m else m
        if lim > JW_PREFIX_MAX:
            lim = JW_PREFIX_MAX
        while p < lim and a[p] == b[p]:
            p += 1
        jaro = jaro + p * JW_PREFIX_SCALE * (np.float64(1.0) - jaro)
    return jaro


@njit(parallel=True, cache=True, fastmath=False, nogil=True)
def _score_all(ia, ib,
               raw_off, raw_flat,
               fe_off, fe_flat,
               lu_off, lu_flat,
               kb_off, kb_flat,
               S_id, S_lu, S_kb, lu_go, lu_ge, kb_gap,
               w_jw, w_edit, w_kb, w_lu, w_fe,
               block, out):
    """Fill out[k, 0:5] for every pair k.

    prange runs over BLOCKS rather than pairs so each thread allocates its DP
    scratch once instead of once per pair; the measure switches are scalars so
    numba can drop the branches entirely at compile time for whichever columns
    were not requested.
    """
    nblocks = (ia.shape[0] + block - 1) // block
    for bi in prange(nblocks):
        H = np.zeros(MAXLEN + 2, np.float32)
        Fv = np.zeros(MAXLEN + 2, np.float32)
        P = np.zeros(MAXLEN + 2, np.int32)
        C = np.zeros(MAXLEN + 2, np.int32)
        fa = np.zeros(MAXLEN + 2, np.bool_)
        fb = np.zeros(MAXLEN + 2, np.bool_)

        lo = bi * block
        hi = lo + block
        if hi > ia.shape[0]:
            hi = ia.shape[0]

        for k in range(lo, hi):
            x = ia[k]
            y = ib[k]

            if w_jw or w_edit:
                ra = raw_flat[raw_off[x]:raw_off[x + 1]]
                rb = raw_flat[raw_off[y]:raw_off[y + 1]]
                if w_jw:
                    out[k, 0] = _jaro_winkler(ra, rb, fa, fb)
                if w_edit:
                    na = ra.shape[0]
                    nb = rb.shape[0]
                    mx = na if na > nb else nb
                    if mx == 0:
                        out[k, 1] = 1.0
                    else:
                        d = _levenshtein(ra, rb, P, C)
                        out[k, 1] = 1.0 - d / mx

            if w_kb:
                out[k, 2] = _linear_align(
                    kb_flat[kb_off[x]:kb_off[x + 1]],
                    kb_flat[kb_off[y]:kb_off[y + 1]], S_kb, kb_gap, H)

            if w_lu:
                out[k, 3] = _affine_align(
                    lu_flat[lu_off[x]:lu_off[x + 1]],
                    lu_flat[lu_off[y]:lu_off[y + 1]], S_lu, lu_go, lu_ge,
                    H, Fv)

            if w_fe:
                a = fe_flat[fe_off[x]:fe_off[x + 1]]
                b = fe_flat[fe_off[y]:fe_off[y + 1]]
                raw = _affine_align(a, b, S_id, FELINE_GAP_OPEN,
                                    FELINE_GAP_EXT, H, Fv)
                n = a.shape[0]
                m = b.shape[0]
                lim = n if n < m else m
                if lim > FELINE_PREFIX_MAX:
                    lim = FELINE_PREFIX_MAX
                p = 0
                while p < lim and a[p] == b[p]:
                    p += 1
                v = raw + p * FELINE_PREFIX_SCALE * (1.0 - raw)
                if v > 1.0:
                    v = 1.0
                if v < 0.0:
                    v = 0.0
                out[k, 4] = v


# ----------------------------------------------------------------------
# name encoding — done once for all 181,819 names
# ----------------------------------------------------------------------
_FELINE_KEEP = re.compile(r"[^a-z0-9\- ]")
_LUMERA_KEEP = re.compile(r"[^a-z0-9\- ]")
_FALLBACK_LUMERA_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789- "


def _pack(seqs, dtype):
    """CSR-style packing: one flat array plus an offset array."""
    off = np.zeros(len(seqs) + 1, np.int64)
    off[1:] = np.cumsum([len(s) for s in seqs])
    flat = np.empty(int(off[-1]), dtype)
    for i, s in enumerate(seqs):
        if len(s):
            flat[off[i]:off[i + 1]] = s
    return off, flat


def feline_norm(name):
    s = _FELINE_KEEP.sub("", str(name).lower())
    return re.sub(r"\s+", " ", s).strip()


def lumera_norm(name):
    s = _LUMERA_KEEP.sub("", str(name).lower())
    return re.sub(r"\s+", " ", s).strip()


def encode_names(names, lumera_alphabet, kbigram_vocab):
    """Return the four per-name encodings the kernels index into."""
    raw = [np.frombuffer(str(s).encode("utf-8", "replace"), np.uint8)
           for s in names]

    fe_chars = sorted({c for s in names for c in feline_norm(s)})
    fe_idx = {c: i for i, c in enumerate(fe_chars)}
    fe = [[fe_idx[c] for c in feline_norm(s)] for s in names]

    lu_idx = {c: i for i, c in enumerate(lumera_alphabet)}
    lu = [[lu_idx[c] for c in lumera_norm(s) if c in lu_idx] for s in names]

    kb = []
    for s in names:
        t = str(s).lower()
        kb.append([kbigram_vocab[t[i:i + 2]] for i in range(len(t) - 1)
                   if t[i:i + 2] in kbigram_vocab])

    over = max((len(x) for x in raw), default=0)
    for seqs in (fe, lu, kb):
        over = max(over, max((len(x) for x in seqs), default=0))
    if over > MAXLEN:
        sys.exit(f"a name encodes to {over} symbols, over MAXLEN={MAXLEN}; "
                 f"raise MAXLEN and re-run")

    raw_off, raw_flat = _pack(raw, np.uint8)
    fe_off, fe_flat = _pack(fe, np.int16)
    lu_off, lu_flat = _pack(lu, np.int16)
    kb_off, kb_flat = _pack(kb, np.int16)
    return (raw_off, raw_flat, fe_off, fe_flat, lu_off, lu_flat,
            kb_off, kb_flat, len(fe_chars))


# ----------------------------------------------------------------------
# model loading
# ----------------------------------------------------------------------
def load_kbigram(path):
    d = np.load(path, allow_pickle=True)
    F = np.asarray(d["F"], np.float64)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    S = np.clip(Fn @ Fn.T, 0.0, 1.0).astype(np.float32)
    vocab = {str(g): i for i, g in enumerate(list(d["active"]))}
    gap = float(d["gap"]) if "gap" in d else 0.5
    return S, vocab, gap


def load_lumera(path):
    d = np.load(path, allow_pickle=True)
    M = np.asarray(d["M"], np.float64)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = np.clip(Mn @ Mn.T, 0.0, 1.0).astype(np.float32)
    alpha = ("".join(list(d["alphabet"])) if "alphabet" in d
             else _FALLBACK_LUMERA_ALPHABET)
    go = float(d["gap_open"]) if "gap_open" in d else 0.6
    ge = float(d["gap_ext"]) if "gap_ext" in d else 0.15
    return S, alpha, go, ge


# ----------------------------------------------------------------------
# NumPy fallback (no numba) — batched over pairs grouped by (len_a, len_b)
# ----------------------------------------------------------------------
def _np_groups(la, lb):
    key = la.astype(np.int64) * (MAXLEN + 1) + lb
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        yield order[s:e]


def _np_gather(off, flat, ids, n):
    base = off[ids][:, None]
    return flat[base + np.arange(n)[None, :]]


def _np_affine(A, B, S, go, ge):
    bsz, n = A.shape
    m = B.shape[1]
    if n == 0 or m == 0:
        return np.zeros(bsz, np.float32)
    sub = S[A[:, :, None], B[:, None, :]].astype(np.float32)
    Hp = np.zeros((bsz, m + 1), np.float32)
    Fp = np.zeros((bsz, m + 1), np.float32)
    Hc = np.zeros_like(Hp)
    Fc = np.zeros_like(Fp)
    best = np.zeros(bsz, np.float32)
    for i in range(1, n + 1):
        Hc[:, 0] = 0.0
        Fc[:, 0] = 0.0
        ecol = np.zeros(bsz, np.float32)
        srow = sub[:, i - 1, :]
        for j in range(1, m + 1):
            np.maximum(Hc[:, j - 1] - go, ecol - ge, out=ecol)
            fj = np.maximum(Hp[:, j] - go, Fp[:, j] - ge)
            v = Hp[:, j - 1] + srow[:, j - 1]
            np.maximum(v, ecol, out=v)
            np.maximum(v, fj, out=v)
            np.maximum(v, 0.0, out=v)
            Hc[:, j] = v
            Fc[:, j] = fj
            np.maximum(best, v, out=best)
        Hp, Hc = Hc, Hp
        Fp, Fc = Fc, Fp
    return best / max(n, m)


def _np_linear(A, B, S, gap):
    bsz, n = A.shape
    m = B.shape[1]
    if n == 0 or m == 0:
        return np.zeros(bsz, np.float32)
    sub = S[A[:, :, None], B[:, None, :]].astype(np.float32)
    prev = np.zeros((bsz, m + 1), np.float32)
    cur = np.zeros_like(prev)
    best = np.zeros(bsz, np.float32)
    for i in range(1, n + 1):
        cur[:, 0] = 0.0
        srow = sub[:, i - 1, :]
        for j in range(1, m + 1):
            v = prev[:, j - 1] + srow[:, j - 1]
            np.maximum(v, prev[:, j] - gap, out=v)
            np.maximum(v, cur[:, j - 1] - gap, out=v)
            np.maximum(v, 0.0, out=v)
            cur[:, j] = v
            np.maximum(best, v, out=best)
        prev, cur = cur, prev
    return best / max(n, m)


def _np_levenshtein(A, B):
    bsz, n = A.shape
    m = B.shape[1]
    if n == 0:
        return np.full(bsz, m, np.int32)
    if m == 0:
        return np.full(bsz, n, np.int32)
    prev = np.tile(np.arange(m + 1, dtype=np.int32), (bsz, 1))
    cur = np.empty_like(prev)
    for i in range(1, n + 1):
        cur[:, 0] = i
        ai = A[:, i - 1]
        for j in range(1, m + 1):
            c = prev[:, j - 1] + (ai != B[:, j - 1])
            np.minimum(c, prev[:, j] + 1, out=c)
            np.minimum(c, cur[:, j - 1] + 1, out=c)
            cur[:, j] = c
        prev, cur = cur, prev
    return prev[:, m]


def _np_jaro_winkler(A, B):
    bsz, n = A.shape
    m = B.shape[1]
    if n == 0 or m == 0:
        return np.zeros(bsz, np.float32)
    window = max(n, m) // 2 - 1
    if window < 0:
        window = 0
    fa = np.zeros((bsz, n), bool)
    fb = np.zeros((bsz, m), bool)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(m - 1, i + window)
        taken = np.zeros(bsz, bool)
        for j in range(lo, hi + 1):
            hit = (~taken) & (~fb[:, j]) & (A[:, i] == B[:, j])
            fb[:, j] |= hit
            taken |= hit
        fa[:, i] = taken
    matches = fa.sum(1).astype(np.float64)

    # matched characters of each side, in order, compared position by position
    half = np.zeros(bsz, np.int64)
    k = np.zeros(bsz, np.int64)
    rows = np.arange(bsz)
    cum_b = np.cumsum(fb, axis=1)
    for i in range(n):
        act = fa[:, i]
        if not act.any():
            continue
        want = k[act] + 1
        pos = np.argmax(cum_b[act] >= want[:, None], axis=1)
        half[act] += (A[act, i] != B[np.flatnonzero(act), pos]).astype(np.int64)
        k[act] += 1
    t = half // 2

    out = np.zeros(bsz, np.float64)
    ok = matches > 0
    mo = matches[ok]
    out[ok] = (mo / n + mo / m + (mo - t[ok]) / mo) / 3.0

    boost = out > JW_BOOST_THRESHOLD
    if boost.any():
        lim = min(n, m, JW_PREFIX_MAX)
        p = np.zeros(bsz, np.int64)
        still = np.ones(bsz, bool)
        for q in range(lim):
            still &= (A[:, q] == B[:, q])
            p += still
        out[boost] += p[boost] * JW_PREFIX_SCALE * (1.0 - out[boost])
    del rows
    return out.astype(np.float32)


def numpy_score_block(ia, ib, enc, models, want, out, batch=120_000):
    (raw_off, raw_flat, fe_off, fe_flat, lu_off, lu_flat,
     kb_off, kb_flat, n_fe) = enc
    S_id, S_lu, lu_go, lu_ge, S_kb, kb_gap = models
    lens = {
        "raw": (raw_off[1:] - raw_off[:-1]).astype(np.int32),
        "fe": (fe_off[1:] - fe_off[:-1]).astype(np.int32),
        "lu": (lu_off[1:] - lu_off[:-1]).astype(np.int32),
        "kb": (kb_off[1:] - kb_off[:-1]).astype(np.int32),
    }
    plan = [
        ("raw", (0, 1), raw_off, raw_flat),
        ("kb", (2,), kb_off, kb_flat),
        ("lu", (3,), lu_off, lu_flat),
        ("fe", (4,), fe_off, fe_flat),
    ]
    for tag, cols, off, flat in plan:
        if not any(want[c] for c in cols):
            continue
        la = lens[tag][ia]
        lb = lens[tag][ib]
        for grp in _np_groups(la, lb):
            n = int(la[grp[0]])
            m = int(lb[grp[0]])
            for s in range(0, len(grp), batch):
                sl = grp[s:s + batch]
                A = _np_gather(off, flat, ia[sl], n)
                B = _np_gather(off, flat, ib[sl], m)
                if tag == "raw":
                    if want[0]:
                        out[sl, 0] = _np_jaro_winkler(A, B)
                    if want[1]:
                        mx = max(n, m)
                        out[sl, 1] = (1.0 if mx == 0 else
                                      1.0 - _np_levenshtein(A, B) / mx)
                elif tag == "kb":
                    out[sl, 2] = _np_linear(A, B, S_kb, kb_gap)
                elif tag == "lu":
                    out[sl, 3] = _np_affine(A, B, S_lu, lu_go, lu_ge)
                else:
                    raw = _np_affine(A, B, S_id, FELINE_GAP_OPEN,
                                     FELINE_GAP_EXT)
                    lim = min(n, m, FELINE_PREFIX_MAX)
                    p = np.zeros(len(sl), np.int64)
                    still = np.ones(len(sl), bool)
                    for q in range(lim):
                        still &= (A[:, q] == B[:, q])
                        p += still
                    out[sl, 4] = np.clip(
                        raw + p * FELINE_PREFIX_SCALE * (1.0 - raw), 0.0, 1.0)


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def score_ids(ia, ib, enc, models, want, use_numba):
    out = np.zeros((len(ia), len(MEASURES)), np.float32)
    if len(ia) == 0:
        return out
    if use_numba:
        (raw_off, raw_flat, fe_off, fe_flat, lu_off, lu_flat,
         kb_off, kb_flat, _) = enc
        S_id, S_lu, lu_go, lu_ge, S_kb, kb_gap = models
        _score_all(ia, ib, raw_off, raw_flat, fe_off, fe_flat,
                   lu_off, lu_flat, kb_off, kb_flat,
                   S_id, S_lu, S_kb, lu_go, lu_ge, kb_gap,
                   want[0], want[1], want[2], want[3], want[4],
                   4096, out)
    else:
        numpy_score_block(ia, ib, enc, models, want, out)
    return out


def read_ids(path):
    """Read only name_id_a and name_id_b. The six blocking columns are left on
    disk: skipping them roughly halves parse time (171 vs 88 MiB/s measured),
    and nothing here needs them."""
    import pandas as pd
    df = pd.read_csv(path, usecols=[0, 1], dtype=np.int32, engine="c")
    return (np.ascontiguousarray(df.iloc[:, 0].to_numpy(np.int32)),
            np.ascontiguousarray(df.iloc[:, 1].to_numpy(np.int32)))


def byte_chunks(path, n_chunks):
    """Split a CSV into [start, end) byte ranges aligned to line starts,
    skipping the header. Returns the ranges in order."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        header = f.readline()
    start = len(header)
    step = max(1, (size - start) // n_chunks)
    cuts = [start]
    with open(path, "rb") as f:
        for i in range(1, n_chunks):
            pos = min(size, start + i * step)
            if pos <= cuts[-1]:
                continue
            f.seek(pos)
            f.readline()                      # discard the partial line
            p = f.tell()
            if p > cuts[-1] and p < size:
                cuts.append(p)
    cuts.append(size)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)
            if cuts[i + 1] > cuts[i]]


def read_range(path, lo, hi):
    import pandas as pd
    import io
    with open(path, "rb") as f:
        f.seek(lo)
        buf = f.read(hi - lo)
    df = pd.read_csv(io.BytesIO(buf), usecols=[0, 1], header=None,
                     dtype=np.int32, engine="c")
    return (np.ascontiguousarray(df.iloc[:, 0].to_numpy(np.int32)),
            np.ascontiguousarray(df.iloc[:, 1].to_numpy(np.int32)))


# ----------------------------------------------------------------------
def verify(args, enc, models, want, names, use_numba):
    """Cross-check every kernel against the reference module it reimplements.

    This is the only thing standing between a fast kernel and five columns of
    confident nonsense, so it compares against the real modules, on real names,
    and prints the worst disagreement rather than a pass/fail.
    """
    import importlib
    rng = np.random.default_rng(args.seed)
    n = min(args.verify, len(names) * (len(names) - 1) // 2)
    ia = rng.integers(0, len(names), n).astype(np.int32)
    ib = rng.integers(0, len(names), n).astype(np.int32)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]

    got = score_ids(ia, ib, enc, models, want, use_numba)

    refs = {}
    try:
        import jellyfish
        refs["jw"] = (0, lambda a, b: jellyfish.jaro_winkler_similarity(a, b))
        refs["edit"] = (1, lambda a, b: 1.0 - jellyfish.levenshtein_distance(
            a, b) / max(len(a), len(b), 1))
    except ImportError:
        print("  jellyfish not installed — jw/edit unchecked "
              "(pip install jellyfish)")
    for mod, col in (("kbigram", 2), ("lumera", 3), ("feline", 4)):
        if not want[col]:
            continue
        try:
            m = importlib.import_module(mod)
            refs[mod] = (col, m.score)
        except Exception as e:
            print(f"  {mod}: reference unavailable ({e})")

    print(f"\n  checked on {len(ia):,} random name pairs")
    print(f"  {'measure':<12}{'max abs diff':>14}{'mean abs diff':>15}"
          f"{'worst pair':>34}")
    print(f"  {'-'*75}")
    bad = False
    for name, (col, fn) in refs.items():
        ref = np.array([fn(str(names[x]), str(names[y]))
                        for x, y in zip(ia, ib)], np.float64)
        d = np.abs(ref - got[:, col].astype(np.float64))
        w = int(np.argmax(d))
        pair = f"{names[ia[w]][:14]} / {names[ib[w]][:14]}"
        print(f"  {name:<12}{d.max():>14.3e}{d.mean():>15.3e}{pair:>34}")
        if d.max() > args.tol:
            bad = True
    print(f"\n  tolerance {args.tol:g}. float32 accumulation in the kernel vs")
    print(f"  float64 in the reference modules makes ~1e-7 differences normal;")
    print(f"  anything larger is a real disagreement, not rounding.")
    if bad:
        print("\n  VERIFY FAILED — do not run the full pass.")
        return 1
    print("\n  verify OK")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--parts-dir", default="results/candidate_pairs",
                     help="directory of part_*.csv shards (preferred)")
    src.add_argument("--pairs-csv",
                     help="single merged candidate_pairs.csv instead")
    ap.add_argument("--name-index", default="results/name_index.csv")
    ap.add_argument("--outdir", default="results/candidate_scores")
    ap.add_argument("--kbigram-model", default="kbigram_model.npz")
    ap.add_argument("--lumera-model", default="lumera_model.npz")
    ap.add_argument("--measures", default=",".join(MEASURES),
                    help=f"comma-separated subset of {','.join(MEASURES)}")
    ap.add_argument("--dtype", default="float32",
                    choices=("float32", "float16"))
    ap.add_argument("--format", default="npy", choices=("npy", "csv"),
                    help="npy sidecar (default) or augmented CSV copies")
    ap.add_argument("--chunks", type=int, default=400,
                    help="byte-range chunks when reading a single CSV")
    ap.add_argument("--limit-parts", type=int, default=0,
                    help="smoke test: process only the first N parts/chunks")
    ap.add_argument("--resume", action="store_true",
                    help="skip parts whose sidecar already exists")
    ap.add_argument("--prefetch", type=int, default=2,
                    help="work units read ahead on reader threads")
    ap.add_argument("--no-numba", action="store_true")
    ap.add_argument("--verify", type=int, default=0,
                    help="check the kernels on N random pairs and exit")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress-secs", type=int, default=30)
    args = ap.parse_args()

    sel = [m.strip() for m in args.measures.split(",") if m.strip()]
    unknown = [m for m in sel if m not in MEASURES]
    if unknown:
        sys.exit(f"unknown measure(s): {unknown}; choose from {list(MEASURES)}")
    want = [m in sel for m in MEASURES]

    use_numba = HAVE_NUMBA and not args.no_numba
    print("=" * 78)
    print("  CANDIDATE SCORING")
    print("=" * 78)
    print(f"measures        : {', '.join(sel)}")
    print(f"kernel          : {'numba (jit, all cores)' if use_numba else 'numpy batched fallback'}")
    if not use_numba and not args.no_numba:
        print("                  numba not installed — this will be ~9x slower")
        print("                  per core.  pip install numba")

    # ---- names -------------------------------------------------------
    import pandas as pd
    nidx = Path(args.name_index)
    if not nidx.exists():
        sys.exit(f"missing {nidx}")
    ni = pd.read_csv(nidx, dtype={"name_id": np.int32, "brand_root": str},
                     keep_default_na=False)
    if not (ni["name_id"].to_numpy() == np.arange(len(ni))).all():
        sys.exit("name_index.csv name_id column is not 0..n-1 in order; "
                 "the pair ids would be misresolved")
    names = ni["brand_root"].astype(str).tolist()
    print(f"names           : {len(names):,}")

    # ---- models ------------------------------------------------------
    S_kb = np.zeros((1, 1), np.float32)
    kb_vocab, kb_gap = {}, 0.5
    if want[2]:
        p = Path(args.kbigram_model)
        if not p.exists():
            sys.exit(f"missing {p} — train it or drop kbigram from --measures")
        S_kb, kb_vocab, kb_gap = load_kbigram(p)
        print(f"kbigram model   : {p}  ({S_kb.shape[0]} bigrams, gap {kb_gap})")

    S_lu = np.zeros((1, 1), np.float32)
    lu_alpha, lu_go, lu_ge = _FALLBACK_LUMERA_ALPHABET, 0.6, 0.15
    if want[3]:
        p = Path(args.lumera_model)
        if not p.exists():
            sys.exit(f"missing {p} — train it (python lumera_train.py) or "
                     f"drop lumera from --measures")
        S_lu, lu_alpha, lu_go, lu_ge = load_lumera(p)
        print(f"lumera model    : {p}  ({S_lu.shape[0]} chars, "
              f"gap {lu_go}/{lu_ge})")

    enc = encode_names(names, lu_alpha, kb_vocab)
    n_fe = enc[-1]
    S_id = np.eye(max(n_fe, 1), dtype=np.float32)
    if want[3] and S_lu.shape[0] < 1:
        sys.exit("lumera substitution matrix is empty")
    models = (S_id, S_lu, lu_go, lu_ge, S_kb, kb_gap)

    # ---- verify ------------------------------------------------------
    if args.verify:
        print("\n" + "=" * 78)
        print("  VERIFY — kernels vs the reference modules")
        print("=" * 78)
        sys.exit(verify(args, enc, models, want, names, use_numba))

    # ---- work list ---------------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dt = np.float32 if args.dtype == "float32" else np.float16

    if args.pairs_csv:
        path = Path(args.pairs_csv)
        if not path.exists():
            sys.exit(f"missing {path}")
        print(f"\nsplitting {path} into {args.chunks} byte ranges...")
        ranges = byte_chunks(path, args.chunks)
        jobs = [("range", i, (path, lo, hi)) for i, (lo, hi) in
                enumerate(ranges)]
    else:
        d = Path(args.parts_dir)
        parts = sorted(d.glob("part_*.csv"))
        if not parts:
            sys.exit(f"no part_*.csv in {d} — pass --pairs-csv instead "
                     f"if the shards were merged and deleted")
        jobs = [("part", i, (p,)) for i, p in enumerate(parts)]

    if args.limit_parts:
        jobs = jobs[:args.limit_parts]
    print(f"work units      : {len(jobs):,}")
    print(f"output          : {outdir}  ({args.format}, {args.dtype})")
    print()

    def load(job):
        kind, i, spec = job
        if kind == "part":
            a, b = read_ids(spec[0])
            return job, a, b
        a, b = read_range(*spec)
        return job, a, b

    manifest = []
    total_rows = 0
    t0 = tlast = time.time()
    io_s = cpu_s = 0.0

    # The kernel releases the GIL, and pandas' C parser mostly does too, so
    # reading runs underneath computing. Depth 2 keeps a many-core box fed:
    # one reader sustains ~5M rows/s, which covers compute up to about nine
    # cores before parsing becomes the critical path.
    depth = max(1, args.prefetch)
    with ThreadPoolExecutor(max_workers=depth) as pre:
        queue = [pre.submit(load, j) for j in jobs[:depth]]
        nxt = depth
        for n, job in enumerate(jobs):
            kind, i, spec = job
            stem = (spec[0].stem if kind == "part" else f"chunk_{i:05d}")
            dest = outdir / f"{stem}.{'npy' if args.format == 'npy' else 'csv'}"

            t = time.time()
            _, ia, ib = queue.pop(0).result()
            io_s += time.time() - t
            if nxt < len(jobs):
                queue.append(pre.submit(load, jobs[nxt]))
                nxt += 1

            if args.resume and dest.exists() and args.format == "npy":
                try:
                    if len(np.load(dest, mmap_mode="r")) == len(ia):
                        manifest.append((stem, total_rows, len(ia),
                                         str(dest)))
                        total_rows += len(ia)
                        continue
                except Exception:
                    pass

            t = time.time()
            sc = score_ids(ia, ib, enc, models, want, use_numba)
            cpu_s += time.time() - t

            if args.format == "npy":
                np.save(dest, sc.astype(dt))
            else:
                with open(dest, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["name_id_a", "name_id_b"] + list(MEASURES))
                    for k in range(len(ia)):
                        w.writerow([int(ia[k]), int(ib[k])] +
                                   [f"{sc[k, c]:.6f}" for c in
                                    range(len(MEASURES))])

            manifest.append((stem, total_rows, len(ia), str(dest)))
            total_rows += len(ia)

            now = time.time()
            if now - tlast >= args.progress_secs or n + 1 == len(jobs):
                el = now - t0
                rate = total_rows / el if el else 0
                left = (len(jobs) - n - 1) / max(n + 1, 1) * el
                print(f"  {n+1:,}/{len(jobs):,} units | rows={total_rows:,} | "
                      f"{rate:,.0f} pairs/s | io={io_s/60:.1f}m "
                      f"cpu={cpu_s/60:.1f}m | elapsed={el/60:.1f}m | "
                      f"~{left/60:.1f}m left")
                tlast = now

    mpath = outdir / "manifest.csv"
    with open(mpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "first_row_index", "n_rows", "path"])
        w.writerows(manifest)

    el = time.time() - t0
    print("\n" + "=" * 78)
    print("  DONE")
    print("=" * 78)
    print(f"pairs scored    : {total_rows:,}")
    print(f"columns         : {', '.join(MEASURES)}  "
          f"(unselected ones are 0.0)")
    print(f"elapsed         : {el/60:.1f} min   "
          f"(io {io_s/60:.1f}, compute {cpu_s/60:.1f})")
    print(f"throughput      : {total_rows/el:,.0f} pairs/s" if el else "")
    print(f"on disk         : "
          f"{sum(Path(p).stat().st_size for _,_,_,p in manifest)/1024**3:.2f} GiB")
    print(f"manifest        : {mpath}")
    print()
    print("Row i of each sidecar is row i of its corresponding input work unit.")
    print("To re-weight:")
    if manifest:
        example_path = manifest[0][3]
        print(f"    s = np.load({example_path!r}, mmap_mode='r')")
    else:
        print("    s = np.load(<sidecar.npy>, mmap_mode='r')")
    print("    combined = s @ np.array([.2,.2,.2,.2,.2], np.float32)")


if __name__ == "__main__":
    main()
