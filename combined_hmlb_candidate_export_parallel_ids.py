#!/usr/bin/env python3
"""
combined_hmlb_candidate_export_parallel.py

Parallel materialization of the FINAL frozen LASA blocker union.

Frozen rules
------------
Orthographic H/M/L/B:
    score = 0.31875*H + 0.28125*M + 0.15*L + 0.25*B
    undirected pair score = max(score(a -> b), score(b -> a))
    keep if >= 0.568

US phonetic:
    input: phones_g2p_arpa
    score = shared distinct ordered phone bigrams /
            min(distinct bigrams in A, distinct bigrams in B)
    keep if >= 0.50

Indian/Hindi-accent phonetic:
    input: predicted_normalized_phonemes
    same score, keep if >= 0.50

Final candidate set:
    orthographic OR phonetic_US OR phonetic_IN

Output
------
results/name_index.csv

    name_id,brand_root

results/candidate_pairs/
    part_00000.csv
    part_00001.csv
    ...

Candidate schema:
    name_id_a
    name_id_b
    orthographic
    phonetic_US
    phonetic_IN
    hmlb_score
    phonetic_US_score
    phonetic_IN_score

Notes
-----
- Each unordered pair is emitted exactly once because workers only emit j > i.
- Workers process dynamic anchor chunks for load balancing.
- On Linux, multiprocessing uses fork by default, so the large precomputed
  bitsets are shared copy-on-write instead of rebuilt in each worker.
- Sharded output avoids locking/contention on one giant CSV.

Example:
    python combined_hmlb_candidate_export_parallel.py \
        --input normalized_inference_output.csv \
        --workers 4

Smoke test:
    python combined_hmlb_candidate_export_parallel.py \
        --input normalized_inference_output.csv \
        --workers 4 \
        --max-anchors 4000
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


ORTHO_THRESHOLD = 0.568
PHONE_THRESHOLD = 0.50

W_H = 0.31875
W_M = 0.28125
W_L = 0.15
W_B = 0.25

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
AIDX = {c: i for i, c in enumerate(ALPHABET)}
NON_ALPHA = re.compile(r"[^a-z]")

# Globals populated once in parent before fork.
G = {}


def normalize_name(value: str) -> str:
    return NON_ALPHA.sub("", str(value).lower())


def make_letter_mask(s: str) -> int:
    mask = 0
    for c in set(s):
        idx = AIDX.get(c)
        if idx is not None:
            mask |= 1 << idx
    return mask


def make_letter_bigram_mask(s: str) -> int:
    mask = 0
    for i in range(len(s) - 1):
        a = AIDX.get(s[i])
        b = AIDX.get(s[i + 1])
        if a is not None and b is not None:
            mask |= 1 << (a * 26 + b)
    return mask


def parse_phone_tokens(value: str) -> tuple[str, ...]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return ()
    return tuple(tok for tok in text.split() if tok)


def build_phone_bigram_vocab(token_rows):
    vocab = {}
    for toks in token_rows:
        for pair in set(zip(toks, toks[1:])):
            if pair not in vocab:
                vocab[pair] = len(vocab)
    return vocab


def phone_bigram_mask(tokens, vocab):
    mask = 0
    for pair in set(zip(tokens, tokens[1:])):
        mask |= 1 << vocab[pair]
    return mask


def build_feature_postings(feature_masks, vocab_size):
    postings = [0] * vocab_size
    for idx, mask in enumerate(feature_masks):
        x = mask
        while x:
            lsb = x & -x
            f = lsb.bit_length() - 1
            postings[f] |= 1 << idx
            x ^= lsb
    return postings


def overlap_at_least(anchor_mask, postings, all_bits):
    m = anchor_mask.bit_count()
    out = [0] * (m + 1)
    out[0] = all_bits

    seen = 0
    x = anchor_mask
    while x:
        lsb = x & -x
        f = lsb.bit_length() - 1
        p = postings[f]
        seen += 1
        for k in range(seen, 0, -1):
            out[k] |= out[k - 1] & p
        x ^= lsb
    return out


def hmlb_directional(shared_chars, shared_bigrams, ux, ua, lx, la, bx):
    h = shared_chars / ux if ux else 0.0

    absent_x = 26 - ux
    if absent_x > 0:
        m = (26 - ux - ua + shared_chars) / absent_x
    else:
        m = 0.0

    l = min(lx, la) / max(lx, la) if max(lx, la) else 0.0
    b = shared_bigrams / bx if bx else 0.0

    return W_H * h + W_M * m + W_L * l + W_B * b


def hmlb_pair_score(shared_chars, shared_bigrams, ux, ua, lx, la, bx, ba):
    return max(
        hmlb_directional(shared_chars, shared_bigrams, ux, ua, lx, la, bx),
        hmlb_directional(shared_chars, shared_bigrams, ua, ux, la, lx, ba),
    )


def phone_pair_score(shared, ba, bb):
    denom = min(ba, bb)
    return shared / denom if denom else 0.0


def minimal_frontier(ux, lx, bx, ua, la, ba):
    passing = []
    for i in range(min(ux, ua) + 1):
        for j in range(min(bx, ba) + 1):
            score = hmlb_pair_score(i, j, ux, ua, lx, la, bx, ba)
            if score + 1e-15 >= ORTHO_THRESHOLD:
                passing.append((i, j))

    if not passing:
        return ()

    frontier = []
    for i, j in passing:
        redundant = False
        for i2, j2 in passing:
            if (i2, j2) == (i, j):
                continue
            if i2 <= i and j2 <= j:
                redundant = True
                break
        if not redundant:
            frontier.append((i, j))

    return tuple(sorted(frontier))


def grouped_phone_plan(anchor_count, target_count_bitsets):
    by_required = defaultdict(int)
    if anchor_count <= 0:
        return ()

    for target_count, bits in target_count_bitsets.items():
        if target_count <= 0:
            continue
        required = math.ceil(PHONE_THRESHOLD * min(anchor_count, target_count) - 1e-15)
        required = max(required, 1)
        if required <= anchor_count:
            by_required[required] |= bits

    return tuple(sorted(by_required.items()))


@lru_cache(maxsize=None)
def ortho_plan(anchor_structure):
    ux, lx, bx = anchor_structure
    by_frontier = defaultdict(int)

    for target_structure, target_bits in G["structure_bits"].items():
        ua, la, ba = target_structure
        frontier = minimal_frontier(ux, lx, bx, ua, la, ba)
        if frontier:
            by_frontier[frontier] |= target_bits

    return tuple((frontier, bits) for frontier, bits in by_frontier.items())


@lru_cache(maxsize=None)
def us_plan(anchor_count):
    return grouped_phone_plan(anchor_count, G["us_count_bits"])


@lru_cache(maxsize=None)
def in_plan(anchor_count):
    return grouped_phone_plan(anchor_count, G["in_count_bits"])


def ortho_candidates(i, char_ge, bg_ge):
    accepted = 0
    for frontier, target_bits in ortho_plan(G["structures"][i]):
        local = 0
        for need_chars, need_bg in frontier:
            if need_chars < len(char_ge) and need_bg < len(bg_ge):
                local |= char_ge[need_chars] & bg_ge[need_bg]
        accepted |= local & target_bits
    return accepted


def phone_candidates(count, overlap_ge, plan):
    accepted = 0
    if count <= 0:
        return 0
    for required, target_bits in plan:
        if required < len(overlap_ge):
            accepted |= overlap_ge[required] & target_bits
    return accepted


def format_score(x):
    return f"{x:.6f}"


def process_chunk(task):
    chunk_id, start_i, end_i, outdir, flush_rows = task

    names = G["names"]
    n = G["n"]
    all_bits = G["all_bits"]

    outpath = Path(outdir) / f"part_{chunk_id:05d}.csv"

    header = [
        "name_id_a",
        "name_id_b",
        "orthographic",
        "phonetic_US",
        "phonetic_IN",
        "hmlb_score",
        "phonetic_US_score",
        "phonetic_IN_score",
    ]

    emitted = 0
    buffer = []

    with open(outpath, "w", newline="", encoding="utf-8", buffering=16 * 1024 * 1024) as raw:
        writer = csv.writer(raw)
        writer.writerow(header)

        for i in range(start_i, min(end_i, n - 1)):
            char_ge = overlap_at_least(G["char_masks"][i], G["char_postings"], all_bits)
            bg_ge = overlap_at_least(G["ortho_bg_masks"][i], G["ortho_bg_postings"], all_bits)
            us_ge = overlap_at_least(G["us_masks"][i], G["us_postings"], all_bits)
            in_ge = overlap_at_least(G["in_masks"][i], G["in_postings"], all_bits)

            o_bits = ortho_candidates(i, char_ge, bg_ge)
            us_bits = phone_candidates(G["us_counts"][i], us_ge, us_plan(G["us_counts"][i]))
            in_bits = phone_candidates(G["in_counts"][i], in_ge, in_plan(G["in_counts"][i]))

            higher = all_bits & ~((1 << (i + 1)) - 1)
            o_bits &= higher
            us_bits &= higher
            in_bits &= higher
            union = o_bits | us_bits | in_bits

            while union:
                lsb = union & -union
                j = lsb.bit_length() - 1
                union ^= lsb

                shared_chars = (G["char_masks"][i] & G["char_masks"][j]).bit_count()
                shared_ortho_bg = (G["ortho_bg_masks"][i] & G["ortho_bg_masks"][j]).bit_count()
                shared_us = (G["us_masks"][i] & G["us_masks"][j]).bit_count()
                shared_in = (G["in_masks"][i] & G["in_masks"][j]).bit_count()

                hmlb = hmlb_pair_score(
                    shared_chars,
                    shared_ortho_bg,
                    G["char_counts"][i],
                    G["char_counts"][j],
                    G["lengths"][i],
                    G["lengths"][j],
                    G["ortho_bg_counts"][i],
                    G["ortho_bg_counts"][j],
                )
                us_score = phone_pair_score(shared_us, G["us_counts"][i], G["us_counts"][j])
                in_score = phone_pair_score(shared_in, G["in_counts"][i], G["in_counts"][j])

                o = 1 if hmlb + 1e-15 >= ORTHO_THRESHOLD else 0
                p_us = 1 if us_score + 1e-15 >= PHONE_THRESHOLD else 0
                p_in = 1 if in_score + 1e-15 >= PHONE_THRESHOLD else 0

                if not (o or p_us or p_in):
                    raise RuntimeError(f"candidate mismatch for {i},{j}")

                buffer.append([
                    i,
                    j,
                    o,
                    p_us,
                    p_in,
                    format_score(hmlb),
                    format_score(us_score),
                    format_score(in_score),
                ])
                emitted += 1

                if len(buffer) >= flush_rows:
                    writer.writerows(buffer)
                    buffer.clear()

        if buffer:
            writer.writerows(buffer)

    size = outpath.stat().st_size
    return {
        "chunk_id": chunk_id,
        "start": start_i,
        "end": end_i,
        "pairs": emitted,
        "bytes": size,
        "path": str(outpath),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="normalized_inference_output.csv")
    ap.add_argument("--outdir", default="results/candidate_pairs")
    ap.add_argument("--name-index", default="results/name_index.csv")
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--chunk-anchors", type=int, default=2000)
    ap.add_argument("--flush-rows", type=int, default=250_000)
    ap.add_argument(
        "--max-anchors",
        type=int,
        default=0,
        help="smoke-test limit on anchor count (0 = full namespace)",
    )
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required")

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    name_index_path = Path(args.name_index)
    outdir.mkdir(parents=True, exist_ok=True)
    name_index_path.parent.mkdir(parents=True, exist_ok=True)

    # Prevent accidental mixing of old and new shards.
    existing = list(outdir.glob("part_*.csv"))
    if existing:
        raise SystemExit(
            f"{outdir} already contains {len(existing)} part_*.csv files. "
            f"Move/delete them before starting a fresh run."
        )

    print("=" * 104)
    print("LASA FINAL HMLB + US + IN BLOCKER — PARALLEL PAIR MATERIALIZATION")
    print("=" * 104)
    print(f"input                   : {input_path}")
    print(f"output directory        : {outdir}")
    print(f"name index              : {name_index_path}")
    print(f"workers                 : {args.workers}")
    print(f"anchor chunk size       : {args.chunk_anchors:,}")
    print(f"orthographic threshold  : {ORTHO_THRESHOLD}")
    print(f"US phonetic threshold   : {PHONE_THRESHOLD}")
    print(f"IN phonetic threshold   : {PHONE_THRESHOLD}")
    if args.max_anchors:
        print(f"max anchors             : {args.max_anchors:,} [SMOKE TEST]")
    print()

    usecols = ["brand_root", "phones_g2p_arpa", "predicted_normalized_phonemes"]
    df = pd.read_csv(input_path, usecols=usecols, encoding="utf-8-sig")

    names = df["brand_root"].fillna("").astype(str).tolist()
    n = len(names)
    print(f"namespace names         : {n:,}")

    # Stable row-ID mapping used by every candidate shard.
    # brand_root is expected to be unique in the namespace.
    if len(set(names)) != n:
        raise SystemExit(
            "brand_root is not unique; refusing to create an ambiguous name_id mapping"
        )

    with open(name_index_path, "w", newline="", encoding="utf-8") as nf:
        nw = csv.writer(nf)
        nw.writerow(["name_id", "brand_root"])
        nw.writerows((i, name) for i, name in enumerate(names))

    print(f"name index rows         : {n:,}")

    norm = [normalize_name(x) for x in names]
    lengths = [len(x) for x in norm]
    char_masks = [make_letter_mask(x) for x in norm]
    char_counts = [x.bit_count() for x in char_masks]
    ortho_bg_masks = [make_letter_bigram_mask(x) for x in norm]
    ortho_bg_counts = [x.bit_count() for x in ortho_bg_masks]

    structures = [
        (char_counts[i], lengths[i], ortho_bg_counts[i])
        for i in range(n)
    ]
    structure_bits = defaultdict(int)
    for i, st in enumerate(structures):
        structure_bits[st] |= 1 << i

    print(f"orthographic structures : {len(structure_bits):,}")

    us_tokens = [parse_phone_tokens(x) for x in df["phones_g2p_arpa"].tolist()]
    in_tokens = [parse_phone_tokens(x) for x in df["predicted_normalized_phonemes"].tolist()]

    us_vocab = build_phone_bigram_vocab(us_tokens)
    in_vocab = build_phone_bigram_vocab(in_tokens)
    us_masks = [phone_bigram_mask(x, us_vocab) for x in us_tokens]
    in_masks = [phone_bigram_mask(x, in_vocab) for x in in_tokens]
    us_counts = [x.bit_count() for x in us_masks]
    in_counts = [x.bit_count() for x in in_masks]

    print(f"US phone-bigram vocab   : {len(us_vocab):,}")
    print(f"IN phone-bigram vocab   : {len(in_vocab):,}")
    print("building shared postings...")

    char_postings = build_feature_postings(char_masks, 26)
    ortho_bg_postings = build_feature_postings(ortho_bg_masks, 26 * 26)
    us_postings = build_feature_postings(us_masks, len(us_vocab))
    in_postings = build_feature_postings(in_masks, len(in_vocab))

    us_count_bits = defaultdict(int)
    in_count_bits = defaultdict(int)
    for i in range(n):
        us_count_bits[us_counts[i]] |= 1 << i
        in_count_bits[in_counts[i]] |= 1 << i

    # Populate globals before forking.
    G.update({
        "names": names,
        "n": n,
        "all_bits": (1 << n) - 1,
        "lengths": lengths,
        "char_masks": char_masks,
        "char_counts": char_counts,
        "ortho_bg_masks": ortho_bg_masks,
        "ortho_bg_counts": ortho_bg_counts,
        "structures": structures,
        "structure_bits": dict(structure_bits),
        "us_masks": us_masks,
        "in_masks": in_masks,
        "us_counts": us_counts,
        "in_counts": in_counts,
        "char_postings": char_postings,
        "ortho_bg_postings": ortho_bg_postings,
        "us_postings": us_postings,
        "in_postings": in_postings,
        "us_count_bits": dict(us_count_bits),
        "in_count_bits": dict(in_count_bits),
    })

    anchor_end = min(n - 1, args.max_anchors if args.max_anchors else n - 1)

    tasks = []
    chunk_id = 0
    for start_i in range(0, anchor_end, args.chunk_anchors):
        end_i = min(start_i + args.chunk_anchors, anchor_end)
        tasks.append((chunk_id, start_i, end_i, str(outdir), args.flush_rows))
        chunk_id += 1

    print(f"work chunks             : {len(tasks):,}")
    print()

    # Linux fork is intentional: share large bitsets copy-on-write.
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    start = time.monotonic()
    done = 0
    total_pairs = 0
    total_bytes = 0

    with ctx.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(process_chunk, tasks, chunksize=1):
            done += 1
            total_pairs += result["pairs"]
            total_bytes += result["bytes"]

            elapsed = time.monotonic() - start
            rate = total_pairs / elapsed if elapsed else 0.0

            print(
                f"chunks={done:,}/{len(tasks):,} | "
                f"last={result['start']:,}-{result['end']-1:,} | "
                f"pairs={total_pairs:,} | "
                f"rate={rate:,.0f} pairs/s | "
                f"csv={total_bytes/(1024**3):,.2f} GiB | "
                f"elapsed={elapsed/60:,.1f} min"
            )

    elapsed = time.monotonic() - start

    print()
    print("=" * 104)
    print("PARALLEL EXPORT COMPLETE")
    print("=" * 104)
    print(f"parts                   : {len(tasks):,}")
    print(f"pairs emitted           : {total_pairs:,}")
    print(f"total CSV GiB           : {total_bytes/(1024**3):,.3f}")
    print(f"runtime seconds         : {elapsed:,.1f}")
    print(f"aggregate pair rate     : {total_pairs/elapsed:,.0f}/s" if elapsed else "")
    print()
    print("Name lookup:")
    print(f"  {name_index_path}")
    print()
    print("Output shards:")
    print(f"  {outdir}/part_*.csv")
    print()
    print("Optional concatenation into one CSV:")
    print(f"  head -n 1 {outdir}/part_00000.csv > results/candidate_pairs.csv")
    print(f"  for f in {outdir}/part_*.csv; do tail -n +2 \"$f\"; done >> results/candidate_pairs.csv")


if __name__ == "__main__":
    main()
