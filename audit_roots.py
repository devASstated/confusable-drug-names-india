#!/usr/bin/env python3
"""
audit_roots.py — systematically audit what clean_roots.py did.

The cleaner is the first stage of the whole pipeline, so a silent defect there
corrupts blocking, scoring and ranking together. Finding problems by chance
(as with "A Plus") is not a process. This reads root_map.csv and sorts every
mapping into named SUSPICION CLASSES so you review by category, not by row.

IT AUDITS block_root, NOT clean_root.
    Since composition-aware dedup, the string handed to the blocker is
    block_root; it differs from clean_root wherever a clean_root covered two
    or more DIFFERENT compositions and the members were kept apart. Auditing
    clean_root there inspects a value the pipeline discards.
    This matters more than it sounds. When the composition change first
    landed, an audit of clean_root returned class counts IDENTICAL to the
    previous run — which read as "nothing broke" but actually meant "this tool
    is not measuring what changed". The real defects (a brand reduced to 'a',
    'ls plus' rebuilt as 'plus', two unrelated brands colliding on 'l') were
    found by reading root_map.csv by hand. An audit that reports clean while
    brands are being destroyed is worse than no audit.
    Older root_map.csv files without a block_root column fall back to
    clean_root, with a warning.

EXAMPLES ARE SAMPLED RANDOMLY, not taken from the top.
    The first N rows of a class are alphabetically clustered and therefore
    unrepresentative — 'a 3', 'a-3', 'a2' tell you about one corner of the
    namespace. --seed makes a sample reproducible so a finding can be quoted
    and re-checked; vary it to probe different parts of a class.

Each class answers a different failure question:
  A DROPPED        - which names never reach the blocker at all?
  B HEAVY SHRINK   - which names lost most of their content?
  C DIGIT LEAK     - which cleaned roots still carry a strength?
  D TAIL LEAK      - which still end in a form/modifier word (cleaning missed)?
  E MODIFIER LOSS  - where a modifier was stripped and the remainder is short
                     (the "A Plus" class: modifier may BE the brand)
  F BIG MERGE      - which cleaned roots absorbed many distinct brand roots?
  G NEAR-MERGE     - brand roots differing ONLY by a modifier that merged
                     (intra-brand collapse: expected, but verify)
  H ODD CHARS      - leftover punctuation / suspicious residue
  I COMP SPLIT     - block_root differs from clean_root: the row was kept
                     distinct because its clean_root covered 2+ compositions
                     (brand-name-extension candidates). Check these carry no
                     strength/form debris — they are light-cleaned, not fully
                     cleaned, so a leak here reaches the blocker.

USAGE
    python audit_roots.py results/root_map.csv
    python audit_roots.py results/root_map.csv --examples 25 --out results/root_audit.csv
"""
import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd

FORM_WORDS = {
    "tablet","tablets","tab","tabs","capsule","capsules","cap","caps","syrup",
    "suspension","oral","orally","solution","injection","inj","infusion","cream",
    "ointment","gel","lotion","paste","powder","granules","drops","drop","eye",
    "ear","nasal","spray","inhaler","sachet","kit","patch","suppository","vial",
    "ampoule","bottle","tube","strip","pack","liquid","shampoo","soap","elixir",
    "emulsion","foam","film","cd","xt","new","soft",
}
MODIFIERS = {"er","xr","cr","sr","dr","od","mr","pr","md","dt","dp","dry",
             "forte","fort","plus","total","active","ls","ds","xl","lb"}
HAS_DIGIT = re.compile(r"\d")
LETTERS = re.compile(r"[^a-z]")


def n_letters(s):
    return len(LETTERS.sub("", str(s).lower()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root_map", nargs="?", default="results/root_map.csv")
    ap.add_argument("--examples", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the random example sample; vary it to probe "
                         "different parts of a class, quote it to make a "
                         "finding re-checkable")
    ap.add_argument("--out", default="", help="write all flagged rows to this CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.root_map, dtype=str, keep_default_na=False)
    for c in ("brand_root", "clean_root", "kept"):
        if c not in df.columns:
            raise SystemExit(f"expected column '{c}' in {args.root_map}")
    df["kept_b"] = df["kept"].astype(str).str.lower().isin(("true", "1", "yes"))

    # block_root is what the blocker actually receives. Audit that.
    if "block_root" in df.columns:
        df["audit_root"] = df["block_root"]
        n_split = int((df["block_root"] != df["clean_root"]).sum())
    else:
        print("  [!] no block_root column — this root_map.csv predates")
        print("      composition-aware dedup. Falling back to clean_root, which")
        print("      is NOT what the blocker consumes. Regenerate it.")
        df["audit_root"] = df["clean_root"]
        n_split = 0
    N = len(df)
    E = args.examples
    flags = defaultdict(list)      # class -> list of (brand_root, clean_root, note)

    print("=" * 74)
    print("  CLEAN-ROOT AUDIT")
    print("=" * 74)
    print(f"\nmappings         : {N:,}")
    print(f"reached blocker  : {df.kept_b.sum():,}")
    print(f"dropped          : {(~df.kept_b).sum():,}")
    print(f"auditing column  : "
          f"{'block_root (what the blocker receives)' if 'block_root' in df.columns else 'clean_root (FALLBACK)'}")
    if n_split:
        print(f"  of which kept distinct by composition-aware dedup: {n_split:,}")

    # ---- A. DROPPED ---------------------------------------------------
    dropped = df[~df.kept_b]
    for r in dropped.itertuples():
        flags["A_DROPPED"].append((r.brand_root, r.audit_root, "never reaches blocker"))

    # I. composition-split rows: light-cleaned, so verify no debris got through
    if "block_root" in df.columns:
        for r in df[df.kept_b].itertuples():
            if str(r.block_root) != str(r.clean_root):
                flags["I_COMP_SPLIT"].append(
                    (r.brand_root, r.block_root,
                     f"kept distinct (clean_root was '{r.clean_root}')"))

    # ---- B/C/D/E/H over kept rows -------------------------------------
    for r in df[df.kept_b].itertuples():
        br, cr = str(r.brand_root), str(r.audit_root)
        lb, lc = n_letters(br), n_letters(cr)
        toks_b = br.lower().split()
        toks_c = cr.lower().split()

        # B. heavy shrink: lost more than half the letters
        if lb >= 6 and lc < lb * 0.5:
            flags["B_HEAVY_SHRINK"].append((br, cr, f"{lb}->{lc} letters"))

        # C. digit leak: a strength may have survived
        if HAS_DIGIT.search(cr):
            flags["C_DIGIT_LEAK"].append((br, cr, "cleaned root still has a digit"))

        # D. tail leak: still ends in a form/modifier word
        if toks_c and (toks_c[-1] in FORM_WORDS or toks_c[-1] in MODIFIERS):
            flags["D_TAIL_LEAK"].append((br, cr, f"ends in '{toks_c[-1]}'"))

        # E. modifier loss with a short remainder (the "A Plus" class)
        removed = [t for t in toks_b if t not in toks_c]
        mods_removed = [t for t in removed if t in MODIFIERS]
        if mods_removed and lc <= 5:
            flags["E_MODIFIER_LOSS"].append(
                (br, cr, f"stripped {mods_removed}, only {lc} letters left"))

        # H. odd residue
        if re.search(r"[^a-z0-9\s\-/+.]", cr.lower()) or cr.strip() != cr:
            flags["H_ODD_CHARS"].append((br, cr, "suspicious characters/whitespace"))

    # ---- F. big merges -------------------------------------------------
    kept = df[df.kept_b]
    grp = kept.groupby("audit_root")["brand_root"].apply(lambda s: sorted(set(s)))
    merges = grp[grp.map(len) > 1].sort_values(key=lambda s: s.map(len),
                                               ascending=False)
    for cr, members in merges.items():
        if len(members) >= 5:
            flags["F_BIG_MERGE"].append(
                (f"{len(members)} roots", cr, "; ".join(members[:4]) + " ..."))

    # ---- G. near-merge: differ only by a modifier ----------------------
    for cr, members in merges.items():
        if len(members) != 2:
            continue
        a, b = [str(m).lower().split() for m in members]
        if len(a) != len(b):
            longer, shorter = (a, b) if len(a) > len(b) else (b, a)
            extra = [t for t in longer if t not in shorter]
            if extra and all(t in MODIFIERS for t in extra):
                flags["G_NEAR_MERGE"].append(
                    (members[0], members[1], f"differ only by {extra}"))

    # ---- REPORT --------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    order = ["A_DROPPED", "B_HEAVY_SHRINK", "C_DIGIT_LEAK", "D_TAIL_LEAK",
             "E_MODIFIER_LOSS", "F_BIG_MERGE", "G_NEAR_MERGE", "H_ODD_CHARS",
             "I_COMP_SPLIT"]
    meaning = {
        "A_DROPPED":       "excluded from the namespace entirely — confirm none is a real brand",
        "B_HEAVY_SHRINK":  "lost >50% of letters — check the cleaner didn't eat the name",
        "C_DIGIT_LEAK":    "strength may have survived cleaning (dose variants will collide)",
        "D_TAIL_LEAK":     "form/modifier word survived — cleaning missed a tail",
        "E_MODIFIER_LOSS": "a modifier was stripped from a SHORT name — it may BE the brand",
        "F_BIG_MERGE":     "many distinct brands collapsed to one blocking string",
        "G_NEAR_MERGE":    "two brands merged that differ only by a modifier (intra-brand)",
        "H_ODD_CHARS":     "leftover punctuation or whitespace residue",
        "I_COMP_SPLIT":    "kept distinct for a differing composition — check for debris",
    }
    print(f"\n{'class':<18}{'count':>9}   what it means")
    print("-" * 74)
    for k in order:
        print(f"{k:<18}{len(flags[k]):>9,}   {meaning[k]}")

    for k in order:
        rows = flags[k]
        if not rows:
            continue
        print(f"\n{'=' * 74}\n  {k}  ({len(rows):,})  — {meaning[k]}\n{'=' * 74}")
        shown = rows if len(rows) <= E else \
            [rows[i] for i in sorted(rng.choice(len(rows), E, replace=False))]
        for a, b, note in shown:
            print(f"    {str(a)[:34]:<36} -> {str(b)[:26]:<28} {note}")
        if len(rows) > E:
            print(f"    ... {len(rows)-E:,} more not shown "
                  f"(random sample, seed {args.seed})")

    if args.out:
        recs = [{"class": k, "brand_root": a, "clean_root": b, "note": n}
                for k in order for a, b, n in flags[k]]
        pd.DataFrame(recs).to_csv(args.out, index=False)
        print(f"\nwrote {len(recs):,} flagged rows -> {args.out}")

    print(f"\n{'=' * 74}")
    print("  Review order: A (exclusions) and E (modifier-as-brand) are the")
    print("  correctness risks. C and D are cleaning LEAKS (dose variants will")
    print("  pollute the ranking). F and G are expected but worth spot-checking.")
    print("=" * 74)


if __name__ == "__main__":
    main()
