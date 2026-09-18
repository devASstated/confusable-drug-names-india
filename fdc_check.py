#!/usr/bin/env python3
"""
fdc_check.py — how badly is composition truncated in the A-Z dataset?

The source schema has only short_composition1 and short_composition2, but Indian
fixed-dose combinations (FDCs) routinely contain 3+ active ingredients. This
script measures, on OUR OWN data, how often that two-field limit is actually a
problem — before deciding whether the national Drug Registry is worth pursuing
as a composition source.

It answers three separate questions:
  1. How many ingredients does each product APPEAR to declare?
     (counted by splitting each composition field on '+' — the dataset's own
      separator for multi-ingredient fields)
  2. How many products use BOTH fields (i.e. sit at the schema's edge)?
  3. Is there evidence of TRUNCATION — products whose own NAME implies more
     ingredients than the composition fields record?

TWO DETECTORS, AND THEY BOUND THE ANSWER FROM OPPOSITE SIDES
  Test A (keyword) only fires on names containing 'Trio', 'Triple' and the
  like. Most 3+ ingredient products do not advertise it that way, so Test A
  is a FLOOR — it misses far more than it catches.

  Test B (strength pattern) fires whenever a name carries more strength values
  than the composition records ingredients. It catches far more, but it
  OVER-counts: strengths written per unit volume look like two values when
  they are one concentration. 'Zoxil CV 400mg/5ml/57mg/5ml' is a TWO
  ingredient product, not four. Test B therefore reports both a raw count and
  a concentration-corrected count; even corrected, treat it as a CEILING,
  because a name can carry repeated strengths for other reasons.

  The honest figure lies between Test A and corrected Test B. What is not in
  doubt is the qualitative finding: a hard cliff at exactly two ingredients
  across a quarter-million Indian products is a schema cap, not a market fact.

USAGE
    python fdc_check.py data/A_Z_medicines_dataset_of_india.csv
    python fdc_check.py <csv> --name-col name --c1 short_composition1 --c2 short_composition2
"""
import argparse
import re
import sys
from collections import Counter

import pandas as pd

# strength parentheses: "Amoxycillin (500mg)" -> ingredient is the part before '('
PAREN = re.compile(r"\([^)]*\)")


def ingredients(field: str):
    """Split one composition field into ingredient names."""
    if not isinstance(field, str) or not field.strip():
        return []
    s = PAREN.sub(" ", field)            # drop strengths
    parts = re.split(r"\s*\+\s*", s)     # the dataset's multi-ingredient separator
    return [p.strip().lower() for p in parts if p.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--name-col", default="name")
    ap.add_argument("--c1", default="short_composition1")
    ap.add_argument("--c2", default="short_composition2")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    for col in (args.c1, args.c2):
        if col not in df.columns:
            sys.exit(f"ERROR: column '{col}' not found. Columns: {list(df.columns)}")
    has_name = args.name_col in df.columns

    n = len(df)
    counts = []          # apparent ingredient count per product
    both_used = 0
    c2_multi = 0         # field-2 itself holds >1 ingredient (schema straining)
    rows_3plus = []

    for r in df.itertuples(index=False):
        f1 = getattr(r, args.c1, "") or ""
        f2 = getattr(r, args.c2, "") or ""
        i1, i2 = ingredients(f1), ingredients(f2)
        total = len(i1) + len(i2)
        counts.append(total)
        if i1 and i2:
            both_used += 1
        if len(i2) > 1:
            c2_multi += 1
        if total >= 3 and len(rows_3plus) < 400:
            nm = getattr(r, args.name_col, "") if has_name else ""
            rows_3plus.append((total, nm, f1, f2))

    dist = Counter(counts)
    print("=" * 70)
    print("  FDC / COMPOSITION-TRUNCATION CHECK  (A-Z dataset)")
    print("=" * 70)
    print(f"\nproducts analysed : {n:,}")
    print(f"\n[APPARENT INGREDIENT COUNT PER PRODUCT]")
    print(f"  (ingredients counted by splitting BOTH composition fields on '+')")
    for k in sorted(dist):
        pct = 100 * dist[k] / n
        bar = "#" * int(pct / 2)
        label = f"{k} ingredient(s)" if k else "0 (empty/unparsed)"
        print(f"    {label:<20} {dist[k]:>7,}  {pct:>5.1f}%  {bar}")

    n3 = sum(v for k, v in dist.items() if k >= 3)
    n2 = dist.get(2, 0)
    print(f"\n[THE QUESTION THAT MATTERS]")
    print(f"  products with 3+ ingredients : {n3:,}  ({100*n3/n:.1f}%)")
    print(f"  products with exactly 2      : {n2:,}  ({100*n2/n:.1f}%)")
    print(f"  both composition fields used : {both_used:,}  ({100*both_used/n:.1f}%)")
    print(f"  field-2 holds >1 ingredient  : {c2_multi:,}  ({100*c2_multi/n:.1f}%)")

    print(f"\n[INTERPRETATION]")
    if n3 == 0:
        print("  No product shows 3+ ingredients. EITHER Indian FDCs with 3+ are")
        print("  absent from this dataset, OR they are silently TRUNCATED to two")
        print("  fields. Check the examples of 2-ingredient products by hand — if")
        print("  a known 3-ingredient brand shows only 2, truncation is real.")
    else:
        print(f"  3+ ingredient products ARE represented ({100*n3/n:.1f}%), so the two")
        print("  fields do NOT hard-cap ingredient count — multiple ingredients are")
        print("  packed inside a field using '+'. Truncation risk is therefore lower")
        print("  than feared, but verify the tail: see the max-count examples below.")

    if rows_3plus:
        rows_3plus.sort(reverse=True)
        print(f"\n[EXAMPLES — highest apparent ingredient counts]")
        for total, nm, f1, f2 in rows_3plus[:args.examples]:
            print(f"    [{total}] {str(nm)[:38]:<40}")
            print(f"         c1: {f1[:80]}")
            if f2:
                print(f"         c2: {f2[:80]}")

    # ---- TEST A: names whose WORDS hint at 3+ ingredients -------------
    suspects = []
    if has_name:
        hint = re.compile(r"\b(trio|triple|tri|quadr|forte\s*plus|3\s*in\s*1|4\s*in\s*1)\b", re.I)
        for r in df.itertuples(index=False):
            nm = str(getattr(r, args.name_col, ""))
            if hint.search(nm):
                tot = len(ingredients(getattr(r, args.c1, ""))) + \
                      len(ingredients(getattr(r, args.c2, "")))
                if tot <= 2:
                    suspects.append((nm, tot))
        print(f"\n[TEST A — KEYWORD SMELL TEST]")
        print(f"  products whose NAME hints at 3+ ingredients but parse to <=2: "
              f"{len(suspects):,}")
        for nm, tot in suspects[:8]:
            print(f"    ({tot} parsed)  {nm[:60]}")
        if not suspects:
            print("    none found — weak evidence AGAINST systematic truncation")

    # ---- TEST B: STRENGTH-PATTERN detector (the strong estimate) -------
    # A name like "Aztel Trio 40mg/5mg/12.5mg Tablet" declares THREE strengths.
    # One strength value per active ingredient is the Indian labelling norm, so
    # if a name carries >=3 strength values but composition parses to <=2
    # ingredients, that row is truncated — regardless of any keyword.
    if has_name:
        # A strength value: number (optional decimal) + unit.
        STRENGTH = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|%|units?)\b", re.I)
        # A CONCENTRATION is one strength expressed per unit volume:
        # '400mg/5ml', '0.291mg/5ml'. The naive counter sees two values and
        # infers two ingredients, which is wrong — suspensions and syrups are
        # routinely labelled this way. Collapse each to a single value before
        # counting, and report both numbers so the over-count is visible.
        CONC = re.compile(
            r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|iu|units?)\s*/\s*\d+(?:\.\d+)?\s*ml\b",
            re.I)

        def declared_strengths(name):
            """(raw count, concentration-corrected count)."""
            raw = len(STRENGTH.findall(name))
            n_conc = len(CONC.findall(name))
            # each concentration contributed 2 matches but represents 1 value
            return raw, raw - n_conc

        trunc, by_declared = [], Counter()
        raw_only = 0                       # flagged by raw count, not corrected
        for r in df.itertuples(index=False):
            nm = str(getattr(r, args.name_col, ""))
            raw, declared = declared_strengths(nm)
            tot = len(ingredients(getattr(r, args.c1, ""))) + \
                  len(ingredients(getattr(r, args.c2, "")))
            if raw >= 3 and tot < raw and declared < 3:
                raw_only += 1             # only the naive counter would flag it
            if declared < 3:
                continue
            if tot < declared:
                trunc.append((declared, tot, nm))
                by_declared[declared] += 1

        print(f"\n[TEST B — STRENGTH-PATTERN DETECTOR]  <- the UPPER bound")
        print(f"  A name carrying N strength values (e.g. '40mg/5mg/12.5mg')")
        print(f"  usually implies N active ingredients. Rows where the name")
        print(f"  declares 3+ and the composition records FEWER are truncated.")
        print(f"  Strengths written per unit volume ('400mg/5ml') are collapsed")
        print(f"  to one value first — otherwise a 2-ingredient suspension looks")
        print(f"  like a 4-ingredient product.\n")
        print(f"  products with >=3 strengths in the name, composition shows fewer:")
        print(f"    {len(trunc):,}  ({100*len(trunc)/n:.2f}% of all products)")
        if raw_only:
            print(f"    (a naive counter would also have flagged {raw_only:,} more")
            print(f"     that are only per-volume concentrations — excluded here)")
        if n2:
            print(f"    = {100*len(trunc)/n2:.1f}% of the {n2:,} products sitting at the 2-field cap")
        if by_declared:
            print(f"\n  by number of strengths declared in the name:")
            for k in sorted(by_declared):
                print(f"    {k} strengths -> {by_declared[k]:>6,} products truncated")
        if trunc:
            trunc.sort(reverse=True)
            print(f"\n  examples (declared vs recorded):")
            for declared, tot, nm in trunc[:args.examples]:
                print(f"    name declares {declared}, composition has {tot}:  {nm[:58]}")
        else:
            print("    none — no strength-pattern evidence of truncation")

        print(f"\n[VERDICT]")
        if n3 == 0 and len(trunc) > 0:
            print(f"  TRUNCATION IS REAL. Zero products record 3+ ingredients,")
            print(f"  yet {len(trunc):,} carry 3+ strength values in their own name.")
            print(f"  A product contradicting itself within one row is not a")
            print(f"  market fact; it is a schema cap.")
            print(f"\n  HOW MANY is bounded, not known:")
            print(f"    lower  {len(suspects) if has_name else 0:,}  (Test A — only names that say")
            print(f"           'Trio'/'Triple'; most FDCs do not advertise it)")
            print(f"    upper  {len(trunc):,}  (Test B — corrected for per-volume")
            print(f"           concentrations, but a name may repeat strengths for")
            print(f"           other reasons)")
            print(f"  Quote the RANGE. Neither endpoint is the count.")
            print(f"\n  The unambiguous cases are tablets with no per-volume")
            print(f"  denominator: a name listing five distinct strengths while the")
            print(f"  composition holds two cannot be explained any other way.")
        elif len(trunc) > 0:
            print(f"  {len(trunc):,} products declare more strengths than their")
            print(f"  composition records. Treat as an upper bound (see above).")
        else:
            print("  No self-contradicting rows found.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
