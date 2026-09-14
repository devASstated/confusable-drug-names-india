#!/usr/bin/env python3
"""
lasa_refset.py — build one clean evaluation set from the ISMP + IMSN files.

Pools three published sources into a single deduped table:
    name_a, name_b, label, source
      label 1 = LASA pair published by a medication-safety body (ISMP 2023 +
                IMSN SALAD Bar 2024). NOTE these lists mix pairs CONFIRMED
                confused in reported errors with pairs judged to have the
                POTENTIAL for confusion — so label=1 means "expert-listed",
                not "observed error".
      label 0 = synthetic negative (random non-confusable pair, from the ML file)

Normalisation (these lists are messy):
    - strip (R) marks and parenthetical glosses:
        'Actimel(R)(dietary supplement)' -> 'actimel'
    - lowercase tall-man caps: 'acetaZOLAMIDE' -> 'acetazolamide'
    - collapse whitespace
    - drop commutative duplicates (A|B == B|A) via a frozenset key

NOTE these are INTERNATIONAL / generic names, not Indian brand roots. That is
deliberate: this set validates the METHOD (scoring + blocking keys), which is
name-agnostic — not coverage of the Indian namespace.

USAGE
    python lasa_refset.py           # writes results/lasa_refset.csv, prints summary
"""

import re
import sys
from pathlib import Path

import pandas as pd

RMARK = re.compile(r"[®™*]")
PARENS = re.compile(r"\([^)]*\)")


def norm(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = RMARK.sub("", name)
    s = PARENS.sub(" ", s)              # drop '(dietary supplement)', '(bisacodyl)'
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load():
    src = Path("data")
    pairs = {}   # frozenset({a,b}) -> (a, b, label, source)   dedups commutative

    def add(a, b, label, source):
        a, b = norm(a), norm(b)
        if not a or not b or a == b:
            return
        key = frozenset((a, b))
        # positives win over negatives if a pair somehow appears as both
        if key in pairs and pairs[key][2] >= label:
            return
        pairs[key] = (a, b, label, source)

    ismp = pd.read_csv(src / "ISMP_2023_LASA_pairs.csv")
    for r in ismp[ismp.LASA == 1].itertuples():
        add(r.drug_A, r.drug_B, 1, "ISMP2023")

    imsn = pd.read_csv(src / "IMSN_SALAD_Bar_2024_pairs.csv")
    for r in imsn[imsn.LASA == 1].itertuples():
        add(r.drug_A, r.drug_B, 1, "IMSN2024")

    ml = pd.read_csv(src / "ISMP_2023_LASA_ML_dataset_no_commutative_pairs.csv")
    for r in ml[ml.label == 0].itertuples():        # negatives only from here
        add(r.drug_A, r.drug_B, 0, "synthetic_negative")

    df = pd.DataFrame([v for v in pairs.values()],
                      columns=["name_a", "name_b", "label", "source"])
    return df


def main():
    df = load()
    df.to_csv("results/lasa_refset.csv", index=False)
    print(f"unique pairs        : {len(df):,}")
    print(f"  positives (label 1): {(df.label==1).sum():,}")
    print(f"  negatives (label 0): {(df.label==0).sum():,}")
    print(f"\nby source:")
    print(df.source.value_counts().to_string())
    print(f"\nsample:")
    print(df.head(8).to_string(index=False))
    print(f"\nwrote -> results/lasa_refset.csv")


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    main()
