#!/usr/bin/env python3
"""
clean_roots.py — blocker-specific root cleaner (v3).

Patched against real leaks found in the 199k namespace:
  - multi-ingredient strength CHAINS: '5mg/500mg/30mg' (3-4 drug FDCs)
    were only half-stripped, leaving '...500mg' -> letters-key 'mgmg'.
  - form+modifier tails: '...tablet dt' stalled on 'dt' (dispersible
    tablet) before reaching 'tablet'.
  - '...plus' / '...forte' modifiers cluster huge blocks.
  - 1-2 letter wreckage roots ('a', 'a2') can't be a blocking unit.

Two-stage design reminder: this feeds the BLOCKER only. The census
brand_root stays as-is and keeps distinctions like 'X' vs 'X Plus'.
"""
import re, sys, pandas as pd
from pathlib import Path

FORM_WORDS = {
    "tablet","tablets","tab","tabs","capsule","capsules","cap","caps","syrup",
    "suspension","oral","solution","injection","inj","infusion","cream","ointment",
    "gel","lotion","paste","powder","granules","drops","drop","eye","ear","nasal",
    "spray","inhaler","rotacaps","respules","sachet","kit","patch","suppository",
    "pessary","vial","ampoule","bottle","tube","strip","pack","liquid","shampoo",
    "soap","mouthwash","gargle","elixir","emulsion","foam","film","sf",
}
# release/formulation + combination modifiers: dosage-noise for BLOCKING.
MODIFIERS = {"er","xr","cr","sr","dr","od","mr","pr","md","dt","dp","dry",
             "forte","fort","plus","total","active","ls","ds"}
STRENGTH_RE = re.compile(
    r"^\d+(\.\d+)?\s*(mg|g|mcg|ml|l|iu|%|w/w|w/v|meq)?/?"
    r"(\d+(\.\d+)?\s*(mg|g|mcg|ml|l|iu|%)?)?$", re.IGNORECASE)
UNIT = r"(?:mg|mcg|g|ml|l|iu|%)"
# a whole strength CHAIN: number-unit ( / number-unit )*  e.g. 5mg/500mg/30mg
STRENGTH_CHAIN = re.compile(
    rf"\b\d[\d.]*\s*{UNIT}?(?:\s*/\s*\d[\d.]*\s*{UNIT}?)+\b", re.IGNORECASE)

def clean_root(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    # 1. glue single strengths: "50 mg" -> "50mg"
    s = re.sub(rf"(\d[\d.]*)\s*({UNIT})", r"\1\2", s)
    # 2. delete whole multi-ingredient strength chains: "5mg/500mg/30mg" -> ""
    s = STRENGTH_CHAIN.sub(" ", s)
    s = re.sub(r"[^a-z0-9\s\-/+.]", " ", s)
    toks = re.sub(r"\s+", " ", s).strip().split()
    # 3. pop trailing form words, modifiers, and any remaining strength tokens
    while toks and (toks[-1] in FORM_WORDS
                    or toks[-1] in MODIFIERS
                    or STRENGTH_RE.match(toks[-1])):
        toks.pop()
    return " ".join(toks) if toks else s

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results/brand_roots.csv"
    df = pd.read_csv(src)
    before = df["brand_root"].fillna("").astype(str)
    df["clean_root"] = before.map(clean_root)

    # drop wreckage: letters-only key must be >= 3 chars to be a blocking unit
    letters = df["clean_root"].str.lower().str.replace(r"[^a-z]","",regex=True)
    keep = letters.str.len() >= 3
    dropped = (~keep).sum()

    changed = (df["clean_root"] != before).sum()
    nb = before.nunique()
    out_roots = df.loc[keep, "clean_root"].drop_duplicates()
    na = out_roots.nunique()
    print(f"rows                 : {len(df):,}")
    print(f"roots changed        : {changed:,} ({100*changed/len(df):.1f}%)")
    print(f"dropped (<3 letters) : {dropped:,}")
    print(f"distinct roots before: {nb:,}")
    print(f"distinct roots after : {na:,}  ({nb-na:,} fewer)")
    print("\nsample of what changed:")
    ch = df[df["clean_root"] != before][["brand_root","clean_root"]].drop_duplicates().head(15)
    for _, r in ch.iterrows():
        print(f"    {str(r['brand_root'])[:38]:<38} -> {r['clean_root']}")
    out = Path(src).with_name("brand_roots_clean.csv")
    out_roots.rename("brand_root").to_frame().to_csv(out, index=False)
    print(f"\nwrote {na:,} clean roots -> {out}")
