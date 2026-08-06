#!/usr/bin/env python3
"""
clean_roots.py — blocker-specific root cleaner (v2, surgical).

Fixes the leak WITHOUT eating meaningful short brand suffixes.

Rules, in order:
  1. Collapse spaced strengths first: '50 mg/31.25 mg' -> '50mg/31.25mg'
     so they strip as ONE token, not three fragments.
  2. Pop trailing tokens that are: a form word, a release/formula modifier
     (er/xr/sr/forte/dry/plus...), or a pure strength.
  3. STOP at anything else -- including short tokens like 'm','d','cv'
     because those mark real product variants (Glimcor-M != Glimcor).
Keeps 'glimcor m forte' -> 'glimcor m'   (drops 'forte', keeps 'm')
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
# release/formulation modifiers: dosage-noise for BLOCKING (not the census)
MODIFIERS = {"er","xr","cr","sr","dr","od","mr","pr","md","dry",
             "forte","fort","total","active"}
STRENGTH_RE = re.compile(
    r"^\d+(\.\d+)?\s*(mg|g|mcg|ml|l|iu|%|w/w|w/v|meq)?/?"
    r"(\d+(\.\d+)?\s*(mg|g|mcg|ml|l|iu|%)?)?$", re.IGNORECASE)

def clean_root(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    # 1. glue spaced strengths: "50 mg / 31.25 mg" -> "50mg/31.25mg"
    s = re.sub(r"(\d[\d.]*)\s*(mg|mcg|g|ml|iu|%)", r"\1\2", s)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"[^a-z0-9\s\-/+.]", " ", s)
    toks = re.sub(r"\s+", " ", s).strip().split()
    # 2/3. pop only form words, modifiers, and strengths -- never short brands
    while toks and (toks[-1] in FORM_WORDS
                    or toks[-1] in MODIFIERS
                    or STRENGTH_RE.match(toks[-1])):
        toks.pop()
    return " ".join(toks) if toks else s

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results/brand_roots.csv"
    df = pd.read_csv(src)
    before = df["brand_root"].astype(str)
    df["clean_root"] = before.map(clean_root)
    changed = (df["clean_root"] != before).sum()
    nb, na = before.nunique(), df["clean_root"].nunique()
    print(f"rows                 : {len(df):,}")
    print(f"roots changed        : {changed:,} ({100*changed/len(df):.1f}%)")
    print(f"distinct roots before: {nb:,}")
    print(f"distinct roots after : {na:,}  ({nb-na:,} merged)")
    print("\nsample of what changed:")
    ch = df[df["clean_root"] != before][["brand_root","clean_root"]].drop_duplicates().head(15)
    for _, r in ch.iterrows():
        print(f"    {r['brand_root'][:42]:<42} -> {r['clean_root']}")
    out = Path(src).with_name("brand_roots_clean.csv")
    (df[["clean_root"]].drop_duplicates()
        .rename(columns={"clean_root":"brand_root"})
        .to_csv(out, index=False))
    print(f"\nwrote deduped clean roots -> {out}")
