#!/usr/bin/env python3
"""
clean_roots.py — blocker-specific root cleaner (v4).

v4 closes the last strength/form leak (1,357 roots still carried 'mg'-type
tails, which floated same-brand dose-variants to the top of the ranking):
  - strengths in grams ('1gm', '1.5gm', '1gm/0.5gm') were missed because
    the unit list had 'g' but not 'gm'.
  - odd units (iu, au, miu, lb, million/spores) and truncated form words
    (gelatin, softgel, disintegrating, vaginal, dusting, wash...) survived.

Core rule change: pop any trailing token that CONTAINS A DIGIT (every
strength does — 1gm, 100iu, 200lb, 40iu/ml — regardless of unit), as long
as we're still in the tail. Stop at the first pure-alphabetic brand token,
so digit-bearing BRANDS (a2, b12) are only stripped if they're the trailing
dose, never the name. Feeds the BLOCKER only; census brand_root untouched.
"""
import re, sys, pandas as pd
from pathlib import Path

FORM_WORDS = {
    "tablet","tablets","tab","tabs","capsule","capsules","cap","caps","syrup",
    "suspension","oral","orally","solution","injection","inj","injecti","injectio",
    "infusion","cream","ointment","gel","gelatin","softgel","lotion","paste",
    "powder","granules","drops","drop","eye","ear","nasal","spray","inhaler",
    "inhale","rotacaps","respules","sachet","kit","patch","suppository","pessary",
    "vial","ampoule","bottle","tube","strip","pack","liquid","shampoo","soap",
    "mouthwash","gargle","elixir","emulsion","foam","film","sf","vaginal","vagin",
    "disintegrating","disin","dusting","wash","prolong","prolonged","dry","spores",
    "orange","tangy","for","hm","soft","gelatin","softgel","new","cd","xt",
}
MODIFIERS = {"er","xr","cr","sr","dr","od","mr","pr","md","dt","dp","dry",
             "forte","fort","plus","total","active","ls","ds","xl","lb"}
STRENGTH_RE = re.compile(
    r"^\d+(\.\d+)?\s*(mg|gm|g|mcg|ml|l|iu|au|miu|%|w/w|w/v|meq|lb)?/?"
    r"(\d+(\.\d+)?\s*(mg|gm|g|mcg|ml|l|iu|au|miu|%|lb)?)?$", re.IGNORECASE)
UNIT = r"(?:mg|gm|mcg|g|ml|l|iu|au|miu|%|lb)"
STRENGTH_CHAIN = re.compile(
    rf"\b\d[\d.]*\s*{UNIT}?(?:\s*/\s*\d[\d.]*\s*{UNIT}?)+\b", re.IGNORECASE)
HAS_DIGIT = re.compile(r"\d")

def clean_root(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(rf"(\d[\d.]*)\s*({UNIT})", r"\1\2", s)   # glue "1 gm" -> "1gm"
    s = STRENGTH_CHAIN.sub(" ", s)                       # delete "5mg/500mg/30mg"
    s = re.sub(r"[^a-z0-9\s\-/+.]", " ", s)
    toks = re.sub(r"\s+", " ", s).strip().split()

    def strippable(t):
        return (t in FORM_WORDS or t in MODIFIERS
                or STRENGTH_RE.match(t)
                or (HAS_DIGIT.search(t) and len(toks) > 1))

    # First pass: pop the form/strength/digit tail from the end.
    while toks and strippable(toks[-1]):
        toks.pop()
    # Second pass: a SHORT (<=2 char) trailing alpha fragment like 'so','p','f'
    # is a truncated form word ONLY if a strength/form tail still sits behind
    # it. Peek: if popping it EXPOSES more strippable tail, it was debris.
    while len(toks) >= 2 and toks[-1].isalpha() and len(toks[-1]) <= 2 \
            and strippable(toks[-2]):
        toks.pop()                       # drop the fragment
        while toks and strippable(toks[-1]):   # then continue the normal strip
            toks.pop()
    return " ".join(toks) if toks else s

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results/brand_roots.csv"
    df = pd.read_csv(src)
    before = df["brand_root"].fillna("").astype(str)
    df["clean_root"] = before.map(clean_root)
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
    out = Path(src).with_name("brand_roots_clean.csv")
    out_roots.rename("brand_root").to_frame().to_csv(out, index=False)
    print(f"wrote {na:,} clean roots -> {out}")
