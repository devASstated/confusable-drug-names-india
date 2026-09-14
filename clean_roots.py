#!/usr/bin/env python3
"""
clean_roots.py — blocker-specific root cleaner.

Produces the blocking representation from census.py's brand_root. Feeds the
BLOCKER only; the census brand_root is left untouched.

Design, in the order the rules apply:
  - strip parentheses, glue "1 gm" -> "1gm", delete strength chains
    ("5mg/500mg/30mg") and concentrations ("40mg/ml", "100iu/ml")
  - pop the trailing form/strength tail, right to left. A digit-bearing token
    counts as a strength only when its alphabetic content is <=2 characters,
    so 'maxx-2m' (the real brand "G Maxx-2M") survives while '2m'/'100iu' go
  - 'soft' is popped only after gelatin/gel/capsule; standalone it is brand
    text ("I Soft" is a marketed eye drop)
  - BACK-OFF GUARD: never strip a name out of existence. If the result cannot
    survive the filter, restore brand-meaningful tokens; if the name was built
    entirely from form-like words ("MD Plus", "New NP"), keep the head plus
    non-debris tokens rather than truncating or merging distinct brands

Survival filter: >=2 alphanumerics and >=1 letter. The threshold is 2, not 3,
because two-character roots ('ad', 'af', 'cv') are real marketed brands and
SHORT names carry MORE confusability risk, not less.

Also writes root_map.csv (brand_root -> clean_root, kept), completing the
provenance chain back to products, compositions and manufacturer.

Known limitations: hyphenated letter+digit suffixes are retained ('achol-d3'
is a MOLECULE, 'acmeglim-m2' is a STRENGTH — indistinguishable without domain
knowledge); brand-meaningful words (Duo, Kid, Plus, Free) are protected at the
cost of a small residual digit leak; ~20 single-character roots are excluded.
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
    "orange","tangy","for","hm","gelatin","softgel","new","cd","xt",
    # NOTE 'soft' is deliberately NOT here. It is a form word only in "soft
    # gelatin"/"soft gel"; on its own it can BE the brand - "I Soft" is a real
    # marketed eye drop (I Soft 0.1% w/v / I Soft Ophthalmic Solution), and an
    # unconditional strip reduced it to 'i'. Handled contextually below.
    # --- v5: added from a data-driven audit of C_DIGIT_LEAK. Each of these was
    # a tail word that HALTED the strip, stranding a strength behind it
    # (e.g. "actopar 250mg oral suspension mango" stopped at 'mango', so
    # '250mg' survived). Counts are how many roots each word was blocking.
    # delivery devices / presentations
    "combipack","combikit","rotacap","respicap","transcaps","transhaler",
    "instacap","cartridge","penfill","kwikpen","flexpen","pen","syringe",
    "rediuse","redimix","readymix","effervescent","chewable","lozenges",
    "jelly","gums","pessaries","depot","shots","nano",
    # routes
    "transdermal","topical","ophthalmic","sublingual","mouth","wash",
    # flavours (these strand strengths constantly)
    "mango","pineapple","strawberry","mint","peppermint","vanilla","fruit",
    "lemon","flavour","flavor","chocolate","butterscotch",
    # --- v6: further strip-blockers found by sampling the audit. Each stranded
    # a strength behind it: "kwitz 4mg chewing gums" halted at 'chewing';
    # "synclar 250mg dry syrup mixed fruit" halted at 'mixed'; "trulicity
    # 0.75mg pre-filled pen" halted at 'pre-filled'.
    "chewing","mixed","gargle","multidose","prefilled","pre-filled","filled",
    # --- v7: last vocabulary gaps from the final audit sample
    "softgels","pellets","scrub","pearls","eazy",
}
# Words that can carry BRAND meaning ("A Plus", "X Forte" may be the real name).
# These are still stripped normally, but the back-off guard MAY restore one if
# stripping left too little — because the modifier might BE part of the brand.
BRAND_MODIFIERS = {"forte", "fort", "plus", "total", "active"}
# Pure dosage-form abbreviations. Stripped, and NEVER restored — 'dt'/'sr' are
# presentation, never brand identity.
FORM_MODIFIERS = {"er","xr","cr","sr","dr","od","mr","pr","md","dt","dp","dry",
                  "ls","ds","xl","lb"}
MODIFIERS = BRAND_MODIFIERS | FORM_MODIFIERS
STRENGTH_RE = re.compile(
    r"^\d+(\.\d+)?\s*(mg|gm|g|mcg|ml|l|iu|au|miu|%|w/w|w/v|meq|lb|million)?/?"
    r"(\d+(\.\d+)?\s*(mg|gm|g|mcg|ml|l|iu|au|miu|%|lb)?)?$", re.IGNORECASE)
UNIT = r"(?:mg|gm|mcg|g|ml|l|iu|au|miu|%|lb|million|millionspores|spores)"
STRENGTH_CHAIN = re.compile(
    rf"\b\d[\d.]*\s*{UNIT}?(?:\s*/\s*\d[\d.]*\s*{UNIT}?)+\b", re.IGNORECASE)
# Concentrations written as <number><unit>/<unit> - '40mg/ml', '100iu/ml',
# '4mg/ml'. STRENGTH_CHAIN above needs a DIGIT after the slash, so it misses
# these entirely and they survive into the blocking root.
CONCENTRATION = re.compile(rf"\b\d[\d.]*\s*{UNIT}\s*/\s*{UNIT}\b", re.IGNORECASE)
HAS_DIGIT = re.compile(r"\d")

def clean_root(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(rf"(\d[\d.]*)\s*({UNIT})", r"\1\2", s)   # glue "1 gm" -> "1gm"
    s = STRENGTH_CHAIN.sub(" ", s)                       # delete "5mg/500mg/30mg"
    s = CONCENTRATION.sub(" ", s)                        # delete "40mg/ml"
    s = re.sub(r"[^a-z0-9\s\-/+.]", " ", s)
    toks = re.sub(r"\s+", " ", s).strip().split()

    # A trailing token containing a digit is USUALLY a strength ('650', '10mg',
    # '2m', '100iu'). But it is NOT always: 'maxx-2m' in "G Maxx-2M" is the
    # BRAND ('G Maxx' is a real marketed family - G Maxx 2mg vs G Maxx-2M,
    # glimepiride vs glimepiride+metformin). Stripping the whole token deletes
    # 4 letters of brand text and leaves 'g'. So a digit-bearing token is only
    # treated as a strength when its ALPHABETIC content is short (<=2 letters):
    # that covers 2m / 10mg / 1gm / 100iu while protecting 'maxx-2m', 'pex-5',
    # '29max' and other brand tokens that merely contain a number.
    MAX_STRENGTH_LETTERS = 2

    def looks_like_strength(t):
        if not HAS_DIGIT.search(t):
            return False
        return len(re.sub(r"[^a-z]", "", t)) <= MAX_STRENGTH_LETTERS

    def strippable(t):
        return (t in FORM_WORDS or t in MODIFIERS
                or STRENGTH_RE.match(t)
                or (looks_like_strength(t) and len(toks) > 1))

    original = list(toks)                # keep, so we can back off if we overstrip

    # First pass: pop the form/strength/digit tail from the end.
    # 'soft' is popped ONLY when what we just removed was gelatin/gel/capsule
    # (i.e. it was "... soft gelatin capsule"), never on its own.
    SOFT_FOLLOWERS = {"gelatin", "gel", "softgel", "capsule", "capsules",
                      "cap", "caps"}
    last_popped = None
    while toks:
        t = toks[-1]
        if t == "soft":
            if last_popped in SOFT_FOLLOWERS:
                last_popped = toks.pop()
                continue
            break                        # a standalone 'soft' is brand text
        if strippable(t):
            last_popped = toks.pop()
            continue
        break
    # Second pass: a SHORT (<=2 char) trailing alpha fragment like 'so','p','f'
    # is a truncated form word ONLY if a strength/form tail still sits behind
    # it. Peek: if popping it EXPOSES more strippable tail, it was debris.
    while len(toks) >= 2 and toks[-1].isalpha() and len(toks[-1]) <= 2 \
            and strippable(toks[-2]):
        toks.pop()                       # drop the fragment
        while toks and strippable(toks[-1]):   # then continue the normal strip
            toks.pop()

    # BACK-OFF GUARD: never strip a name out of existence.
    # A word on the MODIFIER list can also be part of a real brand — "A Plus
    # 100mg/500mg Tablet" is a genuine product whose brand IS "A Plus", but
    # popping 'tablet' then 'plus' leaves "a", which the >=3-letter filter then
    # discards, silently removing the product from the namespace. Whenever
    # stripping leaves too little, restore tokens (rightmost first) until the
    # root carries >=3 letters again, or we are back to the uncleaned name.
    def n_letters(ts):
        return sum(c.isalpha() for t in ts for c in t)

    def survives(ts):
        """Mirror of the survival filter: >=2 alphanumerics AND >=1 letter."""
        s2 = "".join(ts)
        return len(re.sub(r"[^a-z0-9]", "", s2)) >= 2 and bool(re.search(r"[a-z]", s2))

    # Back-off guard, keyed to the SAME test the survival filter uses (not a
    # separate letter count, which fired spuriously on 2-letter roots that
    # would have survived anyway and dragged form words back in).
    # Selective restore is only meaningful when SOMETHING survived stripping.
    # If every token was stripped, picking tokens out of the middle drops the
    # name's head ("md plus" -> "plus", "new np" -> "np"); that case is handled
    # by the whole-name fallback below instead.
    if toks and not survives(toks) and n_letters(original) >= 2:
        # Restore ONLY tokens that carry real brand content. A strength token
        # ('150', '10mg') or a form word ('tablet') must never be dragged back:
        # doing so re-introduces exactly the dose/form debris the cleaner exists
        # to remove, and it re-collides dose variants of the same brand.
        # If no content token is available, leave the short root as-is and let
        # the survival filter decide — a short brand is better than a wrong one.
        restored = list(toks)
        for i in range(len(toks), len(original)):
            tok = original[i]
            if (HAS_DIGIT.search(tok) or tok in FORM_WORDS
                    or tok in FORM_MODIFIERS or STRENGTH_RE.match(tok)):
                continue                      # never restore strength/form debris
            restored.append(tok)
            if survives(restored):
                break
        if survives(restored):                # only adopt if it actually helped
            toks = restored

    # LAST-RESORT FALLBACK. If the root still cannot survive, our word lists do
    # not apply to this name: every token looked like a form word, so the name
    # is most likely BUILT from form-like words ("MD Plus", "New NP", "I
    # Pearls", "LS AM" are brands, not presentations). Keeping only the first
    # token both truncates the brand AND merges distinct ones - "new np" and
    # "new dp" would both collapse to "new". So restore the WHOLE original,
    # dropping only pure strength/number tokens, which are never brand text.
    if not survives(toks) and original:
        # Prefer the name's HEAD plus any tokens that are not form/strength
        # debris. This matters for brands that ARE form words: "PR 1mg Tablet
        # DR" strips to nothing (pr and dr are both in the modifier list), and
        # restoring everything gives 'pr tablet dr' - form junk in the blocking
        # root, and it fragments one brand across several roots ('pr tablet dr'
        # vs 'pr d capsule sr'). Head-plus-content gives 'pr' for both.
        def is_debris(t):
            return (HAS_DIGIT.search(t) or STRENGTH_RE.match(t)
                    or t in FORM_WORDS or t in FORM_MODIFIERS)

        head_plus = original[:1] + [t for t in original[1:] if not is_debris(t)]
        if survives(head_plus):
            toks = head_plus
        else:
            # Nothing but debris after the head: keep the whole original rather
            # than lose the name (e.g. "i pearls" -> 'i' alone cannot survive).
            whole = [t for t in original
                     if not (HAS_DIGIT.search(t) or STRENGTH_RE.match(t))]
            if survives(whole):
                toks = whole
            elif survives(original):
                toks = list(original)

    return " ".join(toks) if toks else s

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results/brand_roots.csv"
    df = pd.read_csv(src)
    before = df["brand_root"].fillna("").astype(str)
    df["clean_root"] = before.map(clean_root)
    # Survival test: >=2 ALPHANUMERICS and at least one letter.
    #
    # Counting alphanumerics (not just letters) keeps short alphanumeric brands
    # like 'a1a', 'b12', 'a-3'. Requiring a letter still drops bare strength
    # residue ('500', '10/20').
    #
    # The threshold is 2, NOT 3, deliberately. An audit found 337 two-character
    # roots being excluded — 'ad', 'af', 'ak', 'cv', 'fm', 'dq' — and these are
    # real marketed brands ('AD 10mg Tablet DT', 'AF 150 Tablet DT'), not junk.
    # Excluding them would be indefensible in a safety screen for a specific
    # reason: SHORT names carry MORE confusability risk, not less (a one- or
    # two-character difference is a larger proportion of a short name). Dropping
    # the highest-risk names because they inflate the candidate count would be
    # optimising the wrong quantity. The extra blocking work is accepted.
    MIN_ALNUM = 2
    alnum = df["clean_root"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    has_letter = df["clean_root"].str.lower().str.contains(r"[a-z]", regex=True,
                                                           na=False)
    keep = (alnum.str.len() >= MIN_ALNUM) & has_letter
    dropped = (~keep).sum()
    changed = (df["clean_root"] != before).sum()
    nb = before.nunique()
    out_roots = df.loc[keep, "clean_root"].drop_duplicates()
    na = out_roots.nunique()
    print(f"rows                 : {len(df):,}")
    print(f"roots changed        : {changed:,} ({100*changed/len(df):.1f}%)")
    print(f"dropped (<{MIN_ALNUM} alnum)   : {dropped:,}")
    print(f"distinct roots before: {nb:,}")
    print(f"distinct roots after : {na:,}  ({nb-na:,} fewer)")
    out = Path(src).with_name("brand_roots_clean.csv")
    out_roots.rename("brand_root").to_frame().to_csv(out, index=False)
    print(f"wrote {na:,} clean roots -> {out}")

    # ---- PROVENANCE MAP ------------------------------------------------
    # brand_roots_clean.csv is DEDUPLICATED, so the link from a blocking
    # string back to the brand roots it came from is lost there. Blocking and
    # scoring work on clean_root strings, so without this map a candidate pair
    # cannot be traced to its products, compositions or manufacturer.
    # Join chain:  candidate clean_root -> root_map.csv -> brand_root
    #              -> product_molecules.csv -> product / molecules / manufacturer
    # 'kept' is False for roots dropped by the <3-letter filter: they never
    # reach the blocker, so this file also records what was excluded and why.
    mp = pd.DataFrame({
        "brand_root": before,
        "clean_root": df["clean_root"],
        "kept": keep,
    })
    map_out = Path(src).with_name("root_map.csv")
    mp.to_csv(map_out, index=False)

    # how many distinct brand roots collapsed onto one blocking string?
    collapsed = (mp[mp.kept]
                 .groupby("clean_root")["brand_root"].nunique())
    n_merged = int((collapsed > 1).sum())
    print(f"wrote {len(mp):,} root mappings -> {map_out}")
    print(f"  clean roots formed from 2+ distinct brand roots: {n_merged:,}")
    if n_merged:
        worst = collapsed.sort_values(ascending=False).head(5)
        print(f"  biggest merges (clean_root <- n brand roots):")
        for cr, cnt in worst.items():
            print(f"    {cnt:>4,}  {str(cr)[:52]}")
