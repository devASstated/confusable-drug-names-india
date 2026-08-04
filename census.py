#!/usr/bin/env python3
"""
census.py — Exact-collision census + molecule inventory
for the A-Z Medicine Dataset of India.

Produces two things:

  1. THE CENSUS  — brand names marketed for two or more DIFFERENT
                   compositions (the "Medzol" pathology). No similarity
                   algorithm needed; this is an exact-match audit.

  2. THE MOLECULE INVENTORY — every distinct active ingredient in the
                   namespace, with frequency. This is the input list for
                   your ATC coverage check.

Usage:
    python census.py data/medicine_dataset.csv
    python census.py data/medicine_dataset.csv --outdir results
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# NORMALISATION TABLES
#
# These are deliberately conservative and deliberately visible. Every entry
# is a judgement call you should be able to defend, so keep them here rather
# than burying them in code.
# --------------------------------------------------------------------------

# Salt / hydrate forms stripped from molecule names.
# "Chlorpheniramine Maleate" -> "chlorpheniramine"
# Rationale: ATC indexes the base, not the salt. Toggle with --keep-salts.
SALT_TOKENS = {
    "hydrochloride", "hcl", "hydrobromide", "hydrate", "dihydrate",
    "monohydrate", "trihydrate", "hemihydrate", "anhydrous",
    "sodium", "potassium", "calcium", "magnesium", "zinc", "aluminium",
    "maleate", "sulphate", "sulfate", "tartrate", "bitartrate", "besylate",
    "mesylate", "citrate", "dihydrogen", "phosphate", "acetate", "succinate",
    "fumarate", "nitrate", "bromide", "chloride", "iodide", "oxalate",
    "lactate", "gluconate", "carbonate", "bicarbonate", "stearate",
    "palmitate", "valerate", "propionate", "dipropionate", "furoate",
    "xinafoate", "aceponate", "pivalate", "sodiumsalt", "base",
}

# British / Indian spellings -> WHO INN spelling.
# Extend this as you find more; it is your INN-reconciliation layer.
SPELLING_MAP = {
    "amoxycillin": "amoxicillin",
    "cephalexin": "cefalexin",
    "cephradine": "cefradine",
    "cefixime": "cefixime",
    "chlorpheniramine": "chlorphenamine",
    "oestradiol": "estradiol",
    "oestrogen": "estrogen",
    "indomethacin": "indometacin",
    "sulphamethoxazole": "sulfamethoxazole",
    "sulphasalazine": "sulfasalazine",
    "frusemide": "furosemide",
    "lignocaine": "lidocaine",
    "beclomethasone": "beclometasone",
    "levosalbutamol": "levosalbutamol",
    "salbutamol": "salbutamol",
    "acetaminophen": "paracetamol",
    "pcm": "paracetamol",
    "thyroxine": "levothyroxine",
    "dicyclomine": "dicycloverine",
}

# Dosage-form words stripped when deriving the brand root.
# "Augmentin 625 Duo Tablet" -> "augmentin duo"
FORM_WORDS = {
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps",
    "syrup", "suspension", "oral", "solution", "injection", "inj", "infusion",
    "cream", "ointment", "gel", "lotion", "paste", "powder", "granules",
    "drops", "drop", "eye", "ear", "nasal", "spray", "inhaler", "rotacaps",
    "respules", "sachet", "kit", "patch", "suppository", "pessary", "vial",
    "ampoule", "bottle", "tube", "strip", "pack", "liquid", "shampoo",
    "soap", "mouthwash", "gargle", "elixir", "emulsion", "foam", "film",
    "sf"
}

# Parses "Amoxycillin  (500mg)" into molecule + strength.
# Strength is optional: "Amoxycillin" alone parses fine.
COMPOSITION_RE = re.compile(
    r"^\s*(?P<mol>[^()]+?)\s*(?:\(\s*(?P<strength>[^)]*?)\s*\))?\s*$"
)

# A token that is purely a strength/number: "625", "500mg", "5ml", "0.1%"
STRENGTH_TOKEN_RE = re.compile(
    r"^\d+(\.\d+)?\s*(mg|g|mcg|ml|l|iu|%|w/w|w/v|meq)?/?"
    r"(\d+(\.\d+)?\s*(mg|g|mcg|ml|l|iu|%)?)?$",
    re.IGNORECASE,
)

PROTECTED = {
    "aluminium hydroxide", "magnesium hydroxide", "calcium carbonate",
    "sodium bicarbonate", "ammonium chloride", "potassium chloride",
    "sodium chloride", "disodium hydrogen phosphate", "ferrous sulphate",
    "ferrous sulfate", "zinc sulphate", "zinc sulfate", "calcium lactate",
    "magnesium sulphate", "magnesium sulfate", "potassium citrate",
    "sodium citrate", "calcium gluconate", "ferrous gluconate",
    "sodium acetate", "sodium lactate", "sodium phosphate",
    "calcium phosphate", "magnesium chloride", "calcium chloride",
}


# --------------------------------------------------------------------------
# NORMALISATION FUNCTIONS
# --------------------------------------------------------------------------

def normalise_molecule(raw, keep_salts=False):
    if not isinstance(raw, str):
        return ""
    s = re.sub(r"\s+", " ",
               re.sub(r"[^a-z0-9\s\-/+]", " ", raw.lower().strip())).strip()
    if not s:
        return ""
    if s in PROTECTED:                       # inorganics: the salt IS the drug
        return SPELLING_MAP.get(s, s)
    if not keep_salts:
        tokens = s.split()
        while len(tokens) > 1 and tokens[-1] in SALT_TOKENS:   # trailing only
            tokens.pop()
        s = " ".join(tokens)
    return SPELLING_MAP.get(s, s).strip()


def normalise_strength(raw) -> str:
    """'500 mg' -> '500mg';  '30mg/5ml' -> '30mg/5ml';  NaN -> ''."""
    if not isinstance(raw, str):
        return ""
    return re.sub(r"\s+", "", raw.lower().strip())


def parse_composition_cell(cell, keep_salts: bool = False):
    """
    'Amoxycillin  (500mg)' -> ('amoxicillin', '500mg')
    Returns (None, None) if the cell is empty or unparseable.
    """
    if not isinstance(cell, str) or not cell.strip():
        return None, None

    m = COMPOSITION_RE.match(cell)
    if not m:
        return None, None

    mol = normalise_molecule(m.group("mol"), keep_salts)
    if not mol:
        return None, None

    return mol, normalise_strength(m.group("strength"))


# def brand_root(name: str) -> str:
#     """
#     Strip dosage form and strength tokens to get the brand root.
#
#       'Augmentin 625 Duo Tablet'  -> 'augmentin duo'
#       'Azithral 500 Tablet'       -> 'azithral'
#       'Ascoril LS Syrup'          -> 'ascoril ls'
#
#     Deliberately keeps marketing modifiers (Duo, LS, DS, CV, SR, OD)
#     because in India those distinguish genuinely different products.
#     """
#     if not isinstance(name, str):
#         return ""
#
#     s = name.lower().strip()
#     s = re.sub(r"\(.*?\)", " ", s)            # drop parenthetical content
#     s = re.sub(r"[^a-z0-9\s\-/+.]", " ", s)
#     s = re.sub(r"\s+", " ", s).strip()
#
#     tokens = [
#         t for t in s.split()
#         if t not in FORM_WORDS and not STRENGTH_TOKEN_RE.match(t)
#     ]
#     return " ".join(tokens) if tokens else s
def brand_root(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = s.replace("sugar free", " ").replace("sugarfree", " ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9\s\-/+.]", " ", s)
    tokens = re.sub(r"\s+", " ", s).strip().split()

    # strip form/strength tokens from the END only —
    # a leading "Oral" belongs to the brand, a trailing one doesn't
    while tokens and (tokens[-1] in FORM_WORDS
                      or STRENGTH_TOKEN_RE.match(tokens[-1])):
        tokens.pop()
    return " ".join(tokens) if tokens else s

# --------------------------------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path to the dataset CSV")
    ap.add_argument("--outdir", default="results", help="output directory")
    ap.add_argument("--keep-salts", action="store_true",
                    help="do NOT strip salt/hydrate forms (sensitivity check)")
    ap.add_argument("--allopathy-only", action="store_true", default=True,
                    help="restrict to type == allopathy (default: on)")
    ap.add_argument("--all-types", dest="allopathy_only", action="store_false",
                    help="include every product type")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  EXACT-COLLISION CENSUS — Indian Pharmaceutical Namespace")
    print("=" * 68)

    # ---- 1. LOAD -----------------------------------------------------
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"\n[1] LOADED")
    print(f"    rows            : {len(df):,}")
    print(f"    columns         : {list(df.columns)}")

    required = {"name", "short_composition1"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"\nERROR: missing required column(s): {missing}")

    if "short_composition2" not in df.columns:
        df["short_composition2"] = ""

    # ---- 2. FILTER ---------------------------------------------------
    if args.allopathy_only and "type" in df.columns:
        before = len(df)
        df = df[df["type"].astype(str).str.strip().str.lower() == "allopathy"]
        print(f"\n[2] FILTERED to allopathy")
        print(f"    kept            : {len(df):,} of {before:,} "
              f"({100*len(df)/max(before,1):.1f}%)")
    else:
        print("\n[2] FILTER skipped (no 'type' column or --all-types)")

    if "Is_discontinued" in df.columns:
        disc = df["Is_discontinued"].astype(str).str.upper().eq("TRUE").sum()
        print(f"    discontinued    : {disc:,} (retained — an old prescription "
              f"still reaches the counter)")

    # ---- 3. PARSE ----------------------------------------------------
    print(f"\n[3] PARSING compositions"
          f"{' (salts retained)' if args.keep_salts else ''} ...")

    parsed = []
    for c1, c2 in zip(df["short_composition1"], df["short_composition2"]):
        m1, s1 = parse_composition_cell(c1, args.keep_salts)
        m2, s2 = parse_composition_cell(c2, args.keep_salts)
        mols, full = [], []
        if m1:
            mols.append(m1); full.append((m1, s1))
        if m2:
            mols.append(m2); full.append((m2, s2))
        parsed.append((mols, full))

    df = df.copy()
    df["molecules"] = [p[0] for p in parsed]
    df["mol_sig"] = [" + ".join(sorted(p[0])) for p in parsed]
    df["full_sig"] = [
        " + ".join(f"{m}@{s}" for m, s in sorted(p[1])) for p in parsed
    ]
    df["n_molecules"] = df["molecules"].str.len()
    df["brand_root"] = df["name"].map(brand_root)

    ok = df["n_molecules"] > 0
    print(f"    parsed OK       : {ok.sum():,} ({100*ok.mean():.2f}%)")
    print(f"    parse failures  : {(~ok).sum():,}")
    print(f"    single-molecule : {(df['n_molecules'] == 1).sum():,}")
    print(f"    two-molecule FDC: {(df['n_molecules'] == 2).sum():,}")

    if (~ok).any():
        df.loc[~ok, ["name", "short_composition1", "short_composition2"]] \
          .to_csv(outdir / "parse_failures.csv", index=False)

    work = df[ok].copy()

    # ---- 4. MOLECULE INVENTORY  (input for ATC coverage) -------------
    inv = (
        work.explode("molecules")
            .groupby("molecules")
            .agg(product_count=("name", "size"),
                 example_brand=("name", "first"))
            .sort_values("product_count", ascending=False)
            .reset_index()
            .rename(columns={"molecules": "molecule"})
    )
    inv.to_csv(outdir / "molecules.csv", index=False)

    print(f"\n[4] MOLECULE INVENTORY")
    print(f"    distinct molecules : {len(inv):,}   -> molecules.csv")
    print(f"    (this is your ATC-lookup input list)")
    print(f"\n    top 10 by product count:")
    for _, r in inv.head(10).iterrows():
        print(f"      {r.product_count:>7,}  {r.molecule}")

    # ---- 5. THE CENSUS -----------------------------------------------
    print(f"\n[5] EXACT-COLLISION CENSUS")

    g = work.groupby("brand_root")
    summary = g.agg(
        n_products=("name", "size"),
        n_distinct_molecule_sets=("mol_sig", "nunique"),
        n_distinct_full_sigs=("full_sig", "nunique"),
    ).reset_index()

    print(f"    distinct brand roots : {len(summary):,}")

    # 5a. DIFFERENT MOLECULES under the same brand root — the dangerous case
    mol_collide = summary[summary["n_distinct_molecule_sets"] >= 2] \
        .sort_values("n_distinct_molecule_sets", ascending=False)

    rows = []
    for _, r in mol_collide.iterrows():
        sub = work[work["brand_root"] == r.brand_root]
        for sig, sg in sub.groupby("mol_sig"):
            rows.append({
                "brand_root": r.brand_root,
                "n_distinct_molecule_sets": r.n_distinct_molecule_sets,
                "composition": sig,
                "n_products": len(sg),
                "example_name": sg["name"].iloc[0],
                "example_manufacturer": sg["manufacturer_name"].iloc[0]
                    if "manufacturer_name" in sg.columns else "",
            })
    mol_detail = pd.DataFrame(rows)
    mol_detail.to_csv(outdir / "collisions_molecule.csv", index=False)

    # 5b. Same molecules, DIFFERENT STRENGTH — the Augmentin 625 / DDS case
    strength_only = summary[
        (summary["n_distinct_molecule_sets"] == 1)
        & (summary["n_distinct_full_sigs"] >= 2)
    ]
    strength_only.to_csv(outdir / "collisions_strength.csv", index=False)

    n_roots = len(summary)
    n_mol = len(mol_collide)
    n_str = len(strength_only)

    print(f"\n    >>> HEADLINE NUMBER <<<")
    print(f"    brand roots sold as 2+ DIFFERENT compositions : {n_mol:,}"
          f"  ({100*n_mol/max(n_roots,1):.2f}% of brand roots)")
    print(f"    products involved                             : "
          f"{mol_detail['n_products'].sum() if len(mol_detail) else 0:,}")
    print(f"    brand roots with same molecules, diff strength : {n_str:,}")

    if len(mol_collide):
        print(f"\n    worst 15 by number of distinct compositions:")
        for _, r in mol_collide.head(15).iterrows():
            sigs = work[work["brand_root"] == r.brand_root]["mol_sig"].unique()
            print(f"\n      {r.brand_root.upper()}  "
                  f"({r.n_distinct_molecule_sets} compositions, "
                  f"{r.n_products} products)")
            for s in list(sigs)[:5]:
                print(f"          - {s}")
            if len(sigs) > 5:
                print(f"          ... and {len(sigs)-5} more")

    # ---- 6. NAMESPACE SHAPE ------------------------------------------
    print(f"\n[6] NAMESPACE SHAPE")
    n_comp = work["mol_sig"].nunique()
    print(f"    distinct compositions        : {n_comp:,}")
    print(f"    distinct brand roots         : {n_roots:,}")
    print(f"    brands per composition (mean): {n_roots/max(n_comp,1):.1f}")

    top = (work.groupby("mol_sig")["brand_root"].nunique()
               .sort_values(ascending=False).head(10))
    print(f"\n    compositions with the most brand names:")
    for sig, n in top.items():
        print(f"      {n:>5,} brands   {sig}")

    print(f"\n{'=' * 68}")
    print(f"  Files written to {outdir.resolve()}/")
    print(f"    molecules.csv            -> feed to ATC lookup")
    print(f"    collisions_molecule.csv  -> THE CENSUS")
    print(f"    collisions_strength.csv")
    print(f"    parse_failures.csv       -> review, then extend the tables")
    print("=" * 68)


if __name__ == "__main__":
    main()
