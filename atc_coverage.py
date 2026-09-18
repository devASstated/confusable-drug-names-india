#!/usr/bin/env python3
"""
atc_coverage.py — map your molecule inventory to WHO ATC codes.

Answers the question that decides your severity method:
    what fraction of the 1,667 molecules in the Indian namespace
    can be mapped to an ATC code by lookup?

Uses the US National Library of Medicine's RxNav / RxClass API:
    free, no key, no registration.
    https://lhncbc.nlm.nih.gov/RxNav/APIs/

Two hops per molecule:
    1. name  -> RxCUI        (RxNorm normalised concept id)
    2. RxCUI -> ATC class    (RxClass, relaSource=ATC)

Results are cached to disk, so re-running is cheap and you can
interrupt with Ctrl-C without losing work.

TWO MEASUREMENT CAVEATS (both matter when quoting these numbers)

  1. COVERAGE IS REPORTED TWO WAYS, and they answer different questions:
       - ingredient-occurrence coverage (always computed): of all
         ingredient-slots across the namespace, weighted by how many
         products each molecule appears in, what share map to ATC?
       - TRUE product-level coverage (only with --products): of all
         PRODUCTS, what share have EVERY ingredient mapped?
     The second is strictly lower and is the honest figure for severity
     work, because a product is only fully classifiable if ALL of its
     ingredients map. The first was previously mislabelled as the second.

  2. NOT EVERY MAPPING IS EQUALLY TRUSTWORTHY. A molecule reaches an ATC code
     either by an EXACT RxNorm name match or by an APPROXIMATE one, and the
     approximate route is unthresholded — RxNav's similarity score has no
     documented scale, so the top candidate is taken whatever it scores. An
     approximate match that happens to carry an ATC code currently counts as
     covered exactly like an exact one.
     That is tolerable for an exploratory coverage figure. It is NOT tolerable
     for divergence (D), where a WRONG therapeutic class is worse than a
     missing one: a missing class is a known gap, a wrong class is confident
     nonsense that propagates into a priority score. Coverage is therefore
     reported twice — exact-only (the trustworthy floor) and including
     approximate (the optimistic ceiling) — and every approximate mapping is
     written to molecules_atc_approx.csv for hand audit before D uses any of
     it.

  3. THE INGREDIENT INVENTORY ITSELF IS INCOMPLETE. The source A-Z dataset
     carries at most TWO composition fields, and measurement shows products
     whose own names declare 3+ strengths while the composition records
     fewer — i.e. ingredients are silently truncated. Any coverage figure
     here is therefore computed over an ingredient set that is known to be
     missing ingredients for an unquantified share of products. Treat these
     percentages as upper bounds on true classifiability.

Usage:
    python atc_coverage.py results/molecules.csv
    python atc_coverage.py results/molecules.csv --outdir results
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RXNAV = "https://rxnav.nlm.nih.gov/REST"
HEADERS = {"User-Agent": "LASA-namespace-study/1.0 (academic research)"}

# NLM asks for <= 20 requests/second. We go far slower to be polite;
# 1,667 molecules x 2 calls is a few minutes either way.
SLEEP = 0.06


# --------------------------------------------------------------------------
# API CALLS
# --------------------------------------------------------------------------

def http_get(url, params=None, retries=3):
    """GET with simple backoff. Returns parsed JSON or None."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def find_rxcui(name: str):
    """
    Molecule name -> RxCUI.

    Exact match first, then approximate. We do NOT threshold on the
    approximate score — its scale is not documented reliably — so we
    record it instead and let the ATC lookup act as the real filter.
    Audit the approx matches by hand afterwards.
    """
    j = http_get(f"{RXNAV}/rxcui.json", {"name": name, "search": 1})
    ids = (j or {}).get("idGroup", {}).get("rxnormId") or []
    if ids:
        return ids[0], "exact"

    j = http_get(f"{RXNAV}/approximateTerm.json",
                 {"term": name, "maxEntries": 1})
    cands = (j or {}).get("approximateGroup", {}).get("candidate") or []
    if cands and cands[0].get("rxcui"):
        try:
            score = float(cands[0].get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        return cands[0]["rxcui"], f"approx:{score:.1f}"

    return None, "none"


def find_atc(rxcui: str):
    """RxCUI -> list of (atc_code, atc_name). Usually ATC level 4."""
    j = http_get(f"{RXNAV}/rxclass/class/byRxcui.json",
                 {"rxcui": rxcui, "relaSource": "ATC"})
    items = (j or {}).get("rxclassDrugInfoList", {}).get("rxclassDrugInfo") or []
    out = []
    for it in items:
        c = it.get("rxclassMinConceptItem", {})
        if c.get("classId"):
            out.append((c["classId"], c.get("className", "")))
    # dedupe, preserve order
    seen, uniq = set(), []
    for code, nm in out:
        if code not in seen:
            seen.add(code)
            uniq.append((code, nm))
    return uniq


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("molecules_csv", help="results/molecules.csv from census.py")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cache", default="results/atc_cache.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the top N molecules (for a quick test)")
    ap.add_argument("--products", default="",
                    help="optional CSV with ONE ROW PER PRODUCT and a column "
                         "listing that product's molecules (see --mol-col). "
                         "Enables the TRUE product-level coverage figure "
                         "(share of products whose ingredients ALL map).")
    ap.add_argument("--mol-col", default="molecules",
                    help="column in --products holding the molecule list, "
                         "separated by '+' or '|' (default: molecules)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache)

    mols = pd.read_csv(args.molecules_csv)
    if "molecule" not in mols.columns:
        sys.exit("ERROR: expected a 'molecule' column")
    if args.limit:
        mols = mols.head(args.limit)

    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"resuming from cache: {len(cache):,} molecules already done")

    print("=" * 68)
    print("  ATC COVERAGE CHECK — Indian namespace molecule inventory")
    print("=" * 68)
    print(f"\nmolecules to map : {len(mols):,}")
    print(f"source           : RxNav / RxClass (US NLM)")
    print(f"cache            : {cache_path}\n")

    try:
        for i, row in enumerate(mols.itertuples(), 1):
            name = row.molecule
            if name in cache:
                continue

            rxcui, how = find_rxcui(name)
            time.sleep(SLEEP)

            atc = []
            if rxcui:
                atc = find_atc(rxcui)
                time.sleep(SLEEP)

            cache[name] = {
                "rxcui": rxcui,
                "match_method": how,
                "atc_codes": [c for c, _ in atc],
                "atc_names": [n for _, n in atc],
            }

            if i % 25 == 0:
                done = sum(1 for m in mols.molecule if m in cache)
                hit = sum(1 for m in mols.molecule
                          if cache.get(m, {}).get("atc_codes"))
                print(f"  {done:>5,}/{len(mols):,}   ATC found: {hit:>5,} "
                      f"({100*hit/max(done,1):.1f}%)   last: {name[:38]}")
                cache_path.write_text(json.dumps(cache, indent=1))

    except KeyboardInterrupt:
        print("\ninterrupted — cache saved, rerun to resume")

    cache_path.write_text(json.dumps(cache, indent=1))

    # ---- ASSEMBLE ----------------------------------------------------
    recs = []
    for row in mols.itertuples():
        c = cache.get(row.molecule, {})
        codes = c.get("atc_codes", [])
        recs.append({
            "molecule": row.molecule,
            "product_count": getattr(row, "product_count", 0),
            "rxcui": c.get("rxcui"),
            "match_method": c.get("match_method", "not_attempted"),
            "n_atc_codes": len(codes),
            "atc_codes": ";".join(codes),
            "atc_names": ";".join(c.get("atc_names", [])),
            "atc_level1": ";".join(sorted({x[0] for x in codes})),
        })

    res = pd.DataFrame(recs)

    # Trust tiers. 'approximate' is not a synonym for 'mapped': the match was
    # made by fuzzy name lookup with no threshold, so it may be a different
    # drug entirely. Keep the tiers separate from here on.
    is_exact = res.match_method == "exact"
    is_approx = res.match_method.str.startswith("approx")
    has_atc = res.n_atc_codes > 0
    res["trust"] = np.where(~has_atc, "unmapped",
                     np.where(is_exact, "exact", "approx_unvalidated"))

    res.to_csv(outdir / "molecules_atc.csv", index=False)
    res[~has_atc].to_csv(outdir / "molecules_no_atc.csv", index=False)
    approx_rows = res[has_atc & is_approx].sort_values("product_count",
                                                       ascending=False)
    approx_rows.to_csv(outdir / "molecules_atc_approx.csv", index=False)

    # ---- REPORT ------------------------------------------------------
    n = len(res)
    n_rx = res.rxcui.notna().sum()
    n_atc = (res.n_atc_codes > 0).sum()

    # ---- ingredient-occurrence coverage (weighted by product frequency) --
    # NOTE this is NOT "products whose molecules all map". Each product with
    # k ingredients contributes k ingredient-occurrences, and a product can be
    # counted in the numerator through a mapped ingredient even when another
    # of its ingredients is unmapped. It is a frequency-weighted INGREDIENT
    # coverage statistic. The true product-level figure needs --products.
    tot_occ = res.product_count.sum()
    cov_occ = res.loc[res.n_atc_codes > 0, "product_count"].sum()

    print(f"\n{'=' * 68}")
    print("  RESULTS")
    print("=" * 68)
    print(f"\n[MOLECULE-LEVEL COVERAGE]")
    print(f"  molecules attempted        : {n:,}")
    print(f"  mapped to an RxCUI         : {n_rx:,}  ({100*n_rx/n:.1f}%)")
    print(f"  mapped to an ATC code      : {n_atc:,}  ({100*n_atc/n:.1f}%)")
    print(f"    of which exact name match: "
          f"{(res.match_method == 'exact').sum():,}")
    print(f"    of which approximate     : "
          f"{res.match_method.str.startswith('approx').sum():,}")

    n_ex = int((res.trust == "exact").sum())
    n_ap = int((res.trust == "approx_unvalidated").sum())
    print(f"\n[MAPPING TRUST]")
    print(f"  exact RxNorm name match    : {n_ex:,}")
    print(f"  approximate, UNVALIDATED   : {n_ap:,}  "
          f"({100*n_ap/max(n_ex+n_ap,1):.1f}% of all mappings)")
    print(f"  -> molecules_atc_approx.csv, audit before D uses these")
    if n_ap:
        print(f"  largest by product count:")
        for r in approx_rows.head(6).itertuples():
            print(f"    {r.product_count:>6,}  {str(r.molecule)[:30]:<32}"
                  f"{r.match_method}")

    cov_occ_ex = res.loc[has_atc & is_exact, "product_count"].sum()
    print(f"\n[INGREDIENT-OCCURRENCE COVERAGE]  (frequency-weighted)")
    print(f"  exact only        : {cov_occ_ex:,} of {tot_occ:,} "
          f"({100*cov_occ_ex/max(tot_occ,1):.1f}%)   <- trustworthy floor")
    print(f"  incl. approximate : {cov_occ:,} of {tot_occ:,} "
          f"({100*cov_occ/max(tot_occ,1):.1f}%)   <- optimistic ceiling")
    print(f"  Reading: of every ingredient-slot in the namespace (a 2-ingredient")
    print(f"  product contributes 2), this share maps to ATC. It is NOT the share")
    print(f"  of products that are fully classifiable — see below.")

    # ---- TRUE product-level coverage (needs a product -> molecules file) ----
    if args.products:
        mapped = set(res.loc[has_atc, "molecule"].astype(str).str.lower())
        mapped_exact = set(res.loc[has_atc & is_exact, "molecule"]
                           .astype(str).str.lower())
        known = set(res["molecule"].astype(str).str.lower())
        prod = pd.read_csv(args.products, dtype=str, keep_default_na=False)
        if args.mol_col not in prod.columns:
            print(f"\n  [!] --products given but column '{args.mol_col}' not found; "
                  f"columns are {list(prod.columns)}")
        else:
            import re as _re
            n_all, n_part, n_none, n_unknown = 0, 0, 0, 0
            for cell in prod[args.mol_col]:
                mols = [m.strip().lower() for m in _re.split(r"\s*[|+]\s*", str(cell))
                        if m.strip()]
                if not mols:
                    continue
                if any(m not in known for m in mols):
                    n_unknown += 1            # molecule never attempted in this run
                    continue
                hits = sum(1 for m in mols if m in mapped)
                if hits == len(mols):
                    n_all += 1
                elif hits:
                    n_part += 1
                else:
                    n_none += 1
            tot = n_all + n_part + n_none
            print(f"\n[TRUE PRODUCT-LEVEL COVERAGE]  <- the honest severity number")
            if tot:
                print(f"  products where ALL ingredients map : {n_all:,} of {tot:,} "
                      f"({100*n_all/tot:.1f}%)")
                print(f"  partially mapped (some, not all)   : {n_part:,}  "
                      f"({100*n_part/tot:.1f}%)  <- NOT fully classifiable")
                print(f"  no ingredient mapped               : {n_none:,}  "
                      f"({100*n_none/tot:.1f}%)")
            if n_unknown:
                print(f"  skipped (a molecule not in this run): {n_unknown:,}")
            # same walk, but only exact mappings count as mapped
            e_all = 0
            for cell in prod[args.mol_col]:
                ms = [m.strip().lower() for m in _re.split(r"\s*[|+]\s*", str(cell))
                      if m.strip()]
                if ms and all(m in known for m in ms) and \
                        all(m in mapped_exact for m in ms):
                    e_all += 1
            if tot:
                print(f"\n  EXACT MAPPINGS ONLY (the number D can rely on today):")
                print(f"    products where ALL ingredients map : {e_all:,} of {tot:,} "
                      f"({100*e_all/tot:.1f}%)")
                print(f"    the gap to the figure above is carried by unvalidated")
                print(f"    approximate matches — audit them and it closes, or it")
                print(f"    does not and that gap was never real coverage.")
            print(f"\n  This is strictly lower than the occurrence figure above, and")
            print(f"  it is the number to quote for divergence/severity coverage.")
    else:
        print(f"\n[TRUE PRODUCT-LEVEL COVERAGE]  not computed")
        print(f"  Pass --products <csv> (one row per product, a column listing its")
        print(f"  molecules) to get the share of products whose ingredients ALL map.")
        print(f"  Without it, only the occurrence figure above is available, and it")
        print(f"  OVERSTATES how many products are fully classifiable.")

    print(f"\n  [caveat] the source dataset truncates composition to two fields;")
    print(f"  products with 3+ ingredients lose some. Every figure above is an")
    print(f"  UPPER BOUND on true classifiability.")

    if n_atc:
        lvl1 = (res[res.n_atc_codes > 0]
                .atc_level1.str.split(";").explode().value_counts())
        print(f"\n[ATC LEVEL-1 DISTRIBUTION]")
        names = {
            "A": "alimentary/metabolism", "B": "blood", "C": "cardiovascular",
            "D": "dermatological", "G": "genitourinary", "H": "hormones",
            "J": "anti-infectives", "L": "antineoplastic/immuno",
            "M": "musculoskeletal", "N": "nervous system",
            "P": "antiparasitic", "R": "respiratory", "S": "sensory organs",
            "V": "various",
        }
        for code, cnt in lvl1.items():
            print(f"    {code}  {cnt:>4,}   {names.get(code, '')}")

    miss = res[res.n_atc_codes == 0].sort_values("product_count",
                                                 ascending=False)
    if len(miss):
        print(f"\n[UNMAPPED — top 20 by product count]")
        print(f"  These fall back to ATC-tree distance or manual assignment.")
        for r in miss.head(20).itertuples():
            print(f"    {r.product_count:>6,}  {r.molecule}")

    print(f"\n{'=' * 68}")
    print(f"  molecules_atc.csv     -> full mapping")
    print(f"  molecules_no_atc.csv  -> the gap, review by hand")
    print(f"  molecules_atc_approx.csv -> UNVALIDATED fuzzy matches; audit")
    print(f"     these before any of it feeds divergence")
    print("=" * 68)


if __name__ == "__main__":
    main()
