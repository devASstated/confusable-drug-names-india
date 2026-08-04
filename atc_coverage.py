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

Usage:
    python atc_coverage.py results/molecules.csv
    python atc_coverage.py results/molecules.csv --outdir results
"""

import argparse
import json
import sys
import time
from pathlib import Path

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
    res.to_csv(outdir / "molecules_atc.csv", index=False)
    res[res.n_atc_codes == 0].to_csv(outdir / "molecules_no_atc.csv", index=False)

    # ---- REPORT ------------------------------------------------------
    n = len(res)
    n_rx = res.rxcui.notna().sum()
    n_atc = (res.n_atc_codes > 0).sum()

    # weight by how many products each molecule appears in — this is the
    # number that actually matters for your severity coverage
    tot_prod = res.product_count.sum()
    cov_prod = res.loc[res.n_atc_codes > 0, "product_count"].sum()

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

    print(f"\n[PRODUCT-WEIGHTED COVERAGE]  <- the number that matters")
    print(f"  products whose molecules all map: {cov_prod:,} of {tot_prod:,} "
          f"({100*cov_prod/max(tot_prod,1):.1f}%)")

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
    print("=" * 68)


if __name__ == "__main__":
    main()
