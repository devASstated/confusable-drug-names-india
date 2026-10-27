#!/usr/bin/env python3
"""
transcribe_compare.py — add a second engine and measure engine agreement.

Stage 2 of the phonetic path. Reads phones_en_us.csv, transcribes the same
roots with eSpeak (via phonemizer), and writes both transcriptions side by
side with an agreement score.

The file it produces, phones_compare.csv, is the input to the Hindi-accent
phoneme model, whose output feeds the H/M/L/B blocker. The blocker reads
phones_g2p_arpa from here as its US-English path; that column is the
transcription from stage 1, carried through unchanged. The eSpeak columns are
diagnostic and nothing downstream reads them.

Why a second engine. There is no ground truth for how an invented name like
'sevflox' is pronounced, so transcription accuracy cannot be measured
directly. What can be measured is whether two independent engines agree. A
rule-based engine fails differently from a neural one, so a sharp divergence
means at least one is wrong. Agreement is weak evidence of correctness;
disagreement is strong evidence of trouble. On the full namespace, 6.0% of
roots fall in the severe-disagreement band, concentrated in abbreviations and
letter suffixes where g2p-en reads letter clusters as words ('fb' as F EY B)
and eSpeak spells them out.

Notation. g2p-en emits ARPABET and eSpeak emits IPA, so ARPABET is mapped to
IPA before comparison; otherwise purely notational differences would read as
disagreement. The mapping is deterministic but the two inventories still
differ over length marks, rhoticity and reduced vowels, so two engines that
agree will not score 1.0. Read the distribution and the severe tail rather
than any single score.

Columns: brand_root, phones_g2p_arpa, g2p_ipa, espeak_raw, espeak_ipa,
         n_g2p, n_espeak, agreement

Usage:
    pip install phonemizer        # and the espeak-ng binary
    python transcribe_compare.py results/phones_en_us.csv
    python transcribe_compare.py results/phones_en_us.csv --limit 5000
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

# ARPABET -> IPA, for comparison only. The pipeline uses phones_g2p_arpa.
ARPA_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɚ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}
# Collapse distinctions the engines write differently by convention rather
# than by sound, so they do not count as disagreement.
FOLD = str.maketrans({"ː": "", "ˈ": "", "ˌ": "", "ʰ": "", "ə": "ʌ",
                      "ɐ": "ʌ", "ɹ": "r", "ɚ": "r", "ɜ": "ʌ", "ɡ": "g"})


def arpa_to_ipa(phones):
    return "".join(ARPA_IPA.get(p, p.lower()) for p in phones)


def fold(s):
    return re.sub(r"[^a-zæɑɔʌʊɛɪŋʃʒθðtʃdʒ]", "", str(s).translate(FOLD).lower())


def edit_sim(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phones_csv", nargs="?", default="results/phones_en_us.csv")
    ap.add_argument("--out", default="results/phones_compare.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lang", default="en-us",
                    help="eSpeak voice. 'en-us' matches g2p-en's General "
                         "American, so a disagreement is an engine "
                         "disagreement rather than an accent one")
    ap.add_argument("--progress-secs", type=int, default=120)
    args = ap.parse_args()

    try:
        from phonemizer.backend import EspeakBackend
    except ImportError:
        sys.exit("needs phonemizer:  pip install phonemizer\n"
                 "and the espeak-ng binary:  sudo pacman -S espeak-ng")
    try:
        esp = EspeakBackend(args.lang, with_stress=False,
                            language_switch="remove-flags")
    except Exception as e:
        sys.exit(f"could not start eSpeak backend for {args.lang}: {e}\n"
                 f"is espeak-ng installed?  sudo pacman -S espeak-ng")

    src = Path(args.phones_csv)
    if not src.exists():
        sys.exit(f"no transcription at {src}")
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)
    roots = df["brand_root"].astype(str).tolist()
    arpa = [p.split() for p in df["phones"].astype(str)]

    print(f"second engine: eSpeak '{args.lang}' over {len(roots):,} roots",
          file=sys.stderr)
    t0 = t_last = time.time()
    esp_out = []
    B = 2000
    for i in range(0, len(roots), B):
        chunk = roots[i:i + B]
        try:
            esp_out.extend(esp.phonemize(chunk, strip=True))
        except Exception:
            esp_out.extend([""] * len(chunk))
        now = time.time()
        if now - t_last >= args.progress_secs or i + B >= len(roots):
            done = min(i + B, len(roots))
            rate = done / max(now - t0, 1e-9)
            print(f"  {done:,}/{len(roots):,}  {rate:.0f} roots/s  "
                  f"~{(len(roots)-done)/max(rate,1e-9)/60:.1f} min left",
                  file=sys.stderr)
            t_last = now

    rows = []
    for r, a, e in zip(roots, arpa, esp_out):
        ia, ie = fold(arpa_to_ipa(a)), fold(e)
        rows.append((r, " ".join(a), ia, str(e), ie,
                     len(a), len(ie), edit_sim(ia, ie)))
    out = pd.DataFrame(rows, columns=["brand_root", "phones_g2p_arpa",
                                      "g2p_ipa", "espeak_raw", "espeak_ipa",
                                      "n_g2p", "n_espeak", "agreement"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    ag = out.agreement
    print("\n" + "=" * 62)
    print("  ENGINE AGREEMENT — g2p-en (neural) vs eSpeak (rule-based)")
    print("=" * 62)
    print(f"roots compared        : {len(out):,}")
    print(f"mean agreement        : {ag.mean():.3f}")
    print(f"median                : {ag.median():.3f}")
    print(f"\n  agreement distribution:")
    for lo, hi, label in [(0.0, 0.4, "severe disagreement"),
                          (0.4, 0.6, "substantial"),
                          (0.6, 0.8, "moderate"),
                          (0.8, 1.01, "close")]:
        n = int(((ag >= lo) & (ag < hi)).sum())
        print(f"    {label:<22}{n:>8,}  ({100*n/len(out):.1f}%)  "
              f"[{lo:.1f}-{hi:.1f})")
    bad = out[ag < 0.4]
    print(f"\n  {len(bad):,} roots where the engines disagree severely — at")
    print(f"  least one is wrong, and these transcriptions should not be")
    print(f"  trusted for blocking or scoring:")
    for r in bad.sort_values("agreement").head(15).itertuples():
        print(f"    {r.brand_root[:18]:<20} g2p: {r.phones_g2p_arpa[:34]:<36}")
        print(f"    {'':<20} esp: {r.espeak_raw[:34]}")
    print(f"\n  length disagreement (a crude second signal):")
    d = (out.n_g2p - out.n_espeak).abs()
    print(f"    identical phone count : {int((d==0).sum()):,}"
          f"  ({100*(d==0).mean():.1f}%)")
    print(f"    differ by 3+          : {int((d>=3).sum()):,}"
          f"  ({100*(d>=3).mean():.1f}%)")
    print(f"\n  CAVEAT: some disagreement is notation, not error — the two")
    print(f"  engines write reduced vowels and rhoticity differently, so")
    print(f"  perfect agreement is not 1.0. Read the distribution and the")
    print(f"  severe tail, not individual scores.")
    print(f"\nwrote -> {args.out}")
    print(f"next: add the predicted_normalized_phonemes column, then run the")
    print(f"      H/M/L/B blocker on the result")


if __name__ == "__main__":
    main()
