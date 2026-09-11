import sys
import os
import re
import json
import time
from collections import Counter
from itertools import product

import pandas as pd

from rapidfuzz.distance import JaroWinkler, Levenshtein

from phonetics import soundex, metaphone, nysiis


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "results/lasa_refset.csv"
)

OUTPUT_DIR = "results"

CANDIDATE_FILE = os.path.join(
    OUTPUT_DIR,
    "lasa_blocking_candidates.csv"
)

REMOVED_POSITIVE_FILE = os.path.join(
    OUTPUT_DIR,
    "removed_positive_pairs.csv"
)

OPTIMIZATION_FILE = os.path.join(
    OUTPUT_DIR,
    "optimization_results.csv"
)

BEST_PARAMS_FILE = os.path.join(
    OUTPUT_DIR,
    "best_blocking_parameters.json"
)


# ============================================================
# SEARCH SETTINGS
# ============================================================

# The optimizer has two phases:
#
#   1. COARSE SEARCH
#   2. FINE SEARCH around the best coarse configuration
#
# This avoids millions of combinations.

COARSE_MAX_CONFIGS = 3000

FINE_MAX_CONFIGS = 3000

PROGRESS_EVERY = 100

# Number of candidate rule families to test.
#
# More rules = potentially higher recall but more negatives.
# The optimizer chooses the best combination.

MAX_RULES = 6


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = re.sub(
        r"\[[^\]]*\]",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def compact_text(text):

    return normalize_text(text).replace(" ", "")


def get_tokens(text):

    text = normalize_text(text)

    if not text:
        return []

    return text.split()


# ============================================================
# BASIC FEATURES
# ============================================================

def consonant_code(text):

    text = compact_text(text)

    return re.sub(
        r"[aeiou]",
        "",
        text
    )


def character_shape(text):

    text = compact_text(text)

    result = []

    for ch in text:

        if ch.isdigit():
            result.append("D")

        elif ch in "aeiou":
            result.append("V")

        else:
            result.append("C")

    return "".join(result)


def match_rating_code(text):

    text = compact_text(text)

    if not text:
        return ""

    consonants = re.sub(
        r"[aeiou]",
        "",
        text
    )

    if len(consonants) > 6:

        consonants = (
            consonants[:3]
            +
            consonants[-3:]
        )

    return consonants.upper()


def get_bigrams(text):

    if not text:
        return set()

    if len(text) == 1:
        return {text}

    return {
        text[i:i + 2]
        for i in range(len(text) - 1)
    }


def bigram_similarity(a, b):

    a_bg = get_bigrams(a)
    b_bg = get_bigrams(b)

    if not a_bg and not b_bg:
        return 1.0

    if not a_bg or not b_bg:
        return 0.0

    intersection = len(
        a_bg & b_bg
    )

    union = len(
        a_bg | b_bg
    )

    if union == 0:
        return 0.0

    return intersection / union


def edit_similarity(a, b):

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    distance = Levenshtein.distance(
        a,
        b
    )

    max_len = max(
        len(a),
        len(b)
    )

    if max_len == 0:
        return 1.0

    return 1.0 - (
        distance / max_len
    )


def length_ratio(a, b):

    if not a or not b:
        return 0.0

    return (
        min(len(a), len(b))
        /
        max(len(a), len(b))
    )


def prefix_ratio(a, b):

    n = min(
        len(a),
        len(b)
    )

    i = 0

    while (
        i < n
        and
        a[i] == b[i]
    ):
        i += 1

    denominator = max(
        len(a),
        len(b)
    )

    if denominator == 0:
        return 0.0

    return i / denominator


def suffix_ratio(a, b):

    n = min(
        len(a),
        len(b)
    )

    i = 0

    while (
        i < n
        and
        a[-1 - i] == b[-1 - i]
    ):
        i += 1

    denominator = max(
        len(a),
        len(b)
    )

    if denominator == 0:
        return 0.0

    return i / denominator


def code_similarity(a, b):

    if not a or not b:
        return 0.0

    return JaroWinkler.normalized_similarity(
        a,
        b
    )


def phonetic_prefix_ratio(a, b):

    if not a or not b:
        return 0.0

    n = min(
        len(a),
        len(b)
    )

    i = 0

    while (
        i < n
        and
        a[i] == b[i]
    ):
        i += 1

    return i / max(
        len(a),
        len(b)
    )


def phonetic_suffix_ratio(a, b):

    if not a or not b:
        return 0.0

    n = min(
        len(a),
        len(b)
    )

    i = 0

    while (
        i < n
        and
        a[-1 - i] == b[-1 - i]
    ):
        i += 1

    return i / max(
        len(a),
        len(b)
    )


# ============================================================
# REPRESENTATIONS
# ============================================================

def build_representation(text):

    normalized = normalize_text(text)

    compact = normalized.replace(
        " ",
        ""
    )

    tokens = get_tokens(
        normalized
    )

    try:
        sx = soundex(compact)
    except Exception:
        sx = ""

    try:
        mp = metaphone(compact)
    except Exception:
        mp = ""

    try:
        ny = nysiis(compact)
    except Exception:
        ny = ""

    return {

        "original": text,

        "normalized":
            normalized,

        "compact":
            compact,

        "tokens":
            tokens,

        "soundex":
            sx,

        "metaphone":
            mp,

        "nysiis":
            ny,

        "match_rating":
            match_rating_code(compact),

        "consonants":
            consonant_code(compact),

        "shape":
            character_shape(compact)
    }


# ============================================================
# TOKEN PHONETIC
# ============================================================

def token_phonetic_similarity(
    tokens_a,
    tokens_b
):

    if not tokens_a or not tokens_b:
        return 0.0

    encoded_a = []

    encoded_b = []

    for token in tokens_a:

        try:
            encoded_a.append(
                metaphone(token) or ""
            )
        except Exception:
            encoded_a.append("")

    for token in tokens_b:

        try:
            encoded_b.append(
                metaphone(token) or ""
            )
        except Exception:
            encoded_b.append("")

    scores = []

    for a in encoded_a:

        if not a:
            continue

        best = 0.0

        for b in encoded_b:

            if not b:
                continue

            score = JaroWinkler.normalized_similarity(
                a,
                b
            )

            if score > best:
                best = score

        scores.append(best)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# ============================================================
# FEATURE COMPUTATION
# ============================================================

def compute_features(
    rep_a,
    rep_b
):

    a = rep_a["compact"]
    b = rep_b["compact"]

    jw = JaroWinkler.normalized_similarity(
        a,
        b
    )

    bg = bigram_similarity(
        a,
        b
    )

    ed = edit_similarity(
        a,
        b
    )

    lr = length_ratio(
        a,
        b
    )

    pr = prefix_ratio(
        a,
        b
    )

    sr = suffix_ratio(
        a,
        b
    )

    sx_a = rep_a["soundex"]
    sx_b = rep_b["soundex"]

    mp_a = rep_a["metaphone"]
    mp_b = rep_b["metaphone"]

    ny_a = rep_a["nysiis"]
    ny_b = rep_b["nysiis"]

    mr_a = rep_a["match_rating"]
    mr_b = rep_b["match_rating"]

    con_a = rep_a["consonants"]
    con_b = rep_b["consonants"]

    shape_a = rep_a["shape"]
    shape_b = rep_b["shape"]

    soundex_exact = (
        1
        if sx_a and sx_b and sx_a == sx_b
        else 0
    )

    metaphone_exact = (
        1
        if mp_a and mp_b and mp_a == mp_b
        else 0
    )

    nysiis_exact = (
        1
        if ny_a and ny_b and ny_a == ny_b
        else 0
    )

    metaphone_similarity = code_similarity(
        mp_a,
        mp_b
    )

    nysiis_similarity = code_similarity(
        ny_a,
        ny_b
    )

    match_rating_similarity = code_similarity(
        mr_a,
        mr_b
    )

    consonant_similarity = code_similarity(
        con_a,
        con_b
    )

    shape_similarity = code_similarity(
        shape_a,
        shape_b
    )

    mp_prefix = phonetic_prefix_ratio(
        mp_a,
        mp_b
    )

    mp_suffix = phonetic_suffix_ratio(
        mp_a,
        mp_b
    )

    sx_prefix = phonetic_prefix_ratio(
        sx_a,
        sx_b
    )

    sx_suffix = phonetic_suffix_ratio(
        sx_a,
        sx_b
    )

    ny_prefix = phonetic_prefix_ratio(
        ny_a,
        ny_b
    )

    ny_suffix = phonetic_suffix_ratio(
        ny_a,
        ny_b
    )

    phonetic_prefix = max(
        mp_prefix,
        sx_prefix,
        ny_prefix
    )

    phonetic_suffix = max(
        mp_suffix,
        sx_suffix,
        ny_suffix
    )

    phonetic_bigram = bigram_similarity(
        mp_a,
        mp_b
    )

    token_phonetic = token_phonetic_similarity(
        rep_a["tokens"],
        rep_b["tokens"]
    )

    phonetic_votes = (

        soundex_exact
        +
        metaphone_exact
        +
        nysiis_exact
        +
        int(metaphone_similarity >= 0.70)
        +
        int(nysiis_similarity >= 0.70)
        +
        int(consonant_similarity >= 0.65)
        +
        int(match_rating_similarity >= 0.70)
    )

    phonetic_score = (

        0.18 * metaphone_similarity

        +

        0.14 * nysiis_similarity

        +

        0.10 * match_rating_similarity

        +

        0.10 * consonant_similarity

        +

        0.08 * shape_similarity

        +

        0.10 * phonetic_bigram

        +

        0.15 * phonetic_prefix

        +

        0.15 * phonetic_suffix
    )

    return {

        "jw": jw,

        "bg": bg,

        "ed": ed,

        "lr": lr,

        "prefix_ratio": pr,

        "suffix_ratio": sr,

        "soundex_exact":
            soundex_exact,

        "metaphone_exact":
            metaphone_exact,

        "nysiis_exact":
            nysiis_exact,

        "metaphone_similarity":
            metaphone_similarity,

        "nysiis_similarity":
            nysiis_similarity,

        "match_rating_similarity":
            match_rating_similarity,

        "consonant_similarity":
            consonant_similarity,

        "shape_similarity":
            shape_similarity,

        "phonetic_prefix":
            phonetic_prefix,

        "phonetic_suffix":
            phonetic_suffix,

        "phonetic_bigram":
            phonetic_bigram,

        "token_phonetic":
            token_phonetic,

        "phonetic_votes":
            phonetic_votes,

        "phonetic_score":
            phonetic_score
    }


# ============================================================
# RULE FUNCTIONS
# ============================================================

def rule_strong_lexical(f, p):

    return (
        f["jw"] >= p["jw_strong"]
        or
        f["ed"] >= p["ed_strong"]
    )


def rule_lexical(f, p):

    return (
        f["jw"] >= p["jw_moderate"]
        and
        (
            f["ed"] >= p["ed_moderate"]
            or
            f["bg"] >= p["bg_moderate"]
        )
    )


def rule_edit(f, p):

    return (
        f["ed"] >= p["edit_rescue"]
        and
        f["lr"] >= p["length_ratio"]
    )


def rule_bigram(f, p):

    return (
        f["bg"] >= p["bigram_rescue"]
        and
        f["lr"] >= p["length_ratio"]
    )


def rule_prefix(f, p):

    return (
        f["prefix_ratio"] >= p["prefix_threshold"]
        and
        f["jw"] >= p["prefix_jw"]
    )


def rule_suffix(f, p):

    return (
        f["suffix_ratio"] >= p["suffix_threshold"]
        and
        f["jw"] >= p["suffix_jw"]
    )


def rule_phonetic(f, p):

    return (
        f["phonetic_score"]
        >= p["phonetic_score"]
    )


def rule_phonetic_ensemble(f, p):

    return (
        f["phonetic_votes"]
        >= p["phonetic_votes"]
    )


def rule_metaphone(f, p):

    return (
        f["metaphone_similarity"]
        >= p["metaphone_threshold"]
    )


def rule_nysiis(f, p):

    return (
        f["nysiis_similarity"]
        >= p["nysiis_threshold"]
    )


def rule_consonant(f, p):

    return (
        f["consonant_similarity"]
        >= p["consonant_threshold"]
    )


def rule_token_phonetic(f, p):

    return (
        f["token_phonetic"]
        >= p["token_phonetic_threshold"]
    )


def rule_boundary_phonetic(f, p):

    return (
        f["phonetic_prefix"]
        >= p["phonetic_boundary"]
        or
        f["phonetic_suffix"]
        >= p["phonetic_boundary"]
    )


# ============================================================
# RULE REGISTRY
# ============================================================

RULES = {

    "strong_lexical":
        rule_strong_lexical,

    "lexical_rescue":
        rule_lexical,

    "edit_rescue":
        rule_edit,

    "bigram_rescue":
        rule_bigram,

    "prefix_rescue":
        rule_prefix,

    "suffix_rescue":
        rule_suffix,

    "phonetic_score":
        rule_phonetic,

    "phonetic_ensemble":
        rule_phonetic_ensemble,

    "metaphone":
        rule_metaphone,

    "nysiis":
        rule_nysiis,

    "consonant":
        rule_consonant,

    "token_phonetic":
        rule_token_phonetic,

    "boundary_phonetic":
        rule_boundary_phonetic
}


# ============================================================
# DEFAULT PARAMETER GRID
# ============================================================

def coarse_parameter_sets():

    jw_strong_values = [
        0.78,
        0.82,
        0.86,
        0.90
    ]

    ed_strong_values = [
        0.70,
        0.75,
        0.80,
        0.85
    ]

    jw_moderate_values = [
        0.55,
        0.60,
        0.65,
        0.70
    ]

    ed_moderate_values = [
        0.30,
        0.40,
        0.50
    ]

    bg_moderate_values = [
        0.20,
        0.30,
        0.40
    ]

    edit_values = [
        0.40,
        0.50,
        0.60,
        0.70
    ]

    bigram_values = [
        0.25,
        0.35,
        0.45
    ]

    prefix_values = [
        0.30,
        0.40,
        0.50,
        0.60
    ]

    phonetic_values = [
        0.24,
        0.28,
        0.32,
        0.36,
        0.40
    ]

    metaphone_values = [
        0.45,
        0.55,
        0.65,
        0.75
    ]

    nysiis_values = [
        0.45,
        0.55,
        0.65,
        0.75
    ]

    consonant_values = [
        0.50,
        0.60,
        0.70
    ]

    token_values = [
        0.35,
        0.45,
        0.55
    ]

    boundary_values = [
        0.40,
        0.50,
        0.60
    ]

    vote_values = [
        2,
        3,
        4
    ]

    # Controlled combinations.
    #
    # We intentionally do NOT create the full Cartesian product.

    configurations = []

    for i in range(4):

        p = {

            "jw_strong":
                jw_strong_values[i],

            "ed_strong":
                ed_strong_values[i],

            "jw_moderate":
                jw_moderate_values[i],

            "ed_moderate":
                ed_moderate_values[
                    i % len(ed_moderate_values)
                ],

            "bg_moderate":
                bg_moderate_values[
                    i % len(bg_moderate_values)
                ],

            "edit_rescue":
                edit_values[i],

            "bigram_rescue":
                bigram_values[
                    i % len(bigram_values)
                ],

            "length_ratio":
                0.45 + 0.05 * i,

            "prefix_threshold":
                prefix_values[
                    i % len(prefix_values)
                ],

            "prefix_jw":
                0.48 + 0.04 * i,

            "suffix_threshold":
                prefix_values[
                    i % len(prefix_values)
                ],

            "suffix_jw":
                0.48 + 0.04 * i,

            "phonetic_score":
                phonetic_values[i],

            "phonetic_votes":
                vote_values[
                    i % len(vote_values)
                ],

            "metaphone_threshold":
                metaphone_values[i],

            "nysiis_threshold":
                nysiis_values[i],

            "consonant_threshold":
                consonant_values[
                    i % len(consonant_values)
                ],

            "token_phonetic_threshold":
                token_values[
                    i % len(token_values)
                ],

            "phonetic_boundary":
                boundary_values[
                    i % len(boundary_values)
                ]
        }

        configurations.append(p)

    # Generate additional controlled combinations.
    #
    # Only a few dimensions change at once.

    bases = list(configurations)

    for base in bases:

        for jw in [
            0.78,
            0.82,
            0.86,
            0.90
        ]:

            q = dict(base)

            q["jw_strong"] = jw

            configurations.append(q)

        for ph in [
            0.24,
            0.28,
            0.32,
            0.36,
            0.40
        ]:

            q = dict(base)

            q["phonetic_score"] = ph

            configurations.append(q)

        for mp in [
            0.45,
            0.55,
            0.65,
            0.75
        ]:

            q = dict(base)

            q["metaphone_threshold"] = mp

            configurations.append(q)

        for ny in [
            0.45,
            0.55,
            0.65,
            0.75
        ]:

            q = dict(base)

            q["nysiis_threshold"] = ny

            configurations.append(q)

        for c in [
            0.50,
            0.60,
            0.70
        ]:

            q = dict(base)

            q["consonant_threshold"] = c

            configurations.append(q)

    # Deduplicate

    unique = {}

    for p in configurations:

        key = tuple(
            sorted(p.items())
        )

        unique[key] = p

    return list(
        unique.values()
    )


# ============================================================
# RULE SELECTION
# ============================================================

RULE_FAMILIES = [

    (
        "strong_lexical",
        1
    ),

    (
        "lexical_rescue",
        1
    ),

    (
        "edit_rescue",
        1
    ),

    (
        "bigram_rescue",
        1
    ),

    (
        "prefix_rescue",
        1
    ),

    (
        "suffix_rescue",
        1
    ),

    (
        "phonetic_score",
        1
    ),

    (
        "phonetic_ensemble",
        1
    ),

    (
        "metaphone",
        1
    ),

    (
        "nysiis",
        1
    ),

    (
        "consonant",
        1
    ),

    (
        "token_phonetic",
        1
    ),

    (
        "boundary_phonetic",
        1
    )
]


# ============================================================
# DECISION
# ============================================================

def blocking_decision(
    features,
    params,
    selected_rules
):

    for rule_name in selected_rules:

        rule_function = RULES[
            rule_name
        ]

        if rule_function(
            features,
            params
        ):

            return True

    return False


# ============================================================
# FAST EVALUATION
# ============================================================

def evaluate_configuration(
    feature_rows,
    params,
    selected_rules
):

    positive_total = 0
    positive_retained = 0

    negative_total = 0
    negative_retained = 0

    for row in feature_rows:

        retained = blocking_decision(
            row["features"],
            params,
            selected_rules
        )

        label = row["label"]

        if label == 1:

            positive_total += 1

            if retained:

                positive_retained += 1

        else:

            negative_total += 1

            if retained:

                negative_retained += 1

    if positive_total == 0:

        recall = 0.0

    else:

        recall = (
            positive_retained
            /
            positive_total
        )

    if negative_total == 0:

        negative_retention = 0.0

    else:

        negative_retention = (
            negative_retained
            /
            negative_total
        )

    return {

        "recall":
            recall,

        "positive_retained":
            positive_retained,

        "positive_total":
            positive_total,

        "negative_retention":
            negative_retention,

        "negative_retained":
            negative_retained,

        "negative_total":
            negative_total
    }


# ============================================================
# SMART EARLY EVALUATION
# ============================================================

def evaluate_configuration_fast(
    feature_rows,
    params,
    selected_rules,
    current_best
):

    positive_total = 0
    positive_retained = 0

    negative_total = 0
    negative_retained = 0

    for row in feature_rows:

        retained = blocking_decision(
            row["features"],
            params,
            selected_rules
        )

        label = row["label"]

        if label == 1:

            positive_total += 1

            if retained:
                positive_retained += 1

        else:

            negative_total += 1

            if retained:
                negative_retained += 1

        # ----------------------------------------------------
        # Early rejection:
        #
        # If recall is already impossible to reach 100%,
        # stop evaluating this configuration.
        # ----------------------------------------------------

        processed_positive = positive_total

        if (
            processed_positive > 0
            and
            positive_total -
            positive_retained
            >
            0
        ):

            # We cannot recover a positive that was already
            # rejected because the decision is deterministic.
            return None

        # ----------------------------------------------------
        # Early rejection for negative retention.
        #
        # If this configuration has already retained more
        # negatives than the current best, it cannot win.
        # ----------------------------------------------------

        if (
            current_best is not None
            and
            negative_total > 0
        ):

            current_rate = (
                negative_retained
                /
                negative_total
            )

            if (
                current_rate
                >
                current_best
                +
                0.05
            ):

                return None

    if positive_total == 0:

        recall = 0.0

    else:

        recall = (
            positive_retained
            /
            positive_total
        )

    if negative_total == 0:

        negative_retention = 0.0

    else:

        negative_retention = (
            negative_retained
            /
            negative_total
        )

    return {

        "recall":
            recall,

        "positive_retained":
            positive_retained,

        "positive_total":
            positive_total,

        "negative_retention":
            negative_retention,

        "negative_retained":
            negative_retained,

        "negative_total":
            negative_total
    }


# ============================================================
# BUILD FEATURES
# ============================================================

def build_feature_rows(df):

    names = set()

    for value in df["name_a"]:

        if not pd.isna(value):
            names.add(str(value))

    for value in df["name_b"]:

        if not pd.isna(value):
            names.add(str(value))

    print(
        f"Unique names   : {len(names):,}"
    )

    print()

    print(
        "Building phonetic representations..."
    )

    representations = {}

    for i, name in enumerate(names):

        representations[name] = (
            build_representation(name)
        )

        if (
            (i + 1) % 250 == 0
            or
            i + 1 == len(names)
        ):

            print(
                f"Representations: "
                f"{i + 1:,}/{len(names):,}",
                flush=True
            )

    print()

    print(
        "Computing pairwise features..."
    )

    feature_rows = []

    total = len(df)

    for i, row in df.iterrows():

        name_a = str(
            row["name_a"]
        )

        name_b = str(
            row["name_b"]
        )

        label = int(
            row["label"]
        )

        features = compute_features(
            representations[name_a],
            representations[name_b]
        )

        feature_rows.append({

            "name_a":
                name_a,

            "name_b":
                name_b,

            "label":
                label,

            "features":
                features
        })

        if (
            (i + 1) % 100 == 0
            or
            i + 1 == total
        ):

            print(
                f"Features: "
                f"{i + 1:,}/{total:,}",
                flush=True
            )

    return feature_rows


# ============================================================
# BUILD SEARCH SPACE
# ============================================================

def build_search_space():

    params = coarse_parameter_sets()

    rule_sets = []

    # Single rules

    for rule_name, _ in RULE_FAMILIES:

        rule_sets.append(
            (rule_name,)
        )

    # Important rule pairs

    important_pairs = [

        (
            "strong_lexical",
            "phonetic_score"
        ),

        (
            "strong_lexical",
            "metaphone"
        ),

        (
            "strong_lexical",
            "nysiis"
        ),

        (
            "strong_lexical",
            "consonant"
        ),

        (
            "strong_lexical",
            "token_phonetic"
        ),

        (
            "lexical_rescue",
            "phonetic_score"
        ),

        (
            "lexical_rescue",
            "metaphone"
        ),

        (
            "lexical_rescue",
            "nysiis"
        ),

        (
            "edit_rescue",
            "phonetic_score"
        ),

        (
            "bigram_rescue",
            "phonetic_score"
        ),

        (
            "prefix_rescue",
            "phonetic_score"
        ),

        (
            "suffix_rescue",
            "phonetic_score"
        ),

        (
            "phonetic_score",
            "metaphone"
        ),

        (
            "phonetic_score",
            "nysiis"
        ),

        (
            "phonetic_score",
            "consonant"
        ),

        (
            "phonetic_score",
            "token_phonetic"
        ),

        (
            "metaphone",
            "nysiis"
        ),

        (
            "metaphone",
            "consonant"
        ),

        (
            "nysiis",
            "consonant"
        ),

        (
            "token_phonetic",
            "boundary_phonetic"
        )
    ]

    for pair in important_pairs:

        rule_sets.append(pair)

    # A few carefully selected triples.

    important_triples = [

        (
            "strong_lexical",
            "metaphone",
            "nysiis"
        ),

        (
            "strong_lexical",
            "phonetic_score",
            "consonant"
        ),

        (
            "strong_lexical",
            "phonetic_score",
            "token_phonetic"
        ),

        (
            "lexical_rescue",
            "metaphone",
            "nysiis"
        ),

        (
            "lexical_rescue",
            "phonetic_score",
            "consonant"
        ),

        (
            "edit_rescue",
            "phonetic_score",
            "metaphone"
        ),

        (
            "edit_rescue",
            "phonetic_score",
            "nysiis"
        ),

        (
            "phonetic_score",
            "metaphone",
            "nysiis"
        ),

        (
            "phonetic_score",
            "metaphone",
            "consonant"
        ),

        (
            "phonetic_score",
            "nysiis",
            "consonant"
        ),

        (
            "metaphone",
            "nysiis",
            "consonant"
        ),

        (
            "metaphone",
            "nysiis",
            "token_phonetic"
        ),

        (
            "phonetic_score",
            "token_phonetic",
            "boundary_phonetic"
        )
    ]

    for triple in important_triples:

        rule_sets.append(triple)

    # Deduplicate

    unique_rules = []

    seen = set()

    for rules in rule_sets:

        key = tuple(rules)

        if key not in seen:

            seen.add(key)

            unique_rules.append(
                rules
            )

    # Limit combinations

    configurations = []

    for p in params:

        for rules in unique_rules:

            configurations.append(
                (
                    p,
                    rules
                )
            )

    return configurations


# ============================================================
# FINE TUNING
# ============================================================

def generate_fine_parameters(
    best_params
):

    candidates = []

    for key, value in best_params.items():

        if not isinstance(
            value,
            (int, float)
        ):

            continue

        if key in [
            "phonetic_votes"
        ]:

            values = [
                max(1, int(value) - 1),
                int(value),
                int(value) + 1
            ]

        else:

            values = [

                max(
                    0.05,
                    value - 0.08
                ),

                max(
                    0.05,
                    value - 0.04
                ),

                value,

                min(
                    0.95,
                    value + 0.04
                ),

                min(
                    0.95,
                    value + 0.08
                )
            ]

        for new_value in values:

            q = dict(best_params)

            q[key] = new_value

            candidates.append(q)

    # Combine only small local modifications.

    for key1 in [
        "jw_strong",
        "phonetic_score",
        "metaphone_threshold",
        "nysiis_threshold",
        "consonant_threshold"
    ]:

        if key1 not in best_params:
            continue

        for key2 in [
            "jw_strong",
            "phonetic_score",
            "metaphone_threshold",
            "nysiis_threshold",
            "consonant_threshold"
        ]:

            if key1 == key2:
                continue

            q = dict(best_params)

            q[key1] = max(
                0.05,
                min(
                    0.95,
                    float(best_params[key1])
                    - 0.04
                )
            )

            q[key2] = max(
                0.05,
                min(
                    0.95,
                    float(best_params[key2])
                    + 0.04
                )
            )

            candidates.append(q)

    unique = {}

    for p in candidates:

        key = tuple(
            sorted(
                p.items()
            )
        )

        unique[key] = p

    return list(
        unique.values()
    )


# ============================================================
# MAIN OPTIMIZATION
# ============================================================

def optimize(
    feature_rows
):

    print()
    print("=" * 70)
    print(
        "CONTROLLED PARAMETER SEARCH"
    )
    print("=" * 70)

    search_space = build_search_space()

    print(
        f"Search configurations: "
        f"{len(search_space):,}"
    )

    print(
        "Goal: 100% LASA recall first, "
        "then minimum negative retention."
    )

    print()

    best = None

    best_score = None

    results = []

    start_time = time.time()

    total = min(
        len(search_space),
        COARSE_MAX_CONFIGS
    )

    # --------------------------------------------------------
    # COARSE SEARCH
    # --------------------------------------------------------

    print(
        "PHASE 1/2: COARSE SEARCH"
    )

    print()

    for index in range(total):

        params, rules = (
            search_space[index]
        )

        result = evaluate_configuration_fast(
            feature_rows,
            params,
            rules,
            (
                best_score
                if best_score is not None
                else None
            )
        )

        if result is None:
            continue

        # We only care about 100% recall.

        if (
            result["recall"]
            < 1.0
        ):
            continue

        negative_retention = (
            result[
                "negative_retention"
            ]
        )

        candidate = {

            "params":
                params,

            "rules":
                rules,

            "metrics":
                result
        }

        if (
            best is None
            or
            negative_retention
            <
            best_score
        ):

            best = candidate

            best_score = (
                negative_retention
            )

            elapsed = (
                time.time()
                -
                start_time
            )

            print(
                f"\n[NEW BEST] "
                f"{index + 1:,}/{total:,} "
                f"| recall=100.0000% "
                f"| negative retention="
                f"{negative_retention * 100:.4f}% "
                f"| rules={rules} "
                f"| elapsed={elapsed:.1f}s",
                flush=True
            )

        results.append({

            "phase":
                "coarse",

            "configuration":
                index + 1,

            "recall":
                result["recall"],

            "negative_retention":
                negative_retention,

            "negative_retained":
                result[
                    "negative_retained"
                ],

            "rules":
                "|".join(rules),

            "parameters":
                json.dumps(
                    params,
                    sort_keys=True
                )
        })

        if (
            (index + 1) % PROGRESS_EVERY == 0
        ):

            elapsed = (
                time.time()
                -
                start_time
            )

            print(
                f"[Progress] "
                f"{index + 1:,}/{total:,} "
                f"({(index + 1) / total * 100:.1f}%) "
                f"| valid 100% recall: "
                f"{len(results):,} "
                f"| best negative retention: "
                f"{best_score * 100:.2f}% "
                if best_score is not None
                else
                f"[Progress] "
                f"{index + 1:,}/{total:,} "
                f"| no 100% configuration yet",
                flush=True
            )

    # --------------------------------------------------------
    # PHASE 2
    # --------------------------------------------------------

    if best is not None:

        print()
        print(
            "=" * 70
        )

        print(
            "PHASE 2/2: FINE SEARCH"
        )

        print(
            "=" * 70
        )

        fine_params = (
            generate_fine_parameters(
                best["params"]
            )
        )

        fine_rules = [
            best["rules"]
        ]

        # Also test nearby rule families.

        nearby_rule_sets = [

            best["rules"]
        ]

        for rules in [
            (
                "strong_lexical",
                "phonetic_score"
            ),

            (
                "strong_lexical",
                "metaphone"
            ),

            (
                "strong_lexical",
                "nysiis"
            ),

            (
                "strong_lexical",
                "consonant"
            ),

            (
                "phonetic_score",
                "metaphone"
            ),

            (
                "phonetic_score",
                "nysiis"
            ),

            (
                "phonetic_score",
                "consonant"
            ),

            (
                "phonetic_score",
                "token_phonetic"
            ),

            (
                "metaphone",
                "nysiis"
            ),

            (
                "metaphone",
                "nysiis",
                "consonant"
            ),

            (
                "strong_lexical",
                "phonetic_score",
                "consonant"
            ),

            (
                "strong_lexical",
                "phonetic_score",
                "token_phonetic"
            )
        ]:

            if rules not in nearby_rule_sets:

                nearby_rule_sets.append(
                    rules
                )

        fine_space = []

        for p in fine_params:

            for rules in nearby_rule_sets:

                fine_space.append(
                    (
                        p,
                        rules
                    )
                )

        if len(fine_space) > FINE_MAX_CONFIGS:

            fine_space = (
                fine_space[
                    :FINE_MAX_CONFIGS
                ]
            )

        print(
            f"Fine configurations: "
            f"{len(fine_space):,}"
        )

        print()

        fine_start = time.time()

        for index, (
            params,
            rules
        ) in enumerate(
            fine_space
        ):

            result = evaluate_configuration_fast(
                feature_rows,
                params,
                rules,
                best_score
            )

            if result is None:
                continue

            if (
                result["recall"]
                < 1.0
            ):
                continue

            negative_retention = (
                result[
                    "negative_retention"
                ]
            )

            results.append({

                "phase":
                    "fine",

                "configuration":
                    index + 1,

                "recall":
                    result["recall"],

                "negative_retention":
                    negative_retention,

                "negative_retained":
                    result[
                        "negative_retained"
                    ],

                "rules":
                    "|".join(rules),

                "parameters":
                    json.dumps(
                        params,
                        sort_keys=True
                    )
            })

            if (
                negative_retention
                <
                best_score
            ):

                best = {

                    "params":
                        params,

                    "rules":
                        rules,

                    "metrics":
                        result
                }

                best_score = (
                    negative_retention
                )

                print(
                    f"\n[NEW FINE BEST] "
                    f"{index + 1:,}/{len(fine_space):,} "
                    f"| recall=100.0000% "
                    f"| negative retention="
                    f"{negative_retention * 100:.4f}% "
                    f"| rules={rules}",
                    flush=True
                )

            if (
                (index + 1) % PROGRESS_EVERY == 0
            ):

                print(
                    f"[Fine progress] "
                    f"{index + 1:,}/"
                    f"{len(fine_space):,} "
                    f"| best negative retention="
                    f"{best_score * 100:.2f}%",
                    flush=True
                )

    # --------------------------------------------------------
    # No solution
    # --------------------------------------------------------

    if best is None:

        print()
        print(
            "WARNING: No searched configuration "
            "achieved 100% recall."
        )

        print(
            "The rule family/search space needs "
            "to be expanded."
        )

        return None, results

    return best, results


# ============================================================
# FINAL EVALUATION
# ============================================================

def final_evaluation(
    feature_rows,
    best
):

    params = best["params"]

    rules = best["rules"]

    retained_rows = []

    removed_positive_rows = []

    rule_counts = Counter()

    positive_total = 0
    positive_retained = 0

    negative_total = 0
    negative_retained = 0

    for row in feature_rows:

        features = row["features"]

        retained = blocking_decision(
            features,
            params,
            rules
        )

        label = row["label"]

        if label == 1:

            positive_total += 1

            if retained:

                positive_retained += 1

            else:

                removed_positive_rows.append({

                    "name_a":
                        row["name_a"],

                    "name_b":
                        row["name_b"],

                    "label":
                        label,

                    **features
                })

        else:

            negative_total += 1

            if retained:

                negative_retained += 1

        if retained:

            for rule in rules:

                if RULES[rule](
                    features,
                    params
                ):

                    rule_counts[
                        rule
                    ] += 1

            retained_rows.append({

                "name_a":
                    row["name_a"],

                "name_b":
                    row["name_b"],

                "label":
                    label,

                "blocking_rules":
                    "|".join(
                        rules
                    ),

                **features
            })

    recall = (
        positive_retained
        /
        positive_total
        if positive_total
        else 0
    )

    negative_retention = (
        negative_retained
        /
        negative_total
        if negative_total
        else 0
    )

    return {

        "retained_rows":
            retained_rows,

        "removed_positive_rows":
            removed_positive_rows,

        "rule_counts":
            rule_counts,

        "positive_total":
            positive_total,

        "positive_retained":
            positive_retained,

        "negative_total":
            negative_total,

        "negative_retained":
            negative_retained,

        "recall":
            recall,

        "negative_retention":
            negative_retention
    }


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)

    print(
        " AUTOMATIC HIGH-RECALL LASA BLOCKING"
    )

    print(
        " CONTROLLED PARAMETER OPTIMIZATION"
    )

    print(
        " 100% RECALL -> MINIMUM NEGATIVE RETENTION"
    )

    print("=" * 70)

    print()

    print(
        f"Reference file : {INPUT_FILE}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    required = [
        "name_a",
        "name_b",
        "label"
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    print(
        f"Rows           : {len(df):,}"
    )

    print(
        "Name column 1  : name_a"
    )

    print(
        "Name column 2  : name_b"
    )

    print(
        "Label column   : label"
    )

    positive_count = (
        df["label"]
        .astype(int)
        .sum()
    )

    negative_count = (
        len(df)
        -
        positive_count
    )

    print(
        f"Positive LASA  : {positive_count:,}"
    )

    print(
        f"Negative pairs : {negative_count:,}"
    )

    print()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_rows = build_feature_rows(
        df
    )

    print()

    print(
        "=" * 70
    )

    print(
        "FEATURE PRECOMPUTATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    # --------------------------------------------------------
    # Optimize
    # --------------------------------------------------------

    best, optimization_results = (
        optimize(
            feature_rows
        )
    )

    if best is None:

        print()
        print(
            "Optimization failed to find "
            "a 100% recall configuration."
        )

        return

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "FINAL EVALUATION OF BEST CONFIGURATION"
    )

    print(
        "=" * 70
    )

    final = final_evaluation(
        feature_rows,
        best
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    positive_total = (
        final[
            "positive_total"
        ]
    )

    positive_retained = (
        final[
            "positive_retained"
        ]
    )

    negative_total = (
        final[
            "negative_total"
        ]
    )

    negative_retained = (
        final[
            "negative_retained"
        ]
    )

    negative_removed = (
        negative_total
        -
        negative_retained
    )

    recall = final["recall"]

    negative_retention = (
        final[
            "negative_retention"
        ]
    )

    print()

    print(
        "BEST RULES"
    )

    for rule in best["rules"]:

        print(
            f"  - {rule}"
        )

    print()

    print(
        "BEST PARAMETERS"
    )

    for key, value in sorted(
        best["params"].items()
    ):

        print(
            f"  {key:<30} = {value}"
        )

    print()

    print(
        "=" * 70
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Positive pairs       : "
        f"{positive_total:,}"
    )

    print(
        f"Positive retained    : "
        f"{positive_retained:,}"
    )

    print(
        f"Positive removed     : "
        f"{positive_total - positive_retained:,}"
    )

    print(
        f"LASA BLOCKING RECALL : "
        f"{recall * 100:.4f}%"
    )

    print()

    print(
        f"Negative pairs       : "
        f"{negative_total:,}"
    )

    print(
        f"Negative retained    : "
        f"{negative_retained:,}"
    )

    print(
        f"Negative removed     : "
        f"{negative_removed:,}"
    )

    print(
        f"NEGATIVE RETENTION   : "
        f"{negative_retention * 100:.4f}%"
    )

    print(
        f"NEGATIVE REMOVAL     : "
        f"{(1 - negative_retention) * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Rule contribution
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "RULE CONTRIBUTION"
    )

    print(
        "=" * 70
    )

    for rule, count in (
        final[
            "rule_counts"
        ].most_common()
    ):

        print(
            f"{rule:<30}: {count:,}"
        )

    # --------------------------------------------------------
    # Save candidates
    # --------------------------------------------------------

    candidates_df = pd.DataFrame(
        final[
            "retained_rows"
        ]
    )

    candidates_df.to_csv(
        CANDIDATE_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save removed positives
    # --------------------------------------------------------

    removed_positive_df = (
        pd.DataFrame(
            final[
                "removed_positive_rows"
            ]
        )
    )

    removed_positive_df.to_csv(
        REMOVED_POSITIVE_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save optimization history
    # --------------------------------------------------------

    optimization_df = pd.DataFrame(
        optimization_results
    )

    if not optimization_df.empty:

        optimization_df = (
            optimization_df
            .sort_values(
                [
                    "recall",
                    "negative_retention"
                ],
                ascending=[
                    False,
                    True
                ]
            )
        )

        optimization_df.to_csv(
            OPTIMIZATION_FILE,
            index=False
        )

    # --------------------------------------------------------
    # Save parameters
    # --------------------------------------------------------

    best_json = {

        "rules":
            list(best["rules"]),

        "parameters":
            best["params"],

        "metrics": {

            "positive_total":
                positive_total,

            "positive_retained":
                positive_retained,

            "recall":
                recall,

            "negative_total":
                negative_total,

            "negative_retained":
                negative_retained,

            "negative_retention":
                negative_retention,

            "negative_removal":
                1 - negative_retention
        }
    }

    with open(
        BEST_PARAMS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            best_json,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 70
    )

    print(
        f"Candidates       : "
        f"{CANDIDATE_FILE}"
    )

    print(
        f"Removed positives: "
        f"{REMOVED_POSITIVE_FILE}"
    )

    print(
        f"Optimization     : "
        f"{OPTIMIZATION_FILE}"
    )

    print(
        f"Best parameters  : "
        f"{BEST_PARAMS_FILE}"
    )

    print()

    if (
        positive_retained
        ==
        positive_total
    ):

        print(
            "SUCCESS: 100% LASA RECALL GUARANTEED "
            "ON THE REFERENCE SET."
        )

    else:

        print(
            "WARNING: Some positive LASA pairs "
            "were removed."
        )

    print()

    print(
        "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()