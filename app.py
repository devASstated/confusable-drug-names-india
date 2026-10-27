#!/usr/bin/env python3
"""LASA reviewer UI: representation-selectable ranking + proposal collision check.

Run:
    streamlit run app.py -- --scores /mnt/nexus/candidate_scores

Score layout (backward-compatible with the already-built orthographic store):
    /mnt/nexus/candidate_scores/                 # orthographic sidecars/shortlists
    /mnt/nexus/candidate_scores/phonetic_us/     # US-token sidecars/shortlists
    /mnt/nexus/candidate_scores/phonetic_hindi/  # Hindi-token sidecars/shortlists
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import rank_dynamic as R

st.set_page_config(page_title="LASA candidate review", page_icon="💊", layout="wide")

MODE_LABELS = {
    "orthographic": "Orthographic",
    "phonetic_us": "Phonetic — US English",
    "phonetic_hindi": "Phonetic — Hindi / Indian accent",
}
MODE_COLUMNS = {
    "phonetic_us": "phones_g2p_arpa",
    "phonetic_hindi": "predicted_normalized_phonemes",
}
DEFAULT_WEIGHTS = dict(jw=.5, edit=.5, kbigram=0.0, lumera=0.0, feline=0.0)


def _full_width():
    import inspect
    try:
        if "width" in inspect.signature(st.dataframe).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


WIDE = _full_width()


def _cli_defaults():
    argv = sys.argv[1:]
    out = {
        "scores": os.environ.get("LASA_SCORES", "/mnt/nexus/candidate_scores"),
        "names": os.environ.get("LASA_NAMES", "results/name_index.csv"),
        "phonemes": os.environ.get("LASA_PHONEMES", "normalized_inference_output.csv"),
        "accent_model": os.environ.get(
            "LASA_ACCENT_MODEL", "byt5_phoneme_advanced/best_byt5_advanced_model")
    }
    for flag, key in (("--scores", "scores"), ("--names", "names"),
                      ("--name-index", "names"), ("--phonemes", "phonemes"),
                      ("--accent-model", "accent_model")):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                out[key] = argv[i + 1]
    return out


DEFAULTS = _cli_defaults()


@st.cache_resource(show_spinner="loading name index…")
def get_names(path: str) -> np.ndarray:
    return pd.read_csv(path, keep_default_na=False)["brand_root"].astype(str).to_numpy()


@st.cache_resource(show_spinner="opening shortlist…")
def get_shortlist(path: str):
    return R.Shortlist(path)


@st.cache_data(show_spinner=False)
def find_stores(scores_dir: str):
    d = Path(scores_dir)
    found = []
    for meta in sorted(d.glob(f"{R.STORE_NAME}*/meta.json")):
        try:
            m = json.loads(meta.read_text())
            found.append((str(meta.parent), float(m["min_max"]), int(m["n_rows"])))
        except Exception:
            pass
    npz = d / R.SHORTLIST_NAME
    if npz.exists() and not found:
        found.append((str(npz), float("nan"), 0))
    return sorted(found, key=lambda t: (-t[1] if t[1] == t[1] else 1e9))


def mode_scores_dir(root: str, mode: str) -> Path:
    root = Path(root)
    if mode == "orthographic":
        return root
    return root / mode


@st.cache_data(show_spinner=False)
def digit_mask_for_names(names_tuple):
    return np.fromiter((any(ch.isdigit() for ch in n) for n in names_tuple),
                       dtype=np.bool_, count=len(names_tuple))


def weights_ui(mode: str):
    """Render only measures that are meaningful for the selected representation.

    Orthographic mode can use all five measures. Phonetic modes currently have
    genuine token-space implementations only for JW, Edit and FELINE; kBigram
    and LUMERA are orthographic learned models and therefore receive exactly
    zero weight rather than diluting the active phonetic weights.
    """
    st.sidebar.header("Similarity weights")

    jw = st.sidebar.slider(
        "Jaro–Winkler", 0.0, 1.0, 0.50, 0.05, key=f"jw_{mode}"
    )
    edit = st.sidebar.slider(
        "Edit similarity", 0.0, 1.0, 0.50, 0.05, key=f"edit_{mode}"
    )

    raw = {
        "jw": float(jw),
        "edit": float(edit),
        "kbigram": 0.0,
        "lumera": 0.0,
        "feline": 0.0,
    }

    with st.sidebar.expander("Experimental measures", expanded=False):
        st.caption(
            "Default 0: these measures have not yet been validated by "
            "medical experts."
        )

        if mode == "orthographic":
            raw["kbigram"] = float(st.slider(
                "kBigram", 0.0, 1.0, 0.0, 0.05, key=f"kb_{mode}"
            ))
            raw["lumera"] = float(st.slider(
                "LUMERA", 0.0, 1.0, 0.0, 0.05, key=f"lu_{mode}"
            ))
        else:
            st.caption(
                "kBigram and LUMERA are unavailable in phonetic mode because "
                "their learned models are orthographic. They are fixed at 0."
            )

        raw["feline"] = float(st.slider(
            "FELINE", 0.0, 1.0, 0.0, 0.05, key=f"fe_{mode}",
            help=(
                "Character-space FELINE"
                if mode == "orthographic"
                else "Affine local alignment over phoneme tokens"
            ),
        ))

    active = (
        ("jw", "edit", "kbigram", "lumera", "feline")
        if mode == "orthographic"
        else ("jw", "edit", "feline")
    )
    total = sum(raw[m] for m in active)

    if total <= 0:
        st.sidebar.error("At least one available measure must have non-zero weight.")
        st.stop()

    # Normalise only over measures that genuinely exist in this representation.
    weights = {m: 0.0 for m in R.MEASURES}
    for m in active:
        weights[m] = raw[m] / total

    st.sidebar.caption(
        "normalised: "
        + "  ".join(f"{m} {weights[m]:.2f}" for m in active)
    )

    if mode != "orthographic":
        st.sidebar.caption("kbigram 0.00  ·  lumera 0.00")

    return weights


def load_mode_shortlists(scores_dir: Path):
    stores = find_stores(str(scores_dir))
    return [get_shortlist(p) for p, _, _ in stores]


def missing_mode_message(mode: str, scores_dir: Path, root: str):
    st.error(f"No shortlist exists for **{MODE_LABELS[mode]}** under `{scores_dir}`.")
    if mode == "orthographic":
        st.code(
            f"python rank_dynamic.py survey --scores {root}\n"
            f"python rank_dynamic.py build --scores {root} --pairs-csv /mnt/nexus/candidate_pairs.csv --min-max 0.60\n"
            f"python rank_dynamic.py compact --scores {root} --tiers 0.70,0.80",
            language="bash")
    else:
        m = "us" if mode == "phonetic_us" else "hindi"
        st.code(
            f"python score_phonetic_candidates.py --mode {m} \\\n"
            f"  --phoneme-csv {DEFAULTS['phonemes']} \\\n"
            f"  --pairs-csv /mnt/nexus/candidate_pairs.csv \\\n"
            f"  --outdir {scores_dir} --dtype float16 --resume\n\n"
            f"python rank_dynamic.py survey --scores {scores_dir}\n"
            f"python rank_dynamic.py build --scores {scores_dir} \\\n"
            f"  --pairs-csv /mnt/nexus/candidate_pairs.csv --min-max 0.60\n"
            f"python rank_dynamic.py compact --scores {scores_dir} --tiers 0.70,0.80",
            language="bash")


# ---------------- proposal helpers -----------------------------------------
def _strip_stress(token: str) -> str:
    return re.sub(r"[012]$", "", token)


@st.cache_resource(show_spinner="loading US G2P…")
def get_g2p():
    from g2p_en import G2p
    return G2p()


def proposal_us_phones(name: str):
    out = []
    for t in get_g2p()(name):
        t = str(t).strip()
        if re.fullmatch(r"[A-Z]+[012]?", t):
            out.append(_strip_stress(t))
    return out


@st.cache_resource(show_spinner="loading Hindi-accent normalizer…")
def get_accent_model(model_path: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()
    return tok, mdl, device


def proposal_hindi_phones(name: str, model_path: str):
    import torch
    us = proposal_us_phones(name)
    canonical = " ".join(us)
    prompt = f"word: {name.lower()} | canonical: {canonical}"
    tok, mdl, device = get_accent_model(model_path)
    x = tok(prompt, return_tensors="pt", truncation=True, max_length=96)
    x = {k: v.to(device) for k, v in x.items()}
    with torch.inference_mode():
        y = mdl.generate(**x, num_beams=4, max_new_tokens=64)
    pred = tok.decode(y[0], skip_special_tokens=True).strip()
    return pred.split() if pred else []


@st.cache_data(show_spinner="loading phoneme namespace…")
def phoneme_namespace(path: str, mode: str, names_tuple):
    col = MODE_COLUMNS[mode]
    d = pd.read_csv(path, usecols=["brand_root", col], keep_default_na=False)
    roots = d["brand_root"].astype(str).to_numpy()
    if not np.array_equal(roots, np.asarray(names_tuple, dtype=str)):
        raise ValueError("phoneme CSV brand_root order does not match name_index.csv")
    return [str(x).strip().split() if str(x).strip() else [] for x in d[col]]


def encode_token_strings(namespace, query):
    vocab = sorted({t for seq in namespace for t in seq} | set(query))
    # Private-use codepoints make each phoneme exactly one character, allowing
    # jellyfish's C implementations to compute token-level JW/edit exactly.
    mp = {t: chr(0xE000 + i) for i, t in enumerate(vocab)}
    enc = ["".join(mp[t] for t in seq) for seq in namespace]
    q = "".join(mp[t] for t in query)
    return enc, q


def token_feline(a, b):
    # Same recurrence/parameters as FELINE, but symbols are phoneme tokens.
    go, ge, ps, pmax = .60, .15, .175, 4
    n, m = len(a), len(b)
    if not n or not m:
        return 0.0
    H = [0.0] * (m + 1); F = [0.0] * (m + 1); best = 0.0
    for i in range(1, n + 1):
        hdiag = 0.0; ecol = 0.0; H[0] = 0.0
        for j in range(1, m + 1):
            e = max(H[j-1] - go, ecol - ge); ecol = e
            f = max(H[j] - go, F[j] - ge); F[j] = f
            v = max(0.0, hdiag + (1.0 if a[i-1] == b[j-1] else 0.0), e, f)
            hdiag, H[j] = H[j], v; best = max(best, v)
    raw = best / max(n, m)
    p = 0
    for x, y in zip(a[:pmax], b[:pmax]):
        if x != y: break
        p += 1
    return min(1.0, max(0.0, raw + p * ps * (1.0 - raw)))


@st.cache_resource(show_spinner="preparing scorer…")
def _ortho_models():
    """The substitution matrices score_candidates already knows how to build."""
    import score_candidates as S
    S_kb, kb_vocab, kb_gap = S.load_kbigram("kbigram_model.npz")
    S_lu, lu_alpha, lu_go, lu_ge = S.load_lumera("lumera_model.npz")
    return S, S_kb, kb_vocab, kb_gap, S_lu, lu_alpha, lu_go, lu_ge


@st.cache_data(show_spinner="encoding namespace…")
def _ortho_encoding(name_path: str, query: str):
    S, S_kb, kb_vocab, kb_gap, S_lu, lu_alpha, lu_go, lu_ge = _ortho_models()
    names = list(get_names(name_path))
    enc = S.encode_names(names + [query], lu_alpha, kb_vocab)
    S_id = np.eye(max(enc[-1], 1), dtype=np.float32)
    return enc, (S_id, S_lu, lu_go, lu_ge, S_kb, kb_gap)


@st.cache_data(show_spinner="encoding phoneme namespace…")
def _phon_encoding(phoneme_path: str, mode: str, names_tuple, qtok: tuple):
    import score_phonetic_candidates as P
    ns = phoneme_namespace(phoneme_path, mode, names_tuple)
    vocab = sorted({t for seq in ns for t in seq} | set(qtok))
    tok = {t: i for i, t in enumerate(vocab)}
    seqs = [[tok[t] for t in seq] for seq in ns] + [[tok[t] for t in qtok]]
    enc, models = P.make_encoding(seqs, len(vocab))
    return enc, models


def proposal_scores(name, mode, names, weights, hide_digits, phoneme_path,
                    accent_model):
    """Score a proposed name against the whole namespace.

    This runs through score_candidates.score_ids — the same numba kernel the
    2.24-billion-pair batch used — rather than calling the scorer modules once
    per name in Python. The modules are identical in result (verified to ~3e-8)
    but cost ~47 s over 181,819 names with all measures active, against 0.13 s
    here. A reviewer typing a candidate name should not wait either way, and
    reusing the batch kernel also removes the chance of the interactive answer
    and the batch answer disagreeing.
    """
    import score_candidates as S
    mask = np.ones(len(names), dtype=bool)
    if hide_digits:
        mask &= ~digit_mask_for_names(tuple(names))
    idx = np.flatnonzero(mask)
    N = len(names)

    if mode == "orthographic":
        q = str(name)
        enc, models = _ortho_encoding(str(name_path), q)
        want = [weights[m] > 0 for m in R.MEASURES]
        representation = q
    else:
        qtok = (proposal_us_phones(name) if mode == "phonetic_us"
                else proposal_hindi_phones(name, accent_model))
        enc, models = _phon_encoding(phoneme_path, mode, tuple(names),
                                     tuple(qtok))
        # kbigram/lumera have no phoneme-space model, so they stay off here
        # exactly as they do in the batch phonetic scorer.
        want = [weights["jw"] > 0, weights["edit"] > 0, False, False,
                weights["feline"] > 0]
        representation = " ".join(qtok)

    ia = np.full(len(idx), N, np.int32)      # the query sits at index N
    ib = idx.astype(np.int32)
    out = S.score_ids(ia, ib, enc, models, want, S.HAVE_NUMBA)

    w = np.array([weights[m] for m in R.MEASURES], np.float32)
    return idx, out, out @ w, representation


# ---------------- UI --------------------------------------------------------
st.sidebar.header("Data")
scores_root = st.sidebar.text_input("scores root", DEFAULTS["scores"])
name_path = st.sidebar.text_input("name index", DEFAULTS["names"])
phoneme_path = st.sidebar.text_input("phoneme namespace", DEFAULTS["phonemes"])

mode = st.sidebar.radio("Similarity mode", list(MODE_LABELS),
                        format_func=lambda x: MODE_LABELS[x])
hide_digits = st.sidebar.checkbox("Hide names containing numeric digits", value=False,
                                  help="Review-time filter only; stored evidence is unchanged.")
weights = weights_ui(mode)

st.sidebar.header("Queue")
page_size = int(st.sidebar.selectbox(
    "Rows per page", [25, 50, 100, 200, 500], index=2
))

# A change in ranking semantics should return the reviewer to page 1 rather
# than leaving them on a deep page whose contents now mean something else.
rank_signature = (
    mode,
    hide_digits,
    tuple(round(float(weights[m]), 6) for m in R.MEASURES),
    page_size,
)
if st.session_state.get("_rank_signature") != rank_signature:
    st.session_state["_rank_signature"] = rank_signature
    st.session_state["queue_page"] = 1

if "queue_page" not in st.session_state:
    st.session_state["queue_page"] = 1

try:
    names = get_names(name_path)
except Exception as e:
    st.error(f"Could not load name index: {e}")
    st.stop()

digit_mask = digit_mask_for_names(tuple(names)) if hide_digits else None
scores_dir = mode_scores_dir(scores_root, mode)

st.title("LASA candidate review")
st.caption("One representation at a time. Default ranking uses established JW + edit; experimental measures start at zero.")

review_tab, proposal_tab = st.tabs(["Existing namespace", "Check proposed name"])

with review_tab:
    stores = load_mode_shortlists(scores_dir)
    if not stores:
        missing_mode_message(mode, scores_dir, scores_root)
    else:
        nav1, nav2, nav3, nav4 = st.columns([1, 1, 2, 1])

        with nav1:
            if st.button("← Previous", disabled=st.session_state["queue_page"] <= 1):
                st.session_state["queue_page"] = max(
                    1, st.session_state["queue_page"] - 1
                )
                st.rerun()

        with nav2:
            widest_n = max((s.n for s in stores), default=0)
            cap = max(1, min(10_000, -(-widest_n // max(page_size, 1))))
            if st.button("Next →",
                         disabled=st.session_state["queue_page"] >= cap):
                st.session_state["queue_page"] = min(
                    cap, st.session_state["queue_page"] + 1)
                st.rerun()

        with nav3:
            # rank() materialises the global top page*page_size rows, so an
            # unbounded page number asks for a DataFrame the size of the whole
            # shortlist — ~40 GiB at 804M rows, which is an OOM, not a slow
            # query. Cap the request at what the widest tier can actually
            # certify-or-return.
            widest = max((s.n for s in stores), default=0)
            max_page = max(1, min(10_000, -(-widest // max(page_size, 1))))
            requested_page = int(st.number_input(
                "Page",
                min_value=1,
                max_value=max_page,
                value=min(int(st.session_state["queue_page"]), max_page),
                step=1,
                key="queue_page_input",
                help=f"deepest page the loaded shortlist can serve: "
                     f"{max_page:,}",
            ))
            if requested_page != st.session_state["queue_page"]:
                st.session_state["queue_page"] = requested_page
                st.rerun()

        with nav4:
            st.metric("Rows/page", f"{page_size:,}")

        page = int(st.session_state["queue_page"])

        with st.spinner(
            f"ranking global page {page:,} "
            f"({MODE_LABELS[mode].lower()})…"
        ):
            df, info = R.rank_page(
                stores,
                weights,
                page=page,
                page_size=page_size,
                exclude_name_mask=digit_mask,
            )

        if len(df) == 0 and page > 1:
            st.warning(
                "No rows were returned for this page. "
                "Use Previous or jump to an earlier page."
            )
        else:
            df = R.attach_names(df, name_path)
            if "ortho" in df.columns:
                df["found_by"] = ["+".join(
                    label for label, col in (
                        ("ortho", "ortho"),
                        ("US", "phon_US"),
                        ("Hindi", "phon_IN"),
                    ) if bool(r[col])
                ) for _, r in df.iterrows()]

            start_rank = info["page_start_rank"]
            end_rank = info["page_end_rank"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("representation", MODE_LABELS[mode])
            c2.metric(
                "showing ranks",
                f"{start_rank:,}–{end_rank:,}" if len(df) else "—",
            )
            c3.metric(
                "page top score",
                f"{float(df.combined.iloc[0]):.4f}" if len(df) else "—",
            )
            c4.metric(
                f"rank #{info['requested_end_rank']:,} score",
                f"{info['nth_score']:.4f}",
            )

            if info["exact"]:
                st.success(
                    f"Exact global ranks {start_rank:,}–{end_rank:,}. "
                    "The app automatically selected the fastest shortlist tier "
                    "that certifies the end of this page."
                )
            else:
                st.warning(
                    f"This page reaches beyond the exactness guarantee of the "
                    f"widest available shortlist. Ranks near the tail may be "
                    f"incomplete; build a lower-threshold shortlist for this mode "
                    f"if you need certified results this deep."
                )

            cols = [c for c in (
                "rank", "root_a", "root_b", "combined", *R.MEASURES,
                "found_by", "hmlb", "ph_US_score", "ph_IN_score"
            ) if c in df.columns]
            st.dataframe(df[cols], hide_index=True, **WIDE)

            file_lo = start_rank if len(df) else 0
            file_hi = end_rank if len(df) else 0
            st.download_button(
                "Download this page as CSV",
                df[cols].to_csv(index=False).encode(),
                file_name=(
                    f"lasa_{mode}_ranks_{file_lo}_{file_hi}.csv"
                ),
                mime="text/csv",
            )

            # Duplicate navigation at the bottom so long pages do not require
            # scrolling back to the top.
            b1, b2, b3 = st.columns([1, 1, 4])
            with b1:
                if st.button(
                    "← Previous page",
                    key="prev_bottom",
                    disabled=page <= 1,
                ):
                    st.session_state["queue_page"] = max(1, page - 1)
                    st.rerun()
            with b2:
                if st.button("Next page →", key="next_bottom",
                             disabled=page >= cap):
                    st.session_state["queue_page"] = min(cap, page + 1)
                    st.rerun()
            with b3:
                st.caption(
                    f"Global queue · page {page:,} · "
                    f"{page_size:,} rows/page"
                )

            with st.expander("Diagnostics", expanded=False):
                st.write({
                    "shortlist": info.get("shortlist_path"),
                    "threshold": info["min_max"],
                    "exact": info["exact"],
                    "weights": info["weights"],
                    "hide_numeric_names": hide_digits,
                    "page": page,
                    "page_size": page_size,
                    "requested_global_rank": info["requested_end_rank"],
                })

with proposal_tab:
    st.subheader("New drug-name collision check")
    proposal = st.text_input("Proposed brand name", placeholder="e.g. Nexaril")
    st.caption(f"Scoring representation: **{MODE_LABELS[mode]}**. This checks the proposal against all {len(names):,} existing roots; it does not modify the namespace.")
    if st.button("Check collisions", type="primary", disabled=not proposal.strip()):
        try:
            with st.spinner("scoring proposed name against the namespace…"):
                idx, sc, combined, rep = proposal_scores(
                    proposal.strip(), mode, names, weights, hide_digits,
                    phoneme_path, DEFAULTS["accent_model"])
            n = min(page_size, len(idx))
            take = np.argpartition(-combined, n-1)[:n] if n else np.array([], int)
            take = take[np.argsort(-combined[take])]
            out = pd.DataFrame({
                "existing_name": names[idx[take]],
                "combined": combined[take],
            })
            for c, m in enumerate(R.MEASURES):
                out[m] = sc[take, c]
            st.caption(f"Proposal representation: `{rep}`")
            st.dataframe(out, hide_index=True, **WIDE)
            st.download_button("Download proposal collisions as CSV",
                               out.to_csv(index=False).encode(),
                               file_name=f"lasa_proposal_{proposal.strip()}_{mode}.csv",
                               mime="text/csv")
        except Exception as e:
            st.error(f"Proposal scoring failed: {e}")
            if mode != "orthographic":
                st.info("US mode requires g2p-en. Hindi mode additionally requires the recovered ByT5 accent model and Transformers/Torch.")
