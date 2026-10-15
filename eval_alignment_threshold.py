"""
Alignment Threshold Evaluation — Method 3: Distribution Inspection
===================================================================
Uses cached article pairs across 4 language pairs (EN-DE, EN-ZH, EN-IT, EN-FR) to:
  1. Chunk (paragraph-based)
  2. Embed (LaBSE)
  3. Align both directions (EN→X, X→EN)
  4. Plot cosine similarity distributions per language pair
  5. Report 1:0 omission rate at multiple threshold values

Run from project root:
    python eval_alignment_threshold.py
"""

import json
import sys
from pathlib import Path
from itertools import groupby

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — saves to PNG, no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
CACHE_DIR  = BASE_DIR / "app" / "cached_articles"
OUTPUT_PNG = BASE_DIR / "alignment_threshold_eval.png"

# ── article pairs to evaluate (must exist in CACHE_DIR) ──────────────────────
# Each entry: (en_file, other_file, lang_label)
# Riley Elf DE excluded: DE version is a stub (11 sentences vs 474 EN) — not a
# genuine parallel article; would bias threshold estimates downward.
ARTICLE_PAIRS = [
    # EN-DE  (2 real parallel articles — science + food)
    ("en_COVID-19_pandemic.json",  "de_COVID-19-Pandemie.json",          "EN-DE"),
    ("en_Gala_(apple).json",       "de_Gala_(Apfel).json",               "EN-DE"),
    # EN-ZH  (3 articles — science, biography, geography)
    ("en_Climate_change.json",     "zh_氣候變化.json",                    "EN-ZH"),
    ("en_Adolf_Hitler.json",       "zh_阿道夫·希特勒.json",              "EN-ZH"),
    ("en_Switzerland.json",        "zh_瑞士.json",                        "EN-ZH"),
    # EN-IT  (3 articles — science, biography, geography)
    ("en_Climate_change.json",     "it_Cambiamento_climatico.json",       "EN-IT"),
    ("en_Adolf_Hitler.json",       "it_Adolf_Hitler.json",                "EN-IT"),
    ("en_Switzerland.json",        "it_Svizzera.json",                    "EN-IT"),
    # EN-FR  (3 articles — science, biography, geography)
    ("en_Climate_change.json",     "fr_Changement_climatique.json",       "EN-FR"),
    ("en_Adolf_Hitler.json",       "fr_Adolf_Hitler.json",                "EN-FR"),
    ("en_Switzerland.json",        "fr_Suisse.json",                      "EN-FR"),
]

THRESHOLDS_TO_TEST = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
CURRENT_THRESHOLD  = 0.45   # old single global threshold (for reference)
LAMBDA             = 0.3
PATH_SIM_THRESHOLD = 0.45

# Calibrated thresholds per language pair (mirrors aligner LANG_THRESHOLDS)
CALIBRATED = {"EN-DE": 0.30, "EN-IT": 0.35, "EN-FR": 0.40, "EN-ZH": 0.45}

# ── step 1: paragraph chunker (mirrors chunker.py) ───────────────────────────
def paragraph_chunk(sentences: list[dict]) -> list[dict]:
    if not sentences:
        return []
    chunks: list[dict] = []
    chunk_counter = 0
    sorted_sents = sorted(sentences, key=lambda x: (x["section"], x["paragraph_index"]))
    for (section, para_idx), group in groupby(
        sorted_sents, key=lambda x: (x["section"], x["paragraph_index"])
    ):
        sents = sorted(list(group), key=lambda x: x["sentence_index"])
        lang  = sents[0]["language"]
        text  = " ".join(s["sentence_text"] for s in sents)
        if len(sents) == 1 and chunks:
            chunks[-1]["chunk_text"]     += " " + text
            chunks[-1]["sentence_count"] += 1
        else:
            chunks.append({
                "chunk_id":        f"{lang}_{chunk_counter:04d}",
                "chunk_text":      text,
                "section":         section,
                "paragraph_index": para_idx,
                "language":        lang,
                "sentence_count":  len(sents),
            })
            chunk_counter += 1
    return chunks


# ── step 2: embed ─────────────────────────────────────────────────────────────
def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> pd.DataFrame:
    df   = pd.DataFrame(chunks)
    df["chunk_embedding"]   = list(model.encode(df["chunk_text"].tolist(),
                                                convert_to_numpy=True,
                                                show_progress_bar=False))
    df["section_embedding"] = list(model.encode(df["section"].tolist(),
                                                convert_to_numpy=True,
                                                show_progress_bar=False))
    return df


# ── step 3: align one direction ───────────────────────────────────────────────
def align(df_query: pd.DataFrame, df_target: pd.DataFrame) -> list[dict]:
    """
    For every chunk in df_query, find best-matching chunks in df_target.
    Returns a list of dicts with all pairwise chunk_sim values (matched +
    unmatched), so we can inspect the full distribution.
    """
    records = []

    tgt_sections = (df_target
                    .drop_duplicates("section")[["section", "section_embedding"]]
                    .reset_index(drop=True))
    section_embs = np.stack(tgt_sections["section_embedding"].to_numpy())

    for _, q_row in df_query.iterrows():
        q_emb      = np.array(q_row["chunk_embedding"]).reshape(1, -1)
        q_sec_emb  = np.array(q_row["section_embedding"]).reshape(1, -1)

        # section routing
        path_sims      = cosine_similarity(q_sec_emb, section_embs)[0]
        best_sec_idx   = int(np.argmax(path_sims))
        best_path_sim  = float(path_sims[best_sec_idx])
        best_section   = tgt_sections.iloc[best_sec_idx]["section"]

        candidates = (df_target[df_target["section"] == best_section]
                      if best_path_sim >= PATH_SIM_THRESHOLD
                      else df_target)

        tgt_embs   = np.stack(candidates["chunk_embedding"].to_numpy())
        chunk_sims = cosine_similarity(q_emb, tgt_embs)[0]

        # store every (query_chunk, best_candidate) similarity
        best_sim = float(np.max(chunk_sims))
        records.append({
            "query_chunk_id": q_row["chunk_id"],
            "best_chunk_sim": best_sim,
            "best_path_sim":  best_path_sim,
            "all_chunk_sims": chunk_sims.tolist(),   # full row, for histogram
        })

    return records


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading LaBSE model…")
    model = SentenceTransformer("sentence-transformers/LaBSE")
    print("Model loaded.\n")

    # ── collect similarity scores grouped by language pair ───────────────────
    # Structure: { "EN-DE": {"fwd_best": [], "bwd_best": [], "fwd_all": []}, ... }
    from collections import defaultdict
    pair_data   = defaultdict(lambda: {"fwd_best": [], "bwd_best": [], "fwd_all": []})
    article_log = []

    for en_file, other_file, lang_label in ARTICLE_PAIRS:
        en_path    = CACHE_DIR / en_file
        other_path = CACHE_DIR / other_file
        if not en_path.exists() or not other_path.exists():
            print(f"  [skip] {en_file} or {other_file} not in cache")
            continue

        other_lang = lang_label.split("-")[1]   # "DE" or "ZH"
        print(f"Processing [{lang_label}]: {en_file}  ↔  {other_file}")

        en_sents    = json.loads(en_path.read_text(encoding="utf-8"))
        other_sents = json.loads(other_path.read_text(encoding="utf-8"))

        en_chunks    = paragraph_chunk(en_sents)
        other_chunks = paragraph_chunk(other_sents)

        # skip stub articles: other language has < 20% of EN chunks
        ratio = len(other_chunks) / max(len(en_chunks), 1)
        if ratio < 0.20:
            print(f"  [skip — stub] EN:{len(en_chunks)} chunks  "
                  f"{other_lang}:{len(other_chunks)} chunks  "
                  f"ratio={ratio:.0%}  (< 20%)")
            continue

        print(f"  Chunks — EN: {len(en_chunks)}, {other_lang}: {len(other_chunks)}")

        df_en    = embed_chunks(en_chunks,    model)
        df_other = embed_chunks(other_chunks, model)

        fwd = align(df_en,    df_other)   # EN → other
        bwd = align(df_other, df_en)      # other → EN

        pd_ = pair_data[lang_label]
        pd_["fwd_best"].extend([r["best_chunk_sim"] for r in fwd])
        pd_["bwd_best"].extend([r["best_chunk_sim"] for r in bwd])
        pd_["fwd_all"].extend( [s for r in fwd for s in r["all_chunk_sims"]])

        article_log.append({
            "label":      lang_label,
            "article":    en_file.replace("en_", "").replace(".json", ""),
            "en_chunks":  len(en_chunks),
            f"{other_lang.lower()}_chunks": len(other_chunks),
        })
        print(f"  Done.\n")

    if not pair_data:
        print("No data collected — check that cached articles exist.")
        sys.exit(1)

    # ── threshold table — per language pair ───────────────────────────────────
    for label, pd_ in sorted(pair_data.items()):
        fwd, bwd = pd_["fwd_best"], pd_["bwd_best"]
        other_lang = label.split("-")[1]
        print(f"\n{'='*65}")
        print(f"  {label}   (EN→{other_lang}: {len(fwd)} chunks | "
              f"{other_lang}→EN: {len(bwd)} chunks)")
        print(f"{'─'*65}")
        print(f"{'Threshold':>10}  {'EN→X 1:0%':>10}  {'X→EN 1:0%':>10}  "
              f"{'Mean matched sim':>17}")
        print(f"{'─'*65}")
        for t in THRESHOLDS_TO_TEST:
            f_omit  = sum(s < t for s in fwd) / len(fwd)
            b_omit  = sum(s < t for s in bwd) / len(bwd)
            matched = [s for s in fwd + bwd if s >= t]
            mean_s  = np.mean(matched) if matched else float("nan")
            marker  = "  ← current" if t == CURRENT_THRESHOLD else ""
            print(f"{t:>10.2f}  {f_omit:>9.1%}  {b_omit:>9.1%}  "
                  f"{mean_s:>17.4f}{marker}")
        print(f"{'='*65}")

    # ── plot — 2 rows × 3 cols ────────────────────────────────────────────────
    # Row 0: best-match histograms (one per language pair)
    # Row 1: omission-rate curves (one per direction per pair)
    lang_pairs  = sorted(pair_data.keys())   # ["EN-DE", "EN-FR", "EN-IT", "EN-ZH"]
    n_pairs     = len(lang_pairs)
    COLORS_FWD  = {"EN-DE": "#4C72B0", "EN-ZH": "#C44E52",
                   "EN-IT": "#8172B2", "EN-FR": "#937860"}
    COLORS_BWD  = {"EN-DE": "#55A868", "EN-ZH": "#DD8452",
                   "EN-IT": "#CCB974", "EN-FR": "#64B5CD"}

    fig, axes = plt.subplots(2, n_pairs, figsize=(7 * n_pairs, 11))
    if n_pairs == 1:
        axes = axes.reshape(2, 1)

    articles_in_run = [a["article"] for a in article_log]
    fig.suptitle(
        "LaBSE Cosine Similarity — Alignment Threshold Evaluation\n"
        f"Articles: {', '.join(articles_in_run)}",
        fontsize=12, fontweight="bold"
    )

    for col, label in enumerate(lang_pairs):
        pd_       = pair_data[label]
        fwd_best  = pd_["fwd_best"]
        bwd_best  = pd_["bwd_best"]
        other     = label.split("-")[1]

        cal_thr   = CALIBRATED.get(label, CURRENT_THRESHOLD)
        color_fwd = COLORS_FWD.get(label, "#333333")
        color_bwd = COLORS_BWD.get(label, "#888888")

        # ── row 0: overlaid histogram fwd + bwd ──────────────────────────────
        ax = axes[0, col]
        ax.hist(fwd_best, bins=50, color=color_fwd, alpha=0.65,
                label=f"EN→{other}", edgecolor="white", linewidth=0.3)
        ax.hist(bwd_best, bins=50, color=color_bwd, alpha=0.65,
                label=f"{other}→EN", edgecolor="white", linewidth=0.3)
        ax.axvline(CURRENT_THRESHOLD, color="crimson",   linewidth=1.6,
                   linestyle="--", label=f"old global ({CURRENT_THRESHOLD})")
        ax.axvline(cal_thr,           color="darkorange", linewidth=1.8,
                   linestyle=":",   label=f"calibrated ({cal_thr})")
        f_omit = sum(s < cal_thr for s in fwd_best) / len(fwd_best)
        b_omit = sum(s < cal_thr for s in bwd_best) / len(bwd_best)
        ax.set_title(
            f"{label}  |  best-match similarity per chunk\n"
            f"1:0 @ calibrated {cal_thr}: EN→{other} {f_omit:.0%}  |  "
            f"{other}→EN {b_omit:.0%}",
            fontsize=10
        )
        ax.set_xlabel("Cosine similarity", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

        # ── row 1: omission-rate curves ───────────────────────────────────────
        ax2 = axes[1, col]
        fwd_rates = [sum(s < t for s in fwd_best) / len(fwd_best)
                     for t in THRESHOLDS_TO_TEST]
        bwd_rates = [sum(s < t for s in bwd_best) / len(bwd_best)
                     for t in THRESHOLDS_TO_TEST]
        ax2.plot(THRESHOLDS_TO_TEST, fwd_rates, "o-",
                 color=color_fwd, label=f"EN→{other}", linewidth=2)
        ax2.plot(THRESHOLDS_TO_TEST, bwd_rates, "s-",
                 color=color_bwd, label=f"{other}→EN", linewidth=2)
        ax2.axvline(CURRENT_THRESHOLD, color="crimson",    linewidth=1.6,
                    linestyle="--", label=f"old global ({CURRENT_THRESHOLD})")
        ax2.axvline(cal_thr,           color="darkorange",  linewidth=1.8,
                    linestyle=":",  label=f"calibrated ({cal_thr})")
        ax2.set_title(f"{label}  |  1:0 rate vs threshold", fontsize=10)
        ax2.set_xlabel("CHUNK_SIM_THRESHOLD", fontsize=9)
        ax2.set_ylabel("Fraction unmatched", fontsize=9)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax2.legend(fontsize=8)
        ax2.set_xlim(0.25, 0.70)
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
