import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm import tqdm

# Directory settings
BASE_DIR = Path(__file__).resolve().parents[2]
EMBED_DIR = BASE_DIR / "outputs"/"embeddings"
OUTPUT_DIR = BASE_DIR / "outputs"/"alignments_source_to_en"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LANG = "en"

# ── Per-language-pair thresholds ──────────────────────────────────────────────
# Calibrated from LaBSE cosine similarity distribution analysis (eval_alignment_threshold.py).
#
# Key finding: LaBSE similarities are NOT uniform across language pairs.
# Distribution analysis on 3 articles per pair (Climate_change, Adolf_Hitler, Switzerland):
#   EN-DE peaks at 0.25–0.40  →  0.45 discards ~47% of real matches → use 0.30
#   EN-IT peaks at 0.40–0.65  →  0.30 has near-zero 1:0 rate, 0.35 is a better balance
#   EN-FR peaks at 0.50–0.70  →  distribution more right-shifted, 0.40 gives ~4% 1:0
#   EN-ZH peaks at 0.65–0.90  →  0.45 is fine (only ~6% 1:0 rate)
#
# Format: lang_code → (chunk_sim_threshold, path_sim_threshold)
LANG_THRESHOLDS: dict[str, tuple[float, float]] = {
    "de": (0.45, 0.35),   # chunk_thr raised (0.30→0.45) after section-name translation fix
    "it": (0.35, 0.35),   # empirically calibrated ✓
    "fr": (0.40, 0.40),   # empirically calibrated ✓
    "zh": (0.45, 0.45),   # empirically calibrated ✓
}
DEFAULT_THRESHOLDS = (0.35, 0.35)   # safe fallback for unknown languages

# Focus languages (Swiss multilingual + Chinese)
SOURCE_LANGS = ["de", "zh", "it", "fr"]

LAMBDA = 0.3


def load_embeddings(lang):
    return pd.read_parquet(EMBED_DIR / f"embedded_{lang}.parquet")


def align_language(src_lang):
    chunk_thr, path_thr = LANG_THRESHOLDS.get(src_lang, DEFAULT_THRESHOLDS)
    print(
        f"align_language called for {src_lang} | "
        f"chunk_thr={chunk_thr}  path_thr={path_thr}  output→{OUTPUT_DIR}")
    df_src = load_embeddings(src_lang)
    df_tgt = load_embeddings(TARGET_LANG)

    results = []

    for _, src_row in tqdm(df_src.iterrows(), total=len(df_src), desc=f"{src_lang} → {TARGET_LANG}"):
        src_chunk_id = src_row["chunk_id"]
        src_chunk_emb = np.array(src_row["chunk_embedding"]).reshape(1, -1)
        src_path_emb = np.array(src_row["section_embedding"]).reshape(1, -1)

        tgt_sections = df_tgt.drop_duplicates(
            subset="section").loc[:, ["section", "section_embedding"]]
        section_embs = list(tgt_sections["section_embedding"])
        path_sims = cosine_similarity(src_path_emb, section_embs)[0]
        best_idx = np.argmax(path_sims)
        best_path_sim = path_sims[best_idx]
        best_section = tgt_sections.iloc[best_idx]["section"]

        # Section filter or fallback
        if best_path_sim >= path_thr:
            candidate_chunks = df_tgt[df_tgt["section"] == best_section]
        else:
            candidate_chunks = df_tgt

        tgt_chunk_embs = list(candidate_chunks["chunk_embedding"])
        chunk_sims = cosine_similarity(src_chunk_emb, tgt_chunk_embs)[0]

        matched_chunks = []
        for idx, sim in enumerate(chunk_sims):
            if sim >= chunk_thr:
                tgt_row = candidate_chunks.iloc[idx]
                combined_score = sim + LAMBDA * best_path_sim
                matched_chunks.append({
                    "tgt_chunk_id": tgt_row["chunk_id"],
                    "tgt_section": tgt_row["section"],
                    "tgt_text": tgt_row["chunk_text"],
                    "chunk_sim": round(sim, 4),
                    "path_sim": round(best_path_sim, 4),
                    "combined_score": round(combined_score, 4)
                })

        # Determine match type
        if not matched_chunks:
            match_type = "1:0"
        elif len(matched_chunks) == 1:
            match_type = "1:1"
        else:
            match_type = "1:N"

        # Build result with extra fields
        results.append({
            "src_lang": src_lang,
            "tgt_lang": TARGET_LANG,
            "src_chunk_id": src_chunk_id,
            "src_section": src_row["section"],
            "paragraph_index": src_row.get("paragraph_index", None),
            "sentence_index": src_row.get("sentence_index", None),
            "src_text": src_row["chunk_text"],
            "match_type": match_type,
            "matched_chunk_ids": [m["tgt_chunk_id"] for m in matched_chunks],
            "matches": matched_chunks
        })

    out_path = OUTPUT_DIR / f"{src_lang}_to_en.json"
    pd.DataFrame(results).to_json(
        out_path, orient="records", indent=2, force_ascii=False)
    print(
        f"[✓] Saved structured alignment: {out_path} ({len(results)} chunks)")


if __name__ == "__main__":
    for lang in SOURCE_LANGS:
        align_language(lang)
