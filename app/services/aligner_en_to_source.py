import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm import tqdm
import json

# 参数设置
BASE_DIR = Path(__file__).resolve().parents[2]
EMBED_DIR = BASE_DIR / "outputs" / "embeddings"
OUTPUT_DIR = BASE_DIR / "outputs"/"alignments_en_to_source"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EN_LANG = "en"

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
TARGET_LANGS = ["de", "zh", "it", "fr"]

LAMBDA = 0.3  # 路径加权因子


def load_embeddings(lang, embed_dir=EMBED_DIR):
    return pd.read_parquet(Path(embed_dir) / f"embedded_{lang}.parquet")


def match_en_to_lang(tgt_lang, embed_dir=EMBED_DIR, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_thr, path_thr = LANG_THRESHOLDS.get(tgt_lang, DEFAULT_THRESHOLDS)
    print(
        f'match_en_to_lang called for {tgt_lang} | '
        f'chunk_thr={chunk_thr}  path_thr={path_thr}  output→{output_dir}')
    df_en = load_embeddings(EN_LANG, embed_dir)
    df_src = load_embeddings(tgt_lang, embed_dir)

    results = []

    for _, en_row in tqdm(df_en.iterrows(), total=len(df_en), desc=f"{EN_LANG} → {tgt_lang}"):
        en_chunk_id = en_row["chunk_id"]
        en_text = en_row["chunk_text"]
        en_emb = np.array(en_row["chunk_embedding"]).reshape(1, -1)
        en_section_emb = np.array(en_row["section_embedding"]).reshape(1, -1)

        # 1. 路径匹配
        section_lookup = df_src.drop_duplicates(
            "section").loc[:, ["section", "section_embedding"]]
        section_embs = np.stack(section_lookup["section_embedding"].to_numpy())
        path_sims = cosine_similarity(en_section_emb, section_embs)[0]
        best_section_idx = np.argmax(path_sims)
        best_section = section_lookup.iloc[best_section_idx]["section"]
        best_path_sim = path_sims[best_section_idx]

        if best_path_sim >= path_thr:
            candidate_chunks = df_src[df_src["section"] == best_section]
        else:
            candidate_chunks = df_src

        # 2. chunk similarity 计算
        src_embs = np.stack(candidate_chunks["chunk_embedding"].to_numpy())
        chunk_sims = cosine_similarity(en_emb, src_embs)[0]

        matched_chunks = []
        for idx, sim in enumerate(chunk_sims):
            if sim >= chunk_thr:
                src_row = candidate_chunks.iloc[idx]
                combined_score = sim + LAMBDA * best_path_sim
                matched_chunks.append({
                    "src_chunk_id": src_row["chunk_id"],
                    "src_text": src_row["chunk_text"],
                    "chunk_sim": float(round(sim, 4)),
                    "path_sim": float(round(best_path_sim, 4)),
                    "combined_score": float(round(combined_score, 4)),
                    "src_section": src_row["section"]
                })

        match_type = (
            "1:0" if len(matched_chunks) == 0
            else "1:1" if len(matched_chunks) == 1
            else "1:N"
        )

        results.append({
            "tgt_chunk_id": en_chunk_id,
            "tgt_text": en_text,
            "from_lang": tgt_lang,
            "match_type": match_type,
            "matched_chunk_ids": [m["src_chunk_id"] for m in matched_chunks],
            "matches": matched_chunks
        })

    out_path = output_dir / f"en_to_{tgt_lang}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved: {out_path} ({len(results)} chunks)")


if __name__ == "__main__":
    for lang in TARGET_LANGS:
        match_en_to_lang(lang)
