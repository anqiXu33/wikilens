import pandas as pd
import numpy as np
from pathlib import Path

from app.services.align_thresholds import LAMBDA, method_for
from app.services.align_scoring import align_pairs
import json

# 参数设置
BASE_DIR = Path(__file__).resolve().parents[2]
EMBED_DIR = BASE_DIR / "outputs" / "embeddings"
OUTPUT_DIR = BASE_DIR / "outputs"/"alignments_en_to_source"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EN_LANG = "en"

# Focus languages (Swiss multilingual + Chinese)
TARGET_LANGS = ["de", "zh", "it", "fr"]


def load_embeddings(lang, embed_dir=EMBED_DIR):
    return pd.read_parquet(Path(embed_dir) / f"embedded_{lang}.parquet")


def match_en_to_lang(tgt_lang, embed_dir=EMBED_DIR, output_dir=OUTPUT_DIR):
    """Align every English chunk to tgt_lang chunks; returns how the pairs were chosen."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method = method_for(tgt_lang)
    print(
        f'match_en_to_lang called for {tgt_lang} | '
        f'score={method.score}  thr={method.threshold}  '
        f'path_thr={"off" if method.path_thr is None else method.path_thr}  output→{output_dir}')
    df_en = load_embeddings(EN_LANG, embed_dir).reset_index(drop=True)
    df_src = load_embeddings(tgt_lang, embed_dir).reset_index(drop=True)

    # 路径匹配 + chunk 对得分 (+ LLM 重排), see align_scoring.align_pairs
    aligned = align_pairs(df_en, df_src, EN_LANG, tgt_lang, method)
    scores, chunk_sims = aligned["scores"], aligned["chunk_sims"]
    best_path_sim, matched, labels = aligned["best_path_sim"], aligned["matched"], aligned["labels"]

    results = []
    for i, en_row in df_en.iterrows():
        path_sim = float(best_path_sim[i])
        matched_chunks = []
        for j in map(int, np.flatnonzero(matched[i])):
            src_row = df_src.iloc[j]
            matched_chunks.append({
                "src_chunk_id": src_row["chunk_id"],
                "src_text": src_row["chunk_text"],
                "score": float(round(scores[i, j], 4)),
                "chunk_sim": float(round(chunk_sims[i, j], 4)),
                "path_sim": float(round(path_sim, 4)),
                "combined_score": float(round(scores[i, j] + LAMBDA * path_sim, 4)),
                "src_section": src_row["section"],
                **({"llm_label": labels[(i, j)]} if (i, j) in labels else {}),
            })

        match_type = (
            "1:0" if len(matched_chunks) == 0
            else "1:1" if len(matched_chunks) == 1
            else "1:N"
        )

        results.append({
            "tgt_chunk_id": en_row["chunk_id"],
            "tgt_text": en_row["chunk_text"],
            "from_lang": tgt_lang,
            "match_type": match_type,
            "matched_chunk_ids": [m["src_chunk_id"] for m in matched_chunks],
            "matches": matched_chunks
        })

    out_path = output_dir / f"en_to_{tgt_lang}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved: {out_path} ({len(results)} chunks)")
    return aligned["info"]


if __name__ == "__main__":
    for lang in TARGET_LANGS:
        match_en_to_lang(lang)
