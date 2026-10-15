import pandas as pd
import numpy as np
from pathlib import Path

from app.services.align_thresholds import LAMBDA, method_for
from app.services.align_scoring import align_pairs

# Directory settings
BASE_DIR = Path(__file__).resolve().parents[2]
EMBED_DIR = BASE_DIR / "outputs"/"embeddings"
OUTPUT_DIR = BASE_DIR / "outputs"/"alignments_source_to_en"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LANG = "en"

# Focus languages (Swiss multilingual + Chinese)
SOURCE_LANGS = ["de", "zh", "it", "fr"]


def load_embeddings(lang, embed_dir=EMBED_DIR):
    return pd.read_parquet(Path(embed_dir) / f"embedded_{lang}.parquet")


def align_language(src_lang, embed_dir=EMBED_DIR, output_dir=OUTPUT_DIR):
    """Align every src_lang chunk to English chunks; returns how the pairs were chosen."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method = method_for(src_lang)
    print(
        f"align_language called for {src_lang} | "
        f"score={method.score}  thr={method.threshold}  "
        f"path_thr={'off' if method.path_thr is None else method.path_thr}  output→{output_dir}")
    df_src = load_embeddings(src_lang, embed_dir).reset_index(drop=True)
    df_tgt = load_embeddings(TARGET_LANG, embed_dir).reset_index(drop=True)

    aligned = align_pairs(df_src, df_tgt, src_lang, TARGET_LANG, method)
    scores, chunk_sims = aligned["scores"], aligned["chunk_sims"]
    best_path_sim, matched, labels = aligned["best_path_sim"], aligned["matched"], aligned["labels"]

    results = []
    for i, src_row in df_src.iterrows():
        path_sim = float(best_path_sim[i])
        matched_chunks = []
        for j in map(int, np.flatnonzero(matched[i])):
            tgt_row = df_tgt.iloc[j]
            matched_chunks.append({
                "tgt_chunk_id": tgt_row["chunk_id"],
                "tgt_section": tgt_row["section"],
                "tgt_text": tgt_row["chunk_text"],
                "score": round(float(scores[i, j]), 4),
                "chunk_sim": round(float(chunk_sims[i, j]), 4),
                "path_sim": round(path_sim, 4),
                "combined_score": round(float(scores[i, j]) + LAMBDA * path_sim, 4),
                **({"llm_label": labels[(i, j)]} if (i, j) in labels else {}),
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
            "src_chunk_id": src_row["chunk_id"],
            "src_section": src_row["section"],
            "paragraph_index": src_row.get("paragraph_index", None),
            "sentence_index": src_row.get("sentence_index", None),
            "src_text": src_row["chunk_text"],
            "match_type": match_type,
            "matched_chunk_ids": [m["tgt_chunk_id"] for m in matched_chunks],
            "matches": matched_chunks
        })

    out_path = output_dir / f"{src_lang}_to_en.json"
    pd.DataFrame(results).to_json(
        out_path, orient="records", indent=2, force_ascii=False)
    print(
        f"[✓] Saved structured alignment: {out_path} ({len(results)} chunks)")
    return aligned["info"]


if __name__ == "__main__":
    for lang in SOURCE_LANGS:
        align_language(lang)
