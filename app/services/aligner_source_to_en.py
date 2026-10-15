import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm import tqdm

from app.services.align_thresholds import LANG_THRESHOLDS, DEFAULT_THRESHOLDS, LAMBDA

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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_thr, path_thr = LANG_THRESHOLDS.get(src_lang, DEFAULT_THRESHOLDS)
    print(
        f"align_language called for {src_lang} | "
        f"chunk_thr={chunk_thr}  path_thr={path_thr}  output→{output_dir}")
    df_src = load_embeddings(src_lang, embed_dir)
    df_tgt = load_embeddings(TARGET_LANG, embed_dir)

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

    out_path = output_dir / f"{src_lang}_to_en.json"
    pd.DataFrame(results).to_json(
        out_path, orient="records", indent=2, force_ascii=False)
    print(
        f"[✓] Saved structured alignment: {out_path} ({len(results)} chunks)")


if __name__ == "__main__":
    for lang in SOURCE_LANGS:
        align_language(lang)
