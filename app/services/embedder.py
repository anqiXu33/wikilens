import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
from tqdm import tqdm
import traceback
from deep_translator import GoogleTranslator

model = SentenceTransformer("sentence-transformers/LaBSE")

# Wikipedia language code -> Google Translate code where they differ.
# deep-translator rejects plain "zh" (LanguageNotSupportedException).
TRANSLATOR_LANG = {"zh": "zh-CN"}


def normalise_section(name: str, lang: str) -> str:
    """Translate a section name to English so all section embeddings live
    in the same semantic space, making cross-lingual path_sim reliable."""
    if lang == "en" or not name or name == "(Lead)":
        return name
    try:
        return GoogleTranslator(source=TRANSLATOR_LANG.get(lang, lang), target="en").translate(name)
    except Exception:
        return name  # fall back to original on translation error

INPUT_DIR = Path("outputs/chunked")
OUTPUT_DIR = Path("outputs/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ["en", "de", "fr", "it", "zh"]


def embed_texts(texts):
    print(f"  [>] Embedding {len(texts)} texts...")
    try:
        embeddings = model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True)
        print(f"  [✓] Successfully embedded {len(texts)} texts")
        return embeddings
    except Exception as e:
        print(f"  [!!] Error during embedding: {e}")
        traceback.print_exc()
        raise


def embed_language(lang):
    in_path = INPUT_DIR / f"chunked_{lang}.json"
    out_path = OUTPUT_DIR / f"embedded_{lang}.parquet"
    if not in_path.exists():
        print(f"[!] Skipped {lang}: {in_path} not found")
        return

    print(f"[*] Processing language '{lang}' from {in_path}")
    try:
        df = pd.read_json(in_path)
    except Exception as e:
        print(f"[!!] Failed to read JSON for {lang}: {e}")
        traceback.print_exc()
        return

    print(f"[*] Embedding {lang}: {len(df)} rows...")

    try:
        chunk_texts = df["chunk_text"].tolist()
        raw_section_texts = df["section"].tolist()
        if lang != "en":
            print(f"[*] Translating {len(set(raw_section_texts))} unique section names → EN ...")
            cache: dict[str, str] = {}
            section_texts = []
            for s in raw_section_texts:
                if s not in cache:
                    cache[s] = normalise_section(s, lang)
                section_texts.append(cache[s])
            print(f"[*] Section translation done.")
        else:
            section_texts = raw_section_texts
    except KeyError as e:
        print(f"[!!] Missing expected column in {lang} dataframe: {e}")
        traceback.print_exc()
        return

    try:
        df["chunk_embedding"] = list(embed_texts(chunk_texts))
        df["section_embedding"] = list(embed_texts(section_texts))
    except Exception as e:
        print(f"[!!] Error during embedding columns for {lang}: {e}")
        traceback.print_exc()
        return

    try:
        df.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"[✓] Saved embedded vectors to {out_path}")
    except Exception as e:
        print(f"[!!] Failed to save parquet file for {lang}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    for lang in LANGS:
        embed_language(lang)
