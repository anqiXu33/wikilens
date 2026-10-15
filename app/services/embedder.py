import json
import threading
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import traceback
from deep_translator import GoogleTranslator

# LaBSE is pinned to an exact Hugging Face commit so every install embeds with
# the same weights. The model (~1.8 GB) is loaded on first use, not on import.
LABSE_MODEL = "sentence-transformers/LaBSE"
LABSE_REVISION = "836121a0533e5664b21c7aacc5d22951f2b8b25b"

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(LABSE_MODEL, revision=LABSE_REVISION)
    return _model


# Wikipedia language code -> Google Translate code where they differ.
# deep-translator rejects plain "zh" (LanguageNotSupportedException).
TRANSLATOR_LANG = {"zh": "zh-CN"}

# Section-name translations are cached on disk ({lang: {name: english}}) and
# committed, so re-running an article doesn't call Google Translate again and
# gets the same section names (and therefore the same section embeddings).
# Failed translations are not cached.
TRANSLATION_CACHE_PATH = Path(__file__).resolve().parents[1] / "cached_translations" / "section_names.json"
_translation_cache: dict[str, dict[str, str]] | None = None
_translation_lock = threading.Lock()


def _load_translation_cache() -> dict[str, dict[str, str]]:
    global _translation_cache
    if _translation_cache is None:
        if TRANSLATION_CACHE_PATH.exists():
            _translation_cache = json.loads(TRANSLATION_CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _translation_cache = {}
    return _translation_cache


def _save_translation_cache() -> None:
    TRANSLATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = TRANSLATION_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(_translation_cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(TRANSLATION_CACHE_PATH)


def normalise_section(name: str, lang: str) -> str:
    """Translate a section name to English so all section embeddings live
    in the same semantic space, making cross-lingual path_sim reliable."""
    if lang == "en" or not name or name == "(Lead)":
        return name
    with _translation_lock:
        cached = _load_translation_cache().get(lang, {}).get(name)
    if cached is not None:
        return cached
    try:
        translated = GoogleTranslator(source=TRANSLATOR_LANG.get(lang, lang), target="en").translate(name)
    except Exception as e:
        # Keep the original name, but say so: an untranslated section name
        # weakens section routing in the aligner.
        print(f"[!] section-name translation failed ({lang}): {name!r} kept untranslated — {e}")
        return name
    if not translated:
        return name
    with _translation_lock:
        _load_translation_cache().setdefault(lang, {})[name] = translated
        _save_translation_cache()
    return translated

INPUT_DIR = Path("outputs/chunked")
OUTPUT_DIR = Path("outputs/embeddings")

LANGS = ["en", "de", "fr", "it", "zh"]


def embed_texts(texts):
    print(f"  [>] Embedding {len(texts)} texts...")
    try:
        embeddings = get_model().encode(
            texts, convert_to_numpy=True, show_progress_bar=True)
        print(f"  [✓] Successfully embedded {len(texts)} texts")
        return embeddings
    except Exception as e:
        print(f"  [!!] Error during embedding: {e}")
        traceback.print_exc()
        raise


def embed_language(lang, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    in_path = Path(input_dir) / f"chunked_{lang}.json"
    out_path = output_dir / f"embedded_{lang}.parquet"
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
