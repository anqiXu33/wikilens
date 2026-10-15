import json
import threading
import time
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


# Google Translate (the free endpoint deep-translator scrapes) allows about
# 5 requests per second and answers 429 beyond that. Names are sent in one
# newline-joined request per language where possible; otherwise one at a
# time, spaced MIN_REQUEST_INTERVAL apart, with exponential backoff on 429.
MIN_REQUEST_INTERVAL = 0.5          # seconds between requests (≤ 2 per second)
RETRY_DELAYS = (2, 4, 8, 16)        # backoff after 429 / server errors
BATCH_MAX_CHARS = 4500              # deep-translator rejects inputs over 5000
_last_request = 0.0


def _is_retryable(e: Exception) -> bool:
    from deep_translator.exceptions import RequestError, TooManyRequests
    return isinstance(e, (TooManyRequests, RequestError))


def _google_translate(text: str, lang: str) -> str:
    """One paced Google Translate request, retried with backoff on rate limits."""
    global _last_request
    translator = GoogleTranslator(source=TRANSLATOR_LANG.get(lang, lang), target="en")
    for attempt in range(len(RETRY_DELAYS) + 1):
        wait = _last_request + MIN_REQUEST_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            return translator.translate(text)
        except Exception as e:
            if attempt == len(RETRY_DELAYS) or not _is_retryable(e):
                raise
            delay = RETRY_DELAYS[attempt]
            print(f"[!] Google Translate rate-limited or unavailable ({type(e).__name__}); retrying in {delay}s")
            time.sleep(delay)


def _translate_batch(names: list[str], lang: str) -> list[str] | None:
    """Translate names in as few requests as possible by joining them with
    newlines. Returns None if the reply doesn't split back into exactly one
    line per name (the caller then translates them one by one)."""
    out: list[str] = []
    batch: list[str] = []
    batches = []
    for name in names:
        if batch and sum(len(n) + 1 for n in batch) + len(name) > BATCH_MAX_CHARS:
            batches.append(batch)
            batch = []
        batch.append(name)
    if batch:
        batches.append(batch)
    for batch in batches:
        translated = _google_translate("\n".join(batch), lang) or ""
        lines = [l.strip() for l in translated.split("\n") if l.strip()]
        if len(lines) != len(batch):
            return None
        out.extend(lines)
    return out


def translate_section_names(names, lang: str) -> dict[str, str]:
    """Translate section names to English so all section embeddings live in
    the same semantic space, making cross-lingual path_sim reliable.

    Returns {name: english} for every name. A name that can't be translated
    maps to itself (with a warning) and is not cached, so a later run retries
    it. Call this once per run and reuse the result.
    """
    names = list(dict.fromkeys(names))
    result = {n: n for n in names}
    if lang == "en":
        return result
    with _translation_lock:
        cache = _load_translation_cache().get(lang, {})
        todo = [n for n in names if n and n != "(Lead)" and n not in cache]
        result.update({n: cache[n] for n in names if n in cache})
    if not todo:
        return result

    print(f"[*] Translating {len(todo)} section names ({lang} → en) ...")
    translated: dict[str, str] = {}
    error = None
    batch = None
    try:
        batch = _translate_batch(todo, lang) if len(todo) > 1 else None
    except Exception as e:
        error = e
    if batch is not None:
        translated = dict(zip(todo, batch))
    elif error is None or not _is_retryable(error):
        # Batch reply didn't split cleanly (or a single name): one at a time
        for name in todo:
            try:
                t = _google_translate(name, lang)
            except Exception as e:
                error = e
                if _is_retryable(e):
                    break           # still rate-limited after backoff: stop asking
                continue
            if t:
                translated[name] = t
    failed = [n for n in todo if n not in translated]
    if failed:
        # Keep the original names, but say so: untranslated section names
        # weaken section routing in the aligner.
        reason = f" — {type(error).__name__}: {error}" if error else ""
        print(f"[!] section-name translation failed ({lang}) for {len(failed)} of {len(todo)} "
              f"names, kept untranslated: {failed}{reason}")

    if translated:
        result.update(translated)
        with _translation_lock:
            _load_translation_cache().setdefault(lang, {}).update(translated)
            _save_translation_cache()
    return result

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


def embed_language(lang, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, section_names=None):
    """Embed chunks and their (English) section names for one language.

    section_names: {original: english} from translate_section_names(). The
    pipeline translates once per run and passes the result here and to the
    scoring step; if omitted, names are translated (or read from cache) here.
    """
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
        if section_names is None:
            section_names = translate_section_names(raw_section_texts, lang)
        section_texts = [section_names.get(sec, sec) for sec in raw_section_texts]
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
