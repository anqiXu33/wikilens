import json
import os
import threading
import time
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import traceback
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

load_dotenv()

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


def _google_translate_names(names: list[str], lang: str) -> tuple[dict[str, str], Exception | None]:
    """Translate names with Google Translate. Returns ({name: english} for
    the names that succeeded, the last error or None)."""
    translated: dict[str, str] = {}
    error = None
    batch = None
    try:
        batch = _translate_batch(names, lang) if len(names) > 1 else None
    except Exception as e:
        error = e
    if batch is not None:
        translated = dict(zip(names, batch))
    elif error is None or not _is_retryable(error):
        # Batch reply didn't split cleanly (or a single name): one at a time
        for name in names:
            try:
                t = _google_translate(name, lang)
            except Exception as e:
                error = e
                if _is_retryable(e):
                    break           # still rate-limited after backoff: stop asking
                continue
            if t:
                translated[name] = t
    return translated, error


# ── Section-name translation with the scoring LLM ────────────────────────────
# TRANSLATION_BACKEND selects how section names are translated:
#   auto   — the scoring LLM if SCORING_BACKEND is openai or ollama and
#            configured, else Google Translate; Google also takes over if the
#            LLM request fails outright (default)
#   llm    — the scoring LLM only
#   google — Google Translate only
#   none   — no translation: section names stay in the original language
TRANSLATION_BACKENDS = ("auto", "llm", "google", "none")
LLM_TRANSLATION_BACKENDS = ("openai", "ollama")

LANGUAGE_NAMES = {
    "de": "German", "fr": "French", "it": "Italian", "zh": "Chinese",
    "es": "Spanish", "eu": "Basque", "ja": "Japanese", "ru": "Russian",
}

SECTION_SEP = " > "     # chunker joins nested headings: "Anbau > Züchtung"


def translation_backend() -> str:
    value = (os.getenv("TRANSLATION_BACKEND", "auto").strip().lower() or "auto")
    if value not in TRANSLATION_BACKENDS:
        raise RuntimeError(
            f"Unknown TRANSLATION_BACKEND={value!r}; expected one of {', '.join(TRANSLATION_BACKENDS)}")
    return value


def _llm_unavailable_reason() -> str | None:
    """None if the scoring LLM can translate section names, else why not."""
    try:
        from app.services import scorer
    except RuntimeError as e:          # e.g. unknown SCORING_BACKEND
        return str(e)
    if scorer.SCORING_BACKEND not in LLM_TRANSLATION_BACKENDS:
        return (f"SCORING_BACKEND={scorer.SCORING_BACKEND} is not one of "
                f"{', '.join(LLM_TRANSLATION_BACKENDS)}")
    try:
        scorer.check_backend_config()
    except RuntimeError as e:
        return str(e)
    return None


def check_translation_config() -> None:
    """Fail before any request if TRANSLATION_BACKEND is invalid, or is
    "llm" without a usable LLM backend."""
    if translation_backend() == "llm":
        reason = _llm_unavailable_reason()
        if reason:
            raise RuntimeError(f"TRANSLATION_BACKEND=llm but no LLM backend is usable: {reason}")


def _llm_messages(names: list[str], lang: str) -> list[dict]:
    language = LANGUAGE_NAMES.get(lang, lang)
    system = (
        f"You translate Wikipedia section headings from {language} to English. "
        "Reply with a single JSON object and nothing else: each key is one heading "
        "exactly as given, its value is the English translation. Translate every "
        f"heading. Keep the \"{SECTION_SEP.strip()}\" separators between nested headings "
        "and translate each part. Use short, natural English headings."
    )
    user = json.dumps(names, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_json_object(text: str) -> dict:
    """The first {...} in the model's reply (models sometimes wrap JSON in
    code fences or add a sentence before it)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    return data


def _validate_llm_translations(names: list[str], data: dict, lang: str) -> dict[str, str]:
    """Keep the valid entries of the model's {original: english} reply and
    warn about the rest (those names keep their original text)."""
    if len(data) != len(names):
        print(f"[!] LLM section-name translation ({lang}): expected {len(names)} entries, got {len(data)}")
    # Match keys exactly, else ignoring surrounding whitespace and case
    loose = {}
    for k, v in data.items():
        if isinstance(k, str):
            loose.setdefault(k.strip().casefold(), v)
    translated, bad = {}, {}
    for name in names:
        value = data[name] if name in data else loose.get(name.strip().casefold())
        if not isinstance(value, str) or not value.strip():
            bad[name] = "missing" if value is None else "empty or not a string"
            continue
        value = " ".join(value.split())
        if name.count(SECTION_SEP) != value.count(SECTION_SEP):
            bad[name] = f"nesting changed: {value!r}"
            continue
        translated[name] = value
    if bad:
        print(f"[!] LLM section-name translation ({lang}): {len(bad)} invalid entries, "
              f"kept untranslated: {bad}")
    return translated


def _llm_translate_names(names: list[str], lang: str) -> dict[str, str]:
    """Translate all names in one request to the scoring LLM. Raises if the
    request fails or the reply isn't a JSON object; invalid entries are
    left out of the result."""
    from app.services import scorer
    response = scorer.inference(_llm_messages(names, lang))
    data = _parse_json_object(scorer._extract_generated_text(response))
    return _validate_llm_translations(names, data, lang)


def translate_section_names(names, lang: str, info: dict | None = None) -> dict[str, str]:
    """Translate section names to English so all section embeddings live in
    the same semantic space, making cross-lingual path_sim reliable.

    Returns {name: english} for every name. A name that can't be translated
    maps to itself (with a warning) and is not cached, so a later run retries
    it. Call this once per run and reuse the result.

    info, if given, is filled with how the names were translated (recorded
    in the run's meta.json): the TRANSLATION_BACKEND setting, how many names
    came from each source (cache / llm / google), the LLM used, and the
    names left untranslated.
    """
    names = list(dict.fromkeys(names))
    result = {n: n for n in names}
    if lang == "en":
        return result
    backend = translation_backend()
    info = info if info is not None else {}
    info.update({"translation_backend": backend, "sources": {}, "untranslated": []})
    todo = [n for n in names if n and n != "(Lead)"]
    if backend == "none":
        info["method"] = "none"
        info["untranslated"] = todo
        return result

    with _translation_lock:
        cache = _load_translation_cache().get(lang, {})
        hits = {n: cache[n] for n in todo if n in cache}
    result.update(hits)
    todo = [n for n in todo if n not in hits]
    if hits:
        info["sources"]["cache"] = len(hits)

    translated: dict[str, str] = {}
    error = None
    use_google = backend == "google"
    if todo and backend in ("auto", "llm"):
        reason = _llm_unavailable_reason()
        if reason is None:
            from app.services import scorer
            info["llm"] = scorer.backend_info()
            print(f"[*] Translating {len(todo)} section names ({lang} → en) with "
                  f"{info['llm']['backend']} ({info['llm']['model']}) ...")
            try:
                translated = _llm_translate_names(todo, lang)
                if translated:
                    info["sources"]["llm"] = len(translated)
            except Exception as e:
                error = e
                print(f"[!] LLM section-name translation failed ({lang}): {type(e).__name__}: {e}")
                use_google = backend == "auto"
        elif backend == "auto":
            print(f"[*] No LLM backend for section-name translation ({reason}); using Google Translate")
            use_google = True
        else:
            error = RuntimeError(reason)

    if todo and use_google:
        print(f"[*] Translating {len(todo)} section names ({lang} → en) with Google Translate ...")
        translated, error = _google_translate_names(todo, lang)
        if translated:
            info["sources"]["google"] = len(translated)

    failed = [n for n in todo if n not in translated]
    if failed:
        # Keep the original names, but say so: untranslated section names
        # weaken section routing in the aligner.
        reason = f" — {type(error).__name__}: {error}" if error else ""
        print(f"[!] section-name translation failed ({lang}) for {len(failed)} of {len(todo)} "
              f"names, kept untranslated: {failed}{reason}")
    info["untranslated"] = failed
    info["method"] = "+".join(info["sources"]) or "none"

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
