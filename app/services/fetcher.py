from nltk.tokenize import sent_tokenize
import wikipediaapi
import pandas as pd
import re
from pathlib import Path
from urllib.parse import urlparse, unquote
import requests
import nltk

from app.services.sections import is_navigation_heading, is_navigation_section

# The German spaCy model is pinned to an exact Hugging Face commit (spaCy 3.7
# model, matching the pinned spacy==3.7.x). Like the NLTK data, it is fetched
# and loaded on first use, not on import.
SPACY_DE_REPO = "spacy/de_core_news_sm"
SPACY_DE_REVISION = "3e486021598de8be095c787aae0211b7f84c42d1"

_nlp_de = None


def get_nlp_de():
    global _nlp_de
    if _nlp_de is None:
        import spacy
        from huggingface_hub import snapshot_download
        path = snapshot_download(SPACY_DE_REPO, revision=SPACY_DE_REVISION,
                                 ignore_patterns=["*.whl"])
        _nlp_de = spacy.load(path)
    return _nlp_de


_punkt_ready = False


def _ensure_punkt():
    global _punkt_ready
    if not _punkt_ready:
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            if not nltk.download("punkt_tab", quiet=True):
                raise RuntimeError(
                    "NLTK 'punkt_tab' data is missing and could not be downloaded. "
                    "Run: python -m nltk.downloader punkt_tab")
        _punkt_ready = True


# Paths
CACHE_DIR = Path("app/cached_articles")
OUTPUT_DIR = Path("outputs/fetched")

# Navigation sections ("See also", "参见", "References", ...) are not prose:
# their headings are listed in app/services/sections.py.


def clean_sentence_text(text: str) -> str:
    text = re.sub(r"^[\s•\-\–\—\*·●◦]+", "", text)
    text = re.sub(r"^[(（]?\d{1,2}[\.\、)]\s*", "", text)
    text = re.sub(r"^[一二三四五六七八九十]+\、", "", text)
    return text.strip()


def clean_text(text):
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text, lang):
    if lang == "zh":
        return re.split(r"(?<=[。！？])(?=[^\n])", text)
    elif lang == "de":
        return [s.text.strip() for s in get_nlp_de()(text).sents if s.text.strip()]
    _ensure_punkt()
    if lang == "it":
        # NLTK punkt has a built-in Italian model — no extra download needed
        return sent_tokenize(text, language="italian")
    if lang == "fr":
        # NLTK punkt has a built-in French model — no extra download needed
        return sent_tokenize(text, language="french")
    else:
        # fallback: English punkt
        return sent_tokenize(text)


def extract_sentences_by_section(page, lang):
    data = []

    def walk(sections, parent_path=""):
        for sec in sections:
            title_raw = sec.title.strip()
            if is_navigation_heading(title_raw, lang):
                continue

            full_path = f"{parent_path} > {title_raw}" if parent_path else title_raw

            paragraphs = [p.strip() for p in sec.text.split("\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                sents = split_sentences(para, lang)
                for j, sent in enumerate(sents):
                    cleaned = clean_sentence_text(sent)
                    if len(cleaned) < 2:
                        continue
                    data.append(
                        {
                            "language": lang,
                            "section": full_path,
                            "paragraph_index": i,
                            "sentence_index": j,
                            "sentence_text": cleaned,
                        }
                    )

            walk(sec.sections, full_path)

    # Handle intro — split into paragraphs like any other section, otherwise
    # the whole lead ends up as a single chunk
    intro_title = "(Lead)"
    intro_paragraphs = [p.strip() for p in (page.summary or "").split("\n") if p.strip()]
    for i, para in enumerate(intro_paragraphs):
        for j, sent in enumerate(split_sentences(para, lang)):
            cleaned = clean_sentence_text(sent)
            if len(cleaned) < 2:
                continue
            data.append(
                {
                    "language": lang,
                    "section": intro_title,
                    "paragraph_index": i,
                    "sentence_index": j,
                    "sentence_text": cleaned,
                }
            )

    walk(page.sections)
    return data


# Bump whenever extract_sentences_by_section changes what it produces, so caches
# written by older code are re-fetched instead of silently reused.
#   v1 (unversioned files): lead labelled "Intro", whole lead as paragraph 0
#   v2: lead labelled "(Lead)" and split into paragraphs
CACHE_VERSION = 2


def get_cache_path(title, lang):
    return CACHE_DIR / f"{lang}_{title.replace(' ', '_')}.v{CACHE_VERSION}.json"


def save_to_cache(df, title, lang):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(
        get_cache_path(title, lang), orient="records", force_ascii=False, indent=2
    )


def load_from_cache(title, lang):
    path = get_cache_path(title, lang)
    if not path.exists():
        return None
    df = pd.read_json(path)
    if not df.empty:
        # Caches written before a heading was added to NAVIGATION_HEADINGS
        # still contain that section.
        df = df[~df["section"].map(lambda s: is_navigation_section(s, lang))].reset_index(drop=True)
    return df


def fetch_and_prepare_article(title, lang="en"):
    cached = load_from_cache(title, lang)
    if cached is not None:
        return cached

    wiki = wikipediaapi.Wikipedia(
        language=lang, user_agent="WikiLens (University of Zurich)"
    )
    page = wiki.page(title)
    if not page.exists():
        return pd.DataFrame()

    # wikipediaapi follows redirects and then reports the target's title.
    # A Wikidata sitelink that is a redirect (e.g. en "Riley Elf" -> "Mini")
    # means this edition has no dedicated article, so comparing against the
    # target would compare two different topics.
    requested = title.replace("_", " ")
    resolved = page.title.replace("_", " ")
    if resolved[:1].upper() + resolved[1:] != requested[:1].upper() + requested[1:]:
        print(f"[!] {lang}: '{requested}' redirects to '{resolved}' — "
              f"no dedicated article in this edition, skipping.")
        return pd.DataFrame()

    data = extract_sentences_by_section(page, lang)
    df = pd.DataFrame(data)
    save_to_cache(df, title, lang)
    return df


def extract_title_from_url(url: str) -> str:
    return unquote(urlparse(url).path.split("/wiki/")[-1])


_HEADERS = {
    "User-Agent": "WikiLens/1.0 (bias analysis research tool; https://github.com/wikilens) python-requests"
}


def _get_json(url: str, params: dict = None, retries: int = 4, timeout: int = 15) -> dict:
    """GET a JSON endpoint with proper User-Agent, timeout, and automatic retry.
    On HTTP 429 the server's Retry-After (capped at 60 s) is honoured."""
    import time
    for attempt in range(1, retries + 1):
        r = None
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            if r is not None and r.status_code == 429:
                try:
                    wait = max(wait, min(int(r.headers.get("Retry-After", 0)), 60))
                except ValueError:
                    pass
            print(f"  [retry {attempt}/{retries}] {e} — retrying in {wait}s")
            time.sleep(wait)


class TitleNotFoundError(LookupError):
    """Wikidata has no item for this title in this language edition."""


def get_qid_from_title(title: str, lang: str = "en") -> str:
    """The Wikidata QID of the {lang}wiki article `title`.

    Wikidata only knows articles by their exact title and does not follow
    redirects: for a missing title it answers with the placeholder id "-1"
    (marked "missing"). That is reported as TitleNotFoundError at once,
    rather than requesting entity "-1" and retrying a request that cannot
    succeed."""
    r = _get_json(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbgetentities",
            "sites": f"{lang}wiki",
            "titles": title,
            "format": "json",
        },
    )
    entities = r.get("entities") or {}
    qid = next(iter(entities), None)
    if qid is None or qid == "-1" or "missing" in entities.get(qid, {}):
        raise TitleNotFoundError(
            f"No Wikidata item for the {lang} Wikipedia article '{title}'. The title may be a "
            f"redirect or misspelt: use the article's exact title, the one shown at the top of "
            f"the Wikipedia page after any redirect (or copy the URL from that page).")
    return qid


def get_langlinks_from_qid(qid: str) -> dict:
    r = _get_json(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
    entity = r["entities"][qid]
    return {
        k.replace("wiki", ""): v["title"]
        for k, v in entity.get("sitelinks", {}).items()
    }


def fetch_articles_in_languages(lang_title_map: dict, output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for lang, title in lang_title_map.items():
        df = fetch_and_prepare_article(title, lang)
        if not df.empty:
            results[lang] = df
            out_path = output_dir / f"fetched_{lang}.json"
            df.to_json(out_path, orient="records", force_ascii=False, indent=2)
            print(f"[✓] {lang}: saved to {out_path}")
        else:
            print(f"[!] {lang}: page not found or empty.")
    return results


def fetch_multilang_from_url(url: str, selected_langs: list, output_dir: Path = OUTPUT_DIR) -> dict:
    title = extract_title_from_url(url)
    print(f"Extracted title: {title}")
    qid = get_qid_from_title(title)
    print(f"Got QID: {qid}")
    lang_map = get_langlinks_from_qid(qid)
    print(f"Available languages from Wikidata: {list(lang_map.keys())}")

    selected_map = {lang: lang_map[lang]
                    for lang in selected_langs if lang in lang_map}
    print(f"Languages to fetch based on selection: {selected_map}")

    if not selected_map:
        print("[!] No languages to fetch after filtering; returning empty dict.")
    return fetch_articles_in_languages(selected_map, output_dir)
