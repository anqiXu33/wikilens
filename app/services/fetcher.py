from nltk.tokenize import sent_tokenize
import wikipediaapi
import pandas as pd
import re
from pathlib import Path
from urllib.parse import urlparse, unquote
import requests
import nltk

import spacy

nltk.download("punkt")
nltk.download("punkt_tab")

# Models
nlp_de = spacy.load("de_core_news_sm")

# Paths
CACHE_DIR = Path("app/cached_articles")
OUTPUT_DIR = Path("outputs/fetched")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Skip headings
EXCLUDE_HEADINGS = {
    "en": [
        "see also",
        "references",
        "literature",
        "notes",
        "external links",
        "sources",
        "bibliography",
        "footnotes",
    ],
    "de": [
        "literatur",
        "einzelnachweise",
        "weblinks",
        "siehe auch",
        "quellen",
        "anmerkungen",
        "belege",
    ],
    "zh": ["参考资料", "參考資料", "參見", "注释", "外部链接", "外部連結"],
    "it": [
        "vedi anche",
        "note",
        "bibliografia",
        "collegamenti esterni",
        "fonti",
        "riferimenti",
        "altri progetti",
    ],
    "fr": [
        "voir aussi",
        "notes",
        "références",
        "liens externes",
        "bibliographie",
        "sources",
        "annexes",
        "notes et références",
    ],
}


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
        return [s.text.strip() for s in nlp_de(text).sents if s.text.strip()]
    elif lang == "it":
        # NLTK punkt has a built-in Italian model — no extra download needed
        return sent_tokenize(text, language="italian")
    elif lang == "fr":
        # NLTK punkt has a built-in French model — no extra download needed
        return sent_tokenize(text, language="french")
    else:
        # fallback: English punkt
        return sent_tokenize(text)


def extract_sentences_by_section(page, lang):
    data = []
    exclude = [h.lower() for h in EXCLUDE_HEADINGS.get(lang, [])]

    def walk(sections, parent_path=""):
        for sec in sections:
            title_raw = sec.title.strip()
            title_lc = title_raw.lower()
            if title_lc in exclude:
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

    # Handle intro
    intro_title = "(Lead)"
    intro_sents = split_sentences(page.summary or "", lang)
    for j, sent in enumerate(intro_sents):
        cleaned = clean_sentence_text(sent)
        if len(cleaned) < 2:
            continue
        data.append(
            {
                "language": lang,
                "section": intro_title,
                "paragraph_index": 0,
                "sentence_index": j,
                "sentence_text": cleaned,
            }
        )

    walk(page.sections)
    return data


def get_cache_path(title, lang):
    return CACHE_DIR / f"{lang}_{title.replace(' ', '_')}.json"


def save_to_cache(df, title, lang):
    df.to_json(
        get_cache_path(title, lang), orient="records", force_ascii=False, indent=2
    )


def load_from_cache(title, lang):
    path = get_cache_path(title, lang)
    return pd.read_json(path) if path.exists() else None


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

    data = extract_sentences_by_section(page, lang)
    df = pd.DataFrame(data)
    save_to_cache(df, title, lang)
    return df


def extract_title_from_url(url: str) -> str:
    return unquote(urlparse(url).path.split("/wiki/")[-1])


_HEADERS = {
    "User-Agent": "WikiLens/1.0 (bias analysis research tool; https://github.com/wikilens) python-requests"
}


def _get_json(url: str, params: dict = None, retries: int = 3, timeout: int = 15) -> dict:
    """GET a JSON endpoint with proper User-Agent, timeout, and automatic retry."""
    import time
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{retries}] {e} — retrying in {wait}s")
            time.sleep(wait)


def get_qid_from_title(title: str, lang: str = "en") -> str:
    r = _get_json(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbgetentities",
            "sites": f"{lang}wiki",
            "titles": title,
            "format": "json",
        },
    )
    return next(iter(r.get("entities", {}).keys()))


def get_langlinks_from_qid(qid: str) -> dict:
    r = _get_json(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
    entity = r["entities"][qid]
    return {
        k.replace("wiki", ""): v["title"]
        for k, v in entity.get("sitelinks", {}).items()
    }


def fetch_articles_in_languages(lang_title_map: dict) -> dict:
    results = {}
    for lang, title in lang_title_map.items():
        df = fetch_and_prepare_article(title, lang)
        if not df.empty:
            results[lang] = df
            out_path = OUTPUT_DIR / f"fetched_{lang}.json"
            df.to_json(out_path, orient="records", force_ascii=False, indent=2)
            print(f"[✓] {lang}: saved to {out_path}")
        else:
            print(f"[!] {lang}: page not found or empty.")
    return results


def fetch_multilang_from_url(url: str, selected_langs: list) -> dict:
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
    return fetch_articles_in_languages(selected_map)
