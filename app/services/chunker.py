import re
from itertools import groupby

import pandas as pd

from app.services.sections import is_navigation_section

# A paragraph whose last character (after closing quotes and brackets) is one
# of these ends a statement. Anything else (no punctuation, ":" or ";") is a
# list item, a definition-list term or a lead-in to a list, so the paragraph
# after it continues the same chunk.
_CLOSING = "\"'”’»）)]」』】"
_SENTENCE_END = re.compile(r"[.!?。！？…]$")


def ends_statement(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.rstrip().rstrip(_CLOSING).rstrip()))


def paragraph_chunk(sentences: list[dict]) -> list[dict]:
    """
    Paragraph-based chunking.

    Groups consecutive sentences by (section, paragraph_index) — each
    paragraph becomes one chunk.  Input order (document order, as produced by
    the fetcher) is preserved, so chunk ids follow the article.  Two kinds of
    paragraph are merged into the previous chunk *of the same section* rather
    than starting a new one:
    - single-sentence paragraphs, which would produce noisy embeddings;
    - any paragraph after one that does not end a statement (see
      ends_statement): Wikipedia lists come out of the API as one line per
      item or term, e.g. "Business trustworthiness system (商务诚信体系)"
      followed by its description, so a list stays in one chunk instead of
      being cut after whichever item happens to have two sentences.

    Navigation sections ("See also", "参见", "References", ...) are skipped,
    also when they come from article caches written before the fetcher
    filtered them.

    Edge cases handled:
    - Paragraph to merge with no previous chunk in its section → kept
      as its own chunk (never merged across a section boundary).
    - Empty input → returns [].

    Args:
        sentences: list of sentence dicts in the fetched format:
                   {language, section, paragraph_index, sentence_index,
                    sentence_text}

    Returns:
        list of chunk dicts:
                   {chunk_id, chunk_text, section, paragraph_index,
                    language, sentence_count, sentences}
        sentences is the list of the chunk's sentence texts, in order.
    """
    if not sentences:
        return []

    chunks: list[dict] = []
    chunk_counter = 0
    continues = False   # the previous paragraph did not end a statement

    # No sorting: the fetcher emits sentences in document order, and sorting by
    # the section *name* would reorder the article alphabetically.
    for (section, para_idx), group in groupby(
        sentences,
        key=lambda x: (x["section"], x["paragraph_index"])
    ):
        sents = list(group)
        if is_navigation_section(section, sents[0]["language"]):
            continue
        sents.sort(key=lambda x: x["sentence_index"])

        lang = sents[0]["language"]
        texts = [s["sentence_text"] for s in sents]
        paragraph_text = " ".join(texts)

        same_section = bool(chunks) and chunks[-1]["section"] == section
        if same_section and (len(sents) == 1 or continues):
            # Single-sentence paragraph, or the next part of a list: absorb
            # into the previous chunk of the same section.
            chunks[-1]["chunk_text"] += " " + paragraph_text
            chunks[-1]["sentence_count"] += len(sents)
            chunks[-1]["sentences"] += texts
        else:
            # Normal paragraph (≥2 sentences), OR a paragraph to merge with
            # no previous chunk in the same section to absorb into.
            chunks.append({
                "chunk_id":        f"{lang}_{chunk_counter:04d}",
                "chunk_text":      paragraph_text,
                "section":         section,
                "paragraph_index": para_idx,
                "language":        lang,
                "sentence_count":  len(sents),
                "sentences":       texts,
            })
            chunk_counter += 1
        continues = not ends_statement(paragraph_text)

    return chunks


def get_chunked_dataframe(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    """
    Wrapper around paragraph_chunk that accepts and returns a DataFrame.

    The `threshold` parameter is kept for API compatibility with main.py
    but is no longer used (paragraph boundaries replace similarity-based
    splitting).

    Returns a DataFrame with one row per chunk and columns:
        chunk_id, chunk_text, section, paragraph_index,
        language, sentence_count, sentences
    """
    sentences = df.to_dict("records")
    chunks = paragraph_chunk(sentences)
    return pd.DataFrame(chunks)
