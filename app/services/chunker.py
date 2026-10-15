import pandas as pd
from itertools import groupby


def paragraph_chunk(sentences: list[dict]) -> list[dict]:
    """
    Paragraph-based chunking.

    Groups sentences by (section, paragraph_index) — each paragraph becomes
    one chunk.  Single-sentence paragraphs are merged into the previous chunk
    rather than becoming isolated chunks, which would produce noisy embeddings.

    Edge cases handled:
    - Single-sentence paragraph at the very start of an article (no previous
      chunk yet) → kept as its own chunk instead of being silently dropped.
    - Empty input → returns [].

    Args:
        sentences: list of sentence dicts in the fetched format:
                   {language, section, paragraph_index, sentence_index,
                    sentence_text}

    Returns:
        list of chunk dicts:
                   {chunk_id, chunk_text, section, paragraph_index,
                    language, sentence_count}
    """
    if not sentences:
        return []

    chunks: list[dict] = []
    chunk_counter = 0

    sorted_sentences = sorted(
        sentences,
        key=lambda x: (x["section"], x["paragraph_index"])
    )

    for (section, para_idx), group in groupby(
        sorted_sentences,
        key=lambda x: (x["section"], x["paragraph_index"])
    ):
        sents = list(group)
        sents.sort(key=lambda x: x["sentence_index"])

        lang = sents[0]["language"]
        paragraph_text = " ".join(s["sentence_text"] for s in sents)

        if len(sents) == 1 and chunks:
            # Single-sentence paragraph: absorb into the previous chunk
            # to avoid noisy one-sentence chunks.
            chunks[-1]["chunk_text"] += " " + paragraph_text
            chunks[-1]["sentence_count"] += 1
        else:
            # Normal paragraph (≥2 sentences), OR single-sentence paragraph
            # with no previous chunk to absorb into.
            chunks.append({
                "chunk_id":        f"{lang}_{chunk_counter:04d}",
                "chunk_text":      paragraph_text,
                "section":         section,
                "paragraph_index": para_idx,
                "language":        lang,
                "sentence_count":  len(sents),
            })
            chunk_counter += 1

    return chunks


def get_chunked_dataframe(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    """
    Wrapper around paragraph_chunk that accepts and returns a DataFrame.

    The `threshold` parameter is kept for API compatibility with main.py
    but is no longer used (paragraph boundaries replace similarity-based
    splitting).

    Returns a DataFrame with one row per chunk and columns:
        chunk_id, chunk_text, section, paragraph_index,
        language, sentence_count
    """
    sentences = df.to_dict("records")
    chunks = paragraph_chunk(sentences)
    return pd.DataFrame(chunks)
