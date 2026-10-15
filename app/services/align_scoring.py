"""Score matrices shared by both aligners.

Each function works on the embedding DataFrames the embedder writes
(outputs/.../embedded_<lang>.parquet), query rows Q against target rows T.
"""
import numpy as np
import pandas as pd

from app.services.align_thresholds import AlignMethod


def unit(rows) -> np.ndarray:
    m = np.stack([np.asarray(r, dtype=np.float64) for r in rows])
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def chunk_cosine(df_q: pd.DataFrame, df_t: pd.DataFrame) -> np.ndarray:
    """(|Q|, |T|) cosine of the whole-chunk embeddings."""
    return unit(df_q["chunk_embedding"]) @ unit(df_t["chunk_embedding"]).T


def _sentence_units(df: pd.DataFrame) -> list[np.ndarray]:
    if "sentence_embeddings" not in df.columns:
        raise ValueError(
            "sentence-level matching needs per-sentence embeddings: the embeddings were "
            "written without them (embed_language(..., sentence_level=True)).")
    return [unit(list(v)) for v in df["sentence_embeddings"]]


def sentence_max(df_q: pd.DataFrame, df_t: pd.DataFrame) -> np.ndarray:
    """(|Q|, |T|) highest cosine between any sentence of q and any sentence of t."""
    q_sents, t_sents = _sentence_units(df_q), _sentence_units(df_t)
    t_all = np.concatenate(t_sents)
    owner = np.repeat(np.arange(len(t_sents)), [len(s) for s in t_sents])
    out = np.full((len(q_sents), len(t_sents)), -np.inf)
    for i, q in enumerate(q_sents):
        best = (q @ t_all.T).max(axis=0)          # each target sentence's best query sentence
        np.maximum.at(out[i], owner, best)
    return out


def score_matrix(df_q: pd.DataFrame, df_t: pd.DataFrame, method: AlignMethod) -> np.ndarray:
    if method.score == "chunk":
        return chunk_cosine(df_q, df_t)
    if method.score == "sentence_max":
        return sentence_max(df_q, df_t)
    raise ValueError(f"unknown alignment score {method.score!r}")


def routing(df_q: pd.DataFrame, df_t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Section routing: (best_path_sim (|Q|,), in_best_section (|Q|, |T|) bool).

    For each query chunk, the target section whose (English) section-name
    embedding is most similar to the query's, and which target chunks lie in it.
    drop_duplicates keeps the first occurrence, so argmax breaks ties by
    document order."""
    sections = df_t.drop_duplicates("section")
    path_sim = unit(df_q["section_embedding"]) @ unit(sections["section_embedding"]).T
    best = path_sim.argmax(axis=1)
    best_section = sections["section"].to_numpy()[best]
    in_best = df_t["section"].to_numpy()[None, :] == best_section[:, None]
    return path_sim[np.arange(len(best)), best], in_best


def match_matrix(scores: np.ndarray, best_path_sim: np.ndarray, in_best: np.ndarray,
                 method: AlignMethod) -> np.ndarray:
    """(|Q|, |T|) bool: the pairs the aligner reports."""
    if method.path_thr is None:
        allowed = np.ones_like(in_best, dtype=bool)
    else:
        routed = best_path_sim >= method.path_thr
        allowed = np.where(routed[:, None], in_best, True)
    return allowed & (scores >= method.threshold)


def align_pairs(df_q: pd.DataFrame, df_t: pd.DataFrame, q_lang: str, t_lang: str,
                method: AlignMethod) -> dict:
    """Everything an aligner writes for queries df_q against targets df_t.

    Returns {"matched": (|Q|, |T|) bool, "scores", "chunk_sims", "best_path_sim",
    "labels": {(i, j): LLM label}, "info": how the pairs were chosen}."""
    scores = score_matrix(df_q, df_t, method)
    chunk_sims = scores if method.score == "chunk" else chunk_cosine(df_q, df_t)
    best_path_sim, in_best = routing(df_q, df_t)
    matched = match_matrix(scores, best_path_sim, in_best, method)
    labels = {}
    info = {"score": method.score, "threshold": method.threshold, "path_thr": method.path_thr}
    if method.rerank_top_n:
        from app.services import llm_rerank
        info["rerank_top_n"] = method.rerank_top_n
        info["rerank_pool"] = method.rerank_pool
        call, llm = llm_rerank.llm_caller()
        if call is None:
            info["rerank"] = {"status": "no LLM, fell back to the score threshold", "reason": llm}
            print(f"[!] No LLM for alignment rerank ({llm}); using {method.score} >= {method.threshold}")
        else:
            if method.rerank_pool == "chunk":
                pool = chunk_sims
            elif method.score == "sentence_max":
                pool = scores
            else:
                pool = sentence_max(df_q, df_t)
            print(f"[*] LLM rerank of the top {method.rerank_top_n} {t_lang} candidates by "
                  f"{method.rerank_pool} for {len(df_q)} {q_lang} chunks with "
                  f"{llm['backend']} ({llm['model']}) ...")
            matched, labels, stats = llm_rerank.rerank(
                pool, df_q["chunk_id"].tolist(), df_q["chunk_text"].tolist(), q_lang,
                df_t["chunk_id"].tolist(), df_t["chunk_text"].tolist(), t_lang,
                method.rerank_top_n, call, fallback=matched)
            status = ("LLM unreachable, fell back to the score threshold" if stats.get("unreachable")
                      else "ok" if not stats["failed"] else "ok, some rows fell back")
            info["rerank"] = {"status": status, "llm": llm, **stats}
            if stats["failed"]:
                print(f"[!] LLM rerank: {stats['failed']} of {stats['calls']} calls failed "
                      f"({stats['fallback_rows']} chunks use the score threshold): {stats['errors']}")
    return {"matched": matched, "scores": scores, "chunk_sims": chunk_sims,
            "best_path_sim": best_path_sim, "labels": labels, "info": info}
