"""Alignment settings shared by both aligners and by `python -m wikilens evaluate`.

Kept in a module with no import-time side effects so the evaluation code can
read the current defaults without importing the aligners.
"""
from dataclasses import dataclass

# ── Per-language-pair thresholds for chunk-level matching ─────────────────────
# Calibrated from LaBSE cosine similarity distribution analysis (eval_alignment_threshold.py).
#
# Key finding: LaBSE similarities are NOT uniform across language pairs.
# Distribution analysis on 3 articles per pair (Climate_change, Adolf_Hitler, Switzerland):
#   EN-DE peaks at 0.25–0.40  →  0.45 discards ~47% of real matches → use 0.30
#   EN-IT peaks at 0.40–0.65  →  0.30 has near-zero 1:0 rate, 0.35 is a better balance
#   EN-FR peaks at 0.50–0.70  →  distribution more right-shifted, 0.40 gives ~4% 1:0
#   EN-ZH peaks at 0.65–0.90  →  0.45 is fine (only ~6% 1:0 rate)
#
# Format: lang_code → (chunk_sim_threshold, path_sim_threshold)
# These were chosen by looking at 1:0 rates, not against labels; see
# docs/annotation_guidelines.md and `python -m wikilens evaluate`.
# EN-ZH no longer uses chunk-level matching by default (see LANG_METHODS);
# its entry is kept for `evaluate`, which sweeps chunk-level thresholds.
LANG_THRESHOLDS: dict[str, tuple[float, float]] = {
    "de": (0.45, 0.35),   # chunk_thr raised (0.30→0.45) after section-name translation fix
    "it": (0.35, 0.35),   # empirically calibrated ✓
    "fr": (0.40, 0.40),   # empirically calibrated ✓
    "zh": (0.45, 0.45),   # chunk-level (no longer the zh default, see LANG_METHODS)
}
DEFAULT_THRESHOLDS = (0.35, 0.35)   # safe fallback for unknown languages

LAMBDA = 0.3  # 路径加权因子 (path weight in combined_score; does not affect which pairs match)


def thresholds_for(lang: str) -> tuple[float, float]:
    """(chunk_thr, path_thr) of chunk-level matching for the non-English language `lang`."""
    return LANG_THRESHOLDS.get(lang, DEFAULT_THRESHOLDS)


# ── Matching method per language pair ─────────────────────────────────────────

@dataclass(frozen=True)
class AlignMethod:
    """How the aligners match chunks for one language pair.

    score:     "chunk"         cosine of the whole-chunk LaBSE embeddings
               "sentence_max"  highest cosine between any sentence of one chunk
                               and any sentence of the other (LaBSE per sentence)
    threshold: a pair matches when its score is >= threshold
    path_thr:  section routing: if the best-matching section name has a
               similarity >= path_thr, only chunks of that section are
               candidates. None turns routing off (every chunk is a candidate).
    rerank_top_n: if set, the top-N targets by rerank_pool are labelled by
               the scoring LLM (app/services/llm_rerank.py) and the pairs it
               calls match or partial are the matches; score, threshold and
               path_thr then only apply when no LLM is available (the fallback).
    rerank_pool: the score that picks the LLM's candidates: "chunk" (chunk
               cosine) or "sentence_max".
    """
    score: str
    threshold: float
    path_thr: float | None
    rerank_top_n: int | None = None
    rerank_pool: str = "chunk"


# EN-ZH: the top 10 targets by chunk cosine go to the LLM, which keeps the
# match / partial ones (one LLM call per chunk and direction): method A of
# docs/alignment_experiments.md, as measured there. Without a usable LLM:
# sentence max >= 0.65, routing off (method B). On 54 hand-labelled chunks from
# 2 articles: A F1 0.65, B 0.50, the old chunk 0.45 / routing 0.45 0.20; see
# README "Alignment method for EN-ZH" for the caveats.
LANG_METHODS: dict[str, AlignMethod] = {
    "zh": AlignMethod("sentence_max", 0.65, None, rerank_top_n=10, rerank_pool="chunk"),
}


def method_for(lang: str) -> AlignMethod:
    """The matching method the aligners use for the non-English language `lang`."""
    if lang in LANG_METHODS:
        return LANG_METHODS[lang]
    chunk_thr, path_thr = thresholds_for(lang)
    return AlignMethod("chunk", chunk_thr, path_thr)
