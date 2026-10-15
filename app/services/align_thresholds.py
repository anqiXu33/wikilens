"""Alignment thresholds shared by both aligners and by `python -m wikilens evaluate`.

Kept in a module with no import-time side effects so the evaluation code can
read the current defaults without importing the aligners.
"""

# ── Per-language-pair thresholds ──────────────────────────────────────────────
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
LANG_THRESHOLDS: dict[str, tuple[float, float]] = {
    "de": (0.45, 0.35),   # chunk_thr raised (0.30→0.45) after section-name translation fix
    "it": (0.35, 0.35),   # empirically calibrated ✓
    "fr": (0.40, 0.40),   # empirically calibrated ✓
    "zh": (0.45, 0.45),   # empirically calibrated ✓
}
DEFAULT_THRESHOLDS = (0.35, 0.35)   # safe fallback for unknown languages

LAMBDA = 0.3  # 路径加权因子 (path weight in combined_score; does not affect which pairs match)


def thresholds_for(lang: str) -> tuple[float, float]:
    """(chunk_thr, path_thr) the aligners use for the non-English language `lang`."""
    return LANG_THRESHOLDS.get(lang, DEFAULT_THRESHOLDS)
