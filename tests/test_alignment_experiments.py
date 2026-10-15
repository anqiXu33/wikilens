"""Alignment method experiments (`python -m wikilens experiment`).
No network, LLM or LaBSE: synthetic vectors, a fake LLM and a saved run.

    python -m pytest tests/test_alignment_experiments.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wikilens import alignment_eval as ae  # noqa: E402
from wikilens import alignment_experiments as ax  # noqa: E402

RUN_DIR = ROOT / "outputs" / "runs" / "20261015_203858_Social_credit_system_en-zh_834a1c"
needs_run = pytest.mark.skipif(not (RUN_DIR / "fetched").exists(), reason="saved run not available")


@needs_run
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_chunk_sentences_rebuild_chunks(lang):
    sents = ax.chunk_sentences(RUN_DIR, lang)
    chunks = ae.load_embeddings(RUN_DIR, lang)
    assert list(sents) == chunks["chunk_id"].tolist()
    assert all(" ".join(sents[c]) == t for c, t in zip(chunks["chunk_id"], chunks["chunk_text"]))


def _unit(*rows):
    m = np.array(rows, dtype=float)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_sentence_scores_max_and_topk():
    # q has two sentences: one equal to t0's only sentence, one orthogonal to everything.
    q = {"q": _unit([1, 0, 0], [0, 0, 1])}
    t = {"t0": _unit([1, 0, 0]), "t1": _unit([0.6, 0.8, 0], [0, 1, 0])}
    mx = ax.sentence_scores(q, t, ["q"], ["t0", "t1"], None)
    assert mx[0].tolist() == pytest.approx([1.0, 0.6])
    # top-2 mean over q's sentences: (1 + 0) / 2 for t0, (0.6 + 0) / 2 for t1.
    top2 = ax.sentence_scores(q, t, ["q"], ["t0", "t1"], 2)
    assert top2[0].tolist() == pytest.approx([0.5, 0.3])
    # k larger than the number of query sentences averages all of them.
    assert ax.sentence_scores(q, t, ["q"], ["t0", "t1"], 5)[0].tolist() == pytest.approx([0.5, 0.3])


def test_parse_rerank_reply():
    ids = ["en_0001", "en_0002", "en_0003", "en_0004"]
    reply = 'Here you go:\n```json\n{"en_0001": "match", "[en_0002]": "Partial", "en_0003": "maybe"}\n```'
    assert ax.parse_rerank_reply(reply, ids) == {
        "en_0001": "match", "en_0002": "partial", "en_0003": "none", "en_0004": "none"}
    assert ax.parse_rerank_reply("no json", ids) == dict.fromkeys(ids, "none")


def _case(scores_row, pos=(), neg=(), elsewhere=(), absent=False, tail_negative=False):
    t_ids = [f"t{j}" for j in range(len(scores_row))]
    base = ae.Similarities(["q"], t_ids, np.array([scores_row], dtype=float),
                           np.array([0.0]), np.ones((1, len(t_ids)), dtype=bool))
    q = ae.LabelledQuery("r", "T", "en-zh", "zh", "en", "q", absent,
                         tail_negative=tail_negative, elsewhere=bool(elsewhere),
                         elsewhere_ids=[t_ids[j] for j in elsewhere])
    return ae.QueryCase(q, base, 0, np.array(pos, dtype=int), np.array(neg, dtype=int),
                        np.array(elsewhere, dtype=int))


def test_coverage_counts_hits_pairs_and_elsewhere():
    a = _case([0.9, 0.8, 0.7, 0.1], pos=[0], neg=[1, 2], elsewhere=[3])
    b = _case([0.9, 0.8, 0.7, 0.1], pos=[], neg=[0, 1, 2], absent=True)
    by_chunk = lambda c: c.sims.chunk_sim[c.row]
    cov = ax.coverage([a, b], by_chunk, ks=(1, 3))
    assert cov["n_chunks"] == 1 and cov["n_positive_pairs"] == 2 and cov["n_elsewhere_named"] == 1
    assert cov["chunks@1"] == 1.0 and cov["pairs@3"] == 0.5 and cov["elsewhere@3"] == 0.0
    # Ranking by another score that puts t3 first finds the elsewhere match.
    cov = ax.coverage([a], lambda c: np.array([0, 0, 0, 1.0]), ks=(1,))
    assert cov["elsewhere@1"] == 1.0 and cov["chunks@1"] == 1.0


def test_as_similarities_disables_routing_and_sweep_counts():
    case = _case([2.0, 1.0, 0.0, -1.0], pos=[0, 1], neg=[2], tail_negative=True)
    case.sims = ax.as_similarities(case.sims, case.sims.chunk_sim)
    rows = ax.sweep([case], [1.0, 2.0])
    assert [(r["tp"], r["fp"], r["fn"]) for r in rows] == [(2, 0, 0), (1, 0, 1)]


def test_llm_rerank_scores_uses_cache(tmp_path):
    q = pd.DataFrame({"chunk_id": ["zh_0"], "section": ["s"], "chunk_embedding": [[1.0, 0.0]],
                      "section_embedding": [[1.0]]})
    t = pd.DataFrame({"chunk_id": ["en_0", "en_1", "en_2"], "section": ["s"] * 3,
                      "chunk_embedding": [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], "section_embedding": [[1.0]] * 3})
    sims = ae.similarities(q, t)
    texts = {"zh": {"zh_0": "源"}, "en": {"en_0": "a", "en_1": "b", "en_2": "c"}}
    calls = []

    def fake_llm(msgs):
        calls.append(msgs)
        assert "[en_0] a" in msgs[1]["content"] and "[en_1]" not in msgs[1]["content"]  # top-2 only
        return '{"en_0": "partial", "en_2": "match"}'

    cache = ax.RerankCache(tmp_path / "cache.json")
    key = ("r", "zh", "en")
    args = ({key: ["zh_0"]}, lambda *k: sims, lambda run, lang: texts[lang], 2, cache, "m")
    scores, stats = ax.llm_rerank_scores(*args, call=fake_llm)
    assert scores[key][0].tolist() == [1.0, ax.NOT_CANDIDATE, 2.0]
    assert stats == {"calls": 1, "cached": 0}
    cache.save({"model": "m"})
    scores2, stats2 = ax.llm_rerank_scores(*args[:4], ax.RerankCache(tmp_path / "cache.json"), "m", call=None)
    assert scores2[key].tolist() == scores[key].tolist() and stats2 == {"calls": 0, "cached": 1}
    assert len(calls) == 1
