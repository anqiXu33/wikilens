"""Pipeline alignment defaults and the fetch / chunk fixes that go with them.
No network, LLM or LaBSE: synthetic vectors, a fake LLM, stubbed HTTP.

    python -m pytest tests/test_alignment_defaults.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services import align_scoring, fetcher, llm_rerank  # noqa: E402
from app.services.align_thresholds import AlignMethod, method_for  # noqa: E402
from app.services.chunker import ends_statement, paragraph_chunk  # noqa: E402
from app.services.sections import is_navigation_section  # noqa: E402


# ── Defaults ──────────────────────────────────────────────────────────────────

def test_zh_default_is_sentence_max_with_llm_rerank():
    m = method_for("zh")
    assert (m.score, m.threshold, m.path_thr, m.rerank_top_n, m.rerank_pool) == \
        ("sentence_max", 0.65, None, 10, "chunk")


@pytest.mark.parametrize("lang, thr", [("de", (0.45, 0.35)), ("it", (0.35, 0.35)),
                                       ("fr", (0.40, 0.40)), ("xx", (0.35, 0.35))])
def test_other_pairs_unchanged(lang, thr):
    m = method_for(lang)
    assert (m.score, m.threshold, m.path_thr, m.rerank_top_n) == ("chunk", *thr, None)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _df(prefix, sections, sentence_vecs):
    """Embedding frame: chunk vector = mean of its sentence vectors."""
    return pd.DataFrame({
        "chunk_id": [f"{prefix}_{i:04d}" for i in range(len(sentence_vecs))],
        "chunk_text": [f"{prefix} text {i}" for i in range(len(sentence_vecs))],
        "section": sections,
        "chunk_embedding": [np.mean(v, axis=0) for v in sentence_vecs],
        "section_embedding": [np.eye(4)[0] if s == "A" else np.eye(4)[1] for s in sections],
        "sentence_embeddings": [[np.asarray(x, dtype=float) for x in v] for v in sentence_vecs],
    })


E = np.eye(4)
Q = _df("zh", ["A", "A"], [[E[0], E[1]], [E[2]]])
T = _df("en", ["A", "B", "B"], [[E[0], E[3]], [E[2], E[3]], [E[3]]])


def test_sentence_max():
    s = align_scoring.sentence_max(Q, T)
    # q0 shares sentence E0 with t0; q1 shares E2 with t1
    assert np.allclose(s, [[1, 0, 0], [0, 1, 0]])


def test_routing_off_reaches_other_sections():
    m = AlignMethod("sentence_max", 0.65, None)
    best, in_best = align_scoring.routing(Q, T)
    got = align_scoring.match_matrix(align_scoring.sentence_max(Q, T), best, in_best, m)
    assert got.tolist() == [[True, False, False], [False, True, False]]
    # With routing on, q1 (section A) could only reach t0.
    routed = align_scoring.match_matrix(align_scoring.sentence_max(Q, T), best, in_best,
                                        AlignMethod("sentence_max", 0.65, 0.5))
    assert not routed[1, 1]


def test_sentence_max_needs_sentence_embeddings():
    with pytest.raises(ValueError, match="per-sentence embeddings"):
        align_scoring.sentence_max(Q.drop(columns="sentence_embeddings"), T)


# ── LLM rerank ────────────────────────────────────────────────────────────────

RERANK = AlignMethod("sentence_max", 0.65, None, rerank_top_n=2)


def test_rerank_keeps_match_and_partial(monkeypatch):
    seen = []

    def call(messages):
        seen.append(messages[1]["content"])
        return '{"en_0000": "partial", "en_0001": "none", "en_0002": "match"}'
    monkeypatch.setattr(llm_rerank, "llm_caller", lambda: (call, {"backend": "fake", "model": "m"}))
    out = align_scoring.align_pairs(Q, T, "zh", "en", RERANK)
    # top 2 by sentence max: q0 → t0, then t1 (stable order); q1 → t1, then t0. t2 is never sent.
    assert all("en_0002" not in s for s in seen) and len(seen) == 2
    assert out["matched"].tolist() == [[True, False, False], [True, False, False]]
    assert out["labels"][(0, 0)] == "partial" and out["labels"][(1, 1)] == "none"
    assert out["info"]["rerank"]["status"] == "ok" and out["info"]["rerank"]["calls"] == 2


def test_rerank_candidates_come_from_chunk_cosine(monkeypatch):
    # tA shares a whole sentence with q (sentence max 1) but its chunk vector is
    # diluted; tB has no identical sentence but the same overall direction.
    q = _df("zh", ["A"], [[E[0], E[1]]])
    t = _df("en", ["A", "A"], [[E[0], E[2], E[3]], [np.array([1.0, 1.0, 0, 0]) / np.sqrt(2)]])
    sent = []

    def call(messages):
        sent.append(messages[1]["content"])
        return '{"en_0001": "match"}'
    monkeypatch.setattr(llm_rerank, "llm_caller", lambda: (call, {"backend": "fake", "model": "m"}))
    out = align_scoring.align_pairs(q, t, "zh", "en", AlignMethod("sentence_max", 0.65, None, 1, "chunk"))
    assert "[en_0001]" in sent[0] and "[en_0000]" not in sent[0]
    assert out["matched"].tolist() == [[False, True]]
    sent.clear()
    align_scoring.align_pairs(q, t, "zh", "en", AlignMethod("sentence_max", 0.65, None, 1, "sentence_max"))
    assert "[en_0000]" in sent[0]


def test_no_llm_falls_back_to_threshold(monkeypatch):
    monkeypatch.setattr(llm_rerank, "llm_caller", lambda: (None, "RuntimeError: no key"))
    out = align_scoring.align_pairs(Q, T, "zh", "en", RERANK)
    assert out["matched"].tolist() == [[True, False, False], [False, True, False]]
    assert out["info"]["rerank"]["status"].startswith("no LLM")


def test_unreachable_llm_falls_back_after_one_call(monkeypatch):
    calls = []

    def call(messages):
        calls.append(1)
        raise ConnectionError("refused")
    monkeypatch.setattr(llm_rerank, "llm_caller", lambda: (call, {"backend": "fake", "model": "m"}))
    out = align_scoring.align_pairs(Q, T, "zh", "en", RERANK)
    assert len(calls) == 1
    assert out["matched"].tolist() == [[True, False, False], [False, True, False]]
    assert out["info"]["rerank"]["unreachable"]


def test_later_failure_falls_back_for_that_row_only(monkeypatch):
    replies = iter(['{"en_0000": "none", "en_0001": "none"}', None])

    def call(messages):
        r = next(replies)
        if r is None:
            raise TimeoutError("slow")
        return r
    monkeypatch.setattr(llm_rerank, "llm_caller", lambda: (call, {"backend": "fake", "model": "m"}))
    out = align_scoring.align_pairs(Q, T, "zh", "en", RERANK)
    assert out["matched"].tolist() == [[False, False, False], [False, True, False]]
    assert out["info"]["rerank"]["fallback_rows"] == 1


# ── Chunking ──────────────────────────────────────────────────────────────────

def _sents(section, paragraphs, lang="en"):
    return [{"language": lang, "section": section, "paragraph_index": p, "sentence_index": i,
             "sentence_text": t} for p, para in enumerate(paragraphs) for i, t in enumerate(para)]


def test_definition_list_stays_in_one_chunk():
    # Social credit system (en), "Organization > Conceptualization": a lead-in,
    # then four "term" / "description" lines, then the next paragraph.
    sec = "Organization > Conceptualization"
    paras = [
        ["The blueprint document set four target areas.", "They are independent of each other."],
        ["Business trustworthiness system (商务诚信体系)"],
        ["Blacklist system for discredited business organizations.", "Regulated by the PBoC."],
        ["Government trustworthiness system (政务诚信体系)"],
        ["Evaluation system targeting civil servants."],
        ["Social trustworthiness system (社会诚信体系)"],
        ["Blacklist system for discredited individuals."],
        ["Judiciary public trust system (司法公信体系)"],
        ["Blacklist system for judgment defaulters.", "Regulated by the Supreme People's Court."],
        ["As of 2023, the government has mainly built a business system.", "More text follows."],
    ]
    chunks = paragraph_chunk(_sents(sec, paras))
    assert len(chunks) == 2
    assert "Business trustworthiness" in chunks[0]["chunk_text"]
    assert chunks[0]["chunk_text"].endswith("Regulated by the Supreme People's Court.")
    assert chunks[0]["sentence_count"] == 12 == len(chunks[0]["sentences"])
    assert chunks[1]["chunk_text"].startswith("As of 2023")


def test_list_after_colon_and_bullets():
    paras = [["Intro one.", "The members are:"], ["Alice"], ["Bob"], ["Carol and Dave", "both founders."],
             ["A new topic starts here.", "And continues."]]
    chunks = paragraph_chunk(_sents("S", paras))
    assert [c["sentence_count"] for c in chunks] == [6, 2]


def test_open_paragraph_does_not_cross_sections():
    s = _sents("A", [["One.", "A heading-like line"]]) + _sents("B", [["Two.", "Three."]])
    assert [c["section"] for c in paragraph_chunk(s)] == ["A", "B"]


@pytest.mark.parametrize("text, closed", [
    ("Ends here.", True), ('He said "stop."', True), ("结束。", True), ("（完）。」", True),
    ("Business trustworthiness system (商务诚信体系)", False), ("The list:", False), ("item;", False),
])
def test_ends_statement(text, closed):
    assert ends_statement(text) is closed


def test_navigation_sections_are_not_chunked():
    s = (_sents("(Lead)", [["社会信用体系是一个系统。", "第二句。"]], "zh")
         + _sents("参见", [["大规模监控"], ["芝麻信用"]], "zh")
         + _sents("See also", [["Mass surveillance"]], "en")
         + _sents("History > Further reading", [["A book."]], "en"))
    chunks = paragraph_chunk(s)
    assert [c["section"] for c in chunks] == ["(Lead)"]


@pytest.mark.parametrize("section, lang, nav", [
    ("参见", "zh", True), ("參見", "zh", True), ("参考文献", "zh", True), ("評價 > 流行文化", "zh", False),
    ("See also", "en", True), ("references", "en", True), ("History > Further reading", "en", True),
    ("Organization", "en", False), ("Siehe auch", "de", True), ("参见", "en", False),
])
def test_is_navigation_section(section, lang, nav):
    assert is_navigation_section(section, lang) is nav


def test_cached_article_drops_navigation_sections(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "CACHE_DIR", tmp_path)
    df = pd.DataFrame(_sents("评价", [["一。", "二。"]], "zh") + _sents("参见", [["芝麻信用"]], "zh"))
    fetcher.save_to_cache(df, "社会信用体系", "zh")
    got = fetcher.load_from_cache("社会信用体系", "zh")
    assert got["section"].unique().tolist() == ["评价"]


# ── Title lookup ──────────────────────────────────────────────────────────────

def test_missing_title_fails_without_requesting_entity(monkeypatch):
    urls = []

    def fake_get_json(url, params=None, **kw):
        urls.append(url)
        return {"entities": {"-1": {"site": "enwiki", "title": "Social Credit System", "missing": ""}},
                "success": 1}
    monkeypatch.setattr(fetcher, "_get_json", fake_get_json)
    with pytest.raises(fetcher.TitleNotFoundError, match="redirect"):
        fetcher.fetch_multilang_from_url("https://en.wikipedia.org/wiki/Social_Credit_System", ["en", "zh"])
    assert len(urls) == 1   # only wbgetentities; Special:EntityData/-1 is never requested


def test_found_title_returns_qid(monkeypatch):
    monkeypatch.setattr(fetcher, "_get_json", lambda url, params=None, **kw: {"entities": {"Q123": {"id": "Q123"}}})
    assert fetcher.get_qid_from_title("Huawei") == "Q123"
