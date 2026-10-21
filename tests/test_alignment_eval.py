"""Alignment annotation and evaluation (`python -m wikilens annotate / evaluate`).
No network, LLM or LaBSE: uses a saved run's embeddings and synthetic data.

    python -m pytest tests/test_alignment_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wikilens import alignment_eval as ae  # noqa: E402
from wikilens.annotate import render_page  # noqa: E402

RUN_DIR = ROOT / "outputs" / "runs" / "20261015_201823_Gala_(apple)_en-de_543608"
needs_run = pytest.mark.skipif(not (RUN_DIR / "embeddings").exists(), reason="saved run not available")


@needs_run
@pytest.mark.parametrize("query, target, align_file, id_key", [
    ("de", "en", "alignments_source_to_en/de_to_en.json", "src_chunk_id"),
    ("en", "de", "alignments_en_to_source/en_to_de.json", "tgt_chunk_id"),
])
def test_vectorised_aligner_matches_saved_output(query, target, align_file, id_key):
    sims = ae.run_similarities(RUN_DIR, query, target)
    pred = sims.align(*ae.default_thresholds(["en", "de"]))
    saved = json.loads((RUN_DIR / align_file).read_text(encoding="utf-8"))
    assert len(saved) == len(sims.query_ids)
    for rec in saved:
        i = sims.query_ids.index(rec[id_key])
        got = {sims.target_ids[j] for j in np.flatnonzero(pred[i])}
        assert got == set(rec["matched_chunk_ids"]), rec[id_key]


def _sims():
    """Two queries, four targets in sections A, A, B, B."""
    q = pd.DataFrame({
        "chunk_id": ["q0", "q1"],
        "section": ["s", "s"],
        "chunk_embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "section_embedding": [[1.0, 0.0], [0.0, 1.0]],
    })
    t = pd.DataFrame({
        "chunk_id": ["t0", "t1", "t2", "t3"],
        "section": ["A", "A", "B", "B"],
        "chunk_embedding": [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
        "section_embedding": [[1.0, 0.0], [1.0, 0.0], [0.6, 0.8], [0.6, 0.8]],
    })
    return ae.similarities(q, t)


def test_align_routing_and_threshold():
    s = _sims()
    # q0 routes to section A (path sim 1.0); t0 = 1.0, t1 = 0.8 there.
    assert s.align(0.7, 0.9)[0].tolist() == [True, True, False, False]
    # q1's best section is B (0.8): routed at path_thr 0.5, so t1 (A, sim 0.6) is excluded.
    assert s.align(0.55, 0.5)[1].tolist() == [False, False, True, False]
    # Routing off (path_thr above any similarity): all of T is searched.
    assert s.align(0.55, np.inf)[1].tolist() == [False, True, True, False]


def _case(labels=None, absent=False, tail_negative=False, partial="positive", elsewhere=False, elsewhere_ids=()):
    q = ae.LabelledQuery("run", "T", "en-de", "de", "en", "q1", absent, labels or {}, tail_negative,
                         elsewhere, list(elsewhere_ids))
    return ae.resolve_cases([q], lambda *a: _sims(), partial)[0]


def test_counts():
    c = _case({"t2": "match", "t1": "partial", "t0": "none"})
    # chunk_thr 0.55, routing off: q1 predicts t1, t2 → both positive.
    assert c.counts(0.55, np.inf).tolist() == [2, 0, 0, 0, 0]
    # Routed to B: t1 missed.
    assert c.counts(0.55, 0.5).tolist() == [1, 0, 1, 0, 0]
    # chunk_thr 0: predicts all four; t0 labelled none (fp), t3 unlabelled.
    assert c.counts(-1, np.inf, "ignore").tolist() == [2, 1, 0, 1, 0]
    assert c.counts(-1, np.inf, "fp").tolist() == [2, 2, 0, 0, 0]
    assert c.counts(-1, np.inf, "auto").tolist() == [2, 1, 0, 1, 0]  # tail not negative
    tail = _case({"t2": "match", "t1": "partial", "t0": "none"}, tail_negative=True)
    assert tail.counts(-1, np.inf, "auto").tolist() == [2, 2, 0, 0, 0]
    # partial as negative / ignored
    assert _case({"t2": "match", "t1": "partial"}, partial="negative").counts(0.55, np.inf).tolist() == [1, 1, 0, 0, 0]
    assert _case({"t2": "match", "t1": "partial"}, partial="ignore").counts(0.55, np.inf, "ignore").tolist() == [1, 0, 0, 1, 0]


def test_absent_counts_every_prediction_as_fp():
    c = _case(absent=True, tail_negative=True)
    assert c.counts(0.55, np.inf).tolist() == [0, 2, 0, 0, 0]
    assert c.counts(0.99, np.inf).tolist() == [0, 0, 0, 0, 0]


def test_elsewhere_is_a_miss_not_absent():
    # Unnamed: one FN at every threshold, even when the aligner matches nothing.
    c = _case({"t0": "none"}, elsewhere=True)
    assert c.counts(0.99, np.inf).tolist() == [0, 0, 1, 0, 1]
    # q1 predicts t1, t2 at 0.55 with routing off; they are outside the labelled
    # pairs and might be the unnamed match, so they stay uncounted under auto.
    assert c.counts(0.55, np.inf).tolist() == [0, 0, 1, 2, 1]
    # Named: t2 is the match outside the candidates; found at 0.55, missed at 0.9.
    named = _case({"t0": "none"}, elsewhere=True, elsewhere_ids=["t2"], tail_negative=True)
    assert named.counts(0.55, np.inf).tolist() == [1, 1, 0, 0, 0]  # t1 → fp via the tail rule
    assert named.counts(0.9, np.inf).tolist() == [0, 0, 1, 0, 1]


def test_load_label_files(tmp_path):
    data = {
        "format": ae.LABELS_FORMAT, "run_id": "r", "title": "T", "pair": "en-zh",
        "query_lang": "zh", "target_lang": "en",
        "items": [
            {"query_id": "zh_0", "absent": False, "candidates": [
                {"chunk_id": "en_1", "chunk_sim": 0.8, "label": "match"},
                {"chunk_id": "en_2", "chunk_sim": 0.6, "label": "none"}]},
            {"query_id": "zh_1", "absent": True, "candidates": [
                {"chunk_id": "en_1", "chunk_sim": 0.5, "label": "none"}]},
            {"query_id": "zh_2", "absent": False, "candidates": [
                {"chunk_id": "en_1", "chunk_sim": 0.5, "label": None}]},
            {"query_id": "zh_3", "absent": False, "elsewhere": True, "elsewhere_ids": [], "candidates": [
                {"chunk_id": "en_1", "chunk_sim": 0.7, "label": "partial"},
                {"chunk_id": "en_2", "chunk_sim": 0.6, "label": "none"}]},
            {"query_id": "zh_4", "absent": False, "elsewhere": True, "elsewhere_ids": ["en_9", "en_1"],
             "candidates": [{"chunk_id": "en_1", "chunk_sim": 0.7, "label": None}]},
        ],
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    queries, warnings = ae.load_label_files([path])
    assert [q.query_id for q in queries] == ["zh_0", "zh_1", "zh_3", "zh_4"]
    assert queries[0].tail_negative and queries[0].labels == {"en_1": "match", "en_2": "none"}
    assert queries[1].absent and queries[1].labels == {} and not queries[1].elsewhere
    # Unnamed match outside the candidates: lower-ranked chunks may be it, so no tail rule.
    assert queries[2].elsewhere and not queries[2].tail_negative and queries[2].elsewhere_ids == []
    # Kept even with no candidate labelled; ids already among the candidates are dropped.
    assert queries[3].elsewhere and queries[3].elsewhere_ids == ["en_9"]
    assert warnings == [f"{path}: 1 unlabelled item(s) ignored"]


def test_prf_and_grid():
    assert ae.prf(np.array([2, 2, 0, 0]))["f1"] == pytest.approx(2 / 3)
    assert ae.prf(np.array([0, 0, 0, 0]))["f1"] == 0.0
    assert ae.parse_grid("0.2:0.3:0.05") == [0.2, 0.25, 0.3]
    assert ae.parse_grid("0.45,0.35") == [0.45, 0.35]
    assert ae.pair_key(["zh", "en"]) == "en-zh" and ae.pair_key(["fr", "de"]) == "de-fr"


@needs_run
def test_candidates_and_page(tmp_path):
    meta = json.loads((RUN_DIR / "meta.json").read_text(encoding="utf-8"))
    data = ae.build_candidates(RUN_DIR, meta, "de", k=3, sample=5, seed=1)
    assert data["query_lang"] == "de" and data["target_lang"] == "en"
    assert len(data["items"]) == 5
    for item in data["items"]:
        sims = [c["chunk_sim"] for c in item["candidates"]]
        assert len(sims) == 3 and sims == sorted(sims, reverse=True)
    assert ae.labels_filename(meta["run_id"], "de", ["en", "de"]) == f"{meta['run_id']}.json"
    assert ae.labels_filename(meta["run_id"], "en", ["en", "de"]) == f"{meta['run_id']}.from-en.json"

    data["items"][0]["text"] = "</script><script>alert(1)</script>"
    data.update(format=ae.LABELS_FORMAT, labels_file="x.json", generated_at="now")
    html = render_page(data)
    assert html.count("</script>") == 2  # only the page's own two script blocks
    payload = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload)["items"][0]["text"] == "</script><script>alert(1)</script>"


def test_extra_candidates_keep_the_tail_rule_on_the_top_k(tmp_path):
    def item(qid, extra_label):
        return {"query_id": qid, "absent": False, "candidates": [
            {"chunk_id": "en_1", "chunk_sim": 0.8, "label": "match"},
            {"chunk_id": "en_2", "chunk_sim": 0.6, "label": "none"},
            {"chunk_id": "en_7", "chunk_sim": 0.4, "label": extra_label, "extra": True, "rank": 7}]}
    data = {"format": ae.LABELS_FORMAT, "run_id": "r", "title": "T", "pair": "en-zh",
            "query_lang": "zh", "target_lang": "en",
            "items": [item("zh_0", None), item("zh_1", "none"), item("zh_2", "partial")]}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    q = {x.query_id: x for x in ae.load_label_files([path])[0]}
    # An unlabelled extra does not undo the tail rule of the top k.
    assert q["zh_0"].tail_negative and q["zh_0"].labels == {"en_1": "match", "en_2": "none"}
    assert q["zh_1"].tail_negative and q["zh_1"].labels["en_7"] == "none"
    # A match below the top k shows the ranking had not run out of matches.
    assert not q["zh_2"].tail_negative and q["zh_2"].labels["en_7"] == "partial"


@needs_run
def test_add_extra_candidates(tmp_path):
    from wikilens.annotate import add_extra_candidates, unlabelled_pairs
    meta = json.loads((RUN_DIR / "meta.json").read_text(encoding="utf-8"))
    data = ae.build_candidates(RUN_DIR, meta, "de", k=3, sample=3, seed=1)
    ids = [it["query_id"] for it in data["items"]]
    all_t = [c["chunk_id"] for c in data["target_chunks"]]
    top = {it["query_id"]: [c["chunk_id"] for c in it["candidates"]] for it in data["items"]}
    below = {q: [t for t in all_t if t not in top[q]] for q in ids}
    labels = {"annotator": "A", "items": [
        {"query_id": ids[0], "absent": False, "elsewhere": False, "elsewhere_ids": [], "note": "n",
         "candidates": [dict(c, label="none") for c in data["items"][0]["candidates"]]},
        {"query_id": ids[1], "absent": True, "candidates": []},
        {"query_id": ids[2], "absent": False, "elsewhere": True, "elsewhere_ids": [below[ids[2]][0]],
         "candidates": []}]}
    pred = lambda q, t, run=meta["run_id"]: {"run_id": run, "query_id": q, "target_id": t}
    experiment = {"methods": [
        {"method": "A: LLM rerank", "unlabelled_predictions": [
            pred(ids[0], top[ids[0]][0]), pred(ids[0], below[ids[0]][1]), pred(ids[0], below[ids[0]][0]),
            pred(ids[1], below[ids[1]][0]),
            pred(ids[2], below[ids[2]][0]),
            pred(ids[0], below[ids[0]][2], run="other")]},
        {"method": "B: sentence max", "unlabelled_predictions": [pred(ids[0], below[ids[0]][2])]}]}
    pairs = unlabelled_pairs(experiment, meta["run_id"], ["llm rerank"])
    assert below[ids[0]][2] not in pairs[ids[0]]  # other method, other run
    assert add_extra_candidates(data, labels, pairs, RUN_DIR) == 2
    extra = [c for c in data["items"][0]["candidates"] if c.get("extra")]
    # Shown candidates are not repeated; extras come in similarity order, ranked below the top k.
    assert {c["chunk_id"] for c in extra} == set(below[ids[0]][:2])
    assert [c["rank"] for c in extra] == sorted(c["rank"] for c in extra) and min(c["rank"] for c in extra) > 3
    assert data["items"][1]["n_extra"] == 0  # absent: every prediction is a false positive already
    assert data["items"][2]["n_extra"] == 0  # already named as the match elsewhere
    assert data["prefill"]["items"][ids[0]]["note"] == "n" and data["prefill"]["annotator"] == "A"
    assert data["prefill"]["items"][ids[0]]["labels"] == {t: "none" for t in top[ids[0]]}
    with pytest.raises(SystemExit):
        add_extra_candidates(data, {"items": labels["items"][:2]}, pairs, RUN_DIR)
