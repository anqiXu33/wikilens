"""`python -m wikilens experiment`: compare two alternatives to chunk-level
LaBSE matching against the human labels in eval/labels/. Experiment only;
the pipeline's aligner is unchanged.

  baseline  chunk-level LaBSE cosine (what the aligner uses), routing on at the
            pipeline default and off, swept over chunk_thr.
  A         LLM rerank: for each query chunk take its top-N (default 10) target
            chunks by chunk-level LaBSE, and ask the configured LLM
            (SCORING_BACKEND, see .env.example) which of them state the same
            fact(s): match / partial / none. Predicted pairs are the candidates
            the LLM calls match or partial ("A, match+partial") or match only.
  B         sentence-level pairing: split both chunks into their sentences
            (the fetcher's sentence split), embed each sentence with LaBSE, and
            score a chunk pair by the highest sentence-pair cosine ("max") or by
            the mean of the k best per-query-sentence matches ("top-k mean":
            each query sentence's best target sentence, then the k highest of
            those), swept over a threshold.

Every method is scored with the same counting as `evaluate`
(wikilens/evaluate.py docstring). Each also gets candidate coverage: the
share of non-absent labelled chunks with a known correct target among the
method's top-k ranked targets, and the share of known positive pairs there.

LLM replies are cached in eval/experiments/llm_rerank_<model>.json, keyed by
the prompt, so re-running scores the saved replies without calling the LLM.
"""
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.llm_rerank import LLM_LABELS, parse_rerank_reply, rerank_messages  # noqa: F401
from wikilens import alignment_eval as ae

EXPERIMENTS_DIR = Path("eval/experiments")
SENTENCE_CACHE_DIR = EXPERIMENTS_DIR / "cache"   # gitignored: re-computable sentence embeddings
COVERAGE_KS = (1, 3, 5, 10)
OFF = math.inf
LLM_SCORE = {"none": 0.0, "partial": 1.0, "match": 2.0}
NOT_CANDIDATE = -1.0


# ── Sentences ─────────────────────────────────────────────────────────────────

def chunk_sentences(run_dir: Path, lang: str) -> dict[str, list[str]]:
    """{chunk_id: [sentence, ...]} from the run's fetched sentences.

    The chunker keeps document order and only concatenates consecutive
    sentences, so chunk i is the next sentence_count sentences; checked
    against chunk_text."""
    run_dir = Path(run_dir)
    sentences = json.loads((run_dir / "fetched" / f"fetched_{lang}.json").read_text(encoding="utf-8"))
    chunks = json.loads((run_dir / "chunked" / f"chunked_{lang}.json").read_text(encoding="utf-8"))
    out, pos = {}, 0
    for c in chunks:
        sents = [s["sentence_text"] for s in sentences[pos:pos + c["sentence_count"]]]
        pos += c["sentence_count"]
        if " ".join(sents) != c["chunk_text"]:
            raise SystemExit(f"{run_dir.name}: sentences of {c['chunk_id']} do not rebuild its chunk_text.")
        out[c["chunk_id"]] = sents
    if pos != len(sentences):
        raise SystemExit(f"{run_dir.name}: {len(sentences) - pos} {lang} sentences not in any chunk.")
    return out


def sentence_embeddings(run_dir: Path, lang: str, cache_dir: Path = SENTENCE_CACHE_DIR) -> dict[str, np.ndarray]:
    """{chunk_id: (n_sentences, d) unit vectors}, LaBSE at the pipeline's pinned revision."""
    sents = chunk_sentences(run_dir, lang)
    flat = [(cid, s) for cid, ss in sents.items() for s in ss]
    cache = Path(cache_dir) / f"{Path(run_dir).name}_{lang}_sentences.parquet"
    texts = [s for _, s in flat]
    if cache.exists():
        df = pd.read_parquet(cache)
        if df["text"].tolist() != texts:
            df = None
    else:
        df = None
    if df is None:
        from app.services.embedder import get_model
        vecs = get_model().encode(texts, batch_size=32, show_progress_bar=False)
        df = pd.DataFrame({"chunk_id": [c for c, _ in flat], "text": texts, "embedding": list(vecs)})
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    unit = ae._unit(df["embedding"])
    out = defaultdict(list)
    for cid, v in zip(df["chunk_id"], unit):
        out[cid].append(v)
    return {cid: np.stack(v) for cid, v in out.items()}


def sentence_scores(q_emb: dict, t_emb: dict, q_ids: list[str], t_ids: list[str], k: int | None) -> np.ndarray:
    """(|Q|, |T|) chunk-pair scores from sentence cosines.

    k=None: max over all sentence pairs. k>=1: for each query sentence its best
    target sentence, then the mean of the k highest of those (all of them when
    the query chunk has fewer than k sentences)."""
    t_all = np.concatenate([t_emb[c] for c in t_ids])
    bounds = np.cumsum([0] + [len(t_emb[c]) for c in t_ids])
    out = np.zeros((len(q_ids), len(t_ids)))
    for i, qc in enumerate(q_ids):
        sim = q_emb[qc] @ t_all.T                                   # (n_q_sents, all target sents)
        for j in range(len(t_ids)):
            block = sim[:, bounds[j]:bounds[j + 1]]                 # (n_q_sents, n_t_sents)
            if k is None:
                out[i, j] = block.max()
            else:
                best = np.sort(block.max(axis=1))[::-1]
                out[i, j] = best[:k].mean()
    return out


# ── LLM rerank ────────────────────────────────────────────────────────────────

# The prompt and reply parsing are the pipeline's (app/services/llm_rerank.py),
# so the cached replies below are the ones the aligner would get.


def prompt_key(messages: list[dict], model: str) -> str:
    return hashlib.sha256(json.dumps([model, messages], ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


class RerankCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries = json.loads(self.path.read_text(encoding="utf-8"))["entries"] if self.path.exists() else {}

    def save(self, model_info: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"llm": model_info, "entries": self.entries},
                                        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def llm_backend():
    """(scorer module, model info) or raise SystemExit with the reason no LLM is usable."""
    try:
        from app.services import scorer
        scorer.check_backend_config()
    except RuntimeError as e:
        raise SystemExit(f"LLM rerank needs a configured LLM: {e}")
    return scorer, scorer.backend_info()


def llm_rerank_scores(cases_by_run, sims_for, texts_for, top_n: int, cache: RerankCache,
                      model: str, call=None) -> tuple[dict, dict]:
    """Scores (match 2 / partial 1 / none 0 / not a candidate -1) for the labelled
    query rows of each run, plus a per-run count of LLM calls made / cached.

    cases_by_run: {(run_id, q_lang, t_lang): [query_id, ...]}
    call(messages) -> reply text, or None to only use the cache (missing entries raise)."""
    scores, stats = {}, {"calls": 0, "cached": 0}
    for key, query_ids in cases_by_run.items():
        run_id, q_lang, t_lang = key
        sims = sims_for(*key)
        q_text, t_text = texts_for(run_id, q_lang), texts_for(run_id, t_lang)
        s = np.full(sims.chunk_sim.shape, NOT_CANDIDATE)
        for qid in query_ids:
            i = sims.query_ids.index(qid)
            top = np.argsort(-sims.chunk_sim[i], kind="stable")[:top_n]
            cand = [(sims.target_ids[j], t_text[sims.target_ids[j]]) for j in top]
            msgs = rerank_messages(q_text[qid], q_lang, cand, t_lang)
            pk = prompt_key(msgs, model)
            entry = cache.entries.get(pk)
            if entry is None:
                if call is None:
                    raise SystemExit(f"No cached LLM reply for {run_id} {qid}; run without --cached-only.")
                reply = call(msgs)
                entry = {"run_id": run_id, "query_id": qid, "candidates": [c for c, _ in cand],
                         "reply": reply, "labels": parse_rerank_reply(reply, [c for c, _ in cand])}
                cache.entries[pk] = entry
                stats["calls"] += 1
            else:
                stats["cached"] += 1
            for j in top:
                s[i, j] = LLM_SCORE[entry["labels"][sims.target_ids[j]]]
        scores[key] = s
    return scores, stats


# ── Scoring a score matrix ────────────────────────────────────────────────────

def as_similarities(base: ae.Similarities, scores: np.ndarray) -> ae.Similarities:
    """A Similarities whose chunk_sim is another score, with routing disabled."""
    return ae.Similarities(base.query_ids, base.target_ids, scores,
                           best_path_sim=np.full(len(base.query_ids), -np.inf),
                           in_best_section=np.ones_like(scores, dtype=bool))


def known_positive_cols(case: ae.QueryCase) -> np.ndarray:
    return np.concatenate([case.pos_cols, case.elsewhere_cols]).astype(int)


def coverage(cases: list[ae.QueryCase], rank_scores, ks=COVERAGE_KS) -> dict:
    """Share of non-absent labelled chunks with a known positive among the top-k
    targets by rank_scores(case) -> (|T|,) (ties broken by LaBSE chunk cosine),
    share of known positive pairs inside the top k, and share of the named
    "elsewhere" matches (outside LaBSE's annotated top 3) inside the top k. An "elsewhere" chunk with
    the match unnamed has no known positive: always counted as not covered."""
    present = [c for c in cases if not c.query.absent]
    n_pairs = sum(len(known_positive_cols(c)) + (c.query.elsewhere and not len(c.elsewhere_cols)) for c in present)
    n_elsewhere = sum(len(c.elsewhere_cols) for c in present)
    out = {"n_chunks": len(present), "n_positive_pairs": int(n_pairs), "n_elsewhere_named": int(n_elsewhere)}
    for k in ks:
        hit = pairs = elsewhere = 0
        for c in present:
            primary = rank_scores(c)
            order = np.lexsort((-c.sims.chunk_sim[c.row], -primary))[:k]
            got = np.isin(known_positive_cols(c), order)
            hit += bool(got.any())
            pairs += int(got.sum())
            elsewhere += int(np.isin(c.elsewhere_cols, order).sum())
        out[f"chunks@{k}"] = hit / len(present) if present else 0.0
        out[f"pairs@{k}"] = pairs / n_pairs if n_pairs else 0.0
        # The named matches LaBSE ranked below the annotated top 3: unbiased by
        # how the candidates were picked, unlike the other positives.
        out[f"elsewhere@{k}"] = elsewhere / n_elsewhere if n_elsewhere else 0.0
    return out


def sweep(cases: list[ae.QueryCase], thresholds, path_thr=OFF, unlabelled="auto") -> list[dict]:
    """Metrics at each threshold on each case's sims (chunk_sim = the method's score)."""
    rows = []
    for t in thresholds:
        per_q = np.array([c.counts(t, path_thr, unlabelled) for c in cases])
        rows.append({"threshold": t, **ae.prf(per_q.sum(axis=0)),
                     "f1_95ci": list(ae.bootstrap_f1(per_q)), "_per_q": per_q})
    return rows


def best_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (r["f1"], r["precision"], -rows.index(r)))


def cross_article(cases: list[ae.QueryCase], rows: list[dict]) -> list[dict] | None:
    """Pick the threshold on the other article(s), score it on the held-out one."""
    titles = sorted({c.query.title for c in cases})
    if len(titles) < 2 or len(rows) < 2:
        return None
    out = []
    for held in titles:
        mask = np.array([c.query.title == held for c in cases])
        train = [ae.prf(r["_per_q"][~mask].sum(axis=0)) for r in rows]
        i = max(range(len(rows)), key=lambda n: (train[n]["f1"], train[n]["precision"], -n))
        test = ae.prf(rows[i]["_per_q"][mask].sum(axis=0))
        out.append({"held_out": held, "threshold": rows[i]["threshold"], **test})
    return out


def unlabelled_predictions(cases: list[ae.QueryCase], threshold: float, path_thr=OFF) -> list[dict]:
    """Predicted pairs no label covers (scored as fp or uncounted): what to label next."""
    out = []
    for c in cases:
        if c.query.absent:
            continue
        pred = np.flatnonzero(c.sims.align(threshold, path_thr)[c.row])
        known = set(c.pos_cols) | set(c.neg_cols) | set(c.elsewhere_cols)
        for j in pred:
            if j not in known:
                out.append({"run_id": c.query.run_id, "query_id": c.query.query_id,
                            "target_id": c.sims.target_ids[j],
                            "counted_as": "fp" if c.query.tail_negative else "uncounted"})
    return out
