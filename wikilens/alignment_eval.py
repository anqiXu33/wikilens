"""Alignment quality: candidate pairs for annotation, and scoring the aligner
against human labels.

The aligner (app/services/aligner_*.py) is re-implemented here in vectorised
form so a whole threshold grid can be scored from one similarity matrix. For a
query chunk q and the other article's chunks T it does:

    1. path routing: find the section of T whose (English) section-name
       embedding is most similar to q's; if that similarity >= path_thr, only
       chunks in that section are candidates, otherwise all of T is.
    2. chunk matching: every candidate with cosine(q, t) >= chunk_thr is a match.

tests/test_alignment_eval.py checks this against the aligners' saved output.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LABELS_FORMAT = "wikilens-alignment-labels/1"
LABEL_VALUES = ("match", "partial", "none")


# ── Run data ──────────────────────────────────────────────────────────────────

def load_embeddings(run_dir: Path, lang: str) -> pd.DataFrame:
    path = Path(run_dir) / "embeddings" / f"embedded_{lang}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found: the run has no embeddings for '{lang}'.")
    return pd.read_parquet(path).reset_index(drop=True)


def load_section_names(run_dir: Path) -> dict[str, dict[str, str]]:
    """{lang: {original section name: English}} if the run recorded it."""
    path = Path(run_dir) / "section_names_en.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def run_languages(meta: dict) -> list[str]:
    langs = list(meta.get("languages_found") or [])
    if len(langs) != 2:
        raise SystemExit(f"Run {meta.get('run_id')} does not record two languages (languages_found={langs}).")
    return langs


def default_query_lang(langs: list[str]) -> str:
    """The non-English side (the aligners' "source" language); else the second language."""
    non_en = [l for l in langs if l != "en"]
    return non_en[0] if len(non_en) == 1 else langs[1]


def pair_key(langs) -> str:
    """Order-independent name for a language pair, English first: 'en-zh'."""
    return "-".join(sorted(set(langs), key=lambda l: (l != "en", l)))


def default_thresholds(langs) -> tuple[float, float]:
    """The thresholds the pipeline uses for this pair (keyed by the non-English language)."""
    from app.services.align_thresholds import DEFAULT_THRESHOLDS, thresholds_for
    non_en = [l for l in set(langs) if l != "en"]
    return thresholds_for(non_en[0]) if len(non_en) == 1 else DEFAULT_THRESHOLDS


def _unit(rows) -> np.ndarray:
    m = np.stack([np.asarray(r, dtype=np.float64) for r in rows])
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


@dataclass
class Similarities:
    """Everything the aligner looks at for queries Q against targets T."""
    query_ids: list[str]
    target_ids: list[str]
    chunk_sim: np.ndarray       # (|Q|, |T|) cosine of chunk embeddings
    best_path_sim: np.ndarray   # (|Q|,) similarity of the best-matching target section
    in_best_section: np.ndarray  # (|Q|, |T|) target chunk lies in that best section

    def align(self, chunk_thr: float, path_thr: float) -> np.ndarray:
        """(|Q|, |T|) bool: pairs the aligner reports as matched."""
        routed = self.best_path_sim >= path_thr
        allowed = np.where(routed[:, None], self.in_best_section, True)
        return allowed & (self.chunk_sim >= chunk_thr)


def similarities(query: pd.DataFrame, target: pd.DataFrame) -> Similarities:
    chunk_sim = _unit(query["chunk_embedding"]) @ _unit(target["chunk_embedding"]).T
    # Same section order as the aligners (drop_duplicates keeps the first
    # occurrence), so argmax breaks ties the same way.
    sections = target.drop_duplicates("section")
    path_sim = _unit(query["section_embedding"]) @ _unit(sections["section_embedding"]).T
    best = path_sim.argmax(axis=1)
    best_section = sections["section"].to_numpy()[best]
    in_best = target["section"].to_numpy()[None, :] == best_section[:, None]
    return Similarities(
        query_ids=query["chunk_id"].tolist(),
        target_ids=target["chunk_id"].tolist(),
        chunk_sim=chunk_sim,
        best_path_sim=path_sim[np.arange(len(best)), best],
        in_best_section=in_best,
    )


def run_similarities(run_dir: Path, query_lang: str, target_lang: str) -> Similarities:
    return similarities(load_embeddings(run_dir, query_lang), load_embeddings(run_dir, target_lang))


# ── Annotation candidates ─────────────────────────────────────────────────────

def _section_label(section: str, names: dict[str, str]) -> dict:
    english = names.get(section)
    return {"section": section, "section_en": english if english and english != section else None}


def build_candidates(run_dir: Path, meta: dict, query_lang: str, k: int = 3,
                     sample: int | None = None, seed: int = 0) -> dict:
    """The data behind the annotation page: each query chunk with its top-k
    target chunks by raw chunk similarity (no routing, no threshold)."""
    langs = run_languages(meta)
    if query_lang not in langs:
        raise SystemExit(f"--from {query_lang}: run {meta['run_id']} has languages {langs}.")
    target_lang = next(l for l in langs if l != query_lang)
    q_df = load_embeddings(run_dir, query_lang)
    t_df = load_embeddings(run_dir, target_lang)
    sims = similarities(q_df, t_df)
    names = load_section_names(run_dir)

    rows = list(range(len(q_df)))
    if sample is not None and sample < len(rows):
        rng = np.random.default_rng(seed)
        rows = sorted(rng.choice(rows, size=sample, replace=False).tolist())

    items = []
    for i in rows:
        q = q_df.iloc[i]
        top = np.argsort(-sims.chunk_sim[i], kind="stable")[:k]
        items.append({
            "query_id": q["chunk_id"],
            "text": q["chunk_text"],
            **_section_label(q["section"], names.get(query_lang, {})),
            "candidates": [{
                "chunk_id": t_df.iloc[j]["chunk_id"],
                "chunk_sim": round(float(sims.chunk_sim[i, j]), 4),
                "text": t_df.iloc[j]["chunk_text"],
                **_section_label(t_df.iloc[j]["section"], names.get(target_lang, {})),
            } for j in top],
        })
    target_chunks = [{"chunk_id": r["chunk_id"], "text": r["chunk_text"],
                      **_section_label(r["section"], names.get(target_lang, {}))}
                     for _, r in t_df.iterrows()]
    return {
        "run_id": meta["run_id"],
        "title": meta.get("title", ""),
        "pair": pair_key(langs),
        "query_lang": query_lang,
        "target_lang": target_lang,
        "k": k,
        "sample": {"n": len(rows), "seed": seed} if sample is not None else None,
        "n_query_chunks": len(q_df),
        "items": items,
        "target_chunks": target_chunks,
    }


def labels_filename(run_id: str, query_lang: str, langs: list[str]) -> str:
    """eval/labels/<run_id>.json for the default direction, <run_id>.from-<lang>.json otherwise."""
    if query_lang == default_query_lang(langs):
        return f"{run_id}.json"
    return f"{run_id}.from-{query_lang}.json"


# ── Labels ────────────────────────────────────────────────────────────────────

@dataclass
class LabelledQuery:
    run_id: str
    title: str
    pair: str
    query_lang: str
    target_lang: str
    query_id: str
    absent: bool
    labels: dict[str, str] = field(default_factory=dict)  # target chunk_id -> match/partial/none
    # The lowest-ranked candidate was labelled "none": the similarity ranking had
    # already run out of matches, so chunks ranked below it are taken as non-matches.
    tail_negative: bool = False
    # "有对应段落，但不在这 k 个候选里": a match exists outside the candidates,
    # optionally named by chunk id. Counted as a miss unless the aligner found it.
    elsewhere: bool = False
    elsewhere_ids: list[str] = field(default_factory=list)


def load_label_files(paths) -> tuple[list[LabelledQuery], list[str]]:
    """Read label files; returns the labelled queries and warnings."""
    queries, warnings = [], []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != LABELS_FORMAT:
            warnings.append(f"{path}: not a {LABELS_FORMAT} file, skipped")
            continue
        skipped = 0
        for it in data["items"]:
            labelled = [c for c in it["candidates"] if c.get("label") in LABEL_VALUES]
            labels = {c["chunk_id"]: c["label"] for c in labelled}
            last = min(labelled, key=lambda c: c.get("chunk_sim", 0), default=None)
            absent = bool(it.get("absent"))
            elsewhere = not absent and bool(it.get("elsewhere"))
            candidate_ids = {c["chunk_id"] for c in it["candidates"]}
            elsewhere_ids = ([cid for cid in dict.fromkeys(it.get("elsewhere_ids") or []) if cid not in candidate_ids]
                             if elsewhere else [])
            if not absent and not elsewhere and not labels:
                skipped += 1
                continue
            queries.append(LabelledQuery(
                run_id=data["run_id"], title=data.get("title", ""), pair=data["pair"],
                query_lang=data["query_lang"], target_lang=data["target_lang"],
                query_id=it["query_id"], absent=absent,
                labels={} if absent else labels,
                tail_negative=absent or (last is not None and last["label"] == "none"
                                         and len(labelled) == len(it["candidates"])
                                         and (not elsewhere or bool(elsewhere_ids))),
                elsewhere=elsewhere, elsewhere_ids=elsewhere_ids))
        if skipped:
            warnings.append(f"{path}: {skipped} unlabelled item(s) ignored")
    return queries, warnings


# ── Scoring ───────────────────────────────────────────────────────────────────

def _positive(label: str, partial: str) -> bool | None:
    """True/False for a labelled pair, None if the pair is left out."""
    if label == "match":
        return True
    if label == "partial":
        return {"positive": True, "negative": False, "ignore": None}[partial]
    return False


@dataclass
class QueryCase:
    """One labelled query, resolved against its run's similarity matrix."""
    query: LabelledQuery
    sims: Similarities
    row: int
    pos_cols: np.ndarray
    neg_cols: np.ndarray
    elsewhere_cols: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def counts(self, chunk_thr: float, path_thr: float, unlabelled: str = "auto") -> np.ndarray:
        """[tp, fp, fn, uncounted, fn_elsewhere] at one threshold setting.

        uncounted: predicted pairs outside the labelled pairs that were not scored.
        fn_elsewhere: the part of fn from matches outside the candidates.
        unlabelled: how to count predicted pairs outside the labelled pairs:
        "fp", "ignore", or "auto" (fp when the lowest-ranked candidate is "none"
        and any match outside the candidates was named by id)."""
        pred = self.sims.align(chunk_thr, path_thr)[self.row]
        n_pred = int(pred.sum())
        if self.query.absent:
            # Nothing in the other article matches, so every prediction is wrong.
            return np.array([0, n_pred, 0, 0, 0])
        tp = int(pred[self.pos_cols].sum())
        fp = int(pred[self.neg_cols].sum())
        fn = len(self.pos_cols) - tp
        fn_elsewhere = 0
        if self.query.elsewhere:
            if len(self.elsewhere_cols):
                found = int(pred[self.elsewhere_cols].sum())
                tp += found
                fn_elsewhere = len(self.elsewhere_cols) - found
            else:
                # The match is somewhere outside the candidates but unnamed:
                # one miss at every threshold (the aligner's recall loss).
                fn_elsewhere = 1
            fn += fn_elsewhere
        outside = n_pred - tp - fp
        if unlabelled == "fp" or (unlabelled == "auto" and self.query.tail_negative):
            return np.array([tp, fp + outside, fn, 0, fn_elsewhere])
        return np.array([tp, fp, fn, outside, fn_elsewhere])


def resolve_cases(queries: list[LabelledQuery], sims_for, partial: str = "positive") -> list[QueryCase]:
    """sims_for(run_id, query_lang, target_lang) -> Similarities."""
    cases = []
    for q in queries:
        sims = sims_for(q.run_id, q.query_lang, q.target_lang)
        try:
            row = sims.query_ids.index(q.query_id)
        except ValueError:
            raise SystemExit(f"Labelled chunk {q.query_id} is not in run {q.run_id}; "
                             "were the labels made from a different run?")
        col = {cid: j for j, cid in enumerate(sims.target_ids)}
        pos, neg = [], []
        for cid, label in q.labels.items():
            if cid not in col:
                raise SystemExit(f"Labelled candidate {cid} is not in run {q.run_id}.")
            y = _positive(label, partial)
            if y is True:
                pos.append(col[cid])
            elif y is False:
                neg.append(col[cid])
        for cid in q.elsewhere_ids:
            if cid not in col:
                raise SystemExit(f"Chunk {cid} (named as a match outside the candidates of {q.query_id}) "
                                 f"is not in run {q.run_id}.")
        cases.append(QueryCase(q, sims, row, np.array(pos, dtype=int), np.array(neg, dtype=int),
                               np.array([col[cid] for cid in q.elsewhere_ids], dtype=int)))
    return cases


def prf(counts: np.ndarray) -> dict:
    tp, fp, fn = (int(x) for x in counts[:3])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "uncounted": int(counts[3]) if len(counts) > 3 else 0,
            "fn_elsewhere": int(counts[4]) if len(counts) > 4 else 0}


def grid_counts(cases: list[QueryCase], chunk_grid, path_grid, unlabelled: str = "auto") -> np.ndarray:
    """(len(chunk_grid), len(path_grid), len(cases), 5) per-query counts (see QueryCase.counts)."""
    out = np.zeros((len(chunk_grid), len(path_grid), len(cases), 5), dtype=np.int64)
    for i, c in enumerate(chunk_grid):
        for j, p in enumerate(path_grid):
            for n, case in enumerate(cases):
                out[i, j, n] = case.counts(c, p, unlabelled)
    return out


def f1_of(counts: np.ndarray) -> float:
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return f1


def bootstrap_f1(per_query: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of micro F1, resampling labelled queries. per_query: (Nq, 4)."""
    if len(per_query) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_query), size=(n, len(per_query)))
    f1s = f1_of(per_query[idx].sum(axis=1))
    return float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))


def best_index(f1_grid: np.ndarray, prec_grid: np.ndarray) -> tuple[int, int]:
    """Highest F1; ties go to the higher precision, then the first grid point."""
    order = np.lexsort((-prec_grid.ravel(), -f1_grid.ravel()))
    return np.unravel_index(order[0], f1_grid.shape)


def parse_grid(spec: str) -> list[float]:
    """'0.2:0.9:0.05' -> [0.2, 0.25, ..., 0.9]; '0.3,0.4' -> [0.3, 0.4]."""
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        n = int(round((hi - lo) / step)) + 1
        return [round(lo + i * step, 4) for i in range(n)]
    return [round(float(x), 4) for x in spec.split(",")]
