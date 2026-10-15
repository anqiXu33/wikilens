"""LLM rerank of alignment candidates (EN-ZH default, see align_thresholds).

For each query chunk, the top-N target chunks by the method's embedding score
are sent to the configured scoring LLM (SCORING_BACKEND) in one request, which
labels each candidate match / partial / none following
docs/annotation_guidelines.md. Candidates labelled match or partial are the
aligned pairs. The prompt is the one scored in docs/alignment_experiments.md
(`python -m wikilens experiment`, method A), so its cached replies stay valid.
"""
import json
import re

import numpy as np

LANG_NAMES = {"en": "English", "zh": "Chinese", "de": "German", "fr": "French", "it": "Italian"}
LLM_LABELS = ("match", "partial", "none")
ACCEPTED = ("match", "partial")


def rerank_messages(query_text: str, q_lang: str, candidates: list[tuple[str, str]], t_lang: str) -> list[dict]:
    """Prompt for one query chunk and its candidates [(chunk_id, text)].
    Mirrors docs/annotation_guidelines.md."""
    ql, tl = LANG_NAMES.get(q_lang, q_lang), LANG_NAMES.get(t_lang, t_lang)
    system = (
        f"You compare a paragraph from the {ql} Wikipedia with candidate paragraphs from the {tl} "
        "Wikipedia article on the same topic, and decide for each candidate whether it states the "
        "same fact(s) as the source paragraph.\n"
        "Labels:\n"
        '  "match": the candidate covers all the facts of the source paragraph.\n'
        '  "partial": the candidate covers at least one fact of the source paragraph, even if that '
        "fact is only a small part of either paragraph.\n"
        '  "none": the candidate shares no fact with the source paragraph; being on the same general '
        "topic is not enough.\n"
        "The same indicator reported for a different year or with a different number counts as the "
        "same fact (language editions are updated at different times).\n"
        "Reply with a single JSON object and nothing else, mapping every candidate id to its label, "
        'e.g. {"en_0001": "none", "en_0002": "partial"}.'
    )
    lines = [f"Source paragraph ({ql}):", query_text, "", f"Candidates ({tl}):"]
    for cid, text in candidates:
        lines += [f"[{cid}] {text}", ""]
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines).rstrip()}]


def parse_rerank_reply(text: str, candidate_ids: list[str]) -> dict[str, str]:
    """{chunk_id: label}; candidates the reply leaves out or garbles count as "none"."""
    start, end = text.find("{"), text.rfind("}")
    data = {}
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            data = {}
    if not isinstance(data, dict):
        data = {}
    loose = {re.sub(r"[\[\]\s]", "", str(k)).casefold(): v for k, v in data.items()}
    out = {}
    for cid in candidate_ids:
        v = data.get(cid, loose.get(cid.casefold()))
        v = str(v).strip().casefold() if v is not None else "none"
        out[cid] = v if v in LLM_LABELS else "none"
    return out


def llm_caller():
    """(call(messages) -> reply text, backend info), or (None, reason) when no
    scoring LLM is configured."""
    try:
        from app.services import scorer
        scorer.check_backend_config()
        token = scorer.get_access_token(scorer.IBM_API_KEY) if scorer.SCORING_BACKEND == "watsonx" else None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    def call(messages):
        return scorer._extract_generated_text(scorer.inference(messages, token))
    return call, scorer.backend_info()


def rerank(scores: np.ndarray, query_ids: list[str], query_texts: list[str], q_lang: str,
           target_ids: list[str], target_texts: list[str], t_lang: str,
           top_n: int, call, fallback: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    """(matched (|Q|, |T|) bool, {(i, j): label}, stats).

    One LLM call per query chunk with its top_n targets by `scores`. If the
    first call fails the LLM is taken to be unreachable and every row uses
    `fallback`; a later failure falls back for that row only."""
    matched = np.zeros_like(fallback, dtype=bool)
    labels: dict[tuple[int, int], str] = {}
    stats = {"calls": 0, "failed": 0, "fallback_rows": 0, "errors": []}
    for i in range(len(query_ids)):
        top = np.argsort(-scores[i], kind="stable")[:top_n]
        cand = [(target_ids[j], target_texts[j]) for j in top]
        try:
            stats["calls"] += 1
            got = parse_rerank_reply(call(rerank_messages(query_texts[i], q_lang, cand, t_lang)),
                                     [c for c, _ in cand])
        except Exception as e:
            stats["failed"] += 1
            if len(stats["errors"]) < 3:
                stats["errors"].append(f"{query_ids[i]}: {type(e).__name__}: {str(e)[:200]}")
            if stats["calls"] == 1:
                stats["unreachable"] = True
                stats["fallback_rows"] = len(query_ids)
                return fallback.copy(), {}, stats
            matched[i] = fallback[i]
            stats["fallback_rows"] += 1
            continue
        for j in top:
            labels[(i, int(j))] = got[target_ids[j]]
            matched[i, j] = got[target_ids[j]] in ACCEPTED
    return matched, labels, stats
