"""`python -m wikilens experiment`: baseline vs A (LLM rerank) vs B (sentence-level
pairing) against eval/labels/. See wikilens/alignment_experiments.py.

Writes eval/experiments/alignment_<pair>_methods.{json,md}. Does not change
the pipeline.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wikilens import alignment_eval as ae
from wikilens import alignment_experiments as ax
from wikilens.evaluate import LABELS_DIR

SENTENCE_KS = (None, 2, 3)   # None = max


def _clean(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _summarise(name: str, cases, rows, rank_scores, kind: str, note: str = "", unlabelled_ignore=None) -> dict:
    best = ax.best_row(rows)
    out = {"method": name, "kind": kind, "note": note,
           "best": _clean(best), "coverage": ax.coverage(cases, rank_scores),
           "cross_article": ax.cross_article(cases, rows) if kind == "sweep" else None,
           "curve": [{k: r[k] for k in ("threshold", "precision", "recall", "f1", "tp", "fp", "fn")} for r in rows]}
    if unlabelled_ignore is not None:
        out["best_unlabelled_ignored"] = _clean(unlabelled_ignore(best["threshold"]))
    out["unlabelled_predictions"] = ax.unlabelled_predictions(cases, best["threshold"],
                                                             best.get("_path_thr", ax.OFF))
    return out


def run_pair(pair: str, queries, args) -> dict:
    from wikilens.cli import find_run

    run_dirs = {q.run_id: find_run(q.run_id)["_dir"] for q in queries}
    base_cache, emb_cache, text_cache = {}, {}, {}

    def base_sims(run_id, q_lang, t_lang):
        key = (run_id, q_lang, t_lang)
        if key not in base_cache:
            base_cache[key] = ae.run_similarities(run_dirs[run_id], q_lang, t_lang)
        return base_cache[key]

    def texts_for(run_id, lang):
        if (run_id, lang) not in text_cache:
            df = ae.load_embeddings(run_dirs[run_id], lang)
            text_cache[(run_id, lang)] = dict(zip(df["chunk_id"], df["chunk_text"]))
        return text_cache[(run_id, lang)]

    def cases_with(score_for):
        """Cases whose sims carry score_for(run_id, q_lang, t_lang) as chunk_sim (routing off)."""
        built = {}

        def sims_for(*key):
            if key not in built:
                built[key] = ax.as_similarities(base_sims(*key), score_for(*key))
            return built[key]
        return ae.resolve_cases(queries, sims_for, partial=args.partial)

    grid = ae.parse_grid(args.grid)
    methods = []

    # Baseline: chunk cosine, as the aligner (pipeline default point, and routing off swept).
    base_cases = ae.resolve_cases(queries, base_sims, partial=args.partial)
    chunk_rank = lambda c: c.sims.chunk_sim[c.row]
    dc, dp = ae.default_thresholds(pair.split("-"))
    default_rows = ax.sweep(base_cases, [dc], path_thr=dp, unlabelled=args.unlabelled)
    default_rows[0]["_path_thr"] = dp
    methods.append(_summarise(f"baseline: chunk cosine, pipeline default ({dc:.2f}/{dp:.2f}, routing on)",
                              base_cases, default_rows, chunk_rank, "fixed",
                              unlabelled_ignore=lambda t: ax.sweep(base_cases, [t], dp, "ignore")[0]))
    rows = ax.sweep(base_cases, grid, unlabelled=args.unlabelled)
    methods.append(_summarise("baseline: chunk cosine, routing off", base_cases, rows, chunk_rank, "sweep",
                              unlabelled_ignore=lambda t: ax.sweep(base_cases, [t], unlabelled="ignore")[0]))

    # B: sentence-level pairing.
    def sent_emb(run_id, lang):
        if (run_id, lang) not in emb_cache:
            emb_cache[(run_id, lang)] = ax.sentence_embeddings(run_dirs[run_id], lang)
        return emb_cache[(run_id, lang)]

    for k in SENTENCE_KS:
        def score_for(run_id, q_lang, t_lang, k=k):
            b = base_sims(run_id, q_lang, t_lang)
            return ax.sentence_scores(sent_emb(run_id, q_lang), sent_emb(run_id, t_lang), b.query_ids, b.target_ids, k)
        cases = cases_with(score_for)
        rows = ax.sweep(cases, grid, unlabelled=args.unlabelled)
        label = "B: sentence max" if k is None else f"B: sentence top-{k} mean"
        methods.append(_summarise(label, cases, rows, lambda c: c.sims.chunk_sim[c.row], "sweep",
                                  unlabelled_ignore=lambda t, cases=cases: ax.sweep(cases, [t], unlabelled="ignore")[0]))

    # A: LLM rerank of the top-N chunk-cosine candidates.
    llm = {"status": "not run"}
    try:
        if args.cached_only:
            raise SystemExit("--cached-only")
        scorer, info = ax.llm_backend()
    except SystemExit as e:
        scorer, info = None, None
        llm["reason"] = str(e)
    model = (info or {}).get("model") or args.llm_model
    cache_path = ax.EXPERIMENTS_DIR / f"llm_rerank_{ax.model_slug(model or 'unknown')}.json"
    cache = ax.RerankCache(cache_path)
    call = None
    if scorer is not None:
        call = lambda msgs: scorer._extract_generated_text(scorer.inference(msgs))
    by_run = defaultdict(list)
    for q in queries:
        by_run[(q.run_id, q.query_lang, q.target_lang)].append(q.query_id)
    try:
        llm_scores, stats = ax.llm_rerank_scores(by_run, base_sims, texts_for, args.top_n, cache, model or "unknown", call)
    except Exception as e:  # SystemExit: no cached reply; else the LLM request failed
        llm_scores = None
        llm.setdefault("reason", f"{type(e).__name__}: {str(e)[:300]}")
    except SystemExit as e:
        llm_scores = None
        llm.setdefault("reason", str(e))
    finally:
        if cache.entries:
            cache.save(info or {"model": model})
    if llm_scores is not None:
        llm = {"status": "ok", "model": model, "cache": str(cache_path), **stats}
        cases = cases_with(lambda *key: llm_scores[key])
        rank = lambda c: c.sims.chunk_sim[c.row]
        for thr, label in ((1.0, "match+partial"), (2.0, "match only")):
            rows = ax.sweep(cases, [thr], unlabelled=args.unlabelled)
            methods.append(_summarise(f"A: LLM rerank top-{args.top_n}, {label}", cases, rows, rank, "fixed",
                                      note=model,
                                      unlabelled_ignore=lambda t: ax.sweep(cases, [t], unlabelled="ignore")[0]))
    else:
        # Without LLM replies only A's candidate pool can be scored: top-N by chunk cosine.
        methods.append({"method": f"A: LLM rerank top-{args.top_n} (not run)", "kind": "pool",
                        "note": llm.get("reason", ""), "best": None,
                        "coverage": ax.coverage(base_cases, chunk_rank, ks=(args.top_n,))})

    return {
        "pair": pair,
        "runs": sorted({q.run_id for q in queries}),
        "n_queries": len(queries),
        "n_absent": sum(q.absent for q in queries),
        "partial": args.partial,
        "unlabelled": args.unlabelled,
        "grid": args.grid,
        "top_n": args.top_n,
        "llm": llm,
        "methods": methods,
    }


def _pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def markdown(res: dict) -> str:
    L = [f"# Alignment methods, {res['pair']}",
         "",
         f"Generated by `python -m wikilens experiment`. {res['n_queries']} labelled chunks "
         f"({res['n_absent']} absent in the other language), partial = {res['partial']}, "
         f"unlabelled predictions = {res['unlabelled']}. "
         f"Thresholds swept over {res['grid']}; \"best\" is chosen on these same labels (optimistic).",
         f"LLM rerank: {res['llm']['status']}"
         + (f" ({res['llm'].get('model')}, {res['llm'].get('calls', 0)} calls, {res['llm'].get('cached', 0)} cached)"
            if res['llm']['status'] == 'ok' else f" — {res['llm'].get('reason', '')}"),
         "",
         "| method | thr | P | R | F1 | F1 95% CI | tp / fp / fn | P, unlabelled ignored | in top-1 | top-3 | top-5 | top-10 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in res["methods"]:
        cov = m["coverage"]
        cells = [_pct(cov.get(f"chunks@{k}")) for k in ax.COVERAGE_KS]
        b = m["best"]
        if b is None:
            L.append(f"| {m['method']} | | | | | | | | {' | '.join(cells)} |")
            continue
        ig = m.get("best_unlabelled_ignored")
        L.append(f"| {m['method']} | {b['threshold']:.2f} | {b['precision']:.3f} | {b['recall']:.3f} | "
                 f"**{b['f1']:.3f}** | {b['f1_95ci'][0]:.2f}–{b['f1_95ci'][1]:.2f} | "
                 f"{b['tp']} / {b['fp']} / {b['fn']} | {ig['precision']:.3f} ({ig['uncounted']} unc.) | "
                 f"{' | '.join(cells)} |")
    L += ["", "Coverage (\"in top-k\"): share of the non-absent labelled chunks with a known correct "
          "target among the method's k highest-scored targets.", "",
          "Coverage in detail: chunks = labelled chunks with a known correct target in the top k; "
          "pairs = known positive pairs in the top k; outside-top-3 = the matches the annotator named "
          "outside chunk cosine's top 3 (the only positives not picked by chunk cosine, so the fairest "
          "test of a different ranking).", "",
          "| method | chunks@3 | pairs@3 | outside-top-3@3 | chunks@10 | pairs@10 | outside-top-3@10 |",
          "|---|---|---|---|---|---|---|"]
    for m in res["methods"]:
        cov = m["coverage"]
        n = cov.get("n_elsewhere_named", 0)
        cell = lambda key: _pct(cov.get(key)) + (f" ({round(cov[key] * n)}/{n})" if "elsewhere" in key and key in cov else "")
        L.append(f"| {m['method']} | " + " | ".join(cell(f"{a}@{k}") for k in (3, 10)
                                                     for a in ("chunks", "pairs", "elsewhere")) + " |")
    L.append("")
    cross = [m for m in res["methods"] if m.get("cross_article")]
    if cross:
        L += ["Cross-article: threshold picked on the other article, scored on the held-out one.", "",
              "| method | held out | thr | P | R | F1 |", "|---|---|---|---|---|---|"]
        for m in cross:
            for c in m["cross_article"]:
                L.append(f"| {m['method']} | {c['held_out'].replace('_', ' ')} | {c['threshold']:.2f} | "
                         f"{c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} |")
    return "\n".join(L) + "\n"


def print_table(res: dict) -> None:
    print(f"\n== {res['pair']} ==  {res['n_queries']} labelled chunks; LLM rerank: {res['llm']['status']}"
          + (f" ({res['llm'].get('reason')})" if res['llm']['status'] != 'ok' else ""))
    print(f"  {'method':<58} {'thr':>5}  {'P':>5}  {'R':>5}  {'F1':>5}   in top-1/3/5/10")
    for m in res["methods"]:
        cov = m["coverage"]
        cells = "/".join(_pct(cov.get(f"chunks@{k}")) for k in ax.COVERAGE_KS)
        b = m["best"]
        if b is None:
            print(f"  {m['method']:<58} {'':>5}  {'':>5}  {'':>5}  {'':>5}   {cells}")
        else:
            print(f"  {m['method']:<58} {b['threshold']:>5.2f}  {b['precision']:.3f}  {b['recall']:.3f}  "
                  f"{b['f1']:.3f}   {cells}")


def cmd_experiment(args) -> int:
    files = [Path(p) for p in args.labels] if args.labels else sorted(LABELS_DIR.glob("*.json"))
    queries, warnings = ae.load_label_files(files)
    for w in warnings:
        print(f"[!] {w}")
    by_pair = defaultdict(list)
    for q in queries:
        if not args.pair or q.pair == ae.pair_key(args.pair.split("-")):
            by_pair[q.pair].append(q)
    if not by_pair:
        raise SystemExit("No labelled chunks.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pair, qs in sorted(by_pair.items()):
        res = run_pair(pair, qs, args)
        stem = out_dir / f"alignment_{pair}_methods"
        stem.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        stem.with_suffix(".md").write_text(markdown(res), encoding="utf-8")
        print_table(res)
        print(f"\n  {stem.with_suffix('.json')}\n  {stem.with_suffix('.md')}")
    return 0
