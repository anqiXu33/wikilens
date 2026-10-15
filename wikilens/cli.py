"""WikiLens command line.

    python -m wikilens run --title "Gala (apple)" --pair en-de             # full pipeline (needs an LLM backend)
    python -m wikilens run --title "Gala (apple)" --pair en-de --no-score  # fetch → align only, no LLM
    python -m wikilens run --title "Gala (apple)" --pair en-de --offline   # replay saved scores, no network
    python -m wikilens runs                                                # list saved runs
    python -m wikilens replay [RUN_ID]                                     # summarise a saved run
    python -m wikilens serve [--offline]                                   # web dashboard

Offline mode never touches the network, an LLM or an API key: it only reads
the run directories already in outputs/runs/.
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path("outputs/runs")
HISTORY_PATH = Path("outputs/history.json")

INDIVIDUAL_FIELDS = ("ideology_score", "sentiment_score", "framing_score")


# ── Saved runs ────────────────────────────────────────────────────────────────

def _norm_title(title: str) -> str:
    return unquote(title).replace(" ", "_").strip("_").casefold()


def _title_from_url(url: str) -> str:
    return unquote(urlparse(url).path.split("/wiki/")[-1])


def load_runs() -> list[dict]:
    """Every run directory with a meta.json, newest first."""
    runs = []
    if RUNS_DIR.exists():
        for meta_path in RUNS_DIR.glob("*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                print(f"[!] skipping {meta_path}: {e}", file=sys.stderr)
                continue
            meta["_dir"] = meta_path.parent
            meta.setdefault("run_id", meta_path.parent.name)
            runs.append(meta)
    runs.sort(key=lambda m: (m.get("timestamp", ""), m["run_id"]), reverse=True)
    return runs


def _is_scored(meta: dict) -> bool:
    return meta.get("scored", True) and bool(meta.get("file1_url"))


def find_run(run: str | None) -> dict:
    """Resolve a run id, a run directory path, or (None) the newest scored run."""
    runs = load_runs()
    if run is None:
        scored = [m for m in runs if _is_scored(m)]
        if not scored:
            raise SystemExit("No scored runs in outputs/runs/.")
        return scored[0]
    for m in runs:
        if run in (m["run_id"], str(m["_dir"])) or Path(run).resolve() == m["_dir"].resolve():
            return m
    raise SystemExit(f"No saved run '{run}'. See: python -m wikilens runs")


def find_run_for(title: str, pair: str) -> dict:
    """Newest scored run for this article and language pair."""
    want_title = _norm_title(title)
    want_langs = set(pair.split("-"))
    runs = load_runs()
    for m in runs:
        if (_is_scored(m) and _norm_title(m.get("title", "")) == want_title
                and set(m.get("languages_found", [])) == want_langs):
            return m
    available = [f"  {m.get('title')} ({'-'.join(m.get('languages_found', []))})"
                 for m in runs if _is_scored(m)]
    raise SystemExit(
        f"No saved scored run for '{title}' ({pair}); offline mode can only replay saved runs.\n"
        "Saved runs:\n" + ("\n".join(available) or "  (none)"))


def _score_file(meta: dict, key: str) -> Path:
    # file URLs look like /runs/<run_id>/scores/<file>.json
    url = meta[key]
    return meta["_dir"] / url.split(f"/runs/{meta['run_id']}/", 1)[1]


# ── Replay ────────────────────────────────────────────────────────────────────

def _mean(values: list, signed: bool = True) -> str:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return "n/a"
    return f"{statistics.fmean(nums):{'+' if signed else ''}.2f}"


def _summarise_scores(records: list[dict], top: int) -> list[str]:
    base = records[0]["base_lang"] if records else "?"
    other = records[0].get("omission_lang", "?") if records else "?"
    unscored = sum(1 for r in records if any(r.get(f) is None for f in INDIVIDUAL_FIELDS))
    no_omission = sum(1 for r in records if r.get("omission_score") is None)
    unaligned = sum(1 for r in records if r.get("omission_basis") == "unaligned")
    lines = [
        f"Base {base.upper()} (omission measured against {other.upper()}): {len(records)} chunks",
        "  mean ideology {}  sentiment {}  framing {}  omission {}".format(
            *(_mean([r.get(f) for r in records]) for f in INDIVIDUAL_FIELDS),
            _mean([r.get("omission_score") for r in records], signed=False)),
        f"  bias scores missing: {unscored}   omission missing: {no_omission}   unaligned (omission = 1): {unaligned}",
    ]
    ranked = sorted((r for r in records if r.get("omission_score") is not None),
                    key=lambda r: r["omission_score"], reverse=True)[:top]
    if ranked:
        lines.append(f"  most omitted in {other.upper()}:")
        for r in ranked:
            text = " ".join(r.get("chunk_text", "").split())
            if len(text) > 90:
                text = text[:87] + "..."
            lines.append(f"    {r['omission_score']:.2f}  {r['chunk_id']}  [{r.get('section', '?')}]  {text}")
    return lines


def replay(meta: dict, top: int = 3, as_json: bool = False) -> None:
    if not _is_scored(meta):
        raise SystemExit(f"Run {meta['run_id']} was not scored (--no-score); nothing to replay.")
    files = [_score_file(meta, k) for k in ("file1_url", "file2_url")]
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    if as_json:
        print(json.dumps({"run": {k: v for k, v in meta.items() if k != "_dir"},
                          "scores": {f.name: r for f, r in zip(files, records)}},
                         ensure_ascii=False, indent=2))
        return
    prov = meta.get("provenance") or {}
    scoring = prov.get("scoring") or {}
    print(f"Run       {meta['run_id']}")
    print(f"Article   {meta.get('title', '?').replace('_', ' ')}   "
          f"pair {'-'.join(meta.get('languages_found', []))}   timestamp {meta.get('timestamp', '?')}")
    print(f"Scoring   {scoring.get('backend', 'not recorded')}"
          + (f" / {scoring['model']}" if scoring.get("model") else ""))
    if prov.get("kind"):
        print(f"Note      {prov['kind']}: {prov.get('note', '')}")
    for r in records:
        print()
        print("\n".join(_summarise_scores(r, top)))
    print()
    print("Scores are unvalidated LLM judgments (see docs/review.md §3). "
          f"Files: {files[0].parent}")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_runs(args) -> int:
    runs = load_runs()
    if not runs:
        print("No saved runs in outputs/runs/.")
        return 0
    for m in runs:
        state = "scored" if _is_scored(m) else "aligned only"
        print(f"{m['run_id']}\n    {m.get('title', '?').replace('_', ' ')}  "
              f"{'-'.join(m.get('languages_found', []))}  {state}")
    return 0


def cmd_replay(args) -> int:
    replay(find_run(args.run), top=args.top, as_json=args.json)
    return 0


def cmd_run(args) -> int:
    if args.title:
        url = f"https://en.wikipedia.org/wiki/{quote(args.title.replace(' ', '_'))}"
        title = args.title
    else:
        url = args.url
        title = _title_from_url(url)

    if args.offline:
        replay(find_run_for(title, args.pair), top=args.top, as_json=args.json)
        return 0

    if args.backend:
        # The scorer reads SCORING_BACKEND when it is imported (after this).
        os.environ["SCORING_BACKEND"] = args.backend
    from app.pipeline import PipelineError, run_analysis
    try:
        result = asyncio.run(run_analysis(url, args.pair, do_score=not args.no_score))
    except PipelineError as e:
        print(f"[✗] {e.detail}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[✗] Analysis failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print()
    print(f"Run directory: {RUNS_DIR / result['run_id']}")
    for lang, s in result["summary"].items():
        print(f"  {lang}: {s['total_chunks']} chunks, {len(s['sections'])} sections")
    for name, counts in result["alignment_summary"].items():
        print(f"  {name}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if result.get("scored"):
        print()
        replay(find_run(result["run_id"]), top=args.top)
    else:
        print("Not scored (--no-score).")
    return 0


def cmd_serve(args) -> int:
    if args.offline:
        os.environ["WIKILENS_OFFLINE"] = "1"
    import uvicorn
    print(f"WikiLens on http://{args.host}:{args.port}"
          + ("  (offline: saved runs only, /analyze disabled)" if args.offline else ""))
    uvicorn.run("main:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m wikilens", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="analyse one article for one language pair")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--title", help='English Wikipedia title, e.g. "Gala (apple)"')
    src.add_argument("--url", help="English Wikipedia URL")
    r.add_argument("--pair", required=True, help="language pair with English on one side, e.g. en-de")
    r.add_argument("--backend", choices=["ollama", "watsonx"],
                   help="scoring backend (default: SCORING_BACKEND from .env, else ollama)")
    mode = r.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true",
                      help="replay the newest saved scores for this article and pair; "
                           "no network, LLM or API key")
    mode.add_argument("--no-score", action="store_true",
                      help="fetch, chunk, embed and align, but skip LLM scoring")
    r.add_argument("--top", type=int, default=3, help="most-omitted chunks to show (default 3)")
    r.add_argument("--json", action="store_true", help="with --offline: print the raw run and scores as JSON")
    r.set_defaults(func=cmd_run)

    ls = sub.add_parser("runs", help="list saved runs in outputs/runs/")
    ls.set_defaults(func=cmd_runs)

    rp = sub.add_parser("replay", help="summarise a saved run (offline)")
    rp.add_argument("run", nargs="?", help="run id or directory (default: newest scored run)")
    rp.add_argument("--top", type=int, default=3, help="most-omitted chunks to show (default 3)")
    rp.add_argument("--json", action="store_true", help="print the raw run and scores as JSON")
    rp.set_defaults(func=cmd_replay)

    sv = sub.add_parser("serve", help="start the web dashboard")
    sv.add_argument("--offline", action="store_true",
                    help="serve saved runs only; new analyses are disabled")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # All paths (outputs/, static/, app/cached_articles/) are relative to the repo root.
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    return args.func(args)
