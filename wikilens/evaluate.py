"""`python -m wikilens evaluate`: score the aligner against the labels in
eval/labels/ over a grid of (chunk_thr, path_thr) settings.

Counting, per labelled query chunk q (see docs/annotation_guidelines.md):
  * q marked "absent" (nothing in the other article matches): every pair the
    aligner reports for q is a false positive.
  * otherwise, over q's labelled top-k candidates: predicted & positive = TP,
    predicted & "none" = FP, positive & not predicted = FN. "partial" counts as
    positive by default (--partial).
  * q marked "elsewhere" (a match exists, but not among the top-k candidates):
    a recall loss for the aligner, never "absent". If the annotator named the
    matching chunk(s), each is a TP when the aligner matched it and an FN when
    not; otherwise q adds one FN at every threshold. Reported as fn_elsewhere.
  * predicted pairs outside the labelled candidates (--unlabelled): with the
    default "auto" they are false positives when q's lowest-ranked candidate
    was labelled "none" (the ranking had already run out of matches), and
    otherwise unknown: reported as "uncounted", not scored.
Precision / recall / F1 are micro-averaged over pairs. Positives outside the
top-k candidates are known only where the annotator marked "elsewhere".
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from wikilens import alignment_eval as ae

LABELS_DIR = Path("eval/labels")
EVAL_DIR = Path("eval")
REFERENCE = (0.45, 0.35)
OFF = math.inf  # path_thr above any similarity: routing never applies


def _fmt_thr(x: float) -> str:
    return "off" if math.isinf(x) else f"{x:.2f}"


def _row(name: str, chunk_thr: float, path_thr: float, m: dict, ci=None) -> str:
    ci_s = f"  [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
    return (f"  {name:<22} {chunk_thr:>6.2f} {_fmt_thr(path_thr):>6}   "
            f"{m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}{ci_s}"
            f"   tp {m['tp']} fp {m['fp']} fn {m['fn']}  fn_elsewhere {m['fn_elsewhere']}  uncounted {m['uncounted']}")


def _with(grid: list[float], extra) -> list[float]:
    return sorted(set(grid) | {round(x, 4) for x in extra})


def evaluate_pair(pair: str, cases: list, args) -> dict:
    langs = pair.split("-")
    default = ae.default_thresholds(langs)
    chunk_grid = _with(ae.parse_grid(args.chunk_grid), [default[0], REFERENCE[0]])
    path_grid = _with(ae.parse_grid(args.path_grid), [default[1], REFERENCE[1]]) + [OFF]

    counts = ae.grid_counts(cases, chunk_grid, path_grid, args.unlabelled)  # (C, P, N, 5)
    total = counts.sum(axis=2)
    metrics = [[ae.prf(total[i, j]) for j in range(len(path_grid))] for i in range(len(chunk_grid))]
    f1 = np.array([[m["f1"] for m in row] for row in metrics])
    prec = np.array([[m["precision"] for m in row] for row in metrics])
    rec = np.array([[m["recall"] for m in row] for row in metrics])

    def at(c, p):
        i, j = chunk_grid.index(round(c, 4)), path_grid.index(p if math.isinf(p) else round(p, 4))
        return {"chunk_thr": c, "path_thr": None if math.isinf(p) else p, **metrics[i][j],
                "f1_95ci": list(ae.bootstrap_f1(counts[i, j], seed=args.seed))}

    bi, bj = ae.best_index(f1, prec)
    off_i = int(np.lexsort((-prec[:, -1], -f1[:, -1]))[0])
    refs = {"pipeline_default": default}
    if REFERENCE != default:
        refs["reference_0.45_0.35"] = REFERENCE
    result = {
        "pair": pair,
        "runs": sorted({c.query.run_id for c in cases}),
        "articles": sorted({c.query.title for c in cases}),
        "n_queries": len(cases),
        "n_absent": sum(c.query.absent for c in cases),
        "n_elsewhere": sum(c.query.elsewhere for c in cases),
        "n_elsewhere_named": sum(c.query.elsewhere and bool(c.query.elsewhere_ids) for c in cases),
        "n_elsewhere_pairs": int(sum(len(c.elsewhere_cols) or c.query.elsewhere for c in cases)),
        "n_labelled_pairs": int(sum(len(c.pos_cols) + len(c.neg_cols) for c in cases)),
        "n_positive_pairs": int(sum(len(c.pos_cols) for c in cases)),
        "partial": args.partial,
        "unlabelled": args.unlabelled,
        "reference_points": {name: at(*thr) for name, thr in refs.items()},
        "best": at(chunk_grid[bi], path_grid[bj]),
        "best_routing_off": at(chunk_grid[off_i], OFF),
        "cross_article": _cross_article(cases, counts, chunk_grid, path_grid, f1),
    }
    result["_grid"] = (chunk_grid, path_grid, f1, prec, rec, metrics)
    return result


def _cross_article(cases, counts, chunk_grid, path_grid, f1_all):
    """Pick thresholds on the other articles, score them on the held-out one."""
    titles = sorted({c.query.title for c in cases})
    if len(titles) < 2:
        return None
    out = []
    for held in titles:
        mask = np.array([c.query.title == held for c in cases])
        train, test = counts[:, :, ~mask].sum(axis=2), counts[:, :, mask].sum(axis=2)
        f1_train = ae.f1_of(train)
        prec_train = np.array([[ae.prf(train[i, j])["precision"] for j in range(train.shape[1])]
                               for i in range(train.shape[0])])
        i, j = ae.best_index(f1_train, prec_train)
        out.append({"held_out": held, "chunk_thr": chunk_grid[i],
                    "path_thr": None if math.isinf(path_grid[j]) else path_grid[j],
                    **ae.prf(test[i, j]),
                    "held_out_best_f1": float(ae.f1_of(test).max())})
    return out


def write_outputs(res: dict, out_dir: Path) -> list[Path]:
    chunk_grid, path_grid, f1, prec, rec, metrics = res["_grid"]
    stem = out_dir / f"alignment_{res['pair']}"
    csv_path = stem.with_name(stem.name + "_grid.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_thr", "path_thr", "precision", "recall", "f1", "tp", "fp", "fn", "fn_elsewhere", "uncounted"])
        for i, c in enumerate(chunk_grid):
            for j, p in enumerate(path_grid):
                m = metrics[i][j]
                w.writerow([f"{c:.4g}", _fmt_thr(p), f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                            f"{m['f1']:.4f}", m["tp"], m["fp"], m["fn"], m["fn_elsewhere"], m["uncounted"]])
    json_path = stem.with_name(stem.name + "_summary.json")
    json_path.write_text(json.dumps({k: v for k, v in res.items() if not k.startswith("_")},
                                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    png_path = stem.with_name(stem.name + "_f1.png")
    plot(res, png_path)
    return [csv_path, json_path, png_path]


def _edges(values: list[float]) -> np.ndarray:
    """Cell edges halfway between grid values, so each cell is centred on its value."""
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        return np.array([v[0] - 0.025, v[0] + 0.025])
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])


def plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chunk_grid, path_grid, f1, *_ = res["_grid"]
    ink, muted, grid_c = "#1b1b1a", "#5f5e5a", "#e4e3de"
    series = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
    default = res["reference_points"]["pipeline_default"]
    best = res["best"]

    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.patch.set_facecolor("white")

    # Heatmap over the routed grid (routing "off" is a line on the right).
    finite = [j for j, p in enumerate(path_grid) if not math.isinf(p)]
    ys = [path_grid[j] for j in finite]
    im = ax_h.pcolormesh(_edges(chunk_grid), _edges(ys), f1[:, finite].T, cmap="Blues",
                         vmin=0, vmax=max(1e-9, float(f1.max())), edgecolors="face")
    for pt, marker, label in ((default, "o", "default"), (best, "*", "best")):
        if pt["path_thr"] is not None:
            ax_h.scatter([pt["chunk_thr"]], [pt["path_thr"]], marker=marker, s=140 if marker == "*" else 70,
                         facecolor="white", edgecolor=ink, linewidth=1.5, zorder=3)
            ax_h.annotate(f"{label} {pt['f1']:.2f}", (pt["chunk_thr"], pt["path_thr"]), xytext=(8, 6),
                          textcoords="offset points", color=ink, fontsize=9,
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax_h.set_xlabel("chunk_thr", color=muted)
    ax_h.set_ylabel("path_thr", color=muted)
    ax_h.set_title("F1 by threshold pair", color=ink, loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.03)
    cb.set_label("F1", color=muted)

    # F1 vs chunk_thr for a few path_thr settings.
    lines = [(f"path_thr {default['path_thr']:.2f} (default)", default["path_thr"])]
    if best["path_thr"] is not None and best["path_thr"] != default["path_thr"]:
        lines.append((f"path_thr {best['path_thr']:.2f} (best)", best["path_thr"]))
    lines.append(("routing off", OFF))
    for (label, p), color in zip(lines, series):
        j = path_grid.index(p)
        ax_l.plot(chunk_grid, f1[:, j], color=color, linewidth=2, marker="o", markersize=4, label=label)
    for pt, marker, label in ((default, "o", "default"), (best, "*", "best")):
        ax_l.scatter([pt["chunk_thr"]], [pt["f1"]], marker=marker, s=140 if marker == "*" else 70,
                     facecolor="white", edgecolor=ink, linewidth=1.5, zorder=3)
        ax_l.annotate(f"{label} ({pt['chunk_thr']:.2f}/{_fmt_thr(pt['path_thr'] or OFF)}) F1 {pt['f1']:.2f}",
                      (pt["chunk_thr"], pt["f1"]), xytext=(6, -14 if label == "default" else 8),
                      textcoords="offset points", color=ink, fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax_l.set_xlabel("chunk_thr", color=muted)
    ax_l.set_ylabel("F1", color=muted)
    ax_l.set_ylim(0, 1)
    ax_l.set_title("F1 vs chunk_thr", color=ink, loc="left", fontsize=11)
    ax_l.grid(True, color=grid_c, linewidth=0.8)
    ax_l.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in (ax_h, ax_l):
        ax.tick_params(colors=muted, labelsize=8)
        for s in ax.spines.values():
            s.set_color(grid_c)
    ax_l.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Alignment F1 against human labels, {res['pair']}  "
                 f"({res['n_queries']} chunks, {res['n_labelled_pairs']} labelled pairs, "
                 f"partial = {res['partial']})", color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_report(res: dict, outputs: list[Path]) -> None:
    print(f"\n== {res['pair']} ==  articles: {', '.join(a.replace('_', ' ') for a in res['articles'])}")
    print(f"  {res['n_queries']} labelled chunks, {res['n_labelled_pairs']} labelled candidate pairs, "
          f"{res['n_positive_pairs']} positive among them (partial = {res['partial']})")
    print(f"  absent in the other language:        {res['n_absent']} chunks")
    print(f"  match exists outside the candidates: {res['n_elsewhere']} chunks "
          f"({res['n_elsewhere_named']} with the matching chunk named, "
          f"{res['n_elsewhere'] - res['n_elsewhere_named']} unnamed = 1 FN each at every threshold); "
          f"{res['n_elsewhere_pairs']} positive pairs")
    print(f"\n  {'setting':<22} {'chunk':>6} {'path':>6}   prec   rec    F1     [F1 95% CI]")
    for name, m in res["reference_points"].items():
        print(_row(name, m["chunk_thr"], m["path_thr"] if m["path_thr"] is not None else OFF, m, m["f1_95ci"]))
    for name in ("best", "best_routing_off"):
        m = res[name]
        print(_row(name, m["chunk_thr"], m["path_thr"] if m["path_thr"] is not None else OFF, m, m["f1_95ci"]))
    print("  (best = chosen on these same labels, so it is optimistic)")
    if res["cross_article"]:
        print("\n  Cross-article: thresholds picked on the other article(s), scored on the held-out one")
        for c in res["cross_article"]:
            print(f"    {c['held_out'].replace('_', ' '):<22} {c['chunk_thr']:.2f}/"
                  f"{_fmt_thr(c['path_thr'] if c['path_thr'] is not None else OFF)}  "
                  f"P {c['precision']:.3f}  R {c['recall']:.3f}  F1 {c['f1']:.3f}  "
                  f"(oracle F1 on held-out {c['held_out_best_f1']:.3f})")
    print("\n  " + "\n  ".join(str(p) for p in outputs))


def cmd_evaluate(args) -> int:
    from wikilens.cli import find_run
    files = [Path(p) for p in args.labels] if args.labels else sorted(LABELS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No label files in {LABELS_DIR}/. Make some with: python -m wikilens annotate RUN_ID")
    queries, warnings = ae.load_label_files(files)
    for w in warnings:
        print(f"[!] {w}")
    by_pair = defaultdict(list)
    for q in queries:
        if not args.pair or q.pair == ae.pair_key(args.pair.split("-")):
            by_pair[q.pair].append(q)
    if not by_pair:
        raise SystemExit("No labelled chunks" + (f" for {args.pair}." if args.pair else "."))

    cache = {}

    def sims_for(run_id, q_lang, t_lang):
        key = (run_id, q_lang, t_lang)
        if key not in cache:
            cache[key] = ae.run_similarities(find_run(run_id)["_dir"], q_lang, t_lang)
        return cache[key]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for pair, qs in sorted(by_pair.items()):
        cases = ae.resolve_cases(qs, sims_for, partial=args.partial)
        res = evaluate_pair(pair, cases, args)
        outputs = write_outputs(res, out_dir)
        print_report(res, outputs)
        summary[pair] = {k: v for k, v in res.items() if not k.startswith("_")}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
