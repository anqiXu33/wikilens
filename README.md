# CMB-AD: Character Memory Bank for Audio Description

Master's thesis code, Anqi Xu, University of Zurich, 2026. Supervisor: Annette Rios.

CMB-AD is a film-grammar-aware, **training-free** audio-description (AD)
generation system for German- and Italian-language Swiss films, following the
SWISS-AD standard. It targets one specific weakness of VLM-based AD,
**cross-segment character-reference coherence**: deciding, for each character
present in an AD segment, whether to **name, describe, or pronominalise** them.

It is built on a fork of [Shot-by-Shot](https://arxiv.org/abs/2504.01020)
(Xie et al., 2025); see [`README_upstream.md`](README_upstream.md) for the
upstream project.

## Data availability

The Swiss German and Italian audio description data used in this thesis was
provided by SWISS TXT and is not included in this repository. Per-film system
outputs, human evaluation files, ground-truth AD and curated character
descriptions are also excluded. Aggregate results reported in the thesis are
available in `results/benchmarks/`.

To run the pipeline on your own data, place it under `data/` (e.g.
`data/annotations/`), following the format of the public benchmark files in
`resources/annotations/`.

| Thesis chapter | Code |
|---|---|
| Ch 3: Corpus analysis & reference rules (threshold calibration) | `scripts/tools/compute_*.py`, cluster `statistics/` |
| Ch 4: System (CMB-AD) | `stage2/` |
| Ch 5: Evaluation methodology | `scripts/eval/` |
| Ch 6: Results | `results/benchmarks/`, `scripts/eval/` |

## What this thesis contributes

The contribution is on the **generation side** (Stage 2). The perception side is
inherited and used unmodified.

**Mine (Stage 2 and evaluation):**
- **Character Memory Bank**: a deterministic, corpus-calibrated discourse-state
  tracker that emits a per-character reference-form flag for each character in
  each AD segment (`stage2/coherence_manager.py`). Thresholds calibrated in Ch 3:
  `HISTORY_WINDOW=3`, `ANCHOR_INTERVAL=4`, cross-AD `lookback=1`, `max_chars=150`.
- **Deterministic identity propagation**: the PRIORITY rule makes the structured
  `pred_ids` signal authoritative over the VLM's free text
  (`stage2/swiss_promptloader.py`).
- **5-block Swiss prompt builder** + the `--no_cmb` ablation (Baseline 1) and the
  Always-ID baseline (`stage2/stage2_qwen3_always_id.py`, Baseline 2).
- **Character-description curation** pipeline
  (`char_recog/generate_char_descriptions.py`,
  `scripts/tools/audit_descriptions.py`; curated outputs not included).
- **Evaluation suite**: the mention-level reference-strategy metrics that supply
  the primary results (RDA, Form Distribution L1, Re-introduction Rate,
  Continuing PRN Rate, PRD-H), plus surface metrics and human-pairwise analysis
  (`scripts/eval/`). `evaluate_coherence.py` provides the `char_continuity` /
  `pred_ids` derivation that the mention-level suite imports; its own
  re-introduction and pro-drop precision metrics are an earlier approach,
  superseded and not reported.

**Inherited (upstream Shot-by-Shot + supervisor's Swiss-adaptation fork):**
- Stage 1 dense visual description (`stage1/`, Qwen2.5-VL-7B).
- Character recognition (`char_recog/main.py`, InsightFace; ported from AutoAD-Zero).
- Film-grammar modules (`film_grammar/`, DINOv2 shot-scale + thread structure;
  both feed Stage-1 prompt construction; `film_grammar/dinov2_model/` is a
  vendored copy of DINOv2).
- The Qwen2.5-VL / Qwen3 model upgrades and the initial Swiss prompt adaptation
  (Annette Rios, before this thesis).

## Pipeline

```
raw video
  → preprocess/        shot detection + context windows
  → char_recog/        InsightFace face ID  → pred_ids
  → film_grammar/      DINOv2 shot scale + thread structure
  → stage1/            Qwen2.5-VL dense description → text_gen
  → stage2/            Qwen3.5-27B + Character Memory Bank → AD
  → scripts/eval/      ROUGE-L / BERTScore + mention-level suite + human pairwise
```

## Models & reproducibility

- **Stage 1:** Qwen2.5-VL-7B-Instruct.
- **Stage 2:** `Qwen/Qwen3.5-27B` (thinking mode), bfloat16; sampled decoding
  (`temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=8192`), `--seed 42`.
- **Environment:** Python 3.12, `torch==2.6.0+cu126`, `transformers==5.3.0`,
  `flash-attn==2.7.4.post1`, CUDA 12.6 (H100/H200). Full freeze in
  `requirements.lock.txt`; context in `repro_meta.txt`.

## Running

The three Stage-2 conditions share one backbone and differ only in the Stage-2
prompt:

```bash
# CMB-AD (proposed)
python stage2/swissAD_qwen3.py --pred_path=<stage1.csv> --anno_path=<anno.csv> \
    --imdb_chars_path=<imdb_chars_enriched_manual.json> --language=German \
    --LR_slope=2.8 --LR_intercept=3 \
    --few_shot_samples_csv=<few_shot_de.csv> \
    --save_dir=results/stage2/cmb_ad/<movie>/

# Baseline 1 (No-CMB ablation): same script + --no_cmb
# Baseline 2 (Always-ID):       stage2/stage2_qwen3_always_id.py
```

Note: the few-shot example files used in the thesis were drawn from the SWISS TXT
data and are not included. `--few_shot_samples_csv` is optional; without it,
Stage 2 runs without few-shot examples.

Evaluation (3 systems × 8 films → final tables):

```bash
python scripts/eval/evaluate_all_models.py     # surface + coherence
python scripts/eval/mention_level_eval.py      # mention-level reference-strategy suite
```

`mention_level_eval.py` writes the five-metric suite (RDA, Form Distribution L1,
Re-introduction Rate, Continuing PRN Rate, PRD-H) to
`results/benchmarks/mention_level_results.csv` (per movie × condition), with
per-(AD, character) diagnostics under `results/benchmarks/mention_alignment/`
(not included in the public repository, as they contain AD text).

Cluster (SLURM) launchers for all 24 Stage-2 jobs are under `scripts/slurm/stage2/`.

## Repository layout

| Path | Role |
|---|---|
| `stage2/swissAD_qwen3.py` | **CMB-AD driver** (`--no_cmb` → Baseline 1) |
| `stage2/coherence_manager.py` | **Character Memory Bank** (state machine) |
| `stage2/swiss_promptloader.py` | 5-block Swiss prompt + PRIORITY rule |
| `stage2/stage2_qwen3_always_id.py` | **Baseline 2** (Always-ID) |
| `char_recog/`, `stage1/`, `preprocess/`, `film_grammar/` | inherited perception pipeline |
| `scripts/eval/` | evaluation suite (Ch 5/6) |
| `scripts/tools/` | corpus stats + curation + repair tools (Ch 3) |
| `scripts/slurm/` | cluster launchers |
| `results/benchmarks/` | final aggregated tables (Ch 6) |
| `evaluate_coherence.py` | `char_continuity` / `pred_ids` derivation for the mention-level suite (its standalone re-intro / pro-drop precision metrics are superseded) |
| `*/_legacy/`, `experiments/` | superseded code, kept for provenance only |

The 8 evaluation films (4 DE / 4 IT) and dataset details are in thesis Ch 5 §5.2.

## Attribution

Forked from [Shot-by-Shot](https://arxiv.org/abs/2504.01020) (Xie et al., 2025)
via the supervisor's Swiss-adaptation fork. See
[`README_upstream.md`](README_upstream.md) for upstream links.
Character recognition ported from
[AutoAD-Zero](https://github.com/Jyxarthur/AutoAD-Zero).
`film_grammar/dinov2_model/` vendors
[DINOv2](https://github.com/facebookresearch/dinov2). See [`LICENSE`](LICENSE).
