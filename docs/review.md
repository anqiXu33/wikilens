# WikiLens: Repository Review

*Reviewed 2026-10-15 at commit `a6c8ea7` (initial commit). Read-only review: I changed no other files and made no LLM calls. Every claim about results is based on the files already in `outputs/` and `app/cached_articles/`.*

---

## TL;DR

WikiLens is an interesting, recruiter-friendly idea: cross-lingual Wikipedia comparison with LaBSE alignment, LLM-as-judge scoring and a polished Wikipedia-style UI. The README tells that story well. **As an ML/NLP portfolio piece, though, the repo is currently weaker than its README suggests**, for four reasons:

1. **Correctness bugs corrupt the results.** The chunker reorders sections alphabetically and merges sentences across section boundaries. The whole lead section becomes one chunk. Runs overwrite each other's files, and the saved outputs show the cross-contamination (details in §3.5).
2. **The omission score is mostly inverted.** In 64% of the real scored records, the LLM returned `preserved/base` instead of `1 − preserved/base`. The prompt specifies both formulas, contradicting itself (§3.2).
3. **There is no evaluation of the system's outputs.** No human labels exist for alignment, omission or bias scores. The threshold "calibration" chooses thresholds by looking at the omission rate, which is the quantity the tool is supposed to measure.
4. **It cannot be reproduced end to end from the repo.** Dependencies are unpinned or missing, the pipeline relies on live Wikipedia, Wikidata and Google Translate calls, and the only scoring path needs either an IBM key or a local Ollama. There is also no offline "replay" mode that uses the saved scores.

An interviewer who clones the repo and opens `outputs/` will find that the "latest" score files are all zeros (§3.5). Fix that first.

Overall, as a portfolio project for ML/NLP roles right now: **5/10**. With the five items in §6 done, it could reach **8/10**. The core idea and the UI are strong. What's missing is the rigor and evidence that ML interviewers look for.

---

## 1. Code structure

### What's there

```
main.py                     FastAPI app: the real end-to-end orchestrator (412 lines)
app/services/
  fetcher.py                Wikidata langlinks → wikipedia-api → sentence split → cache
  chunker.py                paragraph chunking
  embedder.py               LaBSE + Google-Translate section names → parquet
  aligner_en_to_source.py   EN→X alignment          ┐ ~90% duplicated code,
  aligner_source_to_en.py   X→EN alignment          ┘ different output schemas
  analyze_*.py              alignment → CSV (not used by main.py)
  scorer.py                 prompts, watsonx/Ollama clients, parsing, scoring loop (614 lines)
app/api/routes.py           empty
run_*.py (8 files)          ad-hoc step runners
eval_alignment_threshold.py threshold distribution plots
test_pipeline.py            smoke script (not a test suite)
static/index.html           1,282-line single-file frontend
README.ipynb                Chinese module notes (out of date)
```

### Strengths
- The pipeline stages (fetch → chunk → embed → align → score) map cleanly onto modules, and the README diagram matches them.
- The scorer abstracts over two backends (watsonx, Ollama) through a small `_watsonx_body` / `_ollama_body` / `_extract_generated_text` layer. This is a nice design point to mention in interviews.
- The fetcher has sensible engineering touches: Wikidata QID → sitelinks for cross-language titles, retry with backoff, a User-Agent, and per-language heading exclusion lists.

### Problems
- **Stages communicate through fixed global file paths** (`outputs/chunked/chunked_{lang}.json`, `outputs/embeddings/embedded_{lang}.parquet`, …) that contain neither the article name nor a run ID. Every run overwrites the previous one. Two concurrent API requests would clobber each other, and the history view reads alignment files from whichever run happened last (§3.5).
- **Heavy work at import time.** `embedder.py` loads LaBSE (~1.8 GB) on import. `fetcher.py` calls `nltk.download(...)` and `spacy.load("de_core_news_sm")` on import. `scorer.py` reads `.env` on import. Importing `main` therefore takes a long time and needs network access, which makes unit testing very hard.
- **Duplicated logic.**
  - The two aligners are near copies with different field names (`tgt_chunk_id` means the English chunk in one file and the target chunk in the other). That forces `scorer.py` to use string-built keys such as `chunk.get(src_field + "_chunk_id")`.
  - `LANG_THRESHOLDS` is defined twice.
  - `eval_alignment_threshold.py` re-implements the chunker and aligner instead of importing them, and the copies have already drifted (§3.1).
  - `process_file` and `async_process_file` in `scorer.py` duplicate each other, and the sync one is unused.
- **Blocking work inside an `async` endpoint.** `/analyze` runs CPU-bound embedding, alignment and synchronous HTTP (Google Translate) on the event loop, so the server freezes while it runs. The frontend fakes progress with `setTimeout` timers (`index.html:766-768`).
- **Dead or broken code.**
  - `app/api/routes.py` is empty.
  - `run_chunk_threshold_experiment.py` calls `get_chunked_dataframe(df, lang=..., threshold=...)`, but that function no longer accepts `lang`, so the script raises a `TypeError`. Its threshold sweep would be meaningless anyway, because `threshold` is now ignored.
  - `run_*` scripts and `analyze_*` hard-code `["zh","de","sw","eu"]`, which doesn't match the README's supported languages (de, zh, it, fr).
  - The `scorer.py` `__main__` block parses arguments and then does nothing (`# await score(...)` is commented out).
  - `get_chunked_dataframe(threshold=...)` takes a parameter that it ignores.
  - There is a stray top-level `__init__.py`.
- **Silent failure mode.** If scoring raises for any reason (one bad HTTP response anywhere in `asyncio.gather` is enough), `main.py` catches it and returns `create_dummy_response`. That writes all-zero scores under the real filenames and returns `"success": true`. The UI can't distinguish this from a perfectly neutral article.
- **Fragile response parsing.** `parse_response` raises `KeyError` if the model's JSON is missing a field. In the `JSONDecodeError` branch, an unknown `score_type` falls through and hits an unbound `generated_json`.
- **`record_output` re-reads and re-writes the whole JSON file for every chunk.** That's O(n²), and if a run crashes halfway, the file is left partly written.
- **Mixed languages in code and docs.** Chinese comments sit next to English ones, and `README.ipynb` (Chinese) describes modules that no longer exist (`aligner_en_centered_structured.py`, similarity-based chunking, a different chunk ID format). For an English-language job market, translate it or delete it.
- **No tests, no linting, no CI.** `test_pipeline.py` needs the LLM backend and prints "✅ PASSED" unconditionally. Its match preview also reads keys that don't exist (`src_chunk_text`, top-level `chunk_sim`), so it always prints empty strings and `N/A`.

### Verified correctness bugs in the chunker and fetcher

These bugs directly affect what gets aligned and scored:

| Bug | Where | Effect (measured on `app/cached_articles/`) |
|---|---|---|
| Sentences sorted by `(section, paragraph_index)` with **string** ordering of `section` | `chunker.py:38` | Document order is lost. `en_0000` for *Climate change* is "Adaptation", not the lead. Chunk IDs are alphabetical-section positions, not article positions. |
| Single-sentence paragraphs are merged into the *previous chunk in that alphabetical order*, even when it belongs to another section | `chunker.py:52` | Counted **1–38 cross-section merges per article** (e.g. zh Hitler 38, sw COVID 34, it Svizzera 33, de COVID-19 24). The merged text ends up under the wrong section, which also disturbs section routing in the aligner. |
| The lead is split into sentences but every sentence gets `paragraph_index=0` | `fetcher.py:116-130` | The **entire lead becomes one chunk** (*Climate change*: 28 sentences, 3,390 chars; this is the max-length outlier in `chunked_en.json`). The most-read part of an article is scored as one blob. |
| The cache is keyed only by title and never invalidated | `fetcher.py:144` | All cached files use the section label `"Intro"`, but current code emits `"(Lead)"`. The cache was produced by an older version of the code and silently overrides the current logic. |
| `en_Riley_Elf.json` actually contains the English **Mini** article (redirect) | cache | One of the bundled "parallel" pairs isn't parallel. The eval script excludes it, but the cache still ships it. |

---

## 2. Reproducibility

### Dependencies
- `requirements.txt` pins web and utility packages but leaves **the ML-critical ones unpinned**: `sentence-transformers`, `scikit-learn`, `spacy`, `aiohttp`, `pyarrow`, `deep-translator`. The LaBSE model revision isn't pinned either, and neither is the spaCy model version. A fresh install in 2026 may produce different embeddings or tokenization than the published runs did.
- **Missing dependencies:** `matplotlib`, which `eval_alignment_threshold.py` needs. `pytest` would be needed too if tests are added.
- The spaCy German model is a separate manual step, and NLTK data is downloaded at import time.
- No Python version is pinned (README: "3.10+"). There's no `pyproject.toml`, lockfile, Dockerfile or Makefile.

### Fetching data
- Data comes live from Wikipedia and Wikidata, so articles change daily. **No revision IDs are stored**, so nobody can re-fetch the exact text that was analysed. `wikipediaapi` can return `page.lastrevid`. Store it, or fetch through `action=parse&oldid=…`.
- `app/cached_articles/` (24 files, 3.3 MB) serves as a de facto frozen dataset, but nothing documents it: which revision each file came from, when it was fetched, or with which code version (see the `"Intro"` vs `"(Lead)"` mismatch above).
- The embedder calls **Google Translate** through `deep-translator` for every non-English section name. That's an unofficial, rate-limited, non-deterministic network dependency in the middle of the ML pipeline, and translation failures fall back silently to the original text. Cache the translations to a file, or use LaBSE directly with no translation and measure the difference.

### Running end to end
- The only real end-to-end path is the web UI (`POST /analyze`). No CLI exists for "analyse article X for language pair Y and write results to a directory". The `run_*.py` scripts cover individual steps, use inconsistent language lists, and one of them is broken.
- Scoring needs either IBM credentials (expired) or a local Ollama with `llama3.2`. The README states clearly that the 3B fallback produces noisier scores, so **the published results (70B on watsonx) can't be reproduced by anyone**, and Ollama numbers aren't comparable to them.
- Even with the LLM held fixed, runs aren't deterministic. `scorer.py:390` builds `omission_text` from a **`set`** of strings and truncates it at 1,200 chars. Python's string-hash randomization makes the set order, and therefore the truncated text, vary between processes.
- There's no "replay" or "offline" mode that renders the UI from saved score files without an LLM. One would make the project demo-able by any reviewer in about two minutes.

### What's in `outputs/` (data formats, for reference)

| Path | Format | Key fields |
|---|---|---|
| `fetched/fetched_{lang}.json` | list of sentence records | `language, section, paragraph_index, sentence_index, sentence_text` |
| `chunked/chunked_{lang}.json` | list of chunk records | `chunk_id ({lang}_{NNNN}), chunk_text, section, paragraph_index, language, sentence_count` |
| `embeddings/embedded_{lang}.parquet` | chunk records + vectors | adds `chunk_embedding`, `section_embedding` (768-d LaBSE) |
| `alignments_en_to_source/en_to_{x}.json` | one record per **EN** chunk | `tgt_chunk_id` (= EN id!), `tgt_text, from_lang, match_type (1:0/1:1/1:N), matched_chunk_ids, matches[{src_chunk_id, src_text, chunk_sim, path_sim, combined_score, src_section}]` |
| `alignments_source_to_en/{x}_to_en.json` | one record per **X** chunk | `src_lang, tgt_lang, src_chunk_id, src_section, src_text, match_type, matched_chunk_ids, matches[{tgt_chunk_id, tgt_section, tgt_text, chunk_sim, path_sim, combined_score}]` |
| `scores/base_{a}_omit_{b}_scores.json` | one record per base chunk | `chunk_id, base_lang, chunk_text, omission_lang, omission_text, ideology_score, sentiment_score, framing_score, base_points, omission_points, omission_score, section` |
| `scores/base_{a}_omit_{b}_{timestamp}.json` | timestamped copy of the above for `history.json` | same |
| `history.json`, `meta.json` | run metadata for the UI | `title, languages_found, timestamp, file1_url, file2_url` |

The README doesn't document any of this.

---

## 3. Is the evaluation sound?

**Short answer: no. The system's outputs have no evaluation, and the one quantitative analysis (threshold calibration) has a circular objective.**

### 3.1 Alignment threshold calibration

`eval_alignment_threshold.py` plots each chunk's best-match cosine similarity and the "1:0 rate" (fraction of chunks with no match) at each threshold, then thresholds are picked by eye.

- **No ground truth.** No pair is labelled as a true or false alignment, so precision, recall and F1 can't be computed. A threshold that "gives ~4% 1:0" says nothing about whether the matches are correct.
- **The objective is circular.** In this system, a 1:0 rate *is* the omission signal: unmatched chunks are hard-coded to `omission_score = 1.0` (`scorer.py:413`). Choosing thresholds to keep the 1:0 rate low tunes the instrument to find fewer omissions. It's also unclear whether different similarity distributions across language pairs reflect calibration differences in LaBSE or real differences in content overlap. The method can't tell those apart.
- **The eval doesn't match production.**
  - The eval script embeds *untranslated* section names, but production translates them first.
  - The eval uses a global `PATH_SIM_THRESHOLD = 0.45`, but production uses per-language path thresholds.
  - The eval's `CALIBRATED["EN-DE"] = 0.30`, but production uses **0.45**. That change was justified by "section-name translation fix", and no rerun backs it up. The comment in the aligner itself says that "0.45 discards ~47% of real matches" for EN-DE.
- **Small and inconsistent sample.** The README says 5 articles per pair, the docstring says 3, and EN-DE actually uses 2 (COVID-19 pandemic, Gala apple). Those are 11 article pairs in total, with no confidence intervals.
- **The matching rule is weak.** Every candidate above the threshold counts as a match. In the shipped `en_to_de.json`, **60 of 95 EN chunks are "1:N"** and only 20 are "1:1". That's thresholded similarity search, not alignment. Natural baselines to compare against: mutual-best / top-1, a margin-based criterion (Artetxe & Schwenk 2019, standard for LaBSE bitext mining), or monotonic alignment such as Bertalign / Vecalign. Section routing is also a *hard* filter. If the best-matching section passes `path_thr`, only that section is searched, so content that moved to a different section reads as "omitted".

### 3.2 LLM bias scores (ideology, sentiment, framing)

- **No validation of any kind:** no human-annotated subset, no agreement metric (Spearman / Krippendorff's α), no test–retest stability, no cross-model agreement (watsonx 70B vs Ollama), and no sanity checks on synthetic perturbations (e.g. inject loaded adjectives and confirm the sentiment score moves).
- **Construct validity issues.**
  - The three axes overlap heavily. "Sentiment" (negative↔positive) and "Framing" (negative↔positive framing) as defined are almost the same question.
  - "Ideology" (left↔right) is applied to every chunk, including apple cultivars and plate tectonics. In the saved runs, `-0.2` is the most frequent ideology score in almost every file, which is a prior, not a signal.
  - The single few-shot example labels "brave citizens stood up against the corrupt regime" as ideology −0.6 (left), which is debatable and could anchor the model.
- **Scores are absolute per chunk, not paired.** Each language's chunk is scored separately. The UI then subtracts EN and X scores on 1:1 pairs (`index.html:924-938`) and treats `null` as 0. Interviewers will ask about rating-scale drift across languages, because the model may rate German text differently from English text with the same content. A paired prompt ("which of these two framings is more X?") or a translate-then-score control would address this.
- **The prompt has internal problems.**
  - It says "classify each *sentence*", but the input is a multi-sentence chunk.
  - The omission prompt's output spec says `omission_score = omission_points / base_points`, which contradicts the formula given above it (`1 − …`).
  - The few-shot outputs use different `chunk_id`s than their inputs (`en_sub_0004` → `en_cn_0004`).
  - The model computes the omission arithmetic itself, and **it gets it wrong most of the time**. Across the 280 real (non-dummy) records in the timestamped score files, only **92 (33%)** satisfy `omission_score = 1 − omission_points/base_points`. **179 (64%) report the inverted value `omission_points/base_points`**, which is exactly the contradictory spec in the prompt. In the other 9, `omission_points > base_points` or the value is arbitrary. So the omission heatmap is largely backwards: a fully preserved chunk can show as fully omitted. Compute the score in code from the two counts, clip `omission_points ≤ base_points`, and use only the model's counts.

### 3.3 Omission

- Omission depends heavily on the alignment threshold, but the two are never analysed together.
- When a chunk is "1:N", `omission_text` is several concatenated paragraphs cut at 1,200 chars in random order, so whether a fact counts as "preserved" depends on set ordering.
- With the 28-sentence lead chunk, a single omission score covers the whole lead.

### 3.4 UI thresholds
The divergence colours (red ≥ 0.5, amber ≥ 0.25) and coverage-gap colours (red ≥ 15%) are arbitrary and undocumented. The summary cards also use only 1:1 pairs, which are a minority (20 of 95 for EN→DE).

### 3.5 What the saved outputs show

Checked with a script over `outputs/`:

- **The current `base_*_scores.json` files are dummy fallbacks.** All 95 + 65 records have ideology = sentiment = framing = omission = **0.0** and an empty `omission_text`. The latest *Climate change* run hit the silent fallback described in §1. Someone browsing the repo would take these as real results.
- **The history entries are cross-contaminated across articles.**
  - "Gala (apple)" (timestamp `20250320_172800`, while every other run is from 2026): the Gala apple chunks are labelled with *Climate change* sections ("Adaptation", "Anthropogenic climate change").
  - "COVID-19" (`20260320_182949`): the EN file contains COVID text under Climate-change section labels, and the DE file contains **Klimawandel** (German *Climate change*) text, not COVID.
  
  Root cause: chunk IDs (`en_0000`) are positional and not unique per article, and `enrich_sections` / the frontend's alignment fetch (`index.html:842`) always use whichever article ran last.
- **The latest *Climate change* history file covers 29 of 95 EN chunks** (DE: 65/65), so that run was partial.

In short, `outputs/` holds no clean, complete, scored run.

---

## 4. What the README is missing

The README explains the idea and the setup well, and the "Why LaBSE / why per-language thresholds / why paragraphs" notes are good interview material. What's missing:

1. **Results.** There are no numbers, no example finding ("the German *Climate change* article omits X; the Chinese *Switzerland* article frames Y more positively"), no screenshot or GIF of the UI, and no link to a demo. This is the biggest gap for recruiters. (Compare your `dialogsum-qlora` README, which leads with a results table, an ablation, error analysis and a demo link.)
2. **Evaluation methodology and its limits.** It should say plainly that the scores are unvalidated LLM judgments. The threshold claims ("calibrated on 5 articles") need correcting to match the code (2–3 articles, no labels, DE threshold changed without a rerun).
3. **Your contribution.** This was a team project for the IBM watsonx challenge. Say who did what and credit teammates. Interviewers will ask, and it also matters for making the repo public.
4. **A way to run it without credentials.** The README needs an offline demo path (replay saved scores, or run on `app/cached_articles/`). It should also describe the bundled data (which articles, fetch dates, revision IDs).
5. **Data formats and outputs:** the schemas in §2, and what `match_type` 1:0 / 1:1 / 1:N means.
6. **A CLI or script usage section** for running the pipeline without the UI.
7. **Housekeeping.**
   - The clone URL is a placeholder (`your-username`).
   - There's no license file. Pick one before the repo goes public, and check with teammates.
   - No badges.
   - The project-structure tree omits `test_pipeline.py`, the `run_*.py` scripts, `analyze_*` and `README.ipynb`.
   - There's no mention of `sw`/`eu`, which appear in the cache and scripts.
8. **Ethics and limitations of "bias" scoring.** Using a single US-trained LLM to judge left/right ideology across cultures is itself a bias source. One paragraph on this shows maturity.
9. **Cost and runtime for watsonx**, plus model and prompt versions recorded per run.

---

## 5. Portfolio verdict

| Dimension | Now | Notes |
|---|---|---|
| Idea / problem framing | ★★★★☆ | Cross-lingual NLP, multilingual embeddings, LLM-as-judge, a real use case |
| Engineering | ★★★☆☆ | Working full-stack app and a dual LLM backend, but global-state file I/O, duplication, import-time side effects |
| Correctness | ★★☆☆☆ | Chunker ordering and merge bugs, lead blob, inverted omission scores, run contamination, silent dummy fallback |
| ML evaluation | ★☆☆☆☆ | No labels, circular threshold objective, no score validation |
| Reproducibility | ★★☆☆☆ | Unpinned ML deps, live data without revision IDs, Google Translate in the loop, credentialed scorer |
| Presentation | ★★★☆☆ | Good README prose and UI; no results, no screenshot, no contribution statement |

**What it signals today:** "Can build an LLM app." **What ML/NLP hiring managers look for:** "Can measure whether the system works, and knows where it fails." The highest-leverage work is evidence, not more features.

---

## 6. Top 5 improvements, in priority order

| # | Improvement | Effort | Value |
|---|---|---|---|
| 1 | Fix the correctness bugs and make runs isolated and honest | ~1–1.5 days | **Very high** |
| 2 | Build a small gold-labelled evaluation (alignment + omission + score validity) | ~4–6 days | **Very high** |
| 3 | Make it reproducible and demo-able offline | ~1–2 days | **High** |
| 4 | Rewrite the README around results and evidence | ~0.5–1 day (after #1–#3) | **High** |
| 5 | Consolidate code and add unit tests + CI | ~1.5–2 days | **Medium–high** |

### 1. Fix the correctness bugs and make runs isolated and honest (~1–1.5 days, very high value)
- Chunker: keep document order (sort by an explicit section index from the fetcher, not by section name), and never merge across section boundaries.
- Fetcher: split the lead into real paragraphs (`page.summary.split("\n")`), store `revision_id` and `fetched_at`, and version or invalidate the cache. Regenerate `app/cached_articles/`, and drop or fix the mislabelled `Riley_Elf` pair.
- Write each run to `outputs/runs/{title}_{pair}_{timestamp}/` and make chunk IDs article-unique. Have history entries point at their own alignment files.
- Remove the dummy fallback, or at least mark dummy results clearly (`"scored": false`) and return an error. Use `asyncio.gather(..., return_exceptions=True)` and record per-chunk failures as `null` with a reason.
- Compute `omission_score` in code from `base_points` / `omission_points`, clip the counts, and fix the contradictory formula in the prompt (64% of the saved scores are currently inverted).
- Make `omission_text` deterministic: order by similarity, not by `set`.
- Delete the contaminated and dummy files from `outputs/`, then commit **one clean, complete, scored example run** (watsonx if you can still get access, otherwise Ollama, labelled as such).

### 2. Build a small gold-labelled evaluation (~4–6 days, very high value)
This turns the project from an app into ML work.
- **Alignment.** Annotate ~150–200 candidate EN–X chunk pairs per language pair (start with EN-DE and EN-ZH, which you can read yourself) as match / partial / no match. Report precision, recall and F1 against threshold, and compare against baselines: current thresholding, top-1, mutual-best, margin scoring, and section routing on/off. Pick thresholds by F1 rather than by 1:0 rate, and rerun on the *production* code path.
- **Omission.** Annotate ~100 aligned pairs with base_points / preserved_points by hand. Report MAE and correlation of the LLM omission score, and how the score moves as the alignment threshold changes.
- **Bias scores.** On ~100 chunks, measure (a) test–retest stability, (b) agreement between two models (e.g. a 70B model via any API vs Ollama 8B), (c) a perturbation test (insert loaded wording into neutral text and confirm the score moves in the right direction), and (d) a translate-then-score control for cross-language scale drift. Consider merging sentiment and framing, and dropping ideology on non-political topics, if the data says the axes are redundant.
- Commit the labels (`eval/labels/*.csv`) and one script that regenerates every number in the README.

### 3. Make it reproducible and demo-able offline (~1–2 days, high value)
- Add a `pyproject.toml` with pinned versions (or a lockfile via `uv`/`pip-tools`), and pin the LaBSE revision and spaCy model version. Add `matplotlib`, and remove import-time downloads and model loads (load lazily).
- Add a single CLI: `python -m wikilens run --title "Climate change" --pair en-de --backend {watsonx,ollama,replay} --offline`. `--offline` uses the cached articles, and `replay` serves saved scores.
- Cache section-name translations to disk, or replace Google Translate with a deterministic option, and measure whether translation actually helps alignment F1 (from #2).
- Optionally add a Dockerfile, and record the backend, model, prompt hash and git SHA in each run's `meta.json`.

### 4. Rewrite the README around results and evidence (~0.5–1 day after #1–#3, high value)
- Open with a GIF or screenshot, a one-paragraph pitch, and **a results table** (alignment F1 per language pair, omission MAE, score reliability), followed by 2–3 concrete qualitative findings.
- Add sections for: "Run it in 2 minutes without credentials" (replay mode), data and output formats, a precise evaluation methodology, limitations and ethics, **team and your contribution**, and a license.
- Translate or delete `README.ipynb`, fix the clone URL, and update the project tree.

### 5. Consolidate code and add unit tests + CI (~1.5–2 days, medium–high value)
- Merge the two aligners into one `align(query_df, target_df, thresholds)` with a single, clearly named output schema. Move thresholds and constants into one config module, and import that config into the eval script instead of copying it.
- Delete dead code: the empty `routes.py`, the broken `run_chunk_threshold_experiment.py`, the `run_*` scripts with stale language lists, the unused sync `process_file`, and the no-op `__main__` in the scorer.
- Move blocking work in `/analyze` to a background task with real progress polling, so the UI no longer fakes progress with timers.
- Add `pytest` unit tests that need no network or LLM:
  - chunker order and section boundaries
  - lead splitting
  - aligner on tiny synthetic embeddings
  - `parse_response` on malformed or truncated JSON
  - a replay-mode API test
- Add `ruff` and a GitHub Actions workflow. Replace `test_pipeline.py` with an opt-in integration test.

**Suggested order:** #1 → #3 → #2 → #4, with #5 done alongside #1/#3 wherever you're already touching the same code. If time is short, #1 plus a reduced #2 (EN-DE alignment labels only) plus #4 gives most of the value.
