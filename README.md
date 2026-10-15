# WikiLens — Cross-Lingual Bias Analyser for Wikipedia

WikiLens detects editorial bias between Wikipedia language editions by comparing how the same topic is covered across languages. It identifies ideological framing, sentiment shifts, and omitted information at the paragraph level, then visualises the results in a Wikipedia-style interface.

Built for the **IBM watsonx GenAI Challenge 2025** (University of Zurich) as a research tool for Wikipedia editors, it currently supports **English ↔ German, Chinese, Italian, and French**.

LLM scoring runs on any of three interchangeable backends:

- **IBM watsonx.ai** — `meta-llama/llama-3-3-70b-instruct` with HAP/PII masking and Granite Guardian guardrails; this is the configuration used during the challenge
- **Ollama (local)** — any local model (default `llama3.2`), kept as a credential-free fallback so the pipeline stays fully reproducible after challenge access expired
- **OpenAI-compatible API** — OpenRouter, Groq, Together, OpenAI or any other `/chat/completions` endpoint; the way to score with the challenge's Llama 3.3 70B again without IBM access

---

## How it works

```
Wikipedia URL
     │
     ▼
  Fetcher          Retrieves articles in both languages via Wikipedia API
     │              Splits text into sentences, preserving section structure
     ▼
  Chunker          Groups sentences into paragraph-level chunks
     │              Single-sentence paragraphs merged into adjacent chunks
     ▼
  Embedder         Encodes each chunk using LaBSE (Language-agnostic BERT)
     │              Produces language-independent semantic vectors
     ▼
  Aligner          Matches chunks across languages via cosine similarity
     │              Uses per-language calibrated thresholds (EN-DE: 0.45,
     │              EN-ZH: 0.45, EN-IT: 0.35, EN-FR: 0.40)
     ▼
  Scorer           Sends aligned chunk pairs to an LLM — IBM watsonx.ai
     │              (Llama 3.3 70B), local Ollama or an OpenAI-compatible
     │              API (OpenRouter, Groq, ...), selected via .env
     │              Scores three bias dimensions + omission rate
     ▼
  Frontend         Wikipedia-style UI with colour-coded bias heatmap
```

### Bias dimensions

| Score | Range | Meaning |
|-------|-------|---------|
| **Ideology** | −1.0 → +1.0 | Left-leaning ↔ Right-leaning |
| **Sentiment** | −1.0 → +1.0 | Negative ↔ Positive framing |
| **Framing** | −1.0 → +1.0 | Negative ↔ Positive presentation |
| **Omission** | 0.0 → 1.0 | 0 = nothing omitted, 1 = all omitted |

---

## Setup

### Prerequisites

- Python 3.12 (the pinned dependencies are tested on 3.12; see `.python-version`)
- For new analyses, one scoring backend (not needed for offline replay or `--no-score`):
  - an IBM Cloud account with a watsonx.ai project (**watsonx** backend), or
  - [Ollama](https://ollama.com) installed and running (**ollama** backend, default), or
  - an API key for an OpenAI-compatible provider such as OpenRouter or Groq (**openai** backend)

### 0. Try it in one minute, without installing anything

Offline mode replays the scores saved in `outputs/runs/`. It uses only the
Python standard library, needs no network, no LLM and no API key:

```bash
git clone https://github.com/your-username/wikilens.git
cd wikilens
python -m wikilens runs                                            # list saved runs
python -m wikilens run --title "Gala (apple)" --pair en-de --offline
```

### 1. Install dependencies

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a lockfile: every package is pinned to an exact version.
It is generated from the direct dependencies in `requirements.in`:

```bash
uv pip compile --universal --python-version 3.12 requirements.in -o requirements.txt
```

Two version constraints are deliberate: spaCy 3.7 (needed for the German model
below) is built against NumPy 1.x, so `numpy<2`, and pyarrow 26 requires NumPy 2,
so `pyarrow<26`.

On Linux, PyPI's `torch` wheel pulls in the CUDA libraries (several GB). For a
CPU-only machine, install the CPU build of the same version first:
`pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`.

Models and data are pinned too and downloaded on first use (not at import):

| What | Pinned to | Where it comes from |
|---|---|---|
| LaBSE | `sentence-transformers/LaBSE` @ `836121a` | Hugging Face (`app/services/embedder.py`) |
| spaCy German model | `spacy/de_core_news_sm` 3.7.0 @ `3e48602` | Hugging Face (`app/services/fetcher.py`) |
| NLTK `punkt_tab` | latest | NLTK data server; or `python -m nltk.downloader punkt_tab` |

### 2. Configure the scoring backend

Copy `.env.example` to `.env` and pick a backend.

**Option A — IBM watsonx.ai** (used for the challenge; best scoring quality):

```bash
# .env
SCORING_BACKEND=watsonx
IBM_API_KEY=<your IBM Cloud API key>
IBM_PROJECT_ID=<your watsonx.ai project id>
# optional overrides:
# IBM_WATSONX_URL=https://eu-de.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29
# WATSONX_MODEL_ID=meta-llama/llama-3-3-70b-instruct
```

**Option B — local Ollama** (no credentials, runs offline):

```bash
# .env
SCORING_BACKEND=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_URL=http://localhost:11434
```

Then pull the model and start Ollama:

```bash
ollama pull llama3.2
ollama serve
```

Any Ollama-compatible model works. Larger models (e.g. `llama3.1:8b`, `mistral`) give better scoring quality at the cost of speed.

**Option C — OpenAI-compatible API** (OpenRouter, Groq, Together, OpenAI, ...):

Three variables select the provider and model. `OPENAI_BASE_URL` includes the
version prefix (`/v1`); requests go to `$OPENAI_BASE_URL/chat/completions`.
To score with Llama 3.3 70B Instruct, the model used in the challenge:

```bash
# .env — OpenRouter
SCORING_BACKEND=openai
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=<your OpenRouter key>
OPENAI_MODEL=meta-llama/llama-3.3-70b-instruct
```

```bash
# .env — Groq
SCORING_BACKEND=openai
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=<your Groq key>
OPENAI_MODEL=llama-3.3-70b-versatile
OPENAI_MAX_CONCURRENCY=1   # free-tier rate limits are low
```

Together (`https://api.together.xyz/v1`, `meta-llama/Llama-3.3-70B-Instruct-Turbo`)
and OpenAI itself (`https://api.openai.com/v1`, e.g. `gpt-4o-mini`) work the same way.

This backend sends exactly the same request as the Ollama backend (same
prompts, `temperature: 0`) plus an `Authorization: Bearer` header. Up to
`OPENAI_MAX_CONCURRENCY` requests (default 4) run in parallel; HTTP 429 and 5xx
responses are retried with backoff (honouring `Retry-After`). The scores are
not identical to the challenge runs: watsonx used its own text-generation API
with greedy decoding, a `\n}` stop sequence and guardrails, and providers may
serve the model with different quantisation.

### 3. Start the server

```bash
python -m wikilens serve            # or: python -m uvicorn main:app --host 127.0.0.1 --port 8000
python -m wikilens serve --offline  # dashboard over saved runs only; new analyses are disabled
```

Open **http://127.0.0.1:8000** in your browser. `serve --offline` needs the
web dependencies installed but no network, LLM backend or API key.

---

## Usage

1. Paste any English Wikipedia URL (e.g. `https://en.wikipedia.org/wiki/Climate_change`)
2. Select a language pair (e.g. `en-zh`)
3. Click **Analyse** — the pipeline runs automatically
4. Results appear as a section-by-section heatmap with per-chunk scores

**Expected analysis time** (local Ollama with llama3.2 on Apple Silicon):

| Article size | Time |
|---|---|
| Small (~20 chunks) | ~2–3 min |
| Medium (~100 chunks) | ~10–15 min |
| Large (~200 chunks) | ~20–30 min |

The watsonx backend is substantially faster: requests run with up to 8-way concurrency against the hosted 70B model instead of one at a time locally.

### Command line

The CLI runs the same pipeline as the web app (`app/pipeline.py`) and writes
each run to its own `outputs/runs/<run_id>/`. Run it from the repository root.

```bash
# Full analysis: fetch → chunk → embed → align → score. Needs an LLM backend.
python -m wikilens run --title "Gala (apple)" --pair en-de
python -m wikilens run --url https://en.wikipedia.org/wiki/Switzerland --pair en-fr --backend watsonx

# Everything except LLM scoring (real Wikipedia/Wikidata fetch and LaBSE alignment); no API key needed.
python -m wikilens run --title "Gala (apple)" --pair en-de --no-score

# Offline: replay the newest saved scores for this article and pair. No network, LLM or API key.
python -m wikilens run --title "Gala (apple)" --pair en-de --offline

python -m wikilens runs                 # list saved runs (scored / aligned only)
python -m wikilens replay [RUN_ID]      # summarise a saved run (default: newest scored run)
python -m wikilens replay --json        # raw run metadata and scores
```

- `--title` is the **English** Wikipedia title; the other edition's title comes from Wikidata.
- `--pair` needs English on one side (`en-de`, `de-en`, `en-zh`, ...).
- `--backend {ollama,openai,watsonx}` overrides `SCORING_BACKEND` from `.env`.
- Each new run's `meta.json` records the git commit, the LaBSE revision and the scoring backend, model and endpoint (`provenance.scoring`: `backend`, `model`, `base_url`; never the API key), plus how the section names were translated (`provenance.section_names`, see below).
- `--no-score` runs are listed but not added to the dashboard history, since the dashboard needs score files.
- `--no-score` still translates section names as `TRANSLATION_BACKEND` says; with the default `auto` that is the LLM if one is configured. Set `TRANSLATION_BACKEND=google` or `none` to keep LLMs out of a `--no-score` run entirely.

### Alignment evaluation

Human labels for the aligner, and precision / recall / F1 against them. See
[docs/annotation_guidelines.md](docs/annotation_guidelines.md) for what counts as a match.

```bash
# Single-file HTML labelling page: each source chunk with its 3 most similar
# chunks from the other language. --sample N picks N chunks with a fixed --seed (default 0).
python -m wikilens annotate RUN_ID --sample 50
# → open eval/annotation_pages/<run_id>.html, label, "导出 JSON" → save as eval/labels/<run_id>.json

# Grid search over chunk_thr × path_thr (plus routing off) against every file in eval/labels/:
# prints P/R/F1 at the current defaults and at the best setting, and writes
# eval/alignment_<pair>_grid.csv, _summary.json and _f1.png
python -m wikilens evaluate

# Experiment (pipeline unchanged): chunk cosine vs A = LLM rerank of the top 10
# candidates (SCORING_BACKEND) vs B = sentence-level pairing, on the same labels.
# Writes eval/experiments/alignment_<pair>_methods.{md,json}; see docs/alignment_experiments.md
python -m wikilens experiment
```

### Section-name translation

Before embedding, non-English section names are translated to English. This
happens once per run: the same mapping is used for the section embeddings and
for the section labels of the scores, and is saved as `section_names_en.json`
in the run directory.

`TRANSLATION_BACKEND` in `.env` selects the translator:

| Value | Behaviour |
|---|---|
| `auto` (default) | The scoring LLM if `SCORING_BACKEND` is `openai` or `ollama` (and configured); otherwise, or if the LLM request fails, Google Translate |
| `llm` | The scoring LLM only; the run fails at start if `SCORING_BACKEND` is not a configured `openai`/`ollama` backend |
| `google` | Google Translate only (via `deep-translator`) |
| `none` | No translation; section names stay in the original language |

**LLM.** All of an article's uncached section names go to the scoring
backend in a single chat request, using the same request code, model and
retry behaviour as scoring (`temperature: 0`). The model is asked for a JSON
object mapping each original name to its English translation. The reply is
checked: the number of entries must match, every value must be a non-empty
string, and nested names must keep their ` > ` separators. Invalid entries
keep their original name, with a warning. If the request itself fails (backend
unreachable, HTTP error after retries, no JSON in the reply), `auto` falls
back to Google Translate for that article's names; `llm` keeps the original
names.

**Google Translate.** Google's free endpoint allows about 5 requests per
second, so all of a language's uncached names are sent in a single
newline-joined request (falling back to one request per name, at most 2 per
second, if the reply doesn't split back into one line per name). HTTP 429 and
server errors are retried after 2, 4, 8 and 16 s; if Google is still refusing
after that, the remaining names are not requested again in that run. In
practice the free endpoint often refuses even that single request, which is
why `auto` prefers the LLM.

Translations are cached in
`app/cached_translations/section_names.json` (`{lang: {name: english}}`)
whichever translator produced them, so a re-run doesn't request them again and
gets the same section names. Commit the file to make runs reproducible for
others. Failed translations are logged and not cached; the original name is
used for that run. To re-translate names with a different backend, delete
their entries (or the file).

Each run's `meta.json` records how the section names were translated, under
`provenance.section_names`:

```json
"section_names": {
  "translation_backend": "auto",
  "method": "llm",
  "sources": {"llm": 9},
  "llm": {"backend": "openai", "model": "meta-llama/llama-3.3-70b-instruct",
          "base_url": "https://openrouter.ai/api/v1"},
  "untranslated": []
}
```

`sources` counts the names that came from `cache`, `llm` or `google` in this
run; `method` joins those sources (`cache+llm`, ...) or is `none`;
`untranslated` lists names kept in the original language.

---

## Supported languages

| Code | Language | Wikipedia edition |
|------|----------|-------------------|
| `en` | English | Required base language |
| `de` | German | Deutsch |
| `zh` | Chinese | 中文 |
| `it` | Italian | Italiano |
| `fr` | French | Français |

---

## Project structure

```
wikilens/
├── app/
│   ├── pipeline.py                 # End-to-end pipeline (used by the API and the CLI)
│   ├── cached_articles/            # Fetched-article cache
│   ├── cached_translations/        # Section-name translation cache (created on first use)
│   └── services/
│       ├── fetcher.py              # Wikipedia API + sentence splitting
│       ├── chunker.py              # Paragraph-based chunking
│       ├── embedder.py             # LaBSE embedding + section-name translation
│       ├── aligner_en_to_source.py # EN → other language alignment
│       ├── aligner_source_to_en.py # Other language → EN alignment
│       ├── align_thresholds.py     # Per-language alignment thresholds (shared)
│       └── scorer.py               # LLM scoring (watsonx.ai / Ollama / OpenAI-compatible)
├── static/
│   └── index.html                  # Frontend UI
├── outputs/
│   ├── runs/<run_id>/              # One directory per run: fetched/, chunked/, embeddings/,
│   │                               #   alignments_*/, scores/, meta.json
│   └── history.json                # Runs shown in the dashboard
├── wikilens/                       # Command line: python -m wikilens
├── eval/
│   ├── labels/                     # Human alignment labels (annotate → evaluate)
│   ├── alignment_<pair>_*          # evaluate output: grid CSV, summary JSON, F1 plot
│   └── experiments/                # experiment output; llm_rerank_<model>.json caches LLM replies
├── tests/                          # Unit tests, no network (python -m pytest tests)
├── main.py                         # FastAPI server
├── eval_alignment_threshold.py     # Threshold calibration script
├── .env                            # Local config (not committed)
├── requirements.in                 # Direct dependencies
└── requirements.txt                # Pinned lockfile (generated from requirements.in)
```

---

## Technical notes

### Why LaBSE for alignment?

LaBSE (Language-agnostic BERT Sentence Embeddings) produces comparable vectors across languages without requiring translation. A German sentence about the same topic as an English sentence will have a high cosine similarity even though the words are entirely different. This makes it well-suited for cross-lingual paragraph matching.

### Why per-language thresholds?

LaBSE similarity distributions differ significantly across language pairs. Analysis on 5 articles showed:

- **EN-DE** peaks at 0.25–0.40 → initially 0.30, raised to **0.45** after translating section names to English tightened the candidate pool
- **EN-ZH** peaks at 0.65–0.90 → threshold 0.45
- **EN-IT** peaks at 0.40–0.65 → threshold 0.35
- **EN-FR** peaks at 0.50–0.70 → threshold 0.40

A single global threshold would either miss real matches (too high) or create false matches (too low) depending on the language pair.

### Why paragraph-level chunks?

Wikipedia's paragraph breaks are human-authored topic boundaries and are more reliable than fixed-size windows or similarity-based segmentation. Paragraph chunking is also faster, fully deterministic, and produces chunks of consistent semantic scope.

---

### Scoring backends

During the IBM watsonx GenAI Challenge 2025, scoring ran on **watsonx.ai** with `meta-llama/llama-3-3-70b-instruct` (greedy decoding, JSON-stop sequences), with watsonx's HAP/PII masking and **Granite Guardian** guardrails enabled on inputs and outputs. After challenge access expired, an **Ollama** backend was added behind the same interface so the pipeline remains reproducible without IBM credentials, and an **OpenAI-compatible** backend (sharing the Ollama request code) so the same Llama 3.3 70B can be used through providers such as OpenRouter or Groq. All backends share identical prompts and response parsing; switch between them with a single `.env` variable (`SCORING_BACKEND`).

---

## Limitations

- Scoring quality depends on the LLM backend — the local fallback with small models (3B) may produce noisy scores on non-English text; watsonx with Llama 3.3 70B is markedly more reliable
- Alignment thresholds were calibrated on 5 articles per language pair; more data would improve reliability
- The tool assumes one language is always English; direct DE↔ZH comparison is not supported
- Long articles (200+ chunks) require significant local compute time

---

## Acknowledgements

- [IBM watsonx.ai](https://www.ibm.com/watsonx) — hosted LLM inference (Llama 3.3 70B) and Granite Guardian guardrails during the watsonx GenAI Challenge 2025
- [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) — Feng et al., 2022
- [Ollama](https://ollama.com) — local LLM inference
- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page) — article content
- Developed at the University of Zurich for the IBM watsonx GenAI Challenge, spring 2025
