# WikiLens — Cross-Lingual Bias Analyser for Wikipedia

WikiLens detects editorial bias between Wikipedia language editions by comparing how the same topic is covered across languages. It identifies ideological framing, sentiment shifts, and omitted information at the paragraph level, then visualises the results in a Wikipedia-style interface.

Built for the **IBM watsonx GenAI Challenge 2025** (University of Zurich) as a research tool for Wikipedia editors, it currently supports **English ↔ German, Chinese, Italian, and French**.

LLM scoring runs on either of two interchangeable backends:

- **IBM watsonx.ai** — `meta-llama/llama-3-3-70b-instruct` with HAP/PII masking and Granite Guardian guardrails; this is the configuration used during the challenge
- **Ollama (local)** — any local model (default `llama3.2`), kept as a credential-free fallback so the pipeline stays fully reproducible after challenge access expired

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
     │              (Llama 3.3 70B) or local Ollama, selected via .env
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

- Python 3.10+
- One scoring backend:
  - an IBM Cloud account with a watsonx.ai project (**watsonx** backend), or
  - [Ollama](https://ollama.com) installed and running (**ollama** backend, default)

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/wikilens.git
cd wikilens
pip install -r requirements.txt
```

You will also need spaCy's German model for sentence tokenisation:

```bash
python -m spacy download de_core_news_sm
```

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

### 3. Start the server

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in your browser.

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
│   └── services/
│       ├── fetcher.py              # Wikipedia API + sentence splitting
│       ├── chunker.py              # Paragraph-based chunking
│       ├── embedder.py             # LaBSE embedding
│       ├── aligner_en_to_source.py # EN → other language alignment
│       ├── aligner_source_to_en.py # Other language → EN alignment
│       └── scorer.py               # LLM scoring (watsonx.ai / Ollama)
├── static/
│   └── index.html                  # Frontend UI
├── outputs/                        # Generated at runtime
│   ├── chunked/
│   ├── embeddings/
│   ├── alignments_en_to_source/
│   ├── alignments_source_to_en/
│   └── scores/
├── main.py                         # FastAPI server
├── eval_alignment_threshold.py     # Threshold calibration script
├── .env                            # Local config (not committed)
└── requirements.txt
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

During the IBM watsonx GenAI Challenge 2025, scoring ran on **watsonx.ai** with `meta-llama/llama-3-3-70b-instruct` (greedy decoding, JSON-stop sequences), with watsonx's HAP/PII masking and **Granite Guardian** guardrails enabled on inputs and outputs. After challenge access expired, an **Ollama** backend was added behind the same interface so the pipeline remains reproducible without IBM credentials. Both backends share identical prompts and response parsing; switch between them with a single `.env` variable (`SCORING_BACKEND`).

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
