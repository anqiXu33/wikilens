"""The end-to-end analysis pipeline: fetch → chunk → embed → align → score.

Used by both the web API (main.py, POST /analyze) and the command line
(python -m wikilens run). Every analysis writes into its own directory,
outputs/runs/<run_id>/, so runs never read or overwrite each other's files.
"""
import json
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.services.fetcher import fetch_multilang_from_url, extract_title_from_url
from app.services.chunker import paragraph_chunk
from app.services.embedder import embed_language, normalise_section, LABSE_MODEL, LABSE_REVISION
from app.services.aligner_en_to_source import match_en_to_lang
from app.services.aligner_source_to_en import align_language

RUNS_DIR = Path("outputs/runs")
HISTORY_PATH = Path("outputs/history.json")
LAST_META_PATH = Path("outputs/meta.json")


class PipelineError(Exception):
    """An analysis failure, with the HTTP status the API should report."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def parse_language_pair(language_pair: str) -> tuple[str, str]:
    # The aligners and the scorer need English on one side.
    parts = language_pair.split("-")
    if len(parts) != 2 or "en" not in parts or parts[0] == parts[1]:
        raise PipelineError(
            400, f"Invalid language pair '{language_pair}': expected 'en-xx' or 'xx-en'")
    return parts[0], parts[1]


def new_run_dir(title: str, lang1: str, lang2: str) -> tuple[str, str, Path]:
    """Create outputs/runs/<timestamp>_<title>_<pair>_<suffix>/ and return
    (run_id, timestamp, path). Fails if the directory already exists."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^\w().-]+", "_", title).strip("_") or "article"
    run_id = f"{ts}_{safe_title}_{lang1}-{lang2}_{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, ts, run_dir


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def _scoring_info() -> dict:
    from app.services import scorer
    model = scorer.WATSONX_MODEL_ID if scorer.SCORING_BACKEND == "watsonx" else scorer.OLLAMA_MODEL
    return {"backend": scorer.SCORING_BACKEND, "model": model}


def _match_type_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.get("match_type", "?")] = counts.get(r.get("match_type", "?"), 0) + 1
    return dict(sorted(counts.items()))


async def run_analysis(url: str, language_pair: str, do_score: bool = True) -> dict:
    """Run the pipeline for one article and language pair.

    do_score=False stops after alignment: no LLM is called, so no backend or
    API key is needed. Such runs are not added to the history (the dashboard
    needs score files) and their meta.json says "scored": false.
    """
    lang1, lang2 = parse_language_pair(language_pair)
    non_en_lang = lang2 if lang1 == "en" else lang1

    title = extract_title_from_url(url)
    print(f"Analyzing article: {title} for languages: {lang1}, {lang2}")

    run_id, ts, run_dir = new_run_dir(title, lang1, lang2)
    print(f"Run directory: {run_dir}")

    articles = fetch_multilang_from_url(url, [lang1, lang2], output_dir=run_dir / "fetched")
    print(f"Fetched articles keys: {list(articles.keys())}")
    missing = [l for l in (lang1, lang2) if l not in articles]
    if missing:
        raise PipelineError(404, f"No article found for language(s): {', '.join(missing)}")

    # All intermediate files for this run live under run_dir
    outputs_dir = run_dir / "scores"
    chunked_path = run_dir / "chunked"
    embeddings_dir = run_dir / "embeddings"
    align_en_dir = run_dir / "alignments_en_to_source"
    align_src_dir = run_dir / "alignments_source_to_en"
    for d in (outputs_dir, chunked_path, embeddings_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Chunk — paragraph_chunk expects list[dict], articles[lang] is a DataFrame
    chunked_articles = {}
    for lang, df in articles.items():
        chunks = paragraph_chunk(df.to_dict("records"))
        chunked_articles[lang] = pd.DataFrame(chunks)
        print(f"Chunked {lang}: {len(chunks)} chunks")
        # Save JSON so embedder can read it from disk
        chunked_filepath = chunked_path / f"chunked_{lang}.json"
        with open(chunked_filepath, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"[✓] {lang}: chunked → {chunked_filepath}")

    # Step 2: Embed — embed_language reads chunked_{lang}.json from disk
    for lang in chunked_articles:
        print(f"Embedding {lang}...")
        embed_language(lang, input_dir=chunked_path, output_dir=embeddings_dir)
        print(f"[✓] {lang}: embedded")

    # Step 3: Align in both directions
    match_en_to_lang(non_en_lang, embed_dir=embeddings_dir, output_dir=align_en_dir)
    align_language(non_en_lang, embed_dir=embeddings_dir, output_dir=align_src_dir)
    align_en_file = align_en_dir / f"en_to_{non_en_lang}.json"
    align_src_file = align_src_dir / f"{non_en_lang}_to_en.json"
    for af in (align_en_file, align_src_file):
        if not af.exists():
            raise PipelineError(500, f"alignment produced no output file {af}")
    alignment_summary = {
        f"en_to_{non_en_lang}": _match_type_counts(json.loads(align_en_file.read_text(encoding="utf-8"))),
        f"{non_en_lang}_to_en": _match_type_counts(json.loads(align_src_file.read_text(encoding="utf-8"))),
    }
    print(f"Alignments completed: {alignment_summary}")

    url_base = f"/runs/{run_id}"
    entry = {
        "success": True,
        "run_id": run_id,
        "title": title,
        "languages_found": list(articles.keys()),
        "timestamp": ts,
        "scored": do_score,
        "file1_url": f"{url_base}/scores/base_en_omit_{non_en_lang}_scores.json" if do_score else None,
        "file2_url": f"{url_base}/scores/base_{non_en_lang}_omit_en_scores.json" if do_score else None,
        "align_en_url": f"{url_base}/alignments_en_to_source/{align_en_file.name}",
        "align_src_url": f"{url_base}/alignments_source_to_en/{align_src_file.name}",
        "provenance": {
            "git_sha": _git_sha(),
            "labse": f"{LABSE_MODEL}@{LABSE_REVISION}",
            "scoring": _scoring_info() if do_score else None,
        },
    }
    summary = {
        lang: {
            "total_chunks": len(chunked_articles[lang]),
            "sections": list(chunked_articles[lang]["section"].unique()),
        }
        for lang in chunked_articles
    }

    if not do_score:
        with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        return {**entry, "summary": summary, "alignment_summary": alignment_summary}

    # Step 4: Score. Imported here so that the backend choice (SCORING_BACKEND,
    # read by the scorer at import) can be set by the caller first.
    from app.services.scorer import score
    language_pair = f"en-{non_en_lang}"
    print(f"Running scorer for language pair: {language_pair}")
    try:
        # run_dir is fresh, so there are no stale score files to clear
        await score(language_pair, output_dir=run_dir)
        print("Scoring completed successfully")

        score_file1 = outputs_dir / f"base_en_omit_{non_en_lang}_scores.json"
        score_file2 = outputs_dir / f"base_{non_en_lang}_omit_en_scores.json"
        for sf in (score_file1, score_file2):
            if not sf.exists():
                raise RuntimeError(f"scorer produced no output file {sf}")

        # Build chunk_id → section lookup from chunked files.
        # Non-English section names are translated to English so both
        # languages share the same section labels in the frontend chart.
        section_lookup = {}
        for lang in ["en", non_en_lang]:
            cp = chunked_path / f"chunked_{lang}.json"
            if cp.exists():
                raw_chunks = json.load(open(cp, encoding="utf-8"))
                sec_cache: dict[str, str] = {}
                for ch in raw_chunks:
                    raw_sec = ch.get("section", "(Lead)")
                    if raw_sec not in sec_cache:
                        sec_cache[raw_sec] = normalise_section(raw_sec, lang)
                    section_lookup[ch["chunk_id"]] = sec_cache[raw_sec]

        def enrich_sections(scores):
            for s in scores:
                # Always overwrite section with the translated (English) name
                # from the lookup — the scorer writes the original-language
                # section name, which we need to normalise for the chart.
                cid = s.get("chunk_id", "")
                if cid in section_lookup:
                    s["section"] = section_lookup[cid]
                elif "section" not in s:
                    s["section"] = "(Lead)"
            return scores

        with open(score_file1, "r", encoding="utf-8") as f:
            scores1 = enrich_sections(json.load(f))
        print(f"Loaded {len(scores1)} English-base scores")
        with open(score_file2, "r", encoding="utf-8") as f:
            scores2 = enrich_sections(json.load(f))
        print(f"Loaded {len(scores2)} {non_en_lang}-base scores")

        # Write enriched (section-tagged) scores back
        with open(score_file1, "w", encoding="utf-8") as f:
            json.dump(scores1, f, ensure_ascii=False)
        with open(score_file2, "w", encoding="utf-8") as f:
            json.dump(scores2, f, ensure_ascii=False)
    except Exception as e:
        # No placeholder scores: a failed scoring run is reported as
        # an error and is not added to the history.
        print(f"Scoring failed: {str(e)}")
        raise PipelineError(502, f"Scoring failed: {str(e)}") from e

    # History entry: every URL points into this run's own directory,
    # including the alignment files the dashboard reads
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    history = []
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.insert(0, entry)          # newest first
    history = history[:20]            # keep at most 20
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # Also update meta.json for /last-results compatibility
    with open(LAST_META_PATH, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)

    return {
        **entry,
        "summary": summary,
        "alignment_summary": alignment_summary,
        "scoring_summary": {
            "en_base_scores": len(scores1),
            f"{non_en_lang}_base_scores": len(scores2),
            "languages_analyzed": [lang1, lang2],
        },
    }
