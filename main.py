from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl
from pathlib import Path
import json
import shutil
import pandas as pd
import uvicorn
from datetime import datetime

from app.services.fetcher import fetch_multilang_from_url, extract_title_from_url
from app.services.chunker import paragraph_chunk
from app.services.embedder import embed_language, normalise_section
from app.services.aligner_en_to_source import match_en_to_lang
from app.services.aligner_source_to_en import align_language
from app.services.scorer import score

app = FastAPI()

# Serve static files and outputs
app.mount("/static", StaticFiles(directory="static"), name="static")
# More-specific alignment mounts must come BEFORE the generic /outputs mount
app.mount("/outputs/alignments_en_to_source", StaticFiles(directory="outputs/alignments_en_to_source"), name="align_en")
app.mount("/outputs/alignments_source_to_en",  StaticFiles(directory="outputs/alignments_source_to_en"),  name="align_src")
app.mount("/outputs", StaticFiles(directory="outputs/scores"), name="outputs")


class AnalysisRequest(BaseModel):
    url: HttpUrl
    language_pair: str


@app.post("/analyze")
async def analyze_endpoint(request: AnalysisRequest):
    return await analyze(request)


async def analyze(request: AnalysisRequest):
    # Parse and validate the language pair before doing any work. The aligners
    # and the scorer need English on one side.
    parts = request.language_pair.split("-")
    if len(parts) != 2 or "en" not in parts or parts[0] == parts[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language pair '{request.language_pair}': expected 'en-xx' or 'xx-en'",
        )
    lang1, lang2 = parts

    try:

        # Extract title from URL for logging
        title = extract_title_from_url(str(request.url))
        print(f"Analyzing article: {title} for languages: {lang1}, {lang2}")

        # Fetch articles in both languages
        try:
            articles = fetch_multilang_from_url(
                str(request.url), [lang1, lang2])
            print(f"Fetched articles keys: {list(articles.keys())}")
        except Exception as e:
            print(f"Error fetching articles: {e}")
            raise

        missing = [l for l in (lang1, lang2) if l not in articles]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"No article found for language(s): {', '.join(missing)}",
            )

        # Process articles through your analysis pipeline
        outputs_dir = Path("outputs/scores")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        # Process articles through your analysis pipeline
        chunked_path = Path("outputs/chunked")
        chunked_path.mkdir(parents=True, exist_ok=True)

        # Create embeddings directory
        embeddings_dir = Path("outputs/embeddings")
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Chunk — paragraph_chunk expects list[dict], articles[lang] is a DataFrame
        chunked_articles = {}
        for lang, df in articles.items():
            sentences  = df.to_dict("records")
            chunks     = paragraph_chunk(sentences)
            chunked_df = pd.DataFrame(chunks)
            chunked_articles[lang] = chunked_df
            print(f"Chunked {lang}: {len(chunked_df)} chunks")

            # Save JSON so embedder can read it from disk
            chunked_filepath = chunked_path / f"chunked_{lang}.json"
            with open(chunked_filepath, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"[✓] {lang}: chunked → {chunked_filepath}")

        # Step 2: Embed — embed_language reads chunked_{lang}.json from disk
        for lang in chunked_articles:
            print(f"Embedding {lang}...")
            embed_language(lang)
            print(f"[✓] {lang}: embedded")

        # Step 3: Perform alignments using existing aligner functions
        alignments = {}
        # Use your existing aligner functions directly
        if lang1 == "en":
            # English to other language
            match_en_to_lang(lang2)
            alignment_file = (
                Path("outputs/alignments_en_to_source") /
                f"en_to_{lang2}.json"
            )
            if alignment_file.exists():
                with open(alignment_file, "r", encoding="utf-8") as f:
                    alignments[f"{lang1}_to_{lang2}"] = json.load(f)

            # Other language to English
            align_language(lang2)
            alignment_file = (
                Path("outputs/alignments_source_to_en") /
                f"{lang2}_to_en.json"
            )
            if alignment_file.exists():
                with open(alignment_file, "r", encoding="utf-8") as f:
                    alignments[f"{lang2}_to_{lang1}"] = json.load(f)

        elif lang2 == "en":
            # Other language to English
            align_language(lang1)
            alignment_file = (
                Path("outputs/alignments_source_to_en") /
                f"{lang1}_to_en.json"
            )
            if alignment_file.exists():
                with open(alignment_file, "r", encoding="utf-8") as f:
                    alignments[f"{lang1}_to_{lang2}"] = json.load(f)

            # English to other language
            match_en_to_lang(lang1)
            alignment_file = (
                Path("outputs/alignments_en_to_source") /
                f"en_to_{lang1}.json"
            )
            if alignment_file.exists():
                with open(alignment_file, "r", encoding="utf-8") as f:
                    alignments[f"{lang2}_to_{lang1}"] = json.load(f)
        else:
            # Neither language is English - you may need a different aligner or
            # run both languages against English as an intermediate step
            print(
                f"Warning: No direct aligner for {lang1} to {lang2}. Both aligners expect English as one language."
            )

            print(
                f"Alignments completed. Found {len(alignments)} alignment results.")

        # Step 4: Create scoring output using alignment results
        print("Starting bias analysis scoring...")

        # Determine the non-English language for scoring
        non_en_lang = lang1 if lang2 == "en" else lang2 if lang1 == "en" else None

        if non_en_lang:
            # Use the scorer with the correct language pair format
            language_pair = f"en-{non_en_lang}"
            print(f"Running scorer for language pair: {language_pair}")

            try:
                # Clear stale score files so scorer doesn't append to old results
                for stale in [
                    outputs_dir / f"base_en_omit_{non_en_lang}_scores.json",
                    outputs_dir / f"base_{non_en_lang}_omit_en_scores.json",
                ]:
                    if stale.exists():
                        stale.unlink()

                await score(language_pair)
                print("Scoring completed successfully")

                # Read the generated score files
                score_file1 = (
                    Path("outputs/scores") /
                    f"base_en_omit_{non_en_lang}_scores.json"
                )
                score_file2 = (
                    Path("outputs/scores") /
                    f"base_{non_en_lang}_omit_en_scores.json"
                )

                scores1 = []
                scores2 = []

                # Build chunk_id → section lookup from chunked files.
                # Non-English section names are translated to English so both
                # languages share the same section labels in the frontend chart.
                section_lookup = {}
                for lang in ["en", non_en_lang]:
                    cp = Path(f"outputs/chunked/chunked_{lang}.json")
                    if cp.exists():
                        # Build per-language translation cache (unique sections only)
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

                if score_file1.exists():
                    with open(score_file1, "r", encoding="utf-8") as f:
                        scores1 = enrich_sections(json.load(f))
                    print(f"Loaded {len(scores1)} English-base scores")

                if score_file2.exists():
                    with open(score_file2, "r", encoding="utf-8") as f:
                        scores2 = enrich_sections(json.load(f))
                    print(f"Loaded {len(scores2)} {non_en_lang}-base scores")

                # Write enriched (section-tagged) scores back before copying to history
                if scores1 and score_file1.exists():
                    with open(score_file1, "w", encoding="utf-8") as f:
                        json.dump(scores1, f, ensure_ascii=False)
                if scores2 and score_file2.exists():
                    with open(score_file2, "w", encoding="utf-8") as f:
                        json.dump(scores2, f, ensure_ascii=False)

                # Copy score files with timestamp so history entries are permanent
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                hist_file1 = outputs_dir / f"base_en_omit_{non_en_lang}_{ts}.json"
                hist_file2 = outputs_dir / f"base_{non_en_lang}_omit_en_{ts}.json"
                if score_file1.exists():
                    shutil.copy(score_file1, hist_file1)
                if score_file2.exists():
                    shutil.copy(score_file2, hist_file2)

                # History entry (lightweight metadata)
                entry = {
                    "success": True,
                    "title": title,
                    "languages_found": list(articles.keys()),
                    "timestamp": ts,
                    "file1_url": f"/outputs/{hist_file1.name}" if hist_file1.exists() else None,
                    "file2_url": f"/outputs/{hist_file2.name}" if hist_file2.exists() else None,
                }

                # Append to history.json
                history_path = Path("outputs/history.json")
                history = []
                if history_path.exists():
                    try:
                        with open(history_path, encoding="utf-8") as f:
                            history = json.load(f)
                    except Exception:
                        history = []
                history.insert(0, entry)          # newest first
                history = history[:20]            # keep at most 20
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)

                # Also update meta.json for /last-results compatibility
                with open(Path("outputs/meta.json"), "w", encoding="utf-8") as f:
                    json.dump(entry, f, ensure_ascii=False)

                # Build response
                result = {
                    **entry,
                    "summary": {
                        lang: {
                            "total_chunks": len(chunked_articles[lang]),
                            "sections": list(
                                chunked_articles[lang]["section"].unique()
                            ),
                        }
                        for lang in chunked_articles.keys()
                    },
                    "scoring_summary": {
                        "en_base_scores": len(scores1),
                        f"{non_en_lang}_base_scores": len(scores2),
                        "languages_analyzed": [lang1, lang2],
                    },
                }
                return result

            except Exception as e:
                # No placeholder scores: a failed scoring run is reported as
                # an error and is not added to the history.
                print(f"Scoring failed: {str(e)}")
                raise HTTPException(
                    status_code=502, detail=f"Scoring failed: {str(e)}"
                )

        else:
            # Unreachable after the validation above; kept as a guard.
            raise HTTPException(
                status_code=400,
                detail=f"Scorer requires English as one language ({lang1}, {lang2})",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/history")
async def get_history():
    """Return all past analysis runs (newest first, max 20)."""
    history_path = Path("outputs/history.json")
    if not history_path.exists():
        return []
    with open(history_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/last-results")
async def last_results():
    """Return metadata for the most recently completed analysis."""
    meta_path = Path("outputs/meta.json")
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="No previous results found. Run an analysis first.")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
