from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl
from pathlib import Path
import json
import os
import uvicorn

from app.pipeline import RUNS_DIR, PipelineError, run_analysis

app = FastAPI()

# Offline (replay) mode: the dashboard serves the saved runs in outputs/ and
# new analyses are refused, so no network, LLM backend or API key is needed.
# Set by `python -m wikilens serve --offline`.
OFFLINE = os.getenv("WIKILENS_OFFLINE", "").strip().lower() in ("1", "true", "yes")

RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Serve static files and per-run outputs
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


class AnalysisRequest(BaseModel):
    url: HttpUrl
    language_pair: str


@app.post("/analyze")
async def analyze_endpoint(request: AnalysisRequest):
    return await analyze(request)


async def analyze(request: AnalysisRequest):
    if OFFLINE:
        raise HTTPException(
            status_code=503,
            detail="Offline (replay) mode: new analyses are disabled. "
                   "Open a saved run from the history instead.",
        )
    try:
        return await run_analysis(str(request.url), request.language_pair)
    except PipelineError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
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
