"""
End-to-end pipeline smoke test.
Uses Gala (apple) EN-DE — the smallest cached article (~5-9 chunks each).
Runs: fetch → chunk → embed → align → score
"""

import sys, json, asyncio
from pathlib import Path

sys.path.insert(0, ".")
ROOT = Path(__file__).parent

# ── Step 0: paths ─────────────────────────────────────────────────────────────
CACHE_DIR    = ROOT / "app" / "cached_articles"
CHUNKED_DIR  = ROOT / "outputs" / "chunked"
EMBED_DIR    = ROOT / "outputs" / "embeddings"
SCORES_DIR   = ROOT / "outputs" / "scores"

for d in [CHUNKED_DIR, EMBED_DIR, SCORES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LANG       = "de"           # non-English language to test
EN_CACHE   = CACHE_DIR / "en_Gala_(apple).json"
LANG_CACHE = CACHE_DIR / f"{LANG}_Gala_(Apfel).json"


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


# ── Step 1: chunk ─────────────────────────────────────────────────────────────
section("STEP 1 — Paragraph Chunking")

from app.services.chunker import paragraph_chunk

with open(EN_CACHE,   encoding="utf-8") as f:
    en_sentences = json.load(f)
with open(LANG_CACHE, encoding="utf-8") as f:
    lang_sentences = json.load(f)

en_chunks   = paragraph_chunk(en_sentences)
lang_chunks = paragraph_chunk(lang_sentences)

print(f"  EN  chunks : {len(en_chunks)}")
print(f"  {LANG.upper()}  chunks : {len(lang_chunks)}")

# save to outputs/chunked/
en_chunked_path   = CHUNKED_DIR / "chunked_en.json"
lang_chunked_path = CHUNKED_DIR / f"chunked_{LANG}.json"

with open(en_chunked_path,   "w", encoding="utf-8") as f:
    json.dump(en_chunks,   f, ensure_ascii=False, indent=2)
with open(lang_chunked_path, "w", encoding="utf-8") as f:
    json.dump(lang_chunks, f, ensure_ascii=False, indent=2)

print(f"  ✓ Saved chunked files")

# preview first EN chunk
print(f"\n  First EN chunk preview:")
print(f"    section : {en_chunks[0]['section']}")
print(f"    text    : {en_chunks[0]['chunk_text'][:120]}...")


# ── Step 2: embed ─────────────────────────────────────────────────────────────
section("STEP 2 — LaBSE Embedding")

from app.services.embedder import embed_language

embed_language("en")
embed_language(LANG)

en_parquet   = EMBED_DIR / "embedded_en.parquet"
lang_parquet = EMBED_DIR / f"embedded_{LANG}.parquet"
print(f"  ✓ EN   parquet: {en_parquet.stat().st_size // 1024} KB")
print(f"  ✓ {LANG.upper()}   parquet: {lang_parquet.stat().st_size // 1024} KB")


# ── Step 3: align ─────────────────────────────────────────────────────────────
section("STEP 3 — Alignment (bidirectional)")

from app.services.aligner_en_to_source import match_en_to_lang
from app.services.aligner_source_to_en import align_language

match_en_to_lang(LANG)   # EN → DE
align_language(LANG)     # DE → EN

en_to_lang = ROOT / "outputs" / "alignments_en_to_source" / f"en_to_{LANG}.json"
lang_to_en = ROOT / "outputs" / "alignments_source_to_en" / f"{LANG}_to_en.json"

with open(en_to_lang, encoding="utf-8") as f:
    fwd = json.load(f)
with open(lang_to_en, encoding="utf-8") as f:
    bwd = json.load(f)

matched_fwd = [r for r in fwd if r.get("match_type") != "1:0"]
matched_bwd = [r for r in bwd if r.get("match_type") != "1:0"]

print(f"\n  EN → {LANG.upper()}:  {len(fwd)} total  |  {len(matched_fwd)} matched  |  "
      f"{len(fwd)-len(matched_fwd)} unmatched (1:0)")
print(f"  {LANG.upper()} → EN:  {len(bwd)} total  |  {len(matched_bwd)} matched  |  "
      f"{len(bwd)-len(matched_bwd)} unmatched (1:0)")

# preview first match
if matched_fwd:
    m = matched_fwd[0]
    sim = m.get("chunk_sim", "N/A")
    sim_str = f"{sim:.3f}" if isinstance(sim, float) else str(sim)
    print(f"\n  First EN→{LANG.upper()} match:")
    print(f"    chunk_sim : {sim_str}")
    print(f"    EN text   : {m.get('src_chunk_text','')[:80]}...")
    print(f"    {LANG.upper()} text   : {m.get('tgt_chunk_text','')[:80]}...")


# ── Step 4: score ─────────────────────────────────────────────────────────────
from app.services.scorer import score, SCORING_BACKEND

section(f"STEP 4 — LLM Scoring (backend: {SCORING_BACKEND})")

asyncio.run(score(f"en-{LANG}"))

score_file1 = SCORES_DIR / f"base_en_omit_{LANG}_scores.json"
score_file2 = SCORES_DIR / f"base_{LANG}_omit_en_scores.json"

with open(score_file1, encoding="utf-8") as f:
    scores_en = json.load(f)
with open(score_file2, encoding="utf-8") as f:
    scores_lang = json.load(f)

print(f"\n  ✓ base_en_omit_{LANG}: {len(scores_en)} chunks scored")
print(f"  ✓ base_{LANG}_omit_en: {len(scores_lang)} chunks scored")


# ── Summary ───────────────────────────────────────────────────────────────────
section("RESULTS SUMMARY")

import statistics

def print_score_summary(label, records):
    ideo = [r["ideology_score"]  for r in records if r.get("ideology_score")  is not None]
    sent = [r["sentiment_score"] for r in records if r.get("sentiment_score") is not None]
    fram = [r["framing_score"]   for r in records if r.get("framing_score")   is not None]
    omit = [r["omission_score"]  for r in records if r.get("omission_score")  is not None]

    def fmt(lst):
        return f"mean={statistics.mean(lst):+.3f}" if lst else "N/A"

    print(f"\n  [{label}]")
    print(f"    ideology_score  : {fmt(ideo)}")
    print(f"    sentiment_score : {fmt(sent)}")
    print(f"    framing_score   : {fmt(fram)}")
    print(f"    omission_score  : {fmt(omit)}")

print_score_summary(f"EN base, {LANG.upper()} omission", scores_en)
print_score_summary(f"{LANG.upper()} base, EN omission", scores_lang)

print(f"\n  Score files:")
print(f"    {score_file1}")
print(f"    {score_file2}")
print("\n  ✅ Pipeline end-to-end test PASSED\n")
