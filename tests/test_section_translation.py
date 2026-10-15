"""Section-name translation (TRANSLATION_BACKEND) with the LLM and Google
Translate mocked: no network, LLM or API key needed.

    python -m pytest tests/test_section_translation.py
"""
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import embedder, scorer  # noqa: E402

REAL_INFERENCE = scorer.inference

GALA_DE = [
    "Beschreibung > Frucht",
    "Beschreibung > Baum",
    "Geschichte > Ursprung",
    "Anbau",
    "Züchtung",
]
GALA_EN = {
    "Beschreibung > Frucht": "Description > Fruit",
    "Beschreibung > Baum": "Description > Tree",
    "Geschichte > Ursprung": "History > Origin",
    "Anbau": "Cultivation",
    "Züchtung": "Breeding",
}


def chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Fresh translation cache in tmp_path, openai backend configured, and
    every request mocked to fail unless a test replaces it."""
    monkeypatch.setattr(embedder, "TRANSLATION_CACHE_PATH", tmp_path / "section_names.json")
    monkeypatch.setattr(embedder, "_translation_cache", None)
    monkeypatch.setattr(embedder, "RETRY_DELAYS", ())
    monkeypatch.setattr(embedder, "MIN_REQUEST_INTERVAL", 0)
    monkeypatch.delenv("TRANSLATION_BACKEND", raising=False)
    monkeypatch.setattr(scorer, "SCORING_BACKEND", "openai")
    monkeypatch.setattr(scorer, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(scorer, "OPENAI_MODEL", "meta-llama/llama-3.3-70b-instruct")
    monkeypatch.setattr(scorer, "CHAT_MODEL", "meta-llama/llama-3.3-70b-instruct")
    monkeypatch.setattr(scorer, "CHAT_URL", "https://openrouter.ai/api/v1/chat/completions")

    def no_llm(messages, access_token=None):
        raise AssertionError("LLM called unexpectedly")

    def no_google(text, lang):
        raise AssertionError("Google Translate called unexpectedly")

    monkeypatch.setattr(scorer, "inference", no_llm)
    monkeypatch.setattr(embedder, "_google_translate", no_google)


class FakeLLM:
    """Records calls to scorer.inference and replies with fixed content."""

    def __init__(self, content=None, error=None):
        self.content, self.error, self.calls = content, error, []

    def __call__(self, messages, access_token=None):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return chat_response(self.content)


class FakeGoogle:
    def __init__(self, mapping=None, error=None):
        self.mapping, self.error, self.calls = mapping or {}, error, []

    def __call__(self, text, lang):
        self.calls.append(text)
        if self.error:
            raise self.error
        return "\n".join(self.mapping.get(line, line) for line in text.split("\n"))


def cache_on_disk() -> dict:
    path = embedder.TRANSLATION_CACHE_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def too_many_requests():
    from deep_translator.exceptions import TooManyRequests
    return TooManyRequests()


# ── LLM path ──────────────────────────────────────────────────────────────────

def test_auto_uses_llm_in_one_call(monkeypatch):
    llm = FakeLLM(json.dumps(GALA_EN, ensure_ascii=False))
    monkeypatch.setattr(scorer, "inference", llm)
    info = {}
    result = embedder.translate_section_names(["(Lead)"] + GALA_DE + ["Anbau"], "de", info=info)

    assert len(llm.calls) == 1
    system, user = llm.calls[0]
    assert "German" in system["content"] and "JSON" in system["content"]
    assert json.loads(user["content"]) == GALA_DE          # deduplicated, no (Lead)
    assert result == {"(Lead)": "(Lead)", **GALA_EN}
    assert cache_on_disk() == {"de": GALA_EN}
    assert info["translation_backend"] == "auto"
    assert info["method"] == "llm"
    assert info["sources"] == {"llm": 5}
    assert info["llm"] == {"backend": "openai", "model": "meta-llama/llama-3.3-70b-instruct",
                           "base_url": "https://openrouter.ai/api/v1"}
    assert info["untranslated"] == []


def test_llm_reply_in_code_fence(monkeypatch):
    reply = "Here you go:\n```json\n" + json.dumps(GALA_EN, ensure_ascii=False) + "\n```"
    monkeypatch.setattr(scorer, "inference", FakeLLM(reply))
    assert embedder.translate_section_names(GALA_DE, "de") == GALA_EN


def test_llm_invalid_entries_keep_original_and_are_not_cached(monkeypatch, capsys):
    reply = dict(GALA_EN)
    reply["Anbau"] = "  "                                  # empty
    del reply["Züchtung"]                                  # missing
    reply["Geschichte > Ursprung"] = "Origin"              # nesting lost
    monkeypatch.setattr(scorer, "inference", FakeLLM(json.dumps(reply, ensure_ascii=False)))
    info = {}
    result = embedder.translate_section_names(GALA_DE, "de", info=info)

    out = capsys.readouterr().out
    assert "expected 5 entries, got 4" in out
    assert "3 invalid entries" in out
    assert result["Anbau"] == "Anbau"
    assert result["Züchtung"] == "Züchtung"
    assert result["Geschichte > Ursprung"] == "Geschichte > Ursprung"
    assert result["Beschreibung > Frucht"] == "Description > Fruit"
    assert set(cache_on_disk()["de"]) == {"Beschreibung > Frucht", "Beschreibung > Baum"}
    assert info["sources"] == {"llm": 2}
    assert sorted(info["untranslated"]) == ["Anbau", "Geschichte > Ursprung", "Züchtung"]


def test_llm_keys_matched_ignoring_case_and_whitespace(monkeypatch):
    reply = {" anbau ": "Cultivation"}
    monkeypatch.setattr(scorer, "inference", FakeLLM(json.dumps(reply)))
    assert embedder.translate_section_names(["Anbau"], "de") == {"Anbau": "Cultivation"}


def test_cached_names_are_not_requested_again(monkeypatch):
    llm = FakeLLM(json.dumps(GALA_EN, ensure_ascii=False))
    monkeypatch.setattr(scorer, "inference", llm)
    embedder.translate_section_names(GALA_DE, "de")
    monkeypatch.setattr(embedder, "_translation_cache", None)   # reload from disk
    info = {}
    assert embedder.translate_section_names(GALA_DE, "de", info=info) == GALA_EN
    assert len(llm.calls) == 1
    assert info["method"] == "cache"
    assert info["sources"] == {"cache": 5}
    assert "llm" not in info


def test_real_request_code_is_reused(monkeypatch):
    """The translation goes through scorer.inference: same URL, auth header
    and body as scoring."""
    import requests
    sent = []

    class Response:
        status_code = 200
        headers = {}

        def json(self):
            return chat_response(json.dumps({"Anbau": "Cultivation"}))

    def post(url, headers=None, json=None, timeout=None):
        sent.append((url, headers, json))
        return Response()

    monkeypatch.setattr(scorer, "inference", REAL_INFERENCE)
    monkeypatch.setattr(requests, "post", post)
    assert embedder.translate_section_names(["Anbau"], "de") == {"Anbau": "Cultivation"}
    url, headers, body = sent[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "meta-llama/llama-3.3-70b-instruct"
    assert body["temperature"] == 0


# ── Fallback to Google ────────────────────────────────────────────────────────

def test_auto_falls_back_to_google_when_llm_request_fails(monkeypatch):
    monkeypatch.setattr(scorer, "inference", FakeLLM(error=ConnectionError("refused")))
    google = FakeGoogle(GALA_EN)
    monkeypatch.setattr(embedder, "_google_translate", google)
    info = {}
    assert embedder.translate_section_names(GALA_DE, "de", info=info) == GALA_EN
    assert len(google.calls) == 1                         # one batched request
    assert info["method"] == "google"
    assert info["sources"] == {"google": 5}


def test_auto_falls_back_to_google_on_unparseable_reply(monkeypatch):
    monkeypatch.setattr(scorer, "inference", FakeLLM("Sorry, I can't help with that."))
    monkeypatch.setattr(embedder, "_google_translate", FakeGoogle(GALA_EN))
    assert embedder.translate_section_names(GALA_DE, "de") == GALA_EN


@pytest.mark.parametrize("backend,key", [("watsonx", "test-key"), ("openai", "")])
def test_auto_uses_google_without_usable_llm(monkeypatch, backend, key):
    monkeypatch.setattr(scorer, "SCORING_BACKEND", backend)
    monkeypatch.setattr(scorer, "OPENAI_API_KEY", key)
    monkeypatch.setattr(embedder, "_google_translate", FakeGoogle(GALA_EN))
    info = {}
    assert embedder.translate_section_names(GALA_DE, "de", info=info) == GALA_EN
    assert info["method"] == "google"
    assert "llm" not in info


def test_auto_ollama_is_an_llm_backend(monkeypatch):
    monkeypatch.setattr(scorer, "SCORING_BACKEND", "ollama")
    monkeypatch.setattr(scorer, "OPENAI_API_KEY", "")
    llm = FakeLLM(json.dumps(GALA_EN, ensure_ascii=False))
    monkeypatch.setattr(scorer, "inference", llm)
    info = {}
    assert embedder.translate_section_names(GALA_DE, "de", info=info) == GALA_EN
    assert info["llm"]["backend"] == "ollama"


def test_llm_and_google_both_fail_keep_originals(monkeypatch, capsys):
    monkeypatch.setattr(scorer, "inference", FakeLLM(error=ConnectionError("refused")))
    monkeypatch.setattr(embedder, "_google_translate", FakeGoogle(error=too_many_requests()))
    info = {}
    result = embedder.translate_section_names(GALA_DE, "de", info=info)
    assert result == {n: n for n in GALA_DE}
    assert cache_on_disk() == {}
    assert info["method"] == "none"
    assert info["untranslated"] == GALA_DE
    assert "kept untranslated" in capsys.readouterr().out


# ── Explicit TRANSLATION_BACKEND values ───────────────────────────────────────

def test_llm_backend_never_uses_google(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BACKEND", "llm")
    monkeypatch.setattr(scorer, "inference", FakeLLM(error=ConnectionError("refused")))
    info = {}
    assert embedder.translate_section_names(GALA_DE, "de", info=info) == {n: n for n in GALA_DE}
    assert info["translation_backend"] == "llm"
    assert info["untranslated"] == GALA_DE


def test_google_backend_never_uses_llm(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BACKEND", "google")
    monkeypatch.setattr(embedder, "_google_translate", FakeGoogle(GALA_EN))
    info = {}
    assert embedder.translate_section_names(GALA_DE, "de", info=info) == GALA_EN
    assert info["sources"] == {"google": 5}


def test_none_backend_translates_nothing(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BACKEND", "none")
    info = {}
    assert embedder.translate_section_names(GALA_DE, "de", info=info) == {n: n for n in GALA_DE}
    assert info["method"] == "none"
    assert cache_on_disk() == {}


def test_english_is_never_translated():
    info = {}
    assert embedder.translate_section_names(["History"], "en", info=info) == {"History": "History"}
    assert info == {}


def test_check_translation_config(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BACKEND", "deepl")
    with pytest.raises(RuntimeError, match="Unknown TRANSLATION_BACKEND"):
        embedder.check_translation_config()
    monkeypatch.setenv("TRANSLATION_BACKEND", "llm")
    embedder.check_translation_config()                    # openai configured
    monkeypatch.setattr(scorer, "OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        embedder.check_translation_config()
    monkeypatch.setattr(scorer, "SCORING_BACKEND", "watsonx")
    with pytest.raises(RuntimeError, match="SCORING_BACKEND=watsonx"):
        embedder.check_translation_config()
    for value in ("auto", "google", "none"):
        monkeypatch.setenv("TRANSLATION_BACKEND", value)
        embedder.check_translation_config()


# ── meta.json ─────────────────────────────────────────────────────────────────

def test_run_records_translation_in_meta(monkeypatch, tmp_path):
    """run_analysis (--no-score) with fetch, embedding and alignment faked:
    the run's meta.json says how the section names were translated."""
    from app import pipeline

    def sentences(lang, sections):
        return pd.DataFrame([
            {"language": lang, "section": sec, "paragraph_index": i, "sentence_index": 0,
             "sentence_text": f"{sec} text."}
            for i, sec in enumerate(sections)
        ])

    def fetch(url, langs, output_dir):
        return {"en": sentences("en", ["(Lead)", "History"]),
                "de": sentences("de", ["(Lead)"] + GALA_DE)}

    def align(name):
        def write(lang, embed_dir, output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / name.format(lang=lang)).write_text("[]")
        return write

    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(pipeline, "fetch_multilang_from_url", fetch)
    monkeypatch.setattr(pipeline, "embed_language", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "match_en_to_lang", align("en_to_{lang}.json"))
    monkeypatch.setattr(pipeline, "align_language", align("{lang}_to_en.json"))
    monkeypatch.setattr(scorer, "inference", FakeLLM(json.dumps(GALA_EN, ensure_ascii=False)))

    result = asyncio.run(pipeline.run_analysis(
        "https://en.wikipedia.org/wiki/Gala_(apple)", "en-de", do_score=False))
    run_dir = tmp_path / "runs" / result["run_id"]
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["provenance"]["section_names"] == {
        "translation_backend": "auto",
        "sources": {"llm": 5},
        "untranslated": [],
        "llm": {"backend": "openai", "model": "meta-llama/llama-3.3-70b-instruct",
                "base_url": "https://openrouter.ai/api/v1"},
        "method": "llm",
    }
    names = json.loads((run_dir / "section_names_en.json").read_text(encoding="utf-8"))
    assert names["de"] == {"(Lead)": "(Lead)", **GALA_EN}


def test_run_fails_early_on_bad_translation_backend(monkeypatch, tmp_path):
    from app import pipeline
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("TRANSLATION_BACKEND", "deepl")
    with pytest.raises(pipeline.PipelineError) as e:
        asyncio.run(pipeline.run_analysis(
            "https://en.wikipedia.org/wiki/Gala_(apple)", "en-de", do_score=False))
    assert e.value.status_code == 400
    assert not (tmp_path / "runs").exists()
