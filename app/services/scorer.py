import json
from pathlib import Path
import re
import aiohttp
import asyncio
import os
import requests
from dotenv import load_dotenv
load_dotenv()

# === Configuration ===
# Two interchangeable scoring backends, selected via .env → SCORING_BACKEND:
#
#   watsonx  — IBM watsonx.ai text generation (Llama 3.3 70B Instruct by
#              default), as used for the IBM watsonx GenAI Challenge 2025.
#              Requires IBM_API_KEY and IBM_PROJECT_ID.
#   ollama   — local inference via Ollama's OpenAI-compatible endpoint
#              (default; no API key needed, fully reproducible offline).
SCORING_BACKEND = os.getenv("SCORING_BACKEND", "ollama").strip().lower()

# --- Ollama (local) ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
CHAT_URL     = f"{OLLAMA_URL}/v1/chat/completions"

# --- IBM watsonx.ai (cloud) ---
WATSONX_URL = os.getenv(
    "IBM_WATSONX_URL",
    "https://eu-de.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29",
)
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")
IBM_API_KEY      = os.getenv("IBM_API_KEY", "")
IBM_PROJECT_ID   = os.getenv("IBM_PROJECT_ID", "")
IAM_TOKEN_URL    = "https://iam.cloud.ibm.com/identity/token"

# watsonx handles concurrent requests; local Ollama serves one at a time
MAX_CONCURRENCY = 8 if SCORING_BACKEND == "watsonx" else 1

# Base directory is two levels up from scorer.py
BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"


# === Authentication (watsonx only) ===
def get_access_token(apikey: str) -> str:
    """Exchange the IBM Cloud API key for a short-lived IAM access token."""
    if not apikey:
        raise RuntimeError(
            "SCORING_BACKEND=watsonx but IBM_API_KEY is not set. "
            "Add IBM_API_KEY and IBM_PROJECT_ID to .env, or switch to SCORING_BACKEND=ollama."
        )
    r = requests.post(
        IAM_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": apikey},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


# === Prompt Construction ===
def form_message(score_type, chunk_data):
    '''Create messages for model input depending on score_type.'''

    if score_type == 'individual':
        system_prompt = '''You are a multilingual bias analysis assistant.

Your task is to classify each sentence on the following three axes and return a JSON object with unified numeric scores between -1.0 and +1.0 (negative = one side, positive = the opposite side; 0 = neutral). Do not generate explanations.

Please return only the following fields for each sentence:

1. **Ideological leaning**
   - score: from -1.0 (Left) to +1.0 (Right), 0.0 = Center/Neutral

2. **Sentiment shift**
   - score: from -1.0 (Strongly negative) to +1.0 (Strongly positive), 0.0 = Neutral

3. **Framing bias**
   - score: from -1.0 (Strongly negative framing) to +1.0 (Strongly positive framing), 0.0 = No framing

---

Return the result in this exact JSON format:

```json
{
  \"chunk_id\": \"<copy from input>\",
  \"ideology_score\": <float between -1.0 and 1.0>,
  \"sentiment_score\": <float between -1.0 and 1.0>,
  \"framing_score\": <float between -1.0 and 1.0>
}


input: {
  \"chunk_id\": \"ENG_001\",
  \"base_lang\": \"ENG\",
  \"chunk_text\": \"The brave citizens stood up against the corrupt regime.\"
}


output: {
  \"chunk_id\": \"ENG_001\",
  \"ideology_score\": -0.6,
  \"sentiment_score\": 0.4,
  \"framing_score\": -0.8
}


input: '''
        input_data = {
            "chunk_id": chunk_data["chunk_id"],
            "base_lang": chunk_data["base_lang"],
            "chunk_text": chunk_data["chunk_text"]
        }
    elif score_type == 'pair':
        system_prompt = '''You are performing omission bias detection.

Given a JSON input with:

\"chunk_text\" in the baseline language (\"base_lang\")

\"omission_text\" in another language (\"omission_lang\")

The full input format is:

```json
{
  \"chunk_id\": \"<copy from input>\",
  \"base_lang\": \"<e.g., ENG>\",
  \"chunk_text\":  \"<copy from input>\",
  \"omission_lang\": \"<e.g., DEU>\",
  \"omission_text\": \"<copy from input>\"
}

Your task is:

Identify and count the major factual information points in \"chunk_text\".

Determine how many of these points are preserved or mentioned in \"omission_text\".

You can translate the non-English texts into English for clearer comparison of factual points.

Please be aware that information mentioned in \"omission_text\" but does not appear in \"chunk_text\" should NOT affect your omission_points and base_points.

Calculate the omission score as: omission_score = 1 - (omission_points / base_points)

omission_score = 0.0 → no information was omitted

omission_score = 1.0 → all baseline information was omitted

Format your output as JSON with the following fields:

{
  \"chunk_id\": \"...\",
  \"base_points\": <int>,            // number of info points in chunk_text
  \"omission_points\": <int>,        // number of those found in omission_text
  \"omission_score\": <float>        // 1 - (omission_points / base_points)
}


Do not explain your reasoning. Only return the formatted JSON.


input: {
  \"chunk_id\": \"en_sub_0004\",
  \"base_lang\": \"en\",
  \"chunk_text\":  \"On Tuesday, the government passed a controversial education reform bill after a heated debate in parliament. The opposition party walked out in protest, and thousands of students marched through the capital in response.\",
  \"omission_lang\": \"zh\",
  \"omission_text\": \"周二，政府通过了一项教育改革法案。\"
}
output: {
  \"chunk_id\": \"en_cn_0004\",
  \"base_points\": 4,
  \"omission_points\":1,
 \"omission_score\":0.75
}


input: {
  \"chunk_id\": \"en_cn_002\",
  \"base_lang\": \"en\",
  \"chunk_text\":  \"On Tuesday, the government passed a controversial education reform bill after a heated debate in parliament. The opposition party walked out in protest, and thousands of students marched through the capital in response.\",
  \"omission_lang\": \"zh\",
  \"omission_text\": \"周二，政府通过了一项教育改革法案。该法案将影响全国高校约12000名学生。\"
}
output: {
  \"chunk_id\": \"en_cn_0004\",
  \"base_points\": 4,
  \"omission_points\":1,
 \"omission_score\":0.75
}

input: {
  \"chunk_id\": \"en_others_0003\",
  \"base_lang\": \"en\",
  \"chunk_text\":  \"At the climate summit, 195 countries agreed to reduce carbon emissions by 40% by 2030. The agreement was seen as a major diplomatic success and includes penalties for non-compliance.\",
  \"omission_lang\": \"zh\",
  \"omission_text\": \"全球气候峰会在瓦伦西亚召开。\"
}
output: {
  \"chunk_id\": \"en_cn_0006\",
  \"base_points\": 3,
  \"omission_points\":0,
 \"omission_score\":1
}

input: {
  \"chunk_id\": \"en_cn_00007\",
  \"base_lang\": \"en\",
  \"chunk_text\":  \"After the company announced layoffs affecting 10% of its workforce, employee unions staged a protest in front of the headquarters, demanding transparency and severance packages.\",
  \"omission_lang\": \"zh\",
  \"omission_text\": \"三月，该公司宣布裁员10%。工会员工一周后在公司总部举行了抗议活动，诉求包括公司提高透明度并提供合理的遣散费。抗议活动持续了三天。\"
}
output: {
  \"chunk_id\": \"en_cn_0007\",
  \"base_points\": 3,
  \"omission_points\":3,
 \"omission_score\":0
}



input: '''
        input_data = chunk_data
    else:
        raise ValueError(f"Unknown score_type: {score_type}")
    # Return OpenAI-style messages list for Ollama chat completions
    user_content = json.dumps(input_data, ensure_ascii=False) + "\n\noutput:"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


# === Inference ===
def _ollama_body(messages: list[dict]) -> dict:
    return {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0},
    }


def _watsonx_body(messages: list[dict]) -> dict:
    """Build a watsonx.ai text-generation request.

    watsonx's /text/generation endpoint takes a single prompt string, so the
    system + user messages are concatenated. Greedy decoding ≈ temperature 0;
    the "\\n}" stop sequence truncates generation right after the JSON object.
    HAP/PII masking and Granite Guardian run as guardrails on input/output.
    """
    prompt = messages[0]["content"] + " " + messages[1]["content"]
    return {
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 800,
            "min_new_tokens": 0,
            "stop_sequences": ["\n}"],
            "repetition_penalty": 1,
        },
        "model_id": WATSONX_MODEL_ID,
        "project_id": IBM_PROJECT_ID,
        "moderations": {
            "hap": {
                "input":  {"enabled": True, "threshold": 0.5, "mask": {"remove_entity_value": True}},
                "output": {"enabled": True, "threshold": 0.5, "mask": {"remove_entity_value": True}},
            },
            "pii": {
                "input":  {"enabled": True, "threshold": 0.5, "mask": {"remove_entity_value": True}},
                "output": {"enabled": True, "threshold": 0.5, "mask": {"remove_entity_value": True}},
            },
            "granite_guardian": {"input": {"threshold": 1}},
        },
    }


def inference(messages, access_token: str | None = None):
    """Synchronous inference via the configured backend."""
    if SCORING_BACKEND == "watsonx":
        r = requests.post(
            WATSONX_URL,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
            json=_watsonx_body(messages),
            timeout=120,
        )
        if r.status_code != 200:
            raise Exception(f"watsonx error {r.status_code}: {r.text}")
        return r.json()

    r = requests.post(CHAT_URL, json=_ollama_body(messages), timeout=120)
    r.raise_for_status()
    return r.json()


async def async_inference(session, messages, access_token: str | None = None):
    """Async inference via the configured backend."""
    if SCORING_BACKEND == "watsonx":
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        async with session.post(WATSONX_URL, headers=headers, json=_watsonx_body(messages)) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"watsonx error {response.status}: {text}")
            return await response.json()

    async with session.post(CHAT_URL, json=_ollama_body(messages)) as response:
        if response.status != 200:
            text = await response.text()
            raise Exception(f"Ollama error {response.status}: {text}")
        return await response.json()

# === Parse model response ===


def compute_omission_score(base_points, omission_points):
    """omission_score = 1 - preserved/base, computed here from the model's counts.

    The model's own omission_score is not used: it frequently returns the
    inverted ratio (preserved/base). Counts are clipped so that the number of
    preserved points can't exceed the number of base points.
    """
    if isinstance(base_points, bool) or isinstance(omission_points, bool):
        return None
    if not isinstance(base_points, (int, float)) or not isinstance(omission_points, (int, float)):
        return None
    if base_points <= 0:
        return None
    preserved = min(max(omission_points, 0), base_points)
    return round(1 - preserved / base_points, 4)


def _extract_generated_text(response: dict) -> str:
    """Pull the generated text out of either backend's response format.

    watsonx text generation: {"results": [{"generated_text": "..."}]}
    Ollama (OpenAI-compatible): {"choices": [{"message": {"content": "..."}}]}
    """
    if "results" in response:
        return response["results"][0]["generated_text"]
    return response["choices"][0]["message"]["content"]


SCORE_FIELDS = {
    "individual": ("ideology_score", "framing_score", "sentiment_score"),
    "pair": ("base_points", "omission_points"),
}


def _failed(score_type, reason, raw):
    """Explicitly-marked failure: all fields None plus <score_type>_error."""
    print(f"[!] {score_type} scoring failed: {reason}\n    raw: {raw[:300]!r}")
    result = {k: None for k in SCORE_FIELDS[score_type]}
    if score_type == "pair":
        result["omission_score"] = None
    result[f"{score_type}_error"] = reason
    return result


def parse_response(score_type, response):
    if score_type not in SCORE_FIELDS:
        raise ValueError(f"Unknown score_type: {score_type}")
    generated_str = _extract_generated_text(response)

    # Try to find a JSON object — also handle truncated output (no closing brace)
    # First try complete object, then try to auto-close if model truncated output
    match = re.search(r'\{.*?\}', generated_str, re.DOTALL)
    if not match:
        # Try to salvage: find opening brace and close it
        open_match = re.search(r'\{.*', generated_str, re.DOTALL)
        if open_match:
            generated_str = open_match.group(0).rstrip() + "\n}"
            match = re.search(r'\{.*?\}', generated_str, re.DOTALL)
    if not match:
        return _failed(score_type, "no JSON object in model output", generated_str)
    try:
        generated_json = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return _failed(score_type, f"invalid JSON: {e}", match.group(0))

    values = {}
    for key in SCORE_FIELDS[score_type]:
        v = generated_json.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return _failed(score_type, f"missing or non-numeric '{key}'", match.group(0))
        if score_type == "individual" and not -1.0 <= v <= 1.0:
            return _failed(score_type, f"'{key}'={v} outside [-1, 1]", match.group(0))
        values[key] = v

    if score_type == 'individual':
        score_dict = values
    else:
        score_dict = {
            **values,
            "omission_score": compute_omission_score(values["base_points"], values["omission_points"]),
            "omission_basis": "llm_counts",
        }
        if score_dict["omission_score"] is None:
            score_dict["pair_error"] = f"cannot compute omission from base_points={values['base_points']}"
    return score_dict

# === Recording Output ===


def record_output(score_data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = []

    results.append(score_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# === Scoring Loop ===

# A chunk with no aligned counterpart is fully omitted by definition; no LLM
# call is made, so there are no point counts (previously faked as 1 and 0).
UNALIGNED_OMISSION = {
    "base_points": None,
    "omission_points": None,
    "omission_score": 1.0,
    "omission_basis": "unaligned",
}

OMISSION_TEXT_MAX_CHARS = 1200


def build_omission_text(matches, tgt_field, max_chars=OMISSION_TEXT_MAX_CHARS):
    """Concatenate the aligned texts, best match first, up to max_chars.

    Deterministic (ordered by chunk_sim, not by set iteration order, which
    changes between processes), and never empty for an aligned chunk: if
    the best match alone is longer than max_chars it is truncated rather
    than dropped.
    """
    ordered = sorted(matches, key=lambda m: m.get("chunk_sim", 0), reverse=True)
    texts = list(dict.fromkeys(m[tgt_field + "_text"] for m in ordered))
    omission_text = ""
    for text in texts:
        if not omission_text:
            omission_text = text[:max_chars]
            continue
        if len(omission_text) + len(text) + 1 > max_chars:
            break
        omission_text += " " + text
    return omission_text



def process_file(file_path, base_lang, omission_lang, src_field, tgt_field, score_filename, access_token=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seen_chunks = set()
    for chunk in data:
        chunk_id = chunk.get(src_field + "_chunk_id")
        match_type = chunk.get("match_type")
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        omission_text = build_omission_text(chunk["matches"], tgt_field)

        chunk_data = {
            "chunk_id": chunk[src_field + "_chunk_id"],
            "base_lang": base_lang,
            "chunk_text": chunk[src_field + "_text"],
            "omission_lang": omission_lang,
            "omission_text": omission_text
        }

        # Individual scoring
        msg = form_message(score_type="individual", chunk_data=chunk_data)
        response = inference(msg, access_token)
        scores_i = parse_response("individual", response)

        # Pair scoring
        if match_type == "1:0":
            scores_p = UNALIGNED_OMISSION.copy()
        else:
            msg = form_message(score_type="pair", chunk_data=chunk_data)
            response = inference(msg, access_token)
            scores_p = parse_response("pair", response)

        score_data = chunk_data | scores_i | scores_p  # merge two dictionaries
        record_output(score_data, OUTPUT_DIR /
                      'scores' / score_filename)


async def async_process_file(file_path, base_lang, omission_lang, src_field, tgt_field, score_filename, access_token=None):
    # watsonx: up to 8 concurrent requests; local Ollama: 1 (serves one at a time)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seen_chunks = set()

    async with aiohttp.ClientSession() as session:

        async def run_chunk(chunk):
            chunk_id = chunk.get(src_field + "_chunk_id")
            match_type = chunk.get("match_type")
            if chunk_id in seen_chunks:
                return
            seen_chunks.add(chunk_id)

            omission_text = build_omission_text(chunk["matches"], tgt_field)

            chunk_data = {
                "chunk_id": chunk[src_field + "_chunk_id"],
                "base_lang": base_lang,
                "chunk_text": chunk[src_field + "_text"],
                "omission_lang": omission_lang,
                "omission_text": omission_text
            }

            async with semaphore:
                msg = form_message(score_type="individual", chunk_data=chunk_data)
                response = await async_inference(session, msg, access_token)
                scores_i = parse_response("individual", response)

            if match_type == "1:0":
                scores_p = UNALIGNED_OMISSION.copy()
            else:
                async with semaphore:
                    msg = form_message(score_type="pair", chunk_data=chunk_data)
                    response = await async_inference(session, msg, access_token)
                    scores_p = parse_response("pair", response)

            score_data = chunk_data | scores_i | scores_p
            record_output(score_data, OUTPUT_DIR / 'scores' / score_filename)

        await asyncio.gather(*(run_chunk(chunk) for chunk in data))


def reorder_output_by_input(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as f_in:
        input_data = json.load(f_in)
    with open(output_file_path, "r", encoding="utf-8") as f_out:
        output_data = json.load(f_out)

    # Detect chunk ID field based on input file path
    if "en_to_" in str(input_file_path):
        id_field = "tgt_chunk_id"
    elif "_to_en" in str(input_file_path):
        id_field = "src_chunk_id"
    else:
        raise ValueError(
            f"Cannot determine id_field from input file path: {input_file_path}")

    # Build mapping from chunk_id to output entry
    output_map = {entry["chunk_id"]: entry for entry in output_data}
    reordered_output = []
    seen = set()

    for chunk in input_data:
        chunk_id = chunk.get(id_field)
        if chunk_id and chunk_id in output_map and chunk_id not in seen:
            reordered_output.append(output_map[chunk_id])
            seen.add(chunk_id)

    # Save reordered results
    with open(output_file_path, "w", encoding="utf-8") as f_out:
        json.dump(reordered_output, f_out, ensure_ascii=False, indent=2)


# === Entry Point ===
async def score(language_pair):
    lang1, lang2 = language_pair.split("-")
    lang1, lang2 = (lang2, lang1) if lang1 == 'en' else (lang1, lang2)

    if SCORING_BACKEND == "watsonx":
        print(f"  Scoring with watsonx.ai model: {WATSONX_MODEL_ID}  ({WATSONX_URL})")
        access_token = get_access_token(IBM_API_KEY)
    else:
        print(f"  Scoring with Ollama model: {OLLAMA_MODEL}  ({CHAT_URL})")
        access_token = None

    await async_process_file(
        file_path=OUTPUT_DIR/'alignments_en_to_source'/f"en_to_{lang1}.json",
        base_lang='en',
        omission_lang=lang1,
        src_field='tgt',
        tgt_field='src',
        score_filename=f"base_en_omit_{lang1}_scores.json",
        access_token=access_token,
    )

    if SCORING_BACKEND == "watsonx":
        access_token = get_access_token(IBM_API_KEY)  # fresh token for the second pass

    await async_process_file(
        file_path=OUTPUT_DIR/'alignments_source_to_en'/f"{lang1}_to_en.json",
        base_lang=lang1,
        omission_lang='en',
        src_field='src',
        tgt_field='tgt',
        score_filename=f"base_{lang1}_omit_en_scores.json",
        access_token=access_token,
    )

    reorder_output_by_input(
        input_file_path=OUTPUT_DIR /
        'alignments_en_to_source'/f"en_to_{lang1}.json",
        output_file_path=OUTPUT_DIR/'scores' /
        f"base_en_omit_{lang1}_scores.json"
    )
    reorder_output_by_input(
        input_file_path=OUTPUT_DIR /
        'alignments_source_to_en'/f"{lang1}_to_en.json",
        output_file_path=OUTPUT_DIR/'scores' /
        f"base_{lang1}_omit_en_scores.json"
    )


# for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score ideological bias in text chunks.")
    parser.add_argument(
        "lang_pair",
        type=str,
        help="Language pair identifier (e.g., 'en-de', 'en-fr')"
    )
    args = parser.parse_args()

    # await score(args.lang_pair)
   # python app/services/scorer.py en-de
