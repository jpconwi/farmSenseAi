"""
utils/ai_analysis.py
---------------------
This file contains ALL the functions that talk to the AI model.

Two ways to run the AI (chosen in the app sidebar):
  1. Hugging Face:  app.py -> HF API token -> Hugging Face Inference
                    Providers -> Qwen2.5-7B-Instruct
  2. Ollama:        app.py -> Ollama running on your own computer

Keeping these functions in a separate file (instead of putting everything
inside app.py) makes the project easier to read and easier to test.

Beginner note:
    "GenAI" here means we send some text to the Qwen2.5-7B-Instruct model
    (hosted through Hugging Face Inference Providers) and ask it to return
    an answer. We always ask the model to reply in a very strict format
    (JSON) so that Python can read the answer reliably.

    Hugging Face's router is OpenAI-compatible, so we simply send a
    "chat/completions" request to https://router.huggingface.co/v1 with your
    token.
"""

import json
import os
import re
import time

# We talk to Hugging Face with the plain `requests` library (already
# installed together with Streamlit), so there is nothing extra to install.
import requests

# Hugging Face chat models, in order of preference. If the first one fails
# (not served by any provider, out of credits, timeout...), the app
# automatically moves on to the next one. Each must be a model that an
# Inference Provider actually serves - see https://router.huggingface.co/v1/models
MODEL_CHAIN = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
]
MODEL_NAME = MODEL_CHAIN[0]  # default / first choice

# Sentiment model (positive / neutral / negative). This is NOT a chat model:
# it is a text classifier, so it is used only for the "sentiment" column.
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_BASE_URL = "https://router.huggingface.co/v1"


# Allowed categories. We keep this list in ONE place so app.py and this file
# always agree on the same categories.
CATEGORIES = ["Pest", "Disease", "Water", "Weather", "Nutrient", "Other"]
SEVERITIES = ["Low", "Moderate", "High"]

# How many reports we bundle into a single API call. Bundling reports
# together (instead of one request per report) keeps the number of API calls
# small. A 3B model follows instructions best with smaller groups, so we
# keep this modest.
DEFAULT_GROUP_SIZE = 8

# How many times we retry a single API call if we get a 429 (rate limit)
# error, and how long we wait between retries if the server doesn't tell us.
MAX_RETRIES = 1            # was 3 - long retry loops were a cause of "stuck" analysis
DEFAULT_RETRY_SECONDS = 10
MAX_RETRY_WAIT = 15        # never sleep longer than this between retries
REQUEST_TIMEOUT_S = 60     # give up on any single API call after 60 seconds


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if this exception looks like a 429 (rate limit) error."""
    return getattr(exc, "status_code", None) == 429 or "429" in str(exc)


def _extract_retry_delay(exc: Exception, default: int = DEFAULT_RETRY_SECONDS) -> int:
    """If the server sent a Retry-After header, use it (capped); else `default`."""
    try:
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            return min(int(float(retry_after)) + 1, MAX_RETRY_WAIT)
    except Exception:
        pass
    return min(default, MAX_RETRY_WAIT)


def _call_with_retry(fn, max_retries: int = MAX_RETRIES):
    """
    Calls fn() and, if it fails with a rate-limit (429) error, waits and
    retries up to max_retries times. Any other kind of error is raised
    immediately (no point retrying an invalid token or a malformed request).
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if _is_rate_limit_error(e) and attempt < max_retries:
                time.sleep(_extract_retry_delay(e))
                continue
            raise
    raise last_exc


class HFClient:
    """Tiny client for Hugging Face Inference Providers (OpenAI-compatible)."""

    def __init__(self, api_key: str, base_url: str = HF_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")


def get_client(api_key: str):
    """
    Create and return a Hugging Face client using your token.
    Returns None if the token is empty.

    Beginner note: we NEVER hard-code the token in this file. It is always
    passed in from app.py, which reads it from st.secrets["HF_TOKEN"].
    """
    if not api_key:
        return None
    return HFClient(api_key)


# ---------------------------------------------------------------------------
# OLLAMA (local AI) SUPPORT
# ---------------------------------------------------------------------------
OLLAMA_TIMEOUT = 180  # seconds; local models on a laptop can be slow


class OllamaClient:
    """Tiny client for a local Ollama server (default http://localhost:11434)."""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")


def ollama_list_models(host: str = "http://localhost:11434") -> list:
    """Returns the names of models installed in Ollama, or [] if unreachable."""
    try:
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _generate(client, model: str, prompt: str, temperature: float = 0, json_mode: bool = False) -> str:
    """
    Sends one prompt to either Ollama or Hugging Face and returns the reply
    text. json_mode only matters for Ollama (it forces valid JSON output).
    """
    if isinstance(client, OllamaClient):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        r = requests.post(client.host + "/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:200]}")
        return r.json()["message"]["content"].strip()

    r = requests.post(
        client.base_url + "/chat/completions",
        headers={"Authorization": f"Bearer {client.api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Hugging Face error {r.status_code}: {r.text[:300]}")
    text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    # Some models (e.g. Qwen3) print their reasoning inside <think>...</think>
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str):
    """
    Small helper that parses the AI's reply as JSON. Small models sometimes
    add markdown fences (```json ... ```) or a sentence before/after the JSON,
    so if a direct parse fails we cut out the first {...} or [...] block.
    Raises json.JSONDecodeError if no valid JSON can be found.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
        if not starts:
            raise
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end <= start:
            raise
        return json.loads(cleaned[start:end + 1])


def analyze_report(client, report_text: str, model: str = MODEL_NAME) -> dict:
    """
    Send ONE farmer report to the AI and ask it to classify it.

    Returns a dictionary like:
        {
            "category": "Pest",
            "severity": "Moderate",
            "keywords": "insects, rice, leaves",
            "summary": "The rice plants are experiencing insect-related damage."
        }

    If anything goes wrong (no client, API error, bad response), this
    function returns a dictionary with "error" set to a human-readable
    message. It NEVER makes up fake analysis results.
    """
    if client is None:
        return {"error": "No AI client available. Please check your Hugging Face token (or that Ollama is running)."}

    if not report_text or not str(report_text).strip():
        return {"error": "Report text is empty, nothing to analyze."}

    # This is the instruction we send to the AI model.
    # We are very specific about the JSON format we want back, so that
    # Python can reliably read the result every time.
    prompt = f"""You are an agricultural assistant helping analyze farmer reports.

Read the farmer report below and respond with ONLY a valid JSON object
(no extra text, no markdown) with exactly these keys:

- "category": must be exactly one of {CATEGORIES}
- "severity": must be exactly one of {SEVERITIES}
- "keywords": a short comma-separated string of important keywords (3-5 words)
- "summary": one short sentence summarizing the problem

Farmer report:
\"\"\"{report_text}\"\"\"

Respond with ONLY the JSON object."""

    try:
        raw_text = _call_with_retry(lambda: _generate(client, model, prompt, 0, True))
        result = _extract_json(raw_text)

        # --- Validate the AI's response before trusting it ---
        category = result.get("category", "Other")
        if category not in CATEGORIES:
            category = "Other"  # fallback to a safe default

        severity = result.get("severity", "Low")
        if severity not in SEVERITIES:
            severity = "Low"

        return {
            "category": category,
            "severity": severity,
            "keywords": str(result.get("keywords", "")).strip(),
            "summary": str(result.get("summary", "")).strip(),
        }

    except json.JSONDecodeError:
        return {"error": "The AI returned a response that wasn't valid JSON. Please try again."}
    except Exception as e:
        # This catches API errors, internet/connection errors, invalid key errors, etc.
        return {"error": f"AI request failed: {e}"}


def _analyze_group(client, reports: list, model: str) -> list:
    """
    Sends a SMALL GROUP of reports (e.g. 8) to the AI in a single API call
    and asks for a JSON array of results, one per report, in the same order.

    This is the key trick for keeping API usage low: one request classifies
    many reports instead of one request per report.

    Returns a list of dicts (same length/order as `reports`). If the whole
    group call fails, every entry in the returned list has "error" set.
    """
    if client is None:
        return [{"error": "No AI client available. Please check your Hugging Face token (or that Ollama is running)."} for _ in reports]

    numbered = "\n".join(f'{i + 1}. """{text}"""' for i, text in enumerate(reports))

    prompt = f"""You are an agricultural assistant helping analyze farmer reports.

Below is a numbered list of {len(reports)} farmer reports. Analyze EACH ONE
and respond with ONLY a valid JSON array (no extra text, no markdown) with
exactly {len(reports)} elements, in the SAME ORDER as the reports below.

Each element must be an object with exactly these keys:
- "category": must be exactly one of {CATEGORIES}
- "severity": must be exactly one of {SEVERITIES}
- "keywords": a short comma-separated string of important keywords (3-5 words)
- "summary": one short sentence summarizing the problem

Farmer reports:
{numbered}

Respond with ONLY the JSON array, containing exactly {len(reports)} elements."""

    try:
        if isinstance(client, OllamaClient):
            # Ollama's JSON mode must return an object, so wrap the array.
            prompt += '\nReturn the array inside a JSON object like {"results": [ ... ]}.'
        text = _call_with_retry(lambda: _generate(client, model, prompt, 0, True))
        parsed = _extract_json(text)
        if isinstance(parsed, dict):
            # unwrap {"results": [...]} (or any single list value)
            parsed = next((v for v in parsed.values() if isinstance(v, list)), parsed)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON array from the AI.")

        results = []
        for i in range(len(reports)):
            if i >= len(parsed) or not isinstance(parsed[i], dict):
                results.append({"error": "The AI did not return a result for this report."})
                continue
            item = parsed[i]
            category = item.get("category", "Other")
            if category not in CATEGORIES:
                category = "Other"
            severity = item.get("severity", "Low")
            if severity not in SEVERITIES:
                severity = "Low"
            results.append({
                "category": category,
                "severity": severity,
                "keywords": str(item.get("keywords", "")).strip(),
                "summary": str(item.get("summary", "")).strip(),
            })
        return results

    except json.JSONDecodeError:
        return [{"error": "The AI returned a response that wasn't valid JSON. Please try again."} for _ in reports]
    except Exception as e:
        return [{"error": f"AI request failed: {e}"} for _ in reports]


def analyze_reports_batch(client, reports: list, model: str = MODEL_NAME,
                           progress_callback=None, group_size: int = DEFAULT_GROUP_SIZE) -> list:
    """
    Classifies a list of report strings using the AI, sending `group_size`
    reports per API call (instead of one call per report). This dramatically
    cuts the number of requests made.

    progress_callback (optional): a function that accepts a float between
    0 and 1, used to update a Streamlit progress bar.

    Returns a list of dictionaries (same shape as analyze_report's output),
    in the same order as the input list.
    """
    if not reports:
        return []

    results = []
    total_groups = max(1, (len(reports) + group_size - 1) // group_size)

    for group_index in range(total_groups):
        start = group_index * group_size
        chunk = reports[start:start + group_size]
        if not chunk:
            continue
        results.extend(_analyze_group(client, chunk, model=model))
        if progress_callback:
            progress_callback((group_index + 1) / total_groups)

    return results


# ---------------------------------------------------------------------------
# SENTIMENT (cardiffnlp/twitter-roberta-base-sentiment-latest)
# ---------------------------------------------------------------------------
_SENTIMENT_LABELS = ("positive", "neutral", "negative")
_LABEL_FIX = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}


def _norm_sentiment(label) -> str:
    """Turn whatever a model returned into positive / neutral / negative."""
    if isinstance(label, dict):
        label = label.get("sentiment") or label.get("label") or ""
    label = str(label).strip().lower()
    label = _LABEL_FIX.get(label, label)
    return label if label in _SENTIMENT_LABELS else "neutral"


def _sentiment_classifier(client, texts: list, model: str) -> list:
    """
    Sentiment with the cardiffnlp text-classification model. The Hugging Face
    API takes ONE text per request, so we send the (few) unique reports one
    by one. Raises an error on the first failed request.
    """
    url = f"https://router.huggingface.co/hf-inference/models/{model}"

    def post_one(text):
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {client.api_key}"},
            json={"inputs": text[:1500]},
            timeout=REQUEST_TIMEOUT_S,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Hugging Face error {r.status_code}: {r.text[:200]}")
        return r.json()

    labels = []
    for text in texts:
        data = _call_with_retry(lambda t=text: post_one(t))
        scores = data[0] if data and isinstance(data[0], list) else data
        best = max(scores, key=lambda d: d.get("score", 0))
        labels.append(_norm_sentiment(best.get("label", "neutral")))
    return labels


def _sentiment_chat(client, texts: list, models: list):
    """
    Backup: ask a chat model (trying each one in `models` in turn) to label the
    sentiment. Returns (labels, model_used); raises the last error if all fail.
    """
    last_exc = None
    for m in models:
        try:
            labels = []
            for i in range(0, len(texts), DEFAULT_GROUP_SIZE):
                chunk = texts[i:i + DEFAULT_GROUP_SIZE]
                numbered = "\n".join(f'{j + 1}. """{t}"""' for j, t in enumerate(chunk))
                prompt = f"""Label the sentiment of each of the {len(chunk)} farmer reports below as
exactly one of: "positive", "neutral", "negative".
Respond with ONLY a JSON array of {len(chunk)} strings, in the same order.

Reports:
{numbered}"""
                raw = _call_with_retry(lambda: _generate(client, m, prompt, 0, True))
                parsed = _extract_json(raw)
                if isinstance(parsed, dict):
                    parsed = next((v for v in parsed.values() if isinstance(v, list)), parsed)
                if not isinstance(parsed, list) or len(parsed) != len(chunk):
                    raise ValueError("Unexpected sentiment answer from the AI.")
                labels.extend(_norm_sentiment(x) for x in parsed)
            return labels, m
        except Exception as e:
            last_exc = e
    raise last_exc


def analyze_sentiments(client, texts: list, model: str = SENTIMENT_MODEL, fallback_models: list = None):
    """
    Labels each text "positive", "neutral" or "negative".

    First tries the cardiffnlp sentiment model on Hugging Face. If that fails
    (model not available, no permission, etc.) and `fallback_models` is given,
    it asks the chat models instead, one after another.

    Returns (labels, error, source): `labels` is a list the same length as
    `texts` (or [] on failure), `error` is None or a readable message, and
    `source` is the name of the model that produced the labels.
    """
    if not isinstance(client, HFClient):
        return [], "Sentiment needs the Hugging Face provider.", None
    if not texts:
        return [], None, model
    try:
        return _sentiment_classifier(client, texts, model), None, model
    except Exception as first_err:
        if fallback_models:
            try:
                labels, used = _sentiment_chat(client, texts, list(fallback_models))
                return labels, None, used
            except Exception as e:
                return [], f"{model} failed ({str(first_err)[:120]}); chat models also failed ({str(e)[:120]})", None
        return [], f"Sentiment request failed: {first_err}", None


# ---------------------------------------------------------------------------
# PERSISTENT (ON-DISK) CACHE
# ---------------------------------------------------------------------------
# st.session_state only lasts for one browser session - restart the app and
# it's gone, so every restart re-analyzes the whole dataset again even
# though the reports haven't changed. This on-disk cache fixes that: once a
# report has been analyzed successfully, its result is saved to a small
# JSON file and reused forever (until the report text or model changes).
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ai_cache.json")


def _cache_key(report_text: str, model: str) -> str:
    """Build a stable string key for one (report, model) pair."""
    return f"{model}::{report_text.strip()}"


def load_disk_cache(cache_path: str = DEFAULT_CACHE_PATH) -> dict:
    """
    Loads the on-disk cache file into a dict of {cache_key: result}.
    Returns an empty dict if the file doesn't exist yet or is corrupted -
    a missing/bad cache should never crash the app, just mean a cache miss.
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_disk_cache(cache: dict, cache_path: str = DEFAULT_CACHE_PATH) -> None:
    """
    Saves the cache dict to disk as JSON. Any failure here (e.g. read-only
    filesystem on some hosting platforms) is swallowed - losing the cache
    just means slower re-analysis next time, never a crash.
    """
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def build_dataset_summary(df, max_chars: int = 4000) -> str:
    """
    Creates a SHORT, summarized text version of the dataset to send to the
    chatbot, instead of sending the entire raw dataset every time.

    Why: sending huge datasets to the API is slow, expensive, and can hit
    token limits. A summary is usually enough for the AI to answer
    questions like "which crop has the most reports?".
    """
    if df is None or df.empty:
        return "The dataset is empty."

    lines = []
    lines.append(f"Total reports: {len(df)}")

    if "crop" in df.columns:
        crop_counts = df["crop"].value_counts()
        lines.append("Reports per crop: " + ", ".join(f"{c}={n}" for c, n in crop_counts.items()))

    if "location" in df.columns:
        loc_counts = df["location"].value_counts()
        lines.append("Reports per location: " + ", ".join(f"{l}={n}" for l, n in loc_counts.items()))

    if "category" in df.columns:
        cat_counts = df["category"].value_counts()
        lines.append("Reports per category: " + ", ".join(f"{c}={n}" for c, n in cat_counts.items()))

    if "severity" in df.columns:
        sev_counts = df["severity"].value_counts()
        lines.append("Reports per severity: " + ", ".join(f"{s}={n}" for s, n in sev_counts.items()))

    # Cross-tab: crop x category (helps answer "what problems are common for rice?")
    if "crop" in df.columns and "category" in df.columns:
        cross = df.groupby(["crop", "category"]).size()
        cross_lines = [f"{crop}/{cat}={count}" for (crop, cat), count in cross.items()]
        lines.append("Crop-category breakdown: " + ", ".join(cross_lines))

    # Cross-tab: location x category (helps answer "which location has most pest problems?")
    if "location" in df.columns and "category" in df.columns:
        cross2 = df.groupby(["location", "category"]).size()
        cross2_lines = [f"{loc}/{cat}={count}" for (loc, cat), count in cross2.items()]
        lines.append("Location-category breakdown: " + ", ".join(cross2_lines))

    summary = "\n".join(lines)

    # Safety cutoff so we never send a huge amount of text to the API
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n...(summary truncated)"

    return summary


def ask_chatbot(client, question: str, dataset_summary: str, model: str = MODEL_NAME) -> str:
    """
    Sends the user's question, together with the summarized dataset, to the
    AI and returns a plain-text answer.

    The AI is explicitly told to only use the provided summary and to say
    so if it cannot answer - this prevents the chatbot from making up facts.
    """
    if client is None:
        return "⚠️ AI chatbot is unavailable because no valid Hugging Face token was found."

    if not question or not question.strip():
        return "Please type a question first."

    prompt = f"""You are FarmSense AI, a helpful assistant that answers questions
about a dataset of farmer agricultural reports.

You must answer using ONLY the dataset summary below. Do not invent numbers
or facts that are not in the summary. If the answer cannot be determined
from the dataset, say that the information is not available in the dataset.

Dataset summary:
{dataset_summary}

Question: {question}

Give a short, clear, direct answer."""

    try:
        return _call_with_retry(lambda: _generate(client, model, prompt, 0.2))
    except Exception as e:
        return f"⚠️ Could not get an answer from the AI right now. ({e})"
