"""
utils/ai_analysis.py
---------------------
This file contains ALL the functions that talk to the DeepSeek API.

Keeping these functions in a separate file (instead of putting everything
inside app.py) makes the project easier to read and easier to test.

Beginner note:
    "GenAI" here means we send some text to the DeepSeek Chat model
    (deepseek-chat) and ask it to return an answer. We always ask the
    model to reply in a very strict format (JSON) so that Python can read
    the answer reliably.

    DeepSeek's API is OpenAI-compatible, so we use the official `openai`
    Python package and just point it at https://api.deepseek.com.
"""

import json
import os
import re
import time

# We import the OpenAI-compatible client library (used for DeepSeek). If it
# isn't installed, the app will show a friendly error instead of crashing.
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


# Allowed categories. We keep this list in ONE place so app.py and this file
# always agree on the same categories.
CATEGORIES = ["Pest", "Disease", "Water", "Weather", "Nutrient", "Other"]
SEVERITIES = ["Low", "Moderate", "High"]

# How many reports we bundle into a single API call. Bundling reports
# together (instead of one request per report) keeps the number of API calls
# small and the analysis fast.
DEFAULT_GROUP_SIZE = 30

# How many times we retry a single API call if we get a 429 (rate limit)
# error, and how long we wait between retries if the server doesn't tell us.
MAX_RETRIES = 1            # was 3 - long retry loops were a cause of "stuck" analysis
DEFAULT_RETRY_SECONDS = 10
MAX_RETRY_WAIT = 15        # never sleep longer than this between retries
REQUEST_TIMEOUT_S = 60     # give up on any single API call after 60 seconds


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if this exception looks like a 429 (rate limit) error."""
    msg = str(exc)
    return getattr(exc, "status_code", None) == 429 or "429" in msg or "rate limit" in msg.lower()


def _extract_retry_delay(exc: Exception, default: int = DEFAULT_RETRY_SECONDS) -> int:
    """
    If the server sent a Retry-After header, use it (capped); otherwise fall
    back to `default` seconds.
    """
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
    immediately (no point retrying an invalid key or a malformed request).
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


def get_client(api_key: str):
    """
    Create and return a DeepSeek client using the given API key.

    Returns None if:
      - the api_key is empty
      - the openai package failed to import

    Beginner note: we NEVER hard-code the API key in this file. It is always
    passed in from app.py, which reads it from st.secrets["DEEPSEEK_API_KEY"].
    """
    if not api_key:
        return None
    if OpenAI is None:
        return None
    try:
        # The timeout stops a bad connection from freezing the app forever.
        return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=REQUEST_TIMEOUT_S)
    except Exception:
        return None


def _chat(client, model: str, prompt: str, temperature: float = 0, json_mode: bool = False) -> str:
    """
    One place that actually calls DeepSeek. Sends `prompt` as a single user
    message and returns the reply text.

    json_mode=True asks DeepSeek to return a valid JSON *object*.
    """
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = _call_with_retry(lambda: client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        **kwargs,
    ))
    return (response.choices[0].message.content or "").strip()


def _extract_json(text: str):
    """
    Small helper that removes markdown code fences (```json ... ```) if the
    AI added them, then parses the remaining text as JSON.
    Raises ValueError/JSONDecodeError if the text is not valid JSON.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def analyze_report(client, report_text: str, model: str = DEFAULT_MODEL) -> dict:
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
        return {"error": "No DeepSeek client available. Please check your API key."}

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
        raw_text = _chat(client, model, prompt, temperature=0, json_mode=True)
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

    This is the key trick for staying under the free-tier rate limit: one
    request classifies many reports instead of one request per report.

    Returns a list of dicts (same length/order as `reports`). If the whole
    group call fails, every entry in the returned list has "error" set.
    """
    if client is None:
        return [{"error": "No DeepSeek client available. Please check your API key."} for _ in reports]

    numbered = "\n".join(f'{i + 1}. """{text}"""' for i, text in enumerate(reports))

    prompt = f"""You are an agricultural assistant helping analyze farmer reports.

Below is a numbered list of {len(reports)} farmer reports. Analyze EACH ONE
and respond with ONLY a valid JSON object (no extra text, no markdown) with a
single key "results" whose value is an array of exactly {len(reports)}
elements, in the SAME ORDER as the reports below.

Each element of the array must be an object with exactly these keys:
- "category": must be exactly one of {CATEGORIES}
- "severity": must be exactly one of {SEVERITIES}
- "keywords": a short comma-separated string of important keywords (3-5 words)
- "summary": one short sentence summarizing the problem

Farmer reports:
{numbered}

Respond with ONLY the JSON object: {{"results": [ ...exactly {len(reports)} elements... ]}}"""

    try:
        raw_text = _chat(client, model, prompt, temperature=0, json_mode=True)
        parsed = _extract_json(raw_text)
        # DeepSeek's JSON mode returns an object, so unwrap {"results": [...]}
        if isinstance(parsed, dict):
            parsed = parsed.get("results", [])
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


def analyze_reports_batch(client, reports: list, model: str = DEFAULT_MODEL,
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


def ask_chatbot(client, question: str, dataset_summary: str, model: str = DEFAULT_MODEL) -> str:
    """
    Sends the user's question, together with the summarized dataset, to the
    AI and returns a plain-text answer.

    The AI is explicitly told to only use the provided summary and to say
    so if it cannot answer - this prevents the chatbot from making up facts.
    """
    if client is None:
        return "⚠️ AI chatbot is unavailable because no valid DeepSeek API key was found."

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
        return _chat(client, model, prompt, temperature=0.2)
    except Exception as e:
        return f"⚠️ Could not get an answer from the AI right now. ({e})"
