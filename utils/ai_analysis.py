"""
utils/ai_analysis.py
---------------------
This file contains ALL the functions that talk to the Google Gemini API.

Keeping these functions in a separate file (instead of putting everything
inside app.py) makes the project easier to read and easier to test.

Beginner note:
    "GenAI" here means we send some text to Google's Gemini model (e.g.
    gemini-3.6-flash) and ask it to return an answer. We always ask the
    model to reply in a very strict format (JSON) so that Python can read
    the answer reliably.
"""

import json
import re

# We import the Google Gen AI library. If it isn't installed, the app will
# show a friendly error instead of crashing (handled in app.py).
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# Allowed categories. We keep this list in ONE place so app.py and this file
# always agree on the same categories.
CATEGORIES = ["Pest", "Disease", "Water", "Weather", "Nutrient", "Other"]
SEVERITIES = ["Low", "Moderate", "High"]


def get_client(api_key: str):
    """
    Create and return a Gemini client using the given API key.

    Returns None if:
      - the api_key is empty
      - the google-genai package failed to import

    Beginner note: we NEVER hard-code the API key in this file. It is always
    passed in from app.py, which reads it from st.secrets["GEMINI_API_KEY"].
    """
    if not api_key:
        return None
    if genai is None:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


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


def analyze_report(client, report_text: str, model: str = "gemini-3.6-flash") -> dict:
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
        return {"error": "No Gemini client available. Please check your API key."}

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
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,  # temperature=0 makes the answer more consistent
                response_mime_type="application/json",  # asks Gemini to return raw JSON
            ),
        )
        raw_text = response.text
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


def analyze_reports_batch(client, reports: list, model: str = "gemini-3.6-flash", progress_callback=None) -> list:
    """
    Runs analyze_report() for a list of report strings, one at a time.

    progress_callback (optional): a function that accepts a float between
    0 and 1, used to update a Streamlit progress bar.

    Returns a list of dictionaries (same shape as analyze_report's output),
    in the same order as the input list.
    """
    results = []
    total = len(reports) if reports else 1
    for i, text in enumerate(reports):
        result = analyze_report(client, text, model=model)
        results.append(result)
        if progress_callback:
            progress_callback((i + 1) / total)
    return results


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


def ask_chatbot(client, question: str, dataset_summary: str, model: str = "gemini-3.6-flash") -> str:
    """
    Sends the user's question, together with the summarized dataset, to the
    AI and returns a plain-text answer.

    The AI is explicitly told to only use the provided summary and to say
    so if it cannot answer - this prevents the chatbot from making up facts.
    """
    if client is None:
        return "⚠️ AI chatbot is unavailable because no valid Gemini API key was found."

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
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Could not get an answer from the AI right now. ({e})"
