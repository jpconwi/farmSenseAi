"""
app.py
------
🌾 FarmSense AI — Farmer Report Analyzer

This is the main Streamlit application file.
It is organized into clearly labeled sections so a beginner can follow along:

    1.  Page setup
    2.  Choose AI provider (Hugging Face or Ollama) + create client
    3.  Load and clean the dataset (Pandas)
    4.  Sidebar filters
    5.  Dashboard overview (metric cards)
    6.  Charts (Reports by Crop / Location / Category)
    7.  Filtered data table
    8.  AI Report Analyzer (analyze a NEW report typed by the user)
    9.  Dataset chatbot ("Ask FarmSense AI")

Run this file with:  streamlit run app.py
python -m streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
import plotly.express as px

# Streamlit Cloud can keep an OLD copy of utils/ai_analysis.py in memory after
# a code update (causing "cannot import name ..." errors). Reloading it here
# guarantees the app always uses the newest version of that file.
import importlib
import utils.ai_analysis as _ai_module
importlib.reload(_ai_module)

# Our own helper functions live in utils/ai_analysis.py
from utils.ai_analysis import (
    get_client,
    analyze_report,
    analyze_reports_batch,
    build_dataset_summary,
    ask_chatbot,
    CATEGORIES,
    load_disk_cache,
    save_disk_cache,
    _cache_key,
    OllamaClient,
    ollama_list_models,
    analyze_sentiments,
    MODEL_NAME,
    MODEL_CHAIN,
    SENTIMENT_MODEL,
)

# ---------------------------------------------------------------------------
# 1. PAGE SETUP
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FarmSense AI",
    page_icon="🌾",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "farmer_reports.csv")
REQUIRED_COLUMNS = ["date", "location", "crop", "report"]


# ---------------------------------------------------------------------------
# 2. CHOOSE AI PROVIDER + CREATE CLIENT
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    """
    Reads your Hugging Face API token from Streamlit secrets.

    Beginner note: st.secrets reads from the file .streamlit/secrets.toml
    (when running locally) or from the "Secrets" section of your app's
    settings (when deployed on Streamlit Community Cloud).

    We NEVER hard-code the token in this file - that would be unsafe if you
    ever share your code or push it to GitHub.
    """
    try:
        return str(st.secrets["HF_TOKEN"]).strip()
    except Exception:
        return ""


api_key = load_api_key()
PLACEHOLDER_TOKEN = "hf_your_token_here"

provider = st.sidebar.selectbox(
    "AI Provider", ["Hugging Face (cloud)", "Ollama (local)"],
    help="Ollama runs the AI on YOUR computer: no quota, no token. "
         "It only works when the app runs on the same computer as Ollama.",
)
use_ollama = provider.startswith("Ollama")

if use_ollama:
    ollama_host = st.sidebar.text_input("Ollama address", "http://localhost:11434")
    installed = ollama_list_models(ollama_host)
    if installed:
        client = OllamaClient(ollama_host)
        model_name = st.sidebar.selectbox("Ollama model", installed)
    else:
        client = None
        model_name = "(none)"
        st.sidebar.error(
            "⚠️ Cannot reach Ollama (or no models installed).\n\n"
            "Start it with `ollama serve`, install a model with e.g. "
            "`ollama pull llama3.2:3b`, and run this app on the same computer. "
            "Ollama on your laptop is NOT reachable from Streamlit Cloud."
        )
else:
    if not api_key:
        st.sidebar.error(
            "⚠️ No Hugging Face token found.\n\n"
            "Make sure `.streamlit/secrets.toml` exists inside the SAME folder "
            "as `app.py`, contains a line `HF_TOKEN = \"hf_...\"`, then stop and "
            "restart `streamlit run app.py`. (See README.md.)"
        )
    elif api_key == PLACEHOLDER_TOKEN:
        st.sidebar.error("⚠️ `HF_TOKEN` is still the placeholder. Paste your real token in `secrets.toml`.")
    client = get_client(api_key) if api_key and api_key != PLACEHOLDER_TOKEN else None
    model_name = MODEL_NAME  # first choice; the others are automatic fallbacks
    st.sidebar.caption("🤖 AI models (auto-fallback):\n\n" + "\n".join(
        f"{i}. `{m}`" for i, m in enumerate(MODEL_CHAIN, 1)
    ))
    st.sidebar.caption(f"😊 Sentiment model: `{SENTIMENT_MODEL}`")


if st.sidebar.button("🔄 Retry AI analysis"):
    for _k in [k for k in st.session_state if str(k).startswith("ai_failed")]:
        st.session_state.pop(_k)


# ---------------------------------------------------------------------------
# 3. LOAD AND CLEAN THE DATASET (PANDAS)
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(path: str):
    """
    Loads the CSV file and cleans it step by step.
    Returns (dataframe, list_of_warning_messages).

    We return warnings instead of using st.warning() directly here because
    Streamlit calls inside a @st.cache_data function can behave oddly - it's
    safer to show messages from the main app code.
    """
    warnings = []

    # --- Error handling: missing CSV file ---
    if not os.path.exists(path):
        return None, [f"❌ Could not find the dataset file at `{path}`. "
                       f"Make sure `farmer_reports.csv` is inside the `data/` folder."]

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, [f"❌ Failed to read the CSV file: {e}"]

    # --- Error handling: empty dataset ---
    if df.empty:
        return None, ["❌ The dataset file was found but it is empty."]

    # --- Error handling: missing columns ---
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return None, [f"❌ The CSV is missing required column(s): {missing_cols}. "
                       f"Expected columns: {REQUIRED_COLUMNS}"]

    original_count = len(df)

    # Clean up whitespace in text columns (e.g. "  Tago  " -> "Tago")
    for col in ["location", "crop", "report"]:
        df[col] = df[col].astype(str).str.strip()

    # Remove rows where "report" is empty (this is our main text column)
    df = df[df["report"].str.len() > 0]
    df = df[df["report"].str.lower() != "nan"]

    # Remove duplicate rows (exact duplicates)
    df = df.drop_duplicates()

    # Convert the date column to real datetime values.
    # errors="coerce" turns any unreadable date into NaT (missing) instead
    # of crashing the whole app.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Handle missing values: drop rows where date could not be parsed,
    # since we need valid dates for the date filter and charts.
    before_date_drop = len(df)
    df = df.dropna(subset=["date"])
    dropped_for_date = before_date_drop - len(df)

    # Fill any remaining missing location/crop values with "Unknown"
    # instead of dropping the row (we still want to analyze the report text).
    df["location"] = df["location"].replace("", "Unknown").fillna("Unknown")
    df["crop"] = df["crop"].replace("", "Unknown").fillna("Unknown")

    df = df.reset_index(drop=True)

    cleaned_count = len(df)
    removed = original_count - cleaned_count
    if removed > 0:
        warnings.append(
            f"ℹ️ Cleaned dataset: removed {removed} row(s) "
            f"(empty reports, duplicates, or unreadable dates: {dropped_for_date})."
        )

    return df, warnings


raw_df, load_warnings = load_and_clean_data(DATA_PATH)

for msg in load_warnings:
    if msg.startswith("❌"):
        st.error(msg)
    else:
        st.info(msg)

# If loading failed completely, stop here so the rest of the app doesn't crash.
if raw_df is None:
    st.stop()


# ---------------------------------------------------------------------------
# 4. SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")

crop_options = sorted(raw_df["crop"].unique().tolist())
selected_crops = st.sidebar.multiselect("Crop", crop_options, default=crop_options)

location_options = sorted(raw_df["location"].unique().tolist())
selected_locations = st.sidebar.multiselect("Location", location_options, default=location_options)

min_date = raw_df["date"].min().date()
max_date = raw_df["date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Apply the filters chosen in the sidebar
filtered_df = raw_df[
    raw_df["crop"].isin(selected_crops)
    & raw_df["location"].isin(selected_locations)
]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df["date"] >= start_date) & (filtered_df["date"] <= end_date)]


# ---------------------------------------------------------------------------
# ANALYZE THE FILTERED REPORTS WITH AI (classification)
# ---------------------------------------------------------------------------
st.title("🌾 FarmSense AI — Farmer Report Analyzer")
st.caption(
    "GenAI-powered dashboard that classifies farmer reports into problem "
    "categories, shows trends, and answers questions about the dataset."
)

# We only classify reports if we have a working AI client.
# We NEVER make up fake category/severity values - if the AI is unavailable,
# we simply show the raw reports without a category column.
analyzed_df = filtered_df.copy()

if filtered_df.empty:
    st.warning("No reports match the current filters. Try adjusting the sidebar filters.")
    st.stop()

# SPEED DESIGN
#  * We analyze ALL cleaned reports ONCE (not just the filtered ones), so moving
#    a filter never triggers a new AI call - it only slices results we have.
#  * We send only UNIQUE report texts (many reports repeat the same sentence),
#    in as few API calls as possible.
#  * Results are cached in memory and on disk; each call has a timeout, and if
#    the AI fails we show "Not analyzed" instead of waiting.
if "ai_cache" not in st.session_state:
    st.session_state["ai_cache"] = load_disk_cache()
ai_cache = st.session_state["ai_cache"]

unique_reports = list(dict.fromkeys(raw_df["report"].tolist()))

# Models to try, in order. With Hugging Face, if one model fails we
# automatically move on to the next; Ollama uses just the model you picked.
model_order = [model_name] if use_ollama else list(MODEL_CHAIN)


def _cached_under(m: str) -> bool:
    return any(_cache_key(r, m) in ai_cache for r in unique_reports)


todo = [r for r in unique_reports if not any(_cache_key(r, m) in ai_cache for m in model_order)]
ai_error = None
working_model = next((m for m in model_order if _cached_under(m)), model_name)

if client is None:
    st.warning(
        "⚠️ Ollama is not reachable - reports will not be classified. " if use_ollama else
        "⚠️ No valid Hugging Face token - reports will not be classified. Add your token (see README.md)."
    )
elif todo:
    tried_errors = {}
    with st.status(f"Analyzing {len(todo)} unique report(s) with AI...", expanded=True) as status:
        for m in model_order:
            if st.session_state.get(f"ai_failed_{m}"):
                tried_errors[m] = st.session_state[f"ai_failed_{m}"]
                continue
            status.update(label=f"Analyzing {len(todo)} unique report(s) with {m}...")
            progress_bar = st.progress(0.0)
            new_results = analyze_reports_batch(
                client, todo, model=m, progress_callback=progress_bar.progress,
            )
            ok = 0
            err = None
            for text, res in zip(todo, new_results):
                if "error" not in res:
                    ai_cache[_cache_key(text, m)] = res
                    ok += 1
                else:
                    err = res["error"]
            if ok:
                save_disk_cache(ai_cache)
                working_model = m
                note = "" if m == model_order[0] else f" (auto-switched from {model_order[0]})"
                status.update(label=f"AI analysis done with {m}{note}", state="complete")
                break
            st.session_state[f"ai_failed_{m}"] = err
            tried_errors[m] = err
        else:
            status.update(label="AI analysis failed on every model", state="error")
            ai_error = " | ".join(f"{m}: {str(e)[:200]}" for m, e in tried_errors.items())
if ai_error:
    st.error(
        f"❌ AI analysis failed on every model.\n\n{ai_error}\n\n"
        "Check your Hugging Face token and credits (or that Ollama is running), "
        "then use **Retry AI analysis** in the sidebar."
    )
    if "not supported by any provider" in ai_error:
        st.info(
            "💡 This model isn't served by any Hugging Face Inference Provider. "
            "Change `MODEL_NAME` at the top of `utils/ai_analysis.py` to a model that is, "
            "e.g. `Qwen/Qwen3-8B` or `meta-llama/Llama-3.1-8B-Instruct`."
        )


def _result_for(text: str):
    """Return the cached AI result for a report, or a 'Not analyzed' placeholder."""
    for m in model_order:
        key = _cache_key(text, m)
        if key in ai_cache:
            return ai_cache[key]
    return {"category": "Not analyzed", "severity": "—", "keywords": "", "summary": ""}


results = [_result_for(t) for t in analyzed_df["report"]]
analyzed_df["category"] = [r.get("category", "Other") for r in results]
analyzed_df["severity"] = [r.get("severity", "Low") for r in results]
analyzed_df["keywords"] = [r.get("keywords", "") for r in results]
analyzed_df["summary"] = [r.get("summary", "") for r in results]

n_unanalyzed = int((analyzed_df["category"] == "Not analyzed").sum())
if n_unanalyzed:
    st.caption(f"ℹ️ {n_unanalyzed} of {len(analyzed_df)} report(s) have not been analyzed by AI yet.")

# --- Sentiment (cardiffnlp model; works with the Hugging Face provider only) ---
if client is not None and not use_ollama:
    todo_sent = [r for r in unique_reports if _cache_key(r, SENTIMENT_MODEL) not in ai_cache]
    if todo_sent and not st.session_state.get("ai_failed_sentiment"):
        with st.spinner("Analyzing sentiment..."):
            labels, sent_err = analyze_sentiments(client, todo_sent)
        if sent_err:
            st.session_state["ai_failed_sentiment"] = sent_err
        else:
            for text, label in zip(todo_sent, labels):
                ai_cache[_cache_key(text, SENTIMENT_MODEL)] = label
            save_disk_cache(ai_cache)
    if st.session_state.get("ai_failed_sentiment"):
        st.caption(f"⚠️ Sentiment unavailable: {st.session_state['ai_failed_sentiment'][:200]}")

analyzed_df["sentiment"] = [ai_cache.get(_cache_key(t, SENTIMENT_MODEL), "—") for t in analyzed_df["report"]]


# ---------------------------------------------------------------------------
# 5. DASHBOARD OVERVIEW — METRIC CARDS
# ---------------------------------------------------------------------------
st.header("📊 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reports", len(analyzed_df))
col2.metric("Number of Crops", analyzed_df["crop"].nunique())
col3.metric("Number of Locations", analyzed_df["location"].nunique())

_analyzed_only = analyzed_df[analyzed_df["category"] != "Not analyzed"]
most_common = _analyzed_only["category"].mode()
most_common_label = most_common.iloc[0] if not most_common.empty else "N/A"
col4.metric("Most Common Problem", most_common_label)

st.markdown("---")


# ---------------------------------------------------------------------------
# 6. CHARTS
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("🌱 Reports by Crop")
    crop_counts = analyzed_df["crop"].value_counts().reset_index()
    crop_counts.columns = ["crop", "count"]
    fig_crop = px.bar(crop_counts, x="crop", y="count", color="crop", title="Number of Reports per Crop")
    st.plotly_chart(fig_crop, use_container_width=True)

with chart_col2:
    st.subheader("📍 Reports by Location")
    loc_counts = analyzed_df["location"].value_counts().reset_index()
    loc_counts.columns = ["location", "count"]
    fig_loc = px.bar(loc_counts, x="location", y="count", color="location", title="Number of Reports per Location")
    st.plotly_chart(fig_loc, use_container_width=True)

st.subheader("🐛 Problem Categories")
if (analyzed_df["category"] != "Not analyzed").any():
    cat_df = analyzed_df[analyzed_df["category"] != "Not analyzed"]
    cat_counts = cat_df["category"].value_counts().reindex(CATEGORIES).fillna(0).reset_index()
    cat_counts.columns = ["category", "count"]
    fig_cat = px.pie(cat_counts, names="category", values="count", title="Reports by Problem Category")
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("Problem category chart is unavailable because AI analysis has not run (no API key).")

st.subheader("😊 Report Sentiment")
_sent = analyzed_df[analyzed_df["sentiment"] != "—"]
if not _sent.empty:
    sent_counts = _sent["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["sentiment", "count"]
    fig_sent = px.bar(sent_counts, x="sentiment", y="count", color="sentiment", title="Reports by Sentiment")
    st.plotly_chart(fig_sent, use_container_width=True)
else:
    st.info("Sentiment chart is unavailable (needs the Hugging Face provider and a working token).")

st.markdown("---")


# ---------------------------------------------------------------------------
# 7. FILTERED DATA TABLE
# ---------------------------------------------------------------------------
st.header("📋 Farmer Reports")
with st.expander("View cleaned & filtered dataset", expanded=True):
    st.dataframe(
        analyzed_df[["date", "location", "crop", "report", "category", "severity", "keywords", "summary", "sentiment"]],
        use_container_width=True,
        hide_index=True,
    )

with st.expander("📚 Dataset source and citation"):
    st.markdown(
        "**Source:** This is a *synthetic* (computer-generated) sample dataset. "
        "It does not contain real farmer reports. It was created by the author for "
        "CS 315 Activity 3 using the script `gen_data.py` (random seed 7), and "
        "stored in `data/farmer_reports.csv`. The generator deliberately adds a few "
        "messy rows (an empty report, a duplicate, a missing date, extra spaces) so "
        "the data-cleaning step has something to fix."
    )
    st.markdown("**Citation (APA):**")
    st.markdown(
        "Conwi, J. P. (2026). *FarmSense AI farmer reports* [Synthetic dataset]. "
        "Generated with gen_data.py (random seed 7) for CS 315 Activity 3, "
        "North Eastern Mindanao State University."
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# 8. AI REPORT ANALYZER — analyze a brand-new report typed by the user
# ---------------------------------------------------------------------------
st.header("🧠 AI Report Analyzer")
st.caption("Type in a new farmer report and let AI classify it.")

new_report = st.text_area(
    "Enter a farmer report",
    placeholder="Example: The rice leaves have yellow spots and several plants are dying.",
    height=100,
)

def _with_fallback(call):
    """Try the working model first, then the other models, until one works."""
    order = [working_model] + [m for m in model_order if m != working_model]
    out = None
    for m in order:
        out = call(m)
        failed = (isinstance(out, dict) and "error" in out) or (isinstance(out, str) and out.startswith("⚠️"))
        if not failed:
            return out
    return out


if st.button("🔎 Analyze Report"):
    if not new_report.strip():
        st.warning("Please type a report before clicking Analyze.")
    elif client is None:
        st.error("❌ Cannot analyze: no valid Hugging Face token found. See README.md to set it up.")
    else:
        with st.spinner("Analyzing..."):
            result = _with_fallback(lambda m: analyze_report(client, new_report, model=m))

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            st.success("Analysis complete!")
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Category", result["category"])
            r_col2.metric("Severity", result["severity"])
            r_col3.metric("Keywords", result["keywords"] or "—")
            st.markdown(f"**Summary:** {result['summary']}")

st.markdown("---")


# ---------------------------------------------------------------------------
# 9. DATASET CHATBOT — "Ask FarmSense AI"
# ---------------------------------------------------------------------------
st.header("💬 Ask FarmSense AI")
st.caption(
    "Ask a question about the filtered dataset above, e.g. "
    "\"Which crop has the most reports?\" or \"Which location has the most pest problems?\""
)

# We build a short SUMMARY of the dataset (not the full raw data) to send to
# the AI, so we don't waste tokens / API cost. See build_dataset_summary().
dataset_summary = build_dataset_summary(analyzed_df)

with st.expander("See what data summary is sent to the AI (for transparency)"):
    st.code(dataset_summary)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

user_question = st.chat_input("Ask a question about the farmer reports...")

if user_question:
    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        if client is None:
            answer = "⚠️ Chatbot is unavailable because no valid Hugging Face token was found. See README.md."
        else:
            with st.spinner("Thinking..."):
                answer = _with_fallback(lambda m: ask_chatbot(client, user_question, dataset_summary, model=m))
        st.write(answer)

    st.session_state.chat_history.append(("assistant", answer))

st.markdown("---")
st.caption("Built for CS 315 – Application Development and Emerging Technologies · Activity 3 · FarmSense AI")
