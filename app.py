"""
app.py
------
🌾 FarmSense AI — Farmer Report Analyzer

This is the main Streamlit application file.
It is organized into clearly labeled sections so a beginner can follow along:

    1.  Page setup
    2.  Load API key + create Gemini client
    3.  Load and clean the dataset (Pandas)
    4.  Sidebar filters
    5.  Dashboard overview (metric cards)
    6.  Charts (Reports by Crop / Location / Category)
    7.  Filtered data table
    8.  AI Report Analyzer (analyze a NEW report typed by the user)
    9.  Dataset chatbot ("Ask FarmSense AI")

Run this file with:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
import plotly.express as px

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
# 2. LOAD API KEY + CREATE GEMINI CLIENT
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    """
    Reads the Gemini API key from Streamlit secrets.

    Beginner note: st.secrets reads from the file .streamlit/secrets.toml
    (when running locally) or from the "Secrets" section of your app's
    settings (when deployed on Streamlit Community Cloud).

    We NEVER hard-code the key in this file - that would be unsafe if you
    ever share your code or push it to GitHub.
    """
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


def load_default_model() -> str:
    """
    Reads the default Gemini model name from secrets (GEMINI_MODEL), if you
    set one there. Falls back to "gemini-3.6-flash" if it isn't set.
    """
    try:
        return st.secrets["GEMINI_MODEL"]
    except Exception:
        return "gemini-3.6-flash"


api_key = load_api_key()
api_key_missing = not api_key

if api_key_missing:
    st.sidebar.error(
        "⚠️ No Gemini API key found.\n\n"
        "AI features (report analysis + chatbot) will not work until you add "
        "your key to `.streamlit/secrets.toml` (see README.md)."
    )

client = get_client(api_key)

default_model = load_default_model()
model_options = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash"]
if default_model not in model_options:
    model_options.insert(0, default_model)  # make sure your secrets.toml choice is always selectable

model_name = st.sidebar.selectbox(
    "AI Model", model_options, index=model_options.index(default_model),
    help="gemini-3.6-flash is fast and inexpensive, good for this project.",
)


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

if client is not None:
    # Two layers of caching for AI results, keyed by (report text, model):
    #   1. In-memory (st.session_state) - fast, but lost on app restart.
    #   2. On-disk (data/ai_cache.json) - survives restarts, so once your
    #      45 reports have been analyzed once, they never need to be sent
    #      to the AI again unless the report text itself changes.
    # Only SUCCESSFUL results are ever cached - a report that errored out
    # (e.g. a rate limit) is deliberately left uncached so it's retried on
    # the next rerun instead of being stuck as "failed" forever.
    if "ai_cache" not in st.session_state:
        st.session_state["ai_cache"] = load_disk_cache()
    ai_cache = st.session_state["ai_cache"]

    all_reports = filtered_df["report"].tolist()
    uncached_reports = [r for r in all_reports if _cache_key(r, model_name) not in ai_cache]

    fresh_results = {}
    if uncached_reports:
        with st.spinner(
            f"Analyzing {len(uncached_reports)} new farmer report(s) with AI... "
            "this may take a moment."
        ):
            progress_bar = st.progress(0.0)

            def update_progress(pct):
                progress_bar.progress(pct)

            new_results = analyze_reports_batch(
                client, uncached_reports, model=model_name,
                progress_callback=update_progress,
            )
            progress_bar.empty()

        newly_cached = False
        for report_text, result in zip(uncached_reports, new_results):
            if "error" not in result:
                ai_cache[_cache_key(report_text, model_name)] = result
                newly_cached = True
            else:
                fresh_results[report_text] = result

        if newly_cached:
            save_disk_cache(ai_cache)

    results = [
        ai_cache[_cache_key(r, model_name)] if _cache_key(r, model_name) in ai_cache
        else fresh_results.get(r, {"error": "No result returned for this report."})
        for r in all_reports
    ]

    # Check how many analyses failed (e.g. due to a bad API key or network issue)
    error_count = sum(1 for r in results if "error" in r)

    if error_count == len(results):
        first_error = next((r["error"] for r in results if "error" in r), "Unknown error.")
        st.error(
            "❌ AI analysis failed for all reports. The dashboard will show "
            "the raw reports without AI categories.\n\n"
            f"**Underlying error:** {first_error}"
        )
        analyzed_df["category"] = "Not analyzed"
        analyzed_df["severity"] = "Not analyzed"
        analyzed_df["keywords"] = ""
        analyzed_df["summary"] = ""
    else:
        if error_count > 0:
            st.warning(f"⚠️ {error_count} report(s) could not be analyzed by AI and are marked 'Not analyzed'.")
        analyzed_df["category"] = [r.get("category", "Not analyzed") if "error" not in r else "Not analyzed" for r in results]
        analyzed_df["severity"] = [r.get("severity", "Not analyzed") if "error" not in r else "Not analyzed" for r in results]
        analyzed_df["keywords"] = [r.get("keywords", "") if "error" not in r else "" for r in results]
        analyzed_df["summary"] = [r.get("summary", "") if "error" not in r else "" for r in results]
else:
    st.warning(
        "⚠️ AI features are disabled because no Gemini API key was found. "
        "You can still browse and filter the raw dataset below. "
        "See README.md to add your API key."
    )
    analyzed_df["category"] = "Not analyzed"
    analyzed_df["severity"] = "Not analyzed"
    analyzed_df["keywords"] = ""
    analyzed_df["summary"] = ""


# ---------------------------------------------------------------------------
# 5. DASHBOARD OVERVIEW — METRIC CARDS
# ---------------------------------------------------------------------------
st.header("📊 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reports", len(analyzed_df))
col2.metric("Number of Crops", analyzed_df["crop"].nunique())
col3.metric("Number of Locations", analyzed_df["location"].nunique())

if (analyzed_df["category"] != "Not analyzed").any():
    most_common = analyzed_df[analyzed_df["category"] != "Not analyzed"]["category"].mode()
    most_common_label = most_common.iloc[0] if not most_common.empty else "N/A"
else:
    most_common_label = "N/A (AI disabled)"
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

st.markdown("---")


# ---------------------------------------------------------------------------
# 7. FILTERED DATA TABLE
# ---------------------------------------------------------------------------
st.header("📋 Farmer Reports")
with st.expander("View cleaned & filtered dataset", expanded=True):
    st.dataframe(
        analyzed_df[["date", "location", "crop", "report", "category", "severity", "keywords", "summary"]],
        use_container_width=True,
        hide_index=True,
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

if st.button("🔎 Analyze Report"):
    if not new_report.strip():
        st.warning("Please type a report before clicking Analyze.")
    elif client is None:
        st.error("❌ Cannot analyze: no valid Gemini API key found. See README.md to set it up.")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_report(client, new_report, model=model_name)

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
            answer = "⚠️ Chatbot is unavailable because no valid Gemini API key was found. See README.md."
        else:
            with st.spinner("Thinking..."):
                answer = ask_chatbot(client, user_question, dataset_summary, model=model_name)
        st.write(answer)

    st.session_state.chat_history.append(("assistant", answer))

st.markdown("---")
st.caption("Built for CS 315 – Application Development and Emerging Technologies · Activity 3 · FarmSense AI")
