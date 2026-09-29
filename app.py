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

import csv
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
from utils.chat_engine import answer_from_data, build_chat_context
from utils.knowledge import add_diagnosis, diagnose, entry_by_key, KB, TYPE_COLORS, DISCLAIMER
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
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
        # keep only the first 4 fields of every row (ignores stray extra columns)
        df = pd.DataFrame([r[:4] + [""] * (4 - len(r[:4])) for r in rows[1:]], columns=rows[0][:4])
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
    df["date"] = pd.to_datetime(df["date"].replace("", None), errors="coerce")

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

# --- Sentiment (cardiffnlp model, with chat models as backup; Hugging Face only) ---
if client is not None and not use_ollama:
    todo_sent = [r for r in unique_reports if _cache_key(r, SENTIMENT_MODEL) not in ai_cache]
    if todo_sent and not st.session_state.get("ai_failed_sentiment"):
        with st.spinner("Analyzing sentiment..."):
            labels, sent_err, sent_src = analyze_sentiments(client, todo_sent, fallback_models=MODEL_CHAIN)
        if sent_err:
            st.session_state["ai_failed_sentiment"] = sent_err
        else:
            for text, label in zip(todo_sent, labels):
                ai_cache[_cache_key(text, SENTIMENT_MODEL)] = label
            save_disk_cache(ai_cache)
            if sent_src != SENTIMENT_MODEL:
                st.session_state["sentiment_note"] = f"Sentiment labeled by backup model `{sent_src}` ({SENTIMENT_MODEL} was unavailable)."
    if st.session_state.get("ai_failed_sentiment"):
        st.caption(f"⚠️ Sentiment unavailable: {st.session_state['ai_failed_sentiment'][:300]}")
    elif st.session_state.get("sentiment_note"):
        st.caption(f"ℹ️ {st.session_state['sentiment_note']}")

analyzed_df["sentiment"] = [ai_cache.get(_cache_key(t, SENTIMENT_MODEL), "—") for t in analyzed_df["report"]]
analyzed_df = add_diagnosis(analyzed_df)  # specific likely disease/pest for each report (knowledge base)


# ---------------------------------------------------------------------------
# 5. DASHBOARD — KEY FINDINGS, KPI CARDS, TABS
# ---------------------------------------------------------------------------
st.header("📊 Dashboard")
SEV_COLORS = {"Low": "#43A047", "Moderate": "#FB8C00", "High": "#E53935"}
SENT_COLORS = {"positive": "#43A047", "neutral": "#9E9E9E", "negative": "#E53935"}
n_total = len(analyzed_df)
A = analyzed_df[analyzed_df["category"] != "Not analyzed"]
ai_ok = not A.empty


def pct(x, d):
    return f"{x / d:.0%}" if d else "—"


crop_vc, loc_vc = analyzed_df["crop"].value_counts(), analyzed_df["location"].value_counts()
n_high = int((A["severity"] == "High").sum()) if ai_ok else 0
top_cat = A["category"].value_counts().idxmax() if ai_ok else "N/A"
n_neg = int((analyzed_df["sentiment"] == "negative").sum())
weekly = analyzed_df.set_index("date").resample("W").size()

findings = [
    f"📅 **{n_total} reports** from **{analyzed_df['date'].min():%b %d}** to **{analyzed_df['date'].max():%b %d, %Y}**.",
    f"🌱 **{crop_vc.index[0]}** is reported most often ({crop_vc.iloc[0]} reports, {pct(crop_vc.iloc[0], n_total)}).",
    f"📍 **{loc_vc.index[0]}** is the busiest location ({loc_vc.iloc[0]} reports).",
    f"📈 The busiest week ended **{weekly.idxmax():%b %d}** with {int(weekly.max())} reports.",
]
if ai_ok:
    worst = A[A["severity"] == "High"]["crop"].value_counts()
    findings.insert(2, f"🐛 The most common problem is **{top_cat}** ({pct(int((A['category'] == top_cat).sum()), len(A))} of analyzed reports).")
    findings.append(f"🚨 **{n_high} high-severity report(s)**" + (f" — most for **{worst.index[0]}** ({worst.iloc[0]})." if n_high else "."))
st.info("**Key findings (updates with your filters)**\n\n" + "\n\n".join(findings))

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Reports", n_total, help="Cleaned reports after your sidebar filters.")
k2.metric("High Severity", n_high if ai_ok else "N/A", pct(n_high, len(A)) + " of analyzed" if ai_ok else None, delta_color="off",
          help="Reports the AI rated High: they need attention first.")
k3.metric("Top Problem", top_cat, help="The most frequent problem category.")
k4.metric("Hotspot Location", loc_vc.index[0], f"{loc_vc.iloc[0]} reports", delta_color="off", help="Location with the most reports.")
k5.metric("Negative Tone", pct(n_neg, n_total) if (analyzed_df["sentiment"] != "—").any() else "N/A",
          help="Share of reports whose wording sounds negative/urgent.")
if not ai_ok:
    st.warning("AI classification is not available yet, so category and severity charts are hidden. Crop, location and trend charts still work.")

t_over, t_trend, t_prob, t_dx, t_prio, t_data = st.tabs(["📊 Overview", "📈 Trends", "🐛 Problems & Severity", "🩺 Diseases & Pests", "🚨 Priority Reports", "📋 Data"])

with t_over:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌱 Reports by Crop")
        st.caption("Taller bar = more reports about that crop.")
        d = crop_vc.reset_index(); d.columns = ["crop", "reports"]
        st.plotly_chart(px.bar(d, x="crop", y="reports", color="crop", text="reports").update_layout(showlegend=False), use_container_width=True)
    with c2:
        st.subheader("📍 Reports by Location")
        st.caption("Longer bar = more reports from that town.")
        d = loc_vc.reset_index(); d.columns = ["location", "reports"]
        st.plotly_chart(px.bar(d.sort_values("reports"), y="location", x="reports", orientation="h", color="location", text="reports").update_layout(showlegend=False), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🐛 Problem Categories")
        st.caption("Each slice is a type of problem; bigger slice = more common.")
        if ai_ok:
            st.plotly_chart(px.pie(A["category"].value_counts().reset_index(), names="category", values="count", hole=0.45), use_container_width=True)
            with st.expander("📖 What does this chart mean?"):
                st.markdown(
                    "This donut chart groups every farmer report by the **type of problem** the AI found. "
                    "The percentage on a slice is that type's share of all reports shown.\n\n"
                    "- **Disease**: illness caused by fungi, bacteria or viruses (e.g. leaf spot, wilt, bunchy top).\n"
                    "- **Pest**: insects or animals that damage the crop (e.g. aphids, worms, rats, beetles).\n"
                    "- **Nutrient**: the plant lacks food in the soil (e.g. pale leaves, slow growth, little fertilizer).\n"
                    "- **Weather**: damage from typhoon, strong wind or heavy rain.\n"
                    "- **Water**: too little water (drought) or too much water (poor drainage).\n"
                    "- **Other**: reports that do not clearly fit the types above.\n\n"
                    "**How to use it:** the biggest slice is the problem type to focus on first.")
        else:
            st.info("Needs AI analysis.")
    with c4:
        st.subheader("😊 Report Sentiment")
        st.caption("How the farmer's wording sounds: positive, neutral or negative.")
        S = analyzed_df[analyzed_df["sentiment"] != "—"]
        if not S.empty:
            d = S["sentiment"].value_counts().reset_index()
            st.plotly_chart(px.bar(d, x="sentiment", y="count", color="sentiment", color_discrete_map=SENT_COLORS, text="count").update_layout(showlegend=False), use_container_width=True)
            with st.expander("📖 What does this chart mean?"):
                st.markdown(
                    "**Sentiment** is the *tone* of the farmer's words, not how bad the problem is. "
                    "Each bar counts how many reports have that tone (**count**).\n\n"
                    "- **Negative**: the farmer describes damage, loss or worry (e.g. \"plants are dying\").\n"
                    "- **Neutral**: a plain description with no strong feeling (e.g. \"aphids are on the leaves\").\n"
                    "- **Positive**: good news such as recovery or a good harvest.\n\n"
                    "**How to use it:** a tall *negative* bar means many farmers are worried or urgent. "
                    "Reports are usually negative or neutral because farmers mostly write when something goes wrong.")
        else:
            st.info("Needs the Hugging Face provider and a working token.")

with t_trend:
    st.subheader("📈 Reports per Week")
    st.caption("Shows when problems spike. Rising line = more farmers reporting problems.")
    w = weekly.reset_index(); w.columns = ["week", "reports"]
    st.plotly_chart(px.line(w, x="week", y="reports", markers=True), use_container_width=True)
    if ai_ok:
        st.subheader("Weekly Reports by Problem Category")
        st.caption("Colors show which problem types drive each week's total.")
        wc = A.groupby([pd.Grouper(key="date", freq="W"), "category"]).size().reset_index(name="reports")
        st.plotly_chart(px.bar(wc, x="date", y="reports", color="category"), use_container_width=True)

with t_prob:
    if ai_ok:
        st.subheader("Which crops suffer from which problems?")
        st.caption("Darker cell = more reports. Read across a crop's row to see its main problem.")
        ct = pd.crosstab(A["crop"], A["category"])
        st.plotly_chart(px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd"), use_container_width=True)
        st.subheader("Where are the problems?")
        st.caption("Same idea by location: which town has which type of problem.")
        cl = pd.crosstab(A["location"], A["category"])
        st.plotly_chart(px.imshow(cl, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd"), use_container_width=True)
        st.subheader("Severity by Crop")
        st.caption("Red = High severity (act first), orange = Moderate, green = Low.")
        sv = A.groupby(["crop", "severity"]).size().reset_index(name="reports")
        st.plotly_chart(px.bar(sv, x="crop", y="reports", color="severity", color_discrete_map=SEV_COLORS,
                               category_orders={"severity": ["Low", "Moderate", "High"]}), use_container_width=True)
        with st.expander("📖 What does severity mean?"):
            st.markdown("**Severity** is how serious the problem sounds, rated by the AI from the report: "
                        "**High** = act first (crop loss is likely or spreading), "
                        "**Moderate** = needs attention soon, **Low** = minor or early stage.")
    else:
        st.info("These charts need AI analysis (category and severity).")

with t_dx:
    st.subheader("🩺 What kind of problem is it? (specific diseases, pests and stresses)")
    st.caption("Each report is matched to the most likely specific problem, with causes and treatment. "
               "This is general agronomy guidance based on the farmer's words, so confirm with a technician.")
    dxs = analyzed_df.groupby(["likely_issue", "issue_type"]).size().reset_index(name="reports")
    d1, d2 = st.columns([3, 2])
    with d1:
        st.plotly_chart(px.bar(dxs.sort_values("reports"), y="likely_issue", x="reports", color="issue_type", orientation="h",
                               text="reports", color_discrete_map=TYPE_COLORS, labels={"likely_issue": "", "issue_type": "Type"},
                               title="Reports by specific problem"), use_container_width=True)
    with d2:
        tv = analyzed_df["issue_type"].value_counts().reset_index()
        st.plotly_chart(px.pie(tv, names="issue_type", values="count", hole=0.45, color="issue_type",
                               color_discrete_map=TYPE_COLORS, title="Disease vs pest vs weather ..."), use_container_width=True)
    st.subheader("Where is each problem happening?")
    st.caption("Darker cell = more reports of that problem in that location.")
    hm = pd.crosstab(analyzed_df["likely_issue"], analyzed_df["location"])
    st.plotly_chart(px.imshow(hm, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd"), use_container_width=True)
    st.subheader("📖 Problem guide (click a problem for causes and treatment)")
    for _k, _n in analyzed_df.groupby("issue_key").size().sort_values(ascending=False).items():
        _e = entry_by_key(_k)
        _sub = analyzed_df[analyzed_df["issue_key"] == _k]
        with st.expander(f"{_e['name']}  ·  {_e['type']}  ·  {_n} report(s)"):
            st.markdown(f"**In your data:** crops: {', '.join(sorted(_sub['crop'].unique()))} · "
                        f"locations: {', '.join(sorted(_sub['location'].unique()))}")
            st.markdown(f"**What it is:** {_e['agent']}\n\n**Symptoms:** {_e['symptoms']}\n\n**Causes:** {_e['causes']}\n\n"
                        f"**What to do now:** {_e['treatment']}\n\n**Prevention:** {_e['prevention']}")
    st.caption(DISCLAIMER)

with t_prio:
    st.subheader("🚨 High-Severity Reports (newest first)")
    st.caption("These reports were rated High by the AI. Start here.")
    hi = A[A["severity"] == "High"].sort_values("date", ascending=False) if ai_ok else A
    if hi.empty:
        st.success("No high-severity reports in the current filter." if ai_ok else "Needs AI analysis.")
    else:
        st.dataframe(hi[["date", "location", "crop", "likely_issue", "category", "summary", "report"]], use_container_width=True, hide_index=True)

with t_data:
    st.subheader("📋 Cleaned & Filtered Reports")
    cols = ["date", "location", "crop", "report", "likely_issue", "issue_type", "category", "severity", "keywords", "summary", "sentiment"]
    st.dataframe(analyzed_df[cols], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download as CSV", analyzed_df[cols].to_csv(index=False).encode("utf-8"), "farmsense_reports.csv", "text/csv")
    with st.expander("📚 Dataset source and citation"):
        st.markdown(
            "**Source:** *synthetic* (computer-generated) sample data, not real farmer reports, created with `gen_data.py` "
            "and `gen_more_data.py` (random seeds 7, 11 and 21) for CS 315 Activity 3.\n\n**APA:** Conwi, J. P. (2026). *FarmSense AI farmer reports* "
            "[Synthetic dataset]. Generated with gen_data.py and gen_more_data.py (random seeds 7, 11 and 21) for CS 315 Activity 3, North Eastern Mindanao State University."
        )

st.markdown("---")


# ---------------------------------------------------------------------------
# 8. AI REPORT ANALYZER — analyze a brand-new report typed by the user
# ---------------------------------------------------------------------------
st.header("🧠 AI Report Analyzer")
st.caption("Type in a new farmer report and let AI classify it.")

new_crop = st.selectbox("Crop (helps identify the problem)", ["(not sure)"] + sorted(raw_df["crop"].unique().tolist()))
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
            _dx = diagnose(new_crop if new_crop != "(not sure)" else None, new_report, category=result["category"])
            st.markdown(f"#### 🩺 Likely problem: {_dx['name']}  ·  *{_dx['type']}*")
            st.markdown(f"**What it is:** {_dx['agent']}\n\n**Symptoms:** {_dx['symptoms']}\n\n**Causes:** {_dx['causes']}\n\n"
                        f"**What to do now:** {_dx['treatment']}\n\n**Prevention:** {_dx['prevention']}")
            st.caption(DISCLAIMER)

st.markdown("---")


# ---------------------------------------------------------------------------
# 9. DATASET CHATBOT — "Ask FarmSense AI"
# ---------------------------------------------------------------------------
st.header("💬 Ask FarmSense AI")
st.caption(
    "Ask about the filtered data above. Counts, rankings, trends and lists are calculated **exactly** from the data. "
    "Ask about **diseases, causes, treatment or prevention** to get the specific problem and full information."
)
SUGGESTED = ["Which crop has the most reports?", "What kind of diseases are in the reports?", "What are the causes and treatment for the rice disease?",
             "Which location has the most pest problems?", "How many high severity reports are there?", "Show the latest urgent reports"]
_bcols = st.columns(3) + st.columns(3)
for _i, (_c, _q) in enumerate(zip(_bcols, SUGGESTED)):
    if _c.button(_q, use_container_width=True, key=f"sugg_{_i}"):
        st.session_state["pending_q"] = _q

chat_context = build_chat_context(analyzed_df)
with st.expander("See the exact data the chatbot uses (for transparency)"):
    st.code(chat_context)

if "chat_history" not in st.session_state or (st.session_state.chat_history and not isinstance(st.session_state.chat_history[0], dict)):
    st.session_state.chat_history = []
if st.session_state.chat_history and st.button("🗑️ Clear chat"):
    st.session_state.chat_history = []
    st.session_state.pop("last_filters", None)
    st.rerun()
for _n, _m in enumerate(st.session_state.chat_history):
    with st.chat_message(_m["role"]):
        st.markdown(_m["text"])
        if _m.get("source"):
            st.caption(_m["source"])

user_question = st.chat_input("Ask a question about the farmer reports...") or st.session_state.pop("pending_q", None)
if user_question:
    st.session_state.chat_history.append({"role": "user", "text": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
    with st.chat_message("assistant"):
        res = answer_from_data(user_question, analyzed_df, st.session_state.get("last_filters"))
        if res:
            answer, source = res["text"], "🧮 Calculated from the data + agronomy knowledge base (no AI guessing)"
            st.session_state["last_filters"] = res["filters"]
        elif client is None:
            answer, source = "⚠️ This question needs the AI, but no AI provider is available. Add your token (see README.md).", ""
        else:
            past = [m["text"] for m in st.session_state.chat_history[-5:-1] if m["role"] == "user"]
            with st.spinner("Reading the reports..."):
                answer = _with_fallback(lambda m: ask_chatbot(client, user_question, chat_context, model=m, history=past))
            source = "🤖 AI answer based on the report rows above"
        st.markdown(answer)
        if source:
            st.caption(source)
    st.session_state.chat_history.append({"role": "assistant", "text": answer, "source": source})

st.markdown("---")
st.caption("Built for CS 315 – Application Development and Emerging Technologies · Activity 3 · FarmSense AI")
