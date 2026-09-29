"""
utils/chat_engine.py
--------------------
Makes the "Ask FarmSense AI" chatbot accurate, detailed, and visual.

answer_from_data(question, df, last_filters) returns a dict
    {"text": markdown, "fig": plotly figure or None, "filters": dict}
or None when the AI should answer instead.

1. Counts, rankings, percentages, lists and trends are calculated EXACTLY with Pandas.
2. Questions about diseases, causes, treatment or prevention use the knowledge base
   (utils/knowledge.py) to name the specific disease/pest and give full information.
3. Follow-ups such as "what are the causes?" reuse the crop/location from the previous question.
4. Every answer that can be visual comes with a chart.
"""
import re

import pandas as pd
import plotly.express as px

from utils.knowledge import DISCLAIMER, TYPE_COLORS, add_diagnosis, entry_by_key, match_in_text

CATS = ["Pest", "Disease", "Water", "Weather", "Nutrient", "Other"]
SEVS = ["Low", "Moderate", "High"]
SENTS = ["positive", "neutral", "negative"]
NA = ("Not analyzed", "—")
SEV_COLORS = {"Low": "#43A047", "Moderate": "#FB8C00", "High": "#E53935"}
SENT_COLORS = {"positive": "#43A047", "neutral": "#9E9E9E", "negative": "#E53935"}

KNOW_WORDS = (r"cause|why|treat|cure|control|remed|solution|fix|prevent|manage|advice|recommend|"
              r"what disease|which disease|what kind|what type|type of|kind of|what pest|what is wrong|explain|"
              r"what (problems|issues)|information|details?|tell me about|symptom|what should|how (do|to|can)|about the|full info")


def _has(q, *words):
    return any(re.search(rf"\b{w}", q) for w in words)


def _stem(name):
    return r"\b" + re.escape(name.lower().rstrip("s")) + r"s?\b"


def _detect_filters(q, df):
    f = {}
    for col in ("crop", "location"):
        hit = [v for v in df[col].unique() if re.search(_stem(v), q)]
        if hit:
            f[col] = hit
    cats = [c for c in CATS if re.search(_stem(c), q)]
    if _has(q, "insect", "worm", "aphid", "beetle", "caterpillar"):
        cats.append("Pest")
    if _has(q, "fertilizer", "nutrient"):
        cats.append("Nutrient")
    if cats:
        f["category"] = sorted(set(cats))
    sev = [s for s in SEVS if re.search(rf"\b{s.lower()}\b", q)]
    if _has(q, "urgent", "serious", "severe", "critical"):
        sev.append("High")
    if sev:
        f["severity"] = sorted(set(sev))
    sent = [s for s in SENTS if re.search(rf"\b{s}\b", q)]
    if sent:
        f["sentiment"] = sent
    return f


def _detect_dim(q):
    if _has(q, "crop"):
        return "crop"
    if _has(q, "location", "town", "place", "area", "barangay"):
        return "location"
    if _has(q, "categor", "problem", "issue", "type"):
        return "category"
    if _has(q, "severity"):
        return "severity"
    if _has(q, "sentiment"):
        return "sentiment"
    return None


def _describe(f):
    parts = [f"{'/'.join(v)} {k}" for k, v in f.items() if k != "issue_key"]
    return " · ".join(parts) if parts else "all reports"


def _line(r):
    sev = f" · **{r.severity}**" if r.severity not in NA else ""
    return f"- {r.date:%Y-%m-%d} · {r.location} · {r.crop}{sev} — {r.report} → *{r.likely_issue}*"


# ---------------------------------------------------------------- charts
def _count_chart(counts, dim, title):
    d = counts.reset_index()
    d.columns = [dim, "reports"]
    if dim in ("category", "sentiment", "severity") and len(d) <= 8:
        cmap = {"category": TYPE_COLORS, "sentiment": SENT_COLORS, "severity": SEV_COLORS}[dim]
        fig = px.pie(d, names=dim, values="reports", hole=0.45, title=title, color=dim, color_discrete_map=cmap)
        fig.update_traces(textinfo="label+value")
    else:
        fig = px.bar(d.sort_values("reports"), y=dim, x="reports", orientation="h", text="reports",
                     color=dim, title=title).update_layout(showlegend=False)
    return fig


def _issue_chart(sub, title):
    d = sub.groupby(["likely_issue", "issue_type"]).size().reset_index(name="reports")
    fig = px.bar(d.sort_values("reports"), y="likely_issue", x="reports", color="issue_type", orientation="h",
                 text="reports", color_discrete_map=TYPE_COLORS, title=title,
                 labels={"likely_issue": "", "issue_type": "Type"})
    return fig.update_layout(yaxis_title=None, legend_title_text="Type")


def _trend_chart(sub, title):
    w = sub.set_index("date").resample("W").size().reset_index(name="reports")
    fig = px.line(w, x="date", y="reports", markers=True, title=title)
    return fig, w


def quick_chart(question, df):
    if not CHARTS_ENABLED:
        return None
    """A relevant chart for questions the AI answers (subset by any crop/location mentioned)."""
    df = add_diagnosis(df)
    q = question.lower()
    f = _detect_filters(q, df)
    sub = df
    for k in ("crop", "location", "severity"):
        if k in f:
            sub = sub[sub[k].isin(f[k])]
    if sub.empty:
        return None
    return _issue_chart(sub, f"Problems identified · {_describe({k: v for k, v in f.items() if k in ('crop', 'location', 'severity')})}")


# ---------------------------------------------------------------- knowledge answers
def _mix(sub):
    sev = sub["severity"][~sub["severity"].isin(NA)].value_counts()
    return ", ".join(f"{k} {v}" for k, v in sev.items()) if len(sev) else "not analyzed"


def _entry_block(e, sub=None):
    L = [f"#### 🔎 {e['name']}  ·  *{e['type']}*"]
    if sub is not None and len(sub):
        locs = ", ".join(f"{k} ({v})" for k, v in sub["location"].value_counts().head(3).items())
        L.append(f"**In your data:** {len(sub)} report(s) · crop: {', '.join(sorted(sub['crop'].unique()))} · "
                 f"top locations: {locs} · severity: {_mix(sub)}")
    L += [f"- **What it is:** {e['agent']}",
          f"- **Symptoms:** {e['symptoms']}",
          f"- **Causes:** {e['causes']}",
          f"- **What to do now:** {e['treatment']}",
          f"- **Prevention:** {e['prevention']}"]
    return "\n".join(L)


def _knowledge_answer(q, df, f, kt, all_items):
    sub = df
    for k in ("crop", "location", "severity", "sentiment"):
        if k in f:
            sub = sub[sub[k].isin(f[k])]
    scope = _describe({k: v for k, v in f.items() if k in ("crop", "location", "severity", "sentiment")})
    crop = f["crop"][0] if "crop" in f and len(f["crop"]) == 1 else None
    if scope == "all reports" and f.get("issue_key"):
        scope = "the problem you asked about"

    # symptoms typed by the user, e.g. "yellow spots and dying plants"
    typed = match_in_text(q, crop) if not f else []
    if typed:
        f = {**f, "issue_key": [e["key"] for e in typed]}
    if "issue_key" in f:
        sub = sub[sub["issue_key"].isin(f["issue_key"])]
        if scope == "all reports":
            scope = "the problem you asked about"
    if kt:
        sub = sub[sub["issue_type"] == kt]
    if sub.empty and not typed:
        kind = f" of type **{kt}**" if kt else ""
        return {"text": f"No reports{kind} match **{scope}** in the current filter.", "fig": None, "filters": f}

    counts = sub.groupby("issue_key").size().sort_values(ascending=False)
    keys = list(counts.index)
    if not keys:
        keys = [e["key"] for e in typed]
    limit = 8 if all_items else 3
    shown = keys[:limit]
    head = (f"Here is what the data shows for **{scope}**" + (f" ({kt} problems only)" if kt else "") +
            f": **{len(sub)} report(s)** fall into **{len(counts)} kind(s) of problem**.") if len(sub) else \
           "Based on the symptoms you described, this is what it most likely is."
    blocks = [_entry_block(entry_by_key(k), sub[sub["issue_key"] == k] if len(sub) else None) for k in shown]
    more = f"\n\n*{len(keys) - limit} more kind(s) not shown. Ask “show all diseases” to see every one.*" if len(keys) > limit else ""
    fig = _issue_chart(sub, f"Kinds of problems · {scope}") if len(sub) else None
    return {"text": head + "\n\n" + "\n\n".join(blocks) + more + "\n\n" + DISCLAIMER, "fig": fig, "filters": f}


# ---------------------------------------------------------------- main entry
CHARTS_ENABLED = False  # the Ask FarmSense chatbot answers in text only


def answer_from_data(question, df, last_filters=None):
    """Exact answer as {"text", "fig", "filters"}, or None if the AI should answer."""
    res = _answer_impl(question, df, last_filters)
    if res is not None and not CHARTS_ENABLED:
        res["fig"] = None
    return res


def _answer_impl(question, df, last_filters=None):
    q = question.lower().strip()
    if df is None or df.empty:
        return {"text": "There are no reports in the current filter.", "fig": None, "filters": {}}
    df = add_diagnosis(df)
    f = _detect_filters(q, df)
    know = bool(re.search(KNOW_WORDS, q)) or bool(match_in_text(q)) and not f
    wants_chart = _has(q, "chart", "graph", "plot", "visual", "diagram")

    # follow-up: "what are the causes?" reuses the crop/location from the previous question
    is_followup = (len(q.split()) <= 9 and not any(k in f for k in ("crop", "location"))
                   and not re.search(r"\b(all|every|each|reports?|dataset|overall|diseases|pests|kind|type)\b", q))
    if know and is_followup and last_filters and not match_in_text(q):
        for k in ("crop", "location", "severity", "issue_key"):
            if k in last_filters and k not in f:
                f[k] = last_filters[k]

    # ---- disease / cause / treatment questions (knowledge base)
    if know:
        kmap = {"disease": "Disease", "pest": "Pest", "insect": "Pest", "weather": "Weather", "water": "Water", "nutrient": "Nutrient"}
        kt = next((v for w, v in kmap.items() if re.search(rf"\b{w}", q)), None)
        f.pop("category", None)
        return _knowledge_answer(q, df, f, kt, all_items=_has(q, "all", "every", "each", "full", "complete"))

    dim = _detect_dim(q)
    if dim in f:
        dim = None

    needs_ai = [k for k in ("category", "severity") if k in f] + ([dim] if dim in ("category", "severity") else [])
    if any((df[k].isin(NA)).all() for k in set(needs_ai)):
        return {"text": "⚠️ The AI classification (category/severity) hasn't run yet, so I can't answer that exactly. Check the AI provider in the sidebar.", "fig": None, "filters": f}
    if "sentiment" in (dim, *f) and (df["sentiment"] == "—").all():
        return {"text": "⚠️ Sentiment hasn't been calculated yet, so I can't answer that.", "fig": None, "filters": f}

    sub = df.copy()
    for k, vals in f.items():
        if k != dim:
            sub = sub[sub[k].isin(vals)]
    scope = _describe({k: v for k, v in f.items() if k != dim})
    if sub.empty:
        return {"text": f"No reports match **{scope}** in the current filter.", "fig": None, "filters": f}

    # ---- trend over time
    if _has(q, "trend", "over time", "weekly", "per week", "timeline", "by week", "peak", "when"):
        fig, w = _trend_chart(sub, f"Reports per week · {scope}")
        peak = w.loc[w["reports"].idxmax()]
        return {"text": f"**{len(sub)}** report(s) for **{scope}** between {sub['date'].min():%b %d} and {sub['date'].max():%b %d, %Y}. "
                        f"The busiest week ended **{peak['date']:%b %d}** with **{int(peak['reports'])}** reports.",
                "fig": fig, "filters": f}

    # ---- chart of specific problems by crop / location (heatmap)
    if wants_chart and dim in ("crop", "location") and _has(q, "problem", "issue", "disease", "pest", "kind", "type"):
        ct = pd.crosstab(sub["likely_issue"], sub[dim])
        fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd",
                        title=f"Specific problems by {dim} · {scope}")
        top = "\n".join(f"- {c}: **{ct[c].idxmax()}** ({int(ct[c].max())} report(s))" for c in ct.columns)
        return {"text": f"Most common specific problem in each {dim}:\n{top}",
                "fig": fig, "filters": f}

    # ---- breakdown / ranking
    if dim is None and wants_chart:
        dim = "crop"
    if dim and (wants_chart or _has(q, "most", "least", "fewest", "top", "highest", "lowest", "which", "what", "per", "each", "breakdown", "common", "how many")):
        counts = sub[dim].value_counts()
        counts = counts[~counts.index.isin(NA)]
        asc = _has(q, "least", "fewest", "lowest")
        best = counts.min() if asc else counts.max()
        winners = ", ".join(counts[counts == best].index)
        total = int(counts.sum())
        table = "\n".join(f"- {k}: **{v}** ({v / total:.0%})" for k, v in counts.items())
        verb = "have" if len(winners.split(", ")) > 1 else "has"
        where = "across all reports" if scope == "all reports" else f"for {scope}"
        word = "fewest" if asc else "most"
        return {"text": f"**{winners}** {verb} the {word} reports (**{best}** of {total}) {where}.\n\nFull breakdown by {dim}:\n{table}",
                "fig": _count_chart(counts, dim, f"Reports by {dim} · {scope}"), "filters": f}

    def subset_chart():
        for col in ("crop", "location"):
            if col not in f and sub[col].nunique() > 1:
                return _count_chart(sub[col].value_counts(), col, f"Matching reports by {col} · {scope}")
        return _issue_chart(sub, f"Problems identified · {scope}")

    if _has(q, "percent", "percentage", "share", "proportion", "%") and f:
        return {"text": f"**{len(sub)}** of {len(df)} reports (**{len(sub) / len(df):.0%}**) match **{scope}**.",
                "fig": subset_chart(), "filters": f}

    if _has(q, "how many", "count", "number of", "total"):
        return {"text": f"There are **{len(sub)}** report(s) matching **{scope}** (out of {len(df)} in the current filter).",
                "fig": subset_chart(), "filters": f}

    if _has(q, "list", "show", "latest", "recent", "newest", "urgent", "which report", "what report", "give me"):
        rows = sub.sort_values("date", ascending=False).head(8)
        more = f"\n\n…and {len(sub) - 8} more." if len(sub) > 8 else ""
        return {"text": f"**{len(sub)}** report(s) match **{scope}**, newest first:\n" + "\n".join(_line(r) for r in rows.itertuples()) + more,
                "fig": subset_chart(), "filters": f}
    return None


def build_chat_context(df, max_rows=150):
    """Exact statistics + every report row (compact) for the AI to read."""
    df = add_diagnosis(df)
    L = [f"TOTAL REPORTS: {len(df)}", f"DATE RANGE: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}"]
    for col in ("crop", "location", "category", "severity", "sentiment", "likely_issue"):
        vc = df[col].value_counts()
        L.append(f"REPORTS PER {col.upper()}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    L.append("\nALL REPORTS (id | date | location | crop | likely_issue | category | severity | sentiment | report):")
    for i, r in enumerate(df.sort_values("date").head(max_rows).itertuples(), 1):
        L.append(f"{i} | {r.date:%Y-%m-%d} | {r.location} | {r.crop} | {r.likely_issue} | {r.category} | {r.severity} | {r.sentiment} | {r.report}")
    if len(df) > max_rows:
        L.append(f"...({len(df) - max_rows} more rows not shown)")
    return "\n".join(L)
