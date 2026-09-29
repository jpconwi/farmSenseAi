"""
utils/chat_engine.py
--------------------
Makes the "Ask FarmSense AI" chatbot accurate.

1. answer_from_data(): counts, rankings, percentages and report lists are
   calculated EXACTLY with Pandas (no AI guessing) for common questions.
2. build_chat_context(): for everything else, the AI receives exact
   statistics AND every report row, so it reads real data, not a summary.
"""
import re

import pandas as pd

CATS = ["Pest", "Disease", "Water", "Weather", "Nutrient", "Other"]
SEVS = ["Low", "Moderate", "High"]
SENTS = ["positive", "neutral", "negative"]
NA = ("Not analyzed", "—")


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
    if _has(q, "insect", "worm", "aphid", "beetle"):
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
    parts = [f"{'/'.join(v)} {k}" for k, v in f.items()]
    return " · ".join(parts) if parts else "all reports"


def _line(r):
    sev = f" · **{r.severity}**" if r.severity not in NA else ""
    return f"- {r.date:%Y-%m-%d} · {r.location} · {r.crop}{sev} — {r.report}"


def answer_from_data(question, df):
    """Return an exact answer (markdown) or None if the AI should handle it."""
    q = question.lower().strip()
    if df is None or df.empty:
        return "There are no reports in the current filter."
    f = _detect_filters(q, df)
    dim = _detect_dim(q)
    if dim in f:  # e.g. "high severity" is a filter, not a breakdown request
        dim = None

    needs_ai = [k for k in ("category", "severity") if k in f] + ([dim] if dim in ("category", "severity") else [])
    if any((df[k].isin(NA)).all() for k in set(needs_ai)):
        return "⚠️ The AI classification (category/severity) hasn't run yet, so I can't answer that exactly. Check the AI provider in the sidebar."
    if "sentiment" in (dim, *f) and (df["sentiment"] == "—").all():
        return "⚠️ Sentiment hasn't been calculated yet, so I can't answer that."

    sub = df.copy()
    for k, vals in f.items():
        if k != dim:
            sub = sub[sub[k].isin(vals)]
    scope = _describe({k: v for k, v in f.items() if k != dim})
    if sub.empty:
        return f"No reports match **{scope}** in the current filter."

    if dim and _has(q, "most", "least", "fewest", "top", "highest", "lowest", "which", "what", "per", "each", "breakdown", "common"):
        counts = sub[dim].value_counts()
        counts = counts[~counts.index.isin(NA)]
        asc = _has(q, "least", "fewest", "lowest")
        best = counts.min() if asc else counts.max()
        winners = ", ".join(counts[counts == best].index)
        total = int(counts.sum())
        table = "\n".join(f"- {k}: **{v}** ({v / total:.0%})" for k, v in counts.items())
        word = "fewest" if asc else "most"
        verb = "have" if len(winners.split(", ")) > 1 else "has"
        where = "across all reports" if scope == "all reports" else f"for {scope}"
        return f"**{winners}** {verb} the {word} reports (**{best}** of {total}) {where}.\n\nFull breakdown by {dim}:\n{table}"

    if _has(q, "percent", "percentage", "share", "proportion", "%") and f:
        return f"**{len(sub)}** of {len(df)} reports ({len(sub) / len(df):.0%}) match **{scope}**."

    if _has(q, "how many", "count", "number of", "total"):
        return f"There are **{len(sub)}** report(s) matching **{scope}** (out of {len(df)} in the current filter)."

    if _has(q, "list", "show", "latest", "recent", "newest", "urgent", "which report", "what report", "give me"):
        rows = sub.sort_values("date", ascending=False).head(8)
        more = f"\n\n…and {len(sub) - 8} more." if len(sub) > 8 else ""
        return f"**{len(sub)}** report(s) match **{scope}**, newest first:\n" + "\n".join(_line(r) for r in rows.itertuples()) + more
    return None


def build_chat_context(df, max_rows=150):
    """Exact statistics + every report row (compact) for the AI to read."""
    L = [f"TOTAL REPORTS: {len(df)}", f"DATE RANGE: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}"]
    for col in ("crop", "location", "category", "severity", "sentiment"):
        vc = df[col].value_counts()
        L.append(f"REPORTS PER {col.upper()}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    L.append("\nALL REPORTS (id | date | location | crop | category | severity | sentiment | report):")
    for i, r in enumerate(df.sort_values("date").head(max_rows).itertuples(), 1):
        L.append(f"{i} | {r.date:%Y-%m-%d} | {r.location} | {r.crop} | {r.category} | {r.severity} | {r.sentiment} | {r.report}")
    if len(df) > max_rows:
        L.append(f"...({len(df) - max_rows} more rows not shown)")
    return "\n".join(L)
