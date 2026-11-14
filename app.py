"""Streamlit UI for quorum.  Run:  .venv/bin/streamlit run app.py"""
import asyncio
import html

import streamlit as st

import quorum

BADGE_COLORS = {"Official": "#1a7f37", "Agreeing engines": "#0969da", "Web": "#6e7781"}
EXAMPLES = [
    "What was Tesla's revenue in Q3 2026?",
    "When was the ISS launched?",
    "Does semaglutide reduce major adverse cardiovascular events in adults with obesity?",
]

st.set_page_config(page_title="quorum", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading the classifier model (one time)...")
def _warm_classifier():
    import classifier
    classifier._load()  # embeds the 40 examples once per server process
    return True


_warm_classifier()
st.title("quorum")
st.caption("Routes a question to a specialist source when one fits, and checks it against the web.")

ex = st.selectbox("Example", ["(type your own)"] + EXAMPLES)
question = st.text_input("Question", value="" if ex == "(type your own)" else ex)

if st.button("Answer", type="primary", disabled=not question.strip()):
    with st.spinner("Routing and searching..."):
        try:
            r = asyncio.run(quorum.answer(question.strip()))
        except Exception as e:  # surface API/key problems in the UI instead of a stack trace
            st.error(f"{type(e).__name__}: {e}")
            st.stop()

    k = r["classifier"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Category", r["category"],
              delta=None if r["label"] == r["category"] else f"labeled {r['label']}", delta_color="off")
    c2.metric("Classifier", f"{k['method']} · {k['latency_ms']} ms",
              delta=f"conf {k['confidence']:.2f} · margin {k['margin']:.2f}", delta_color="off")
    c3.metric("Provider calls", f"{r['provider_calls']} vs {r['default_calls']} default")
    c4.metric("LLM calls", r["llm_calls"])
    c5.metric("Latency", f"{r['latency_ms'] / 1000:.1f}s")

    st.markdown(f"**Route:** `{r['route']}`")
    if r["entity"] or r["keywords"]:
        st.markdown(f"**Entity:** {r['entity']}" + (f" · **Period:** {r['period']}" if r["period"] else "")
                    + f" · **Keywords:** `{r['keywords']}`")

    st.subheader("Answer")
    verified = r["status"].startswith(("verified", "agreed", "resolved"))
    (st.success if verified else st.warning)(f"**{r['answer']}** — {r['status']}")
    if r.get("explanation"):
        st.caption(r["explanation"])

    st.subheader("Sources")
    for s in r["sources"]:
        color = BADGE_COLORS.get(s["badge"], "#6e7781")
        value = html.escape(s["value"] or "")
        if s["struck"]:
            value = f"<s>{value}</s>"
        meta = " · ".join(filter(None, [html.escape(s["provider"]), html.escape(s["date"] or "")]))
        st.markdown(
            f"<div style='padding:8px 0;border-bottom:1px solid rgba(128,128,128,.25)'>"
            f"<span style='background:{color};color:#fff;border-radius:10px;padding:2px 8px;font-size:12px'>"
            f"{html.escape(s['badge'])}</span> &nbsp;<b>{value}</b> &nbsp;"
            f"<a href='{html.escape(s['url'])}' target='_blank'>{html.escape(s['title'] or s['url'])}</a>"
            f"<br><span style='font-size:12px;opacity:.7'>{meta}</span></div>",
            unsafe_allow_html=True,
        )
        if s.get("quote"):
            st.caption(f"“{s['quote'][:300]}”")

    with st.expander("Call log"):
        st.dataframe(r["events"], width="stretch")
