"""Streamlit UI for Canon.   Run:  .venv/bin/streamlit run app.py"""
import asyncio
import datetime as dt
import html
import time

import streamlit as st

from canon import canon, memory

EXAMPLES = [
    "Latest version of next.js",
    "What was Tesla's revenue in Q3 2026?",
    "Does semaglutide reduce cardiovascular events in obesity?",
    "What's new in the latest langchain release?",
    "When was the ISS launched?",
]
GREEN, YELLOW, GREY, BLUE = "#1a7f37", "#9a6700", "#6e7781", "#0969da"

st.set_page_config(page_title="Canon", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model (one time)…")
def _warm():
    from canon import classifier
    classifier._load()  # MiniLM + the 40 category examples, once per server process
    return True


_warm()


def badge(text, color):
    return (f"<span style='background:{color};color:#fff;border-radius:10px;padding:2px 9px;font-size:12px;"
            f"margin-right:6px'>{html.escape(text)}</span>")


st.title("Canon")
st.caption("Telem finds the sources. Canon keeps the facts.")
ask_tab, mem_tab = st.tabs(["Ask", "Memory"])

with ask_tab:
    ex = st.selectbox("Example", ["(type your own)"] + EXAMPLES)
    question = st.text_input("Question", value="" if ex == "(type your own)" else ex)
    compare = st.toggle("Show what a web-only agent would say", value=False,
                        help="Tavily alone, side by side. Not counted in Canon's totals.")

    if st.button("Answer", type="primary", disabled=not question.strip()):
        q = question.strip()
        with st.status("Checking memory…", expanded=True) as status:
            t0 = time.perf_counter()

            def on_step(msg):
                # runs on the same thread as asyncio.run, so writing to the status box is safe
                status.update(label=msg)
                if msg != "Done":
                    st.write(f"`{time.perf_counter() - t0:5.1f}s`  {msg}")

            async def go():
                if compare:
                    return await asyncio.gather(canon.answer(q, on_step=on_step), canon.web_only(q))
                return await canon.answer(q, on_step=on_step), None

            try:
                r, web = asyncio.run(go())
            except Exception as e:  # surface API/key problems in the UI instead of a stack trace
                status.update(label="Failed", state="error")
                st.error(f"{type(e).__name__}: {e}")
                st.stop()
            status.update(label=f"Done in {r['latency_ms'] / 1000:.1f}s — {r['route']}", state="complete",
                          expanded=False)

        cols = st.columns(2) if web else [st.container()]
        with cols[0]:
            # ---- answer card
            if r["verified"] and r.get("less_precise"):
                b = badge("✅ consistent, less precise", YELLOW)
            elif r["verified"]:
                b = badge("✅ verified", GREEN)
            else:
                b = badge("unverified", GREY)
            if r["from_memory"]:
                b += badge("from memory", BLUE)
            st.markdown(b, unsafe_allow_html=True)
            st.markdown(f"### {html.escape(r['answer'])}")
            if r.get("quote"):
                st.markdown(f"> {html.escape(r['quote'][:400])}")
            src = html.escape(r.get("source") or "source")
            link = f"<a href='{html.escape(r['url'])}' target='_blank'>{src}</a>" if r.get("url") else src
            st.markdown(link + (f" · {html.escape(r['date'])}" if r.get("date") else ""), unsafe_allow_html=True)
            if r.get("web_disagrees"):
                st.warning(f"The web disagrees with the official source; the official value wins. {r['explanation']}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Calls", f"{r['calls_used']} vs {r['calls_default']} default")
            m2.metric("Tokens saved", f"{r['tokens_saved']:,}",
                      help=f"Canon read {r['tokens_read']:,} tokens of sources and returned {r['tokens_returned']:,}.")
            m3.metric("Latency", f"{r['latency_ms'] / 1000:.1f}s")
            if r.get("monitor") and r["monitor"].get("id"):
                st.info(f"Hot fact: Telem monitor registered on {r['monitor']['domain']} to catch changes.")

        if web:
            with cols[1]:
                st.markdown(badge("web-only agent (tavily)", GREY), unsafe_allow_html=True)
                st.markdown(f"### {html.escape(web['answer'] or '(no answer)')}")
                st.caption(f"1 call · the agent reads ~{web['tokens_read']:,} tokens of results · "
                           f"{web['latency_ms'] / 1000:.1f}s · not counted in Canon's totals")
                for u in web["urls"][:5]:
                    st.markdown(f"- {html.escape(u)}")

        with st.expander("Details: route, classifier, sources, call log"):
            k = r.get("classifier")
            st.markdown(f"**Route:** `{r['route']}`  \n**Category:** {r['category']}"
                        + (f" · classifier {k['method']} ({k['ms']} ms, conf {k['confidence']:.2f})" if k else ""))
            for s in r.get("sources", []):
                val = html.escape(s.get("value") or "")
                if s.get("struck"):
                    val = f"<s>{val}</s>"
                color = GREEN if s["badge"] == "Official" else BLUE if s["badge"] == "Agreeing engines" else GREY
                st.markdown(f"{badge(s['badge'], color)}<b>{val}</b> <a href='{html.escape(s['url'])}' "
                            f"target='_blank'>{html.escape(s.get('title') or s['url'])}</a>", unsafe_allow_html=True)
            if r.get("events"):
                st.dataframe(r["events"], width="stretch")

with mem_tab:
    s = memory.stats()
    a, b, c, d = st.columns(4)
    a.metric("Facts stored", s["facts_stored"])
    b.metric("Memory hits", s["memory_hits"])
    c.metric("Searches saved", s["searches_saved"])
    d.metric("Tokens saved", f"{s['tokens_saved']:,}")
    facts = memory.all_facts()
    if not facts:
        st.caption("No facts yet. Verified answers are saved here; unverified ones never are.")
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for f in facts:
        left = dt.datetime.fromisoformat(f["expires_at"]) - now
        fresh = "expired" if left.total_seconds() <= 0 else (
            f"{left.days}d left" if left.days else f"{int(left.total_seconds() // 3600)}h left")
        rows.append({"answer": f["answer"], "question": f["question"], "category": f["category"],
                     "source": f["source_url"], "verified": f["verified_at"][:16].replace("T", " "),
                     "freshness": fresh, "hits": f["hit_count"], "calls saved": f["calls_saved"]})
    if rows:
        st.dataframe(rows, width="stretch", column_config={"source": st.column_config.LinkColumn("source")})
