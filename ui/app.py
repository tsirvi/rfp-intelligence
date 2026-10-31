"""Streamlit front end; talks to the REST API only.

streamlit run ui/app.py        (expects the API on API_URL, default http://127.0.0.1:8000)
"""

from __future__ import annotations

import json
import os
import re
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
DOC_TYPES = ["rfp", "addendum", "bid_page", "specs", "affidavit", "other"]
DOC_COLOURS = {"rfp": "blue", "addendum": "orange", "bid_page": "violet", "specs": "green", "affidavit": "gray"}
ALL_BIDS = "All bids"

# Display grouping of the 20 fields (mirrors agents/fields.py; the UI only talks to the API).
FIELD_GROUPS = {
    "Identity": ["Bid Number", "Title", "company_name", "contact_info"],
    "Dates": ["Due Date", "Pre Bid Meeting", "Delivery Date"],
    "Submission & terms": ["Bid Submission Type", "Term of Bid", "Payment Terms"],
    "Compliance": [
        "Bid Bond Requirement",
        "Any Additional Documentation Required",
        "Contract or Cooperative to use",
        "MFG for Registration",
    ],
    "Product": ["Product", "Model_no", "Part_no", "Product Specification", "Installation"],
}
STATUS_ICON = {"passed": "✅", "failed": "⚠️", "not_found": "—"}
EXAMPLE_QUESTIONS = [
    "What is the submission deadline for Bid1 after all addendums?",
    "Which affidavits are required for the Dell laptop bid?",
    "Compare the warranty requirements of both bids.",
    "Is a bid bond required, and if so, how much?",
    "What changed in Addendum 2 compared to the original RFP?",
]
ANSWER_MODE = {
    "record": "⚡ From the validated record",
    "llm": "🤖 Written by the model",
    "passages": "⏱️ Relevant passages (the model could not answer in time)",
    "chat": "💬 Help",
    "error": "❌ Request failed",
}
EXAMPLE_SEARCHES = ["proposal due date", "warranty requirements", "delivery", "pre-proposal meeting", "insurance"]

st.set_page_config(page_title="RFP Intelligence", page_icon="📑", layout="wide")


# ---- API helpers --------------------------------------------------------------------------------------


def api(method: str, path: str, quiet: bool = False, **kwargs):
    """Call the API; show errors in the page unless quiet. Returns parsed JSON (or text) or None."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 120), **kwargs)
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {API_URL}. Start it with: `uvicorn api.app:app --port 8000`")
        st.stop()
    if resp.status_code >= 400:
        if not quiet:
            is_json = "json" in resp.headers.get("content-type", "")
            st.error(resp.json().get("detail", resp.text) if is_json else resp.text)
        return None
    return resp.json() if "json" in resp.headers.get("content-type", "") else resp.text


OCR_NOTE = re.compile(r"no text layer, used OCR")


def problems(doc: dict) -> list[str]:
    """Warnings that need attention; successful OCR is reported separately as information."""
    return [w for w in doc["warnings"] if not OCR_NOTE.search(w)]


def doc_label(doc_type: str, addendum_number: int | None) -> str:
    label = doc_type.replace("_", " ")
    return f"{label} {addendum_number}" if addendum_number is not None else label


def highlight(text: str, query: str) -> str:
    """Markdown-safe passage with the query's words in bold."""
    safe = re.sub(r"([\\`*_{}\[\]()#+!|>~])", r"\\\1", text)
    words = sorted({w for w in re.findall(r"[A-Za-z0-9-]{3,}", query)}, key=len, reverse=True)
    if words:
        pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)
        safe = pattern.sub(r"**\1**", safe)
    return safe.replace("\n", "  \n")


# ---- sidebar: status, bid picker, add a bid -------------------------------------------------------------


@st.fragment(run_every=5)
def sidebar_status() -> None:
    health = api("GET", "/health", timeout=10)
    if health is None:
        return
    st.caption(f"🟢 API connected · {len(health['bids'])} bids · {health['chunks']} passages indexed")
    for job_id in st.session_state.get("index_jobs", []):
        job = api("GET", f"/index/{job_id}", timeout=10, quiet=True)
        if not job:
            continue
        if job["status"] in ("queued", "running"):
            st.info(f"Indexing {job['bid_id']}…", icon="⏳")
        elif job["status"] == "failed":
            st.error(f"{job['bid_id']}: {job['error']}")
        else:
            r = job["report"]
            st.success(
                f"{job['bid_id']} indexed: {len(r['added'])} new, {len(r['updated'])} updated, "
                f"{len(r['unchanged'])} unchanged ({r['seconds']}s)"
            )
            for f in r["failures"]:
                st.warning(f"{f['file_name']}: {f['error']}")
    if health["indexing"] and not st.session_state.get("index_jobs"):
        st.info("Indexing bids from the data folder…", icon="⏳")


bids = api("GET", "/bids") or []
bid_ids = [b["bid_id"] for b in bids]

with st.sidebar:
    st.title("📑 RFP Intelligence")
    sidebar_status()
    selected = st.selectbox("Bid", [ALL_BIDS, *bid_ids], index=1 if bid_ids else 0, help="Applies to every tab")
    scope = bid_ids if selected == ALL_BIDS else [selected]
    with st.expander("➕ Add a bid folder"), st.form("index", border=False):
        path = st.text_input("Folder path", placeholder="data/Bid3 or /full/path/to/bid")
        new_id = st.text_input("Bid ID", placeholder="defaults to the folder name")
        force = st.checkbox("Re-parse files that have not changed")
        if st.form_submit_button("Index folder", type="primary") and path.strip():
            job = api("POST", "/index", json={"path": path.strip(), "bid_id": new_id.strip() or None, "force": force})
            if job:
                st.session_state.setdefault("index_jobs", []).append(job["job_id"])
                st.rerun()
    with st.expander("📈 Usage: cost and latency"):
        usage = api("GET", "/usage", quiet=True, timeout=10)
        if usage and usage["runs"]:
            rows = [
                {"run": kind, "count": k["runs"], "avg s": k["avg_seconds"], "LLM calls": k["llm_calls"],
                 "tokens": k["prompt_tokens"] + k["output_tokens"], f"cost ({usage['currency']})": k["est_cost"]}
                for kind, k in usage["by_kind"].items()
            ]  # fmt: skip
            st.dataframe(rows, hide_index=True)
            if usage["ask_cache_hit_rate"] is not None:
                st.caption(f"Q&A semantic cache hit rate: {usage['ask_cache_hit_rate']:.0%}")
            st.caption("Local model: cost is 0 unless token prices are set in config.yaml (`costs`).")
        else:
            st.caption("No runs recorded yet.")
    st.caption(
        "Answers and extracted fields are generated by an AI model; check the cited sources before relying on them."
    )

if not bid_ids:
    st.info("No bids indexed yet. Add a bid folder from the sidebar to get started.", icon="👈")
    st.stop()

overview_tab, ask_tab, compare_tab, search_tab, docs_tab = st.tabs(
    ["📋 Bid overview", "💬 Ask", "⚖️ Compare", "🔎 Search", "📂 Documents"]
)


# ---- bid overview: the extracted record ------------------------------------------------------------------


@st.fragment(run_every=10)
def extraction_progress(bid: str) -> None:
    jobs = [j for j in st.session_state.get("extract_jobs", []) if j["bid_id"] == bid]
    for job in jobs[-1:]:
        status = api("GET", f"/extract/{job['job_id']}", timeout=10, quiet=True)
        if not status:
            continue
        minutes = (time.time() - job["started"]) / 60
        if status["status"] in ("queued", "running"):
            st.info(f"Extracting {bid}… {minutes:.0f} min so far (usually 10-30 min on a CPU-only model)", icon="⏳")
        elif status["status"] == "failed":
            st.error(f"Extraction failed: {status['error']}")
        elif not job.get("shown"):
            job["shown"] = True
            st.success("Extraction finished.", icon="✅")
            st.rerun()  # reload the page so the new record shows


def field_status(f: dict) -> str:
    if "status" in f:
        return f["status"]
    return "not_found" if f["value"] is None else ("failed" if "validation failed" in f["notes"] else "passed")


DECISION_STYLE = {"GO": st.success, "REVIEW": st.warning, "NO-GO": st.error}
CHECK_ICON = {"pass": "✅", "review": "⚠️", "fail": "❌", "skip": "—"}


def go_no_go_panel(bid: str) -> None:
    """Bid / no-bid recommendation against company_profile.yaml (rule-based, instant)."""
    key = f"go_no_go_{bid}"
    if st.button(
        "🎯 Go / no-go recommendation", key=f"btn_{key}", help="Checks the record against company_profile.yaml"
    ):
        st.session_state[key] = api("GET", f"/go-no-go/{bid}")
    d = st.session_state.get(key)
    if not d:
        return
    DECISION_STYLE[d["decision"]](f"**{d['decision']}** · {d['summary']}", icon="🎯")
    with st.expander(f"Checks for {d['company']} as of {d['evaluation_date']}"):
        for c in d["checks"]:
            source = f" · _{c['source']}_" if c.get("source") else ""
            st.markdown(f"{CHECK_ICON[c['outcome']]} **{c['name']}**: {c['reason']}{source}")
        st.download_button(
            "⬇️ Recommendation (Markdown)", d["markdown"], file_name=f"{bid}_go_no_go.md", key=f"dl_{key}"
        )


def render_record(bid: str) -> None:
    rec = api("GET", f"/records/{bid}", quiet=True)
    if rec is None:
        st.info(f"No extracted record for {bid} yet. Run the extraction above, or `python main.py --bid data/{bid}`.")
        return
    fields = rec["fields"]
    statuses = {name: field_status(f) for name, f in fields.items() if name != "Bid Summary"}

    with st.container(border=True):
        title = (fields["Title"]["value"] or bid).replace("**", "")  # portal titles carry literal markdown
        st.markdown(f"#### {title}")
        st.write(fields["Bid Summary"]["value"] or "")
        due = fields["Due Date"]
        if due["value"]:
            st.markdown(f"**Due:** {due['value']}" + (f"  \n_{due['notes']}_" if "Addendum" in due["notes"] else ""))

    go_no_go_panel(bid)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fields verified", sum(s == "passed" for s in statuses.values()))
    m2.metric("Needs review", sum(s == "failed" for s in statuses.values()))
    m3.metric("Not in documents", sum(s == "not_found" for s in statuses.values()))
    m4.metric("Run time", f"{(rec['run'].get('seconds') or 0) / 60:.0f} min", help=f"Model {rec['model']}")

    show = st.segmented_control(
        "Show", ["All fields", "Needs review", "Not found"], default="All fields", key=f"show_{bid}"
    )
    wanted = {"All fields": {"passed", "failed", "not_found"}, "Needs review": {"failed"}, "Not found": {"not_found"}}
    for group, names in FIELD_GROUPS.items():
        names = [n for n in names if n in fields and statuses[n] in wanted[show or "All fields"]]
        if not names:
            continue
        st.markdown(f"##### {group}")
        for name in names:
            f = fields[name]
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 6, 2])
                c1.markdown(f"**{name.replace('_', ' ')}**")
                c1.caption(f"{STATUS_ICON[statuses[name]]} confidence {f['confidence']:.2f}")
                c2.markdown(f["value"] or "_Not found in documents_")
                if f["notes"] and f["notes"] != "Not found in documents":
                    c2.caption(f["notes"])
                c3.caption("\n\n".join(f"📄 {s['file']}, p. {s['page']}" for s in f["sources"][:3]) or "No source")

    changes = rec.get("addendum_changes") or []
    if changes:
        st.markdown("##### Addendum changes")
        relevant = [c for c in changes if c["field"] != "Other"]
        for c in relevant:
            icon = "✏️" if c["change_type"] == "modifies" else "💬"
            new = f" → **{c['new_value']}**" if c.get("new_value") else ""
            st.markdown(f"{icon} **{c['addendum']}** · {c['field'].replace('_', ' ')}: {c['description']}{new}")
        others = [c for c in changes if c["field"] == "Other"]
        if others:
            with st.expander(f"{len(others)} other addendum notes"):
                for c in others:
                    st.markdown(f"- **{c['addendum']}**: {c['description']}")

    v = rec["validation"]
    st.caption(
        f"Validator: {v['passed']} passed, {v['failed']} failed, {v['not_found']} not found · model {rec['model']} · "
        f"{rec['run'].get('llm_calls', '?')} LLM calls · generated {rec['generated_at']} · trace {rec['run'].get('trace')}"
    )
    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button("⬇️ JSON", json.dumps(rec, indent=2), file_name=f"{bid}.json", mime="application/json")
    report = api("GET", f"/records/{bid}/report", quiet=True)
    if report:
        d2.download_button("⬇️ Report (Markdown)", report, file_name=f"{bid}.md", mime="text/markdown")


with overview_tab:
    if selected == ALL_BIDS:
        st.info("Pick a bid in the sidebar to see its extracted record.", icon="👈")
    else:
        c1, c2 = st.columns([5, 1])
        c1.subheader(selected)
        if c2.button("▶️ Run extraction", help="Re-extract all 20 fields with the agents (10-30 min on CPU)"):
            job = api("POST", "/extract", json={"bid_id": selected})
            known = {j["job_id"] for j in st.session_state.get("extract_jobs", [])}
            if job and job["job_id"] not in known:  # the API returns the running job instead of a duplicate
                st.session_state.setdefault("extract_jobs", []).append(
                    {"job_id": job["job_id"], "bid_id": selected, "started": time.time()}
                )
        extraction_progress(selected)
        render_record(selected)


# ---- ask: chat with cited answers -------------------------------------------------------------------------

with ask_tab:
    history = st.session_state.setdefault("chat", [])
    st.caption(
        "Questions about an extracted field (due date, bid bond, contact…) are answered instantly from the "
        "validated record; others are answered by the model from cited passages, usually within 1-1.5 minutes."
    )
    example = st.pills("Try an example", EXAMPLE_QUESTIONS, key="example_q")
    for turn in history:
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.write(turn["content"])
                continue
            a = turn["content"]
            if a["answer"].startswith("Not found in documents"):
                st.warning(a["answer"], icon="🔍")
            else:
                st.markdown(a["answer"])
            if a["citation_labels"]:
                st.caption("Sources: " + " · ".join(f"📄 {c}" for c in a["citation_labels"]))
            details = [ANSWER_MODE.get(a.get("mode", "llm"), "")]
            if a.get("cached"):
                details.append(f"♻️ cached answer to an equivalent question (similarity {a['cache_similarity']:.2f})")
            if a.get("mode") != "chat":
                details += [f"Bids: {', '.join(a['bids']) or '-'}", f"confidence {a['confidence']:.2f}"]
            st.caption(" · ".join([*details, f"{a['seconds']:.0f}s"]))

    question = st.chat_input("Ask about the bids, e.g. When is Bid1 due after all addendums?") or example
    if question and question != st.session_state.get("last_question"):
        st.session_state["last_question"] = question
        history.append({"role": "user", "content": question})
        bids_scope = None if selected == ALL_BIDS else [selected]
        with st.status("The agents are working: planning, searching, answering…", expanded=False) as status:
            started = time.time()
            result = api("POST", "/ask", json={"question": question, "bids": bids_scope}, timeout=900)
            status.update(label=f"Done in {time.time() - started:.0f}s", state="complete" if result else "error")
        if not result:  # keep the failure in the chat; an error message alone would vanish on rerun
            result = {
                "answer": "The question could not be answered; check that the API and Ollama are running.",
                "citation_labels": [],
                "bids": bids_scope or [],
                "confidence": 0.0,
                "mode": "error",
            }
        history.append({"role": "assistant", "content": result | {"seconds": time.time() - started}})
        st.rerun()
    if history and st.button("Clear conversation"):
        st.session_state["chat"], st.session_state["last_question"] = [], None
        st.rerun()


# ---- compare: side-by-side report of extracted bids --------------------------------------------------------

with compare_tab:
    extracted = [b for b in bid_ids if api("GET", f"/records/{b}", quiet=True, timeout=10)]
    if len(extracted) < 2:
        st.info("Extract at least two bids to compare them (Bid overview → Run extraction).", icon="⚖️")
    else:
        chosen = st.multiselect("Bids to compare", extracted, default=extracted)
        summary = st.toggle("Add a model-written summary", value=False, help="About a minute on a CPU-only model")
        if st.button("⚖️ Compare", type="primary", disabled=len(chosen) < 2):
            with st.spinner("Comparing…"):
                st.session_state["comparison"] = api(
                    "POST", "/compare", json={"bids": chosen, "summary": summary}, timeout=900
                )
        report = st.session_state.get("comparison")
        if report:
            st.markdown(report["markdown"])
            c1, c2, _ = st.columns([1, 1, 4])
            name = "_vs_".join(report["bids"])
            c1.download_button("⬇️ Report (Markdown)", report["markdown"], file_name=f"{name}.md")
            data = json.dumps({k: v for k, v in report.items() if k != "markdown"}, indent=2)
            c2.download_button("⬇️ JSON", data, file_name=f"{name}.json", mime="application/json")


# ---- search: hybrid retrieval with citations -----------------------------------------------------------------

with search_tab:
    c1, c2 = st.columns([5, 1])
    query = c1.text_input("Search", placeholder="e.g. proposal due date", label_visibility="collapsed")
    with c2.popover("⚙️ Options", width="stretch"):
        k = st.slider("Results", 1, 20, 6)
        doc_types = st.multiselect("Document types", DOC_TYPES, format_func=lambda t: t.replace("_", " "))
        expand = st.toggle("Query expansion", value=True, help="Also search procurement synonyms (deadline → due date)")
        rerank = st.toggle("Rerank (slower)", value=False, help="Cross-encoder reranking; seconds per query on CPU")
    picked = st.pills("Examples", EXAMPLE_SEARCHES, key="example_s", label_visibility="collapsed")
    query = query.strip() or (picked or "")

    if len(query) >= 2:
        params = {"q": query, "k": k, "rerank": rerank, "expand": expand, "bid_id": scope, "doc_type": doc_types}
        with st.spinner("Searching…"):
            result = api("GET", "/search", params=params)
        if result:
            also = f" · also searched: {'; '.join(result['variants'][1:])}" if len(result["variants"]) > 1 else ""
            st.caption(f"{len(result['hits'])} passages in {result['timings'].get('total', 0):.2f}s{also}")
            if not result["hits"]:
                st.info("No matching passages. Try other words or fewer filters.")
            for hit in result["hits"]:
                with st.container(border=True):
                    h1, h2 = st.columns([4, 1])
                    h1.markdown(f"**{hit['rank']}. {hit['citation']}**")
                    with h2:
                        st.badge(
                            doc_label(hit["doc_type"], hit["addendum_number"]),
                            color=DOC_COLOURS.get(hit["doc_type"], "gray"),
                        )
                    if hit["section"]:
                        st.caption(f"{hit['bid_id']} · {hit['section']}")
                    if hit["kind"] == "table":
                        st.markdown(hit["text"])
                    elif len(hit["text"]) > 700:
                        st.markdown(highlight(hit["text"][:700], query) + " …")
                        with st.expander("Show full passage"):
                            st.markdown(highlight(hit["text"], query))
                    else:
                        st.markdown(highlight(hit["text"], query))
                    why = [f"dense #{hit['dense_rank']}"] if hit["dense_rank"] else []
                    why += [f"keyword #{hit['bm25_rank']}"] if hit["bm25_rank"] else []
                    st.caption("Matched by " + " and ".join(why) if why else "")
    else:
        st.caption("Search every indexed passage. Results show the file and page they come from.")


# ---- documents: what was ingested -------------------------------------------------------------------------------

with docs_tab:
    for bid in [b for b in bids if b["bid_id"] in scope]:
        docs = bid["documents"]
        st.subheader(bid["bid_id"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Documents", len(docs))
        m2.metric("Pages", sum(d["pages"] for d in docs))
        m3.metric("Tables", sum(d["tables"] for d in docs))
        m4.metric("OCR pages", sum(d["ocr_pages"] for d in docs))
        st.dataframe(
            [
                {
                    "File": d["file_name"],
                    "Type": doc_label(d["doc_type"], d["addendum_number"]),
                    "Title": (d["title"] or "").replace("**", ""),  # portal titles carry literal markdown
                    "Date": d["document_date"],
                    "Date from": (d["document_date_source"] or "").replace("_", " "),
                    "Pages": d["pages"],
                    "Tables": d["tables"],
                    "OCR pages": d["ocr_pages"],
                    "Warnings": len(problems(d)),
                }
                for d in docs
            ],
            hide_index=True,
            width="stretch",
        )
        ocr = {d["file_name"]: [w for w in d["warnings"] if OCR_NOTE.search(w)] for d in docs}
        ocr = {name: notes for name, notes in ocr.items() if notes}
        if ocr:
            pages = sum(len(n) for n in ocr.values())
            with st.expander(
                f"ℹ️ {pages} page{'s' * (pages > 1)} read with OCR (scanned or drawn pages, no text layer)"
            ):
                st.caption(
                    "These pages had no extractable text, so they were read from an image of the page. "
                    "OCR text can contain recognition errors."
                )
                for name, notes in ocr.items():
                    numbers = ", ".join(re.findall(r"page (\d+)", " ".join(notes)))
                    st.markdown(f"**{name}**: pages {numbers}")
        warned = [d for d in docs if problems(d)]
        if warned:
            with st.expander(f"⚠️ Parsing warnings ({sum(len(problems(d)) for d in warned)})"):
                for d in warned:
                    st.markdown(f"**{d['file_name']}**\n" + "\n".join(f"- {w}" for w in problems(d)))
