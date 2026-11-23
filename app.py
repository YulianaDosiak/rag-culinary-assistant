import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

import eval_store as es
from vector_store import VectorStore
from llm_client import LLMClient
from graph_retriever import graph_stats, find_entities, recipes_for, recipe_facts, REL_BY_LABEL
from rag import answer, MODES

st.set_page_config(page_title="Culinary Graph RAG", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.8rem;
        background:#2b3a55; color:#cfe0ff; margin-right:6px;}
</style>
""", unsafe_allow_html=True)

COLORS = {"HAS_INGREDIENT": "#f6a04d", "NEEDS_EQUIPMENT": "#6ab0f3", "BELONGS_TO": "#8bd17c"}
GRADE_OPTS = ["—", "correct", "partial", "wrong", "no_info_but_in_sources"]
STAGES = ["—", "extraction", "retrieval", "generation"]
SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0, "no_info_but_in_sources": 0.0}
EXAMPLES = [
    "Which recipes contain peanut butter?",
    "Which recipes contain both tuna and mayonnaise?",
    "Which recipes use a food processor?",
    "What equipment do I need for Vietnamese fish cakes?",
    "How do I make beef bourguignon?",
]


@st.cache_resource
def load_resources():
    return VectorStore(name="tech_knowledge_base"), LLMClient()


@st.cache_data(ttl=300)
def cached_stats():
    return graph_stats()


store, llm = load_resources()


# ------------------------------------------------------------------ helpers
def subgraph_html(triples):
    net = Network(height="400px", width="100%", bgcolor="#0e1117", font_color="white", cdn_resources="in_line")
    seen = set()
    for r, rel, n in triples[:80]:
        if r not in seen:
            net.add_node(r, label=r, color="#e8d44d", shape="dot", size=18); seen.add(r)
        if n not in seen:
            net.add_node(n, label=n, color=COLORS[rel], shape="dot", size=11); seen.add(n)
        net.add_edge(r, n, color=COLORS[rel], title=rel)
    return net.generate_html()


def render_result(res, mode):
    if "error" in res:
        st.error(res["error"])
        return
    st.markdown(res["answer"])
    t1, t2, t3 = st.tabs(["Sources", "Graph facts", "Subgraph"])
    with t1:
        cited = set(res["cited"])
        for c in res["chunks"]:
            dist = f" · distance {c['distance']:.3f}" if c["distance"] is not None else ""
            with st.expander(f"[{c['id']}] ({c['origin']}){dist}{' (cited)' if c['id'] in cited else ''}"):
                st.write(c["text"])
    with t2:
        if mode == "vector":
            st.info("Vector-only mode: no graph facts.")
        elif res["facts"]:
            st.markdown(" ".join(f"<span class='badge'>{l}: {n}</span>" for l, n in res["entities"]),
                        unsafe_allow_html=True)
            for f in res["facts"]:
                st.markdown(f)
        else:
            st.info("No graph facts.")
    with t3:
        if res["triples"]:
            components.html(subgraph_html(res["triples"]), height=420)
        else:
            st.info("Nothing to draw.")


def run_missing(keys_with_q, status):
    """keys_with_q: list of (key, question, mode). Stops after 2 errors in a row (quota)."""
    cache = es.load_cache()
    bar = st.progress(0.0)
    fails = 0
    for i, (key, q, mode) in enumerate(keys_with_q):
        status.write(f"Running {key} ({i + 1}/{len(keys_with_q)})...")
        try:
            cache[key] = es.entry_from_result(answer(q, store, llm, mode=mode))
            es.save_cache(cache)
            fails = 0
        except Exception as e:
            fails += 1
            status.warning(f"{key} failed: {e}")
            if fails >= 2:
                status.error("Two failures in a row: Gemini quota is probably exhausted. "
                             "Stop and try later (daily quota resets ~10:00 Kyiv time).")
                break
        bar.progress((i + 1) / len(keys_with_q))
        time.sleep(13)
    status.success("Finished. Reloading...")
    time.sleep(1)
    st.rerun()


def grade_widgets(key, grades):
    cur = grades.get(key, {"grade": "—", "stage": "—"})
    g = st.selectbox("Grade", GRADE_OPTS, index=GRADE_OPTS.index(cur["grade"]), key="g_" + key)
    s = "—"
    if g in ("partial", "wrong", "no_info_but_in_sources"):
        s = st.selectbox("Error stage", STAGES, index=STAGES.index(cur.get("stage", "—")), key="s_" + key)
    if (g, s) != (cur["grade"], cur.get("stage", "—")):
        grades[key] = {"grade": g, "stage": s}
        es.save_grades(grades)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Knowledge graph")
    try:
        stats = cached_stats()
        cols = st.columns(2)
        for i, row in enumerate(stats["nodes"]):
            cols[i % 2].metric(row["label"], row["c"])
        st.caption("Relationships: " + ", ".join(f"{r['type']} {r['c']}" for r in stats["rels"]))
    except Exception as e:
        st.error(f"Neo4j not reachable: {e}")
    st.divider()
    st.subheader("Try these")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.q = ex
            st.session_state.run_now = True

st.title("Culinary Graph RAG Assistant")
tab_ask, tab_tests, tab_sum, tab_graph, tab_agent, tab_agent_tests = st.tabs(
    ["Ask", "Test set", "Summary", "Graph", "Agent", "Agent tests"])

# ------------------------------------------------------------------ ASK
with tab_ask:
    st.session_state.setdefault("q", "")
    q = st.text_input("Your question", key="q", placeholder="Which recipes contain peanut butter?")
    go = st.button("Ask both systems", type="primary") or st.session_state.pop("run_now", False)
    if go and q.strip():
        out = {}
        with st.spinner("Vector search + graph search (2 LLM calls)..."):
            for mode in ("vector", "graph"):
                try:
                    out[mode] = answer(q, store, llm, mode=mode)
                except Exception as e:
                    out[mode] = {"error": str(e)}
        st.session_state.last = {"q": q, "out": out}

    last = st.session_state.get("last")
    if last:
        st.markdown(f"**Q:** {last['q']}")
        c1, c2 = st.columns(2)
        for col, mode in ((c1, "vector"), (c2, "graph")):
            with col:
                st.subheader(MODES[mode])
                render_result(last["out"][mode], mode)

        with st.expander("Add this question to the test set"):
            with st.form("add_test"):
                ttype = st.selectbox("Type", ["positive", "negative", "relational"])
                expected = st.text_input("Expected answer (what the sources actually say)")
                if st.form_submit_button("Add") and expected:
                    tests = es.load_tests()
                    new_id = f"{ttype[0].upper()}{sum(1 for t in tests if t['type'] == ttype) + 1}"
                    while any(t["id"] == new_id for t in tests):
                        new_id += "x"
                    tests.append({"id": new_id, "type": ttype, "question": last["q"], "expected": expected})
                    es.save_tests(tests)
                    cache = es.load_cache()
                    for mode in ("vector", "graph"):
                        if "error" not in last["out"][mode]:
                            cache[f"{new_id}|{mode}|1"] = es.entry_from_result(last["out"][mode])
                    es.save_cache(cache)
                    st.success(f"Added as {new_id} (run 1 already saved, run 2 can be added in Test set).")

# ------------------------------------------------------------------ TEST SET
with tab_tests:
    tests, cache, grades = es.load_tests(), es.load_cache(), es.load_grades()
    missing = [(f"{t['id']}|{m}|{r}", t["question"], m)
               for t in tests for m in ("vector", "graph") for r in (1, 2)
               if not es.is_ok(cache.get(f"{t['id']}|{m}|{r}"))]
    done = len(tests) * 4 - len(missing)
    st.progress(done / max(len(tests) * 4, 1), text=f"{done} of {len(tests) * 4} answers collected")
    status = st.empty()
    if missing and st.button(f"Run all {len(missing)} missing answers (~{len(missing) * 13 // 60 + 1} min)"):
        run_missing(missing, status)

    for t in tests:
        keys = [f"{t['id']}|{m}|{r}" for m in ("vector", "graph") for r in (1, 2)]
        n_ok = sum(es.is_ok(cache.get(k)) for k in keys)
        n_gr = sum(grades.get(k, {}).get("grade", "—") != "—" for k in keys)
        with st.expander(f"{t['id']} · {t['type']} · {t['question']}   [{n_ok}/4 answers, {n_gr}/4 graded]"):
            st.caption(f"Expected: {t['expected']}")
            cols = st.columns(2)
            for col, mode in zip(cols, ("vector", "graph")):
                with col:
                    st.markdown(f"#### {MODES[mode]}")
                    for run in (1, 2):
                        key = f"{t['id']}|{mode}|{run}"
                        e = cache.get(key)
                        st.markdown(f"**Run {run}**")
                        if not e:
                            st.warning("not run yet")
                        elif not es.is_ok(e):
                            st.error(e["answer"][:200])
                        else:
                            st.write(e["answer"])
                            st.caption("retrieved: " + ", ".join(e["retrieved"]))
                            grade_widgets(key, grades)
                        st.divider()
            miss = [(f"{t['id']}|{m}|{r}", t["question"], m) for m in ("vector", "graph") for r in (1, 2)
                    if not es.is_ok(cache.get(f"{t['id']}|{m}|{r}"))]
            if miss and st.button(f"Run {len(miss)} missing for {t['id']}", key="run_" + t["id"]):
                run_missing(miss, st.empty())

# ------------------------------------------------------------------ SUMMARY
with tab_sum:
    tests, cache, grades = es.load_tests(), es.load_cache(), es.load_grades()
    rows = []
    for t in tests:
        for m in ("vector", "graph"):
            for r in (1, 2):
                key = f"{t['id']}|{m}|{r}"
                g = grades.get(key, {})
                rows.append({"id": t["id"], "type": t["type"], "question": t["question"],
                             "mode": MODES[m], "mode_key": m, "run": r,
                             "answer": (cache.get(key) or {}).get("answer", "NOT RUN"),
                             "grade": g.get("grade", "—"), "stage": g.get("stage", "—")})
    df = pd.DataFrame(rows)
    graded = df[df["grade"] != "—"] if not df.empty else df
    if graded.empty:
        st.info("Grade some answers in the Test set tab first.")
    else:
        graded = graded.assign(score=graded["grade"].map(SCORE))
        st.subheader("Score by system (correct = 1, partial = 0.5)")
        by_mode = graded.groupby("mode")["score"].agg(["mean", "count"]).rename(
            columns={"mean": "avg score", "count": "graded answers"})
        st.dataframe(by_mode, use_container_width=True)

        st.subheader("Score by question type")
        pv = graded.pivot_table(index="type", columns="mode", values="score", aggfunc="mean")
        st.bar_chart(pv)
        st.dataframe(pv.round(2), use_container_width=True)

        st.subheader("Grades count")
        st.dataframe(pd.crosstab(graded["mode"], graded["grade"]), use_container_width=True)

        errs = graded[graded["grade"].isin(["partial", "wrong", "no_info_but_in_sources"])]
        if not errs.empty:
            st.subheader("Errors by pipeline stage")
            st.dataframe(pd.crosstab(errs["mode"], errs["stage"]), use_container_width=True)
            st.dataframe(errs[["id", "mode", "run", "grade", "stage", "question", "answer"]],
                         use_container_width=True)
    if not df.empty:
        st.download_button("Download results.csv", df.drop(columns="mode_key").to_csv(index=False).encode("utf-8-sig"),
                           "results.csv", "text/csv")

# ------------------------------------------------------------------ GRAPH
with tab_graph:
    name = st.text_input("Entity or recipe name (e.g. peanut butter, food processor, lentil tacos)")
    if name:
        ents = find_entities(name)
        if not ents:
            st.warning("Nothing found in the graph.")
        triples = []
        for label, n in ents:
            if label == "Recipe":
                f = recipe_facts(n)
                st.markdown(f"**Recipe: {n}**")
                st.write("Ingredients: " + ", ".join(i["name"] for i in f["ings"]))
                st.write("Equipment: " + (", ".join(e["name"] for e in f["eqs"]) or "—"))
                st.write("Categories: " + (", ".join(c["name"] for c in f["cats"]) or "—"))
                triples += [(n, "HAS_INGREDIENT", i["name"]) for i in f["ings"]]
                triples += [(n, "NEEDS_EQUIPMENT", e["name"]) for e in f["eqs"]]
                triples += [(n, "BELONGS_TO", c["name"]) for c in f["cats"]]
            else:
                found = recipes_for(label, n)
                st.markdown(f"**{label}: {n}** is used in {len(found)} recipes")
                st.write(", ".join(r["recipe"] for r in found))
                triples += [(r["recipe"], REL_BY_LABEL[label], n) for r in found]
        if triples:
            components.html(subgraph_html(triples), height=420)


import agent_ui

with tab_agent:
    agent_ui.render_chat(store, llm)
with tab_agent_tests:
    agent_ui.render_tests(store, llm)