import json
import os
import time

import pandas as pd
import streamlit as st

from agent import Agent
from memory_store import MemoryStore

TESTS_FILE = "agent_tests.json"
EVAL_FILE = "agent_eval.json"
GRADES = ["—", "correct", "partial", "wrong"]
STAGES = ["—", "tool choice", "arguments", "retrieval", "generation"]


def _load(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def render_trace(res):
    st.caption(f"Status: {res['status']} · model calls: {res['model_calls']} · tool calls: {res['tool_calls']}")
    for s in res["trace"]:
        st.markdown(f"**Model call {s['call']}**: {s['thought']}")
        if s["tool"]:
            st.markdown(f"Tool `{s['tool']}` with args `{s['args']}`")
            st.code(s["observation"], language=None)
        else:
            st.markdown("Final answer given.")


# ------------------------------------------------------------------ chat tab
def render_chat(store, llm):
    if "agent" not in st.session_state:
        st.session_state.agent = Agent(store, llm, MemoryStore("memory.json"))
        st.session_state.agent_msgs = []
    agent: Agent = st.session_state.agent

    with st.expander("Agent settings"):
        c1, c2, c3, c4 = st.columns(4)
        agent.max_model_calls = c1.number_input("Max model calls per request", 1, 10, 5, key="ag_mc")
        agent.max_calls_per_tool = c2.number_input("Max calls per tool", 1, 5, 2, key="ag_tc")
        agent.history_turns = c3.number_input("History turns kept", 1, 20, 6, key="ag_ht")
        agent.pause = c4.number_input("Pause between model calls (s)", 0, 30, 8, key="ag_p")
        st.caption("Each model call sees: instruction + tool list, all memory facts, the last N conversation turns "
                   "(questions and final answers only), the current question and the tool steps of this request.")

    facts = agent.memory.all()
    with st.expander(f"Memory ({len(facts)} facts)"):
        if not facts:
            st.info("Nothing stored yet. Try: \"I'm vegetarian.\"")
        for k, v in facts.items():
            a, b = st.columns([6, 1])
            a.markdown(f"**{k}**: {v['value']}  _(updated {v['updated']})_")
            if b.button("Delete", key="ag_del_" + k):
                agent.memory.delete(k)
                st.rerun()

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("New conversation", key="ag_new"):
        agent.reset_conversation()
        st.session_state.agent_msgs = []
        st.rerun()
    if b2.button("Clear memory", key="ag_clr"):
        agent.memory.clear()
        st.rerun()

    for m in st.session_state.agent_msgs:
        with st.chat_message("user"):
            st.write(m["q"])
        with st.chat_message("assistant"):
            if m["res"]["status"] != "ok":
                st.warning(f"Run status: {m['res']['status']}")
            st.write(m["res"]["answer"])
            with st.expander(f"Steps ({m['res']['model_calls']} model calls)"):
                render_trace(m["res"])

    with st.form("agent_form", clear_on_submit=True):
        q = st.text_input("Message to the agent", key="ag_q")
        sent = st.form_submit_button("Send")
    if sent and q.strip():
        with st.spinner("Agent is working (several model calls)..."):
            res = agent.run(q.strip())
        st.session_state.agent_msgs.append({"q": q.strip(), "res": res})
        st.rerun()


# ------------------------------------------------------------------ tests tab
def _run_test(test, store, llm, pause):
    mem = MemoryStore("memory_test.json")
    mem.clear()
    agent = Agent(store, llm, mem, pause=pause)
    total = 0
    for s in test.get("setup", []):
        total += agent.run(s)["model_calls"]
    if test.get("new_conversation"):
        agent.reset_conversation()
    for k, v in test.get("limits", {}).items():
        setattr(agent, k, v)
    res = agent.run(test["question"])
    res["setup_model_calls"] = total
    res["memory_after"] = mem.all()
    return res


def render_tests(store, llm):
    tests, ev = _load(TESTS_FILE, []), _load(EVAL_FILE, {})
    pause = st.number_input("Pause between model calls (s)", 0, 30, 8, key="agt_pause")
    todo = [t for t in tests if t["id"] not in ev]
    status = st.empty()
    if todo and st.button(f"Run all {len(todo)} missing tests", key="agt_all"):
        for i, t in enumerate(todo):
            status.write(f"Running {t['id']} ({i + 1}/{len(todo)})...")
            ev[t["id"]] = _run_test(t, store, llm, pause)
            _save(EVAL_FILE, ev)
        st.rerun()

    for t in tests:
        r = ev.get(t["id"])
        title = f"{t['id']} · {t['type']} · {t['question']}"
        with st.expander(title + (f"  [{r.get('grade', '—')}]" if r else "  [not run]")):
            if t.get("setup"):
                st.caption("Setup messages: " + " | ".join(t["setup"]) +
                           ("  → then a NEW conversation" if t.get("new_conversation") else ""))
            if t.get("limits"):
                st.caption(f"Limits for this test: {t['limits']}")
            st.caption(f"Expected: {t['expected']}")
            if st.button("Run / re-run", key="agt_run_" + t["id"]):
                with st.spinner("Running..."):
                    ev[t["id"]] = _run_test(t, store, llm, pause)
                    _save(EVAL_FILE, ev)
                st.rerun()
            if r:
                st.markdown(f"**Answer:** {r['answer']}")
                render_trace(r)
                if r.get("memory_after"):
                    st.caption("Memory after test: " + json.dumps(
                        {k: v["value"] for k, v in r["memory_after"].items()}, ensure_ascii=False))
                g = st.selectbox("Grade", GRADES, index=GRADES.index(r.get("grade", "—")), key="agt_g_" + t["id"])
                s = st.selectbox("Error stage", STAGES, index=STAGES.index(r.get("stage", "—")), key="agt_s_" + t["id"])
                if (g, s) != (r.get("grade", "—"), r.get("stage", "—")):
                    r["grade"], r["stage"] = g, s
                    _save(EVAL_FILE, ev)

    rows = [{"id": t["id"], "type": t["type"], "grade": ev[t["id"]].get("grade", "—"),
             "stage": ev[t["id"]].get("stage", "—"), "model calls": ev[t["id"]]["model_calls"],
             "status": ev[t["id"]]["status"], "tools": json.dumps(ev[t["id"]]["tool_calls"])}
            for t in tests if t["id"] in ev]
    if rows:
        st.subheader("Summary")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)