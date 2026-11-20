import re
from vector_store import VectorStore
from llm_client import LLMClient
from graph_retriever import hybrid_search, vector_search

SYSTEM_PROMPT = (
    "You are a culinary assistant. Answer ONLY from the provided context.\n"
    "Cite sources inline as [chunk_id] after each claim, using the chunk ids shown in the context.\n"
    "If the context does not contain the answer, reply exactly: "
    "\"I don't have this information in the provided sources.\" Never use outside knowledge.\n"
    "For questions like 'which recipes contain/need X', list every matching recipe from GRAPH FACTS "
    "and the text chunks, and mention amounts when known."
)

MODES = {"vector": "Vectors only (Lab 1)", "graph": "Graph + vectors (Lab 2)"}


def cited_ids(answer: str) -> list[str]:
    seen = []
    for cid in re.findall(r"[\w\-.]+_chunk_\d+", answer):
        if cid not in seen:
            seen.append(cid)
    return seen


def answer(question: str, store: VectorStore, llm: LLMClient, mode: str = "graph", top_k: int = 5) -> dict:
    retrieval = hybrid_search(question, store, top_k) if mode == "graph" else vector_search(question, store, top_k)
    prompt = f"CONTEXT:\n{retrieval['context']}\n\nQUESTION: {question}"
    text = llm.chat(prompt=prompt, system=SYSTEM_PROMPT, temperature=0.0).text
    return {"answer": text, "cited": cited_ids(text), **retrieval}