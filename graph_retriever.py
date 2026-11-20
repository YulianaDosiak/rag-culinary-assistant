import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase, READ_ACCESS

from vector_store import VectorStore
from normalize import expand_question

load_dotenv()
_driver = None

REL_BY_LABEL = {"Ingredient": "HAS_INGREDIENT", "Equipment": "NEEDS_EQUIPMENT", "Category": "BELONGS_TO"}


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
    return _driver


def read(query: str, **params) -> list[dict]:
    """The ONLY way this module talks to Neo4j: a read-only session + read transaction."""
    def work(tx):
        return [r.data() for r in tx.run(query, **params)]
    with get_driver().session(default_access_mode=READ_ACCESS) as session:
        return session.execute_read(work)


# ---------------------------------------------------------------- stats
def graph_stats() -> dict:
    nodes = read("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY c DESC")
    rels = read("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS c ORDER BY c DESC")
    return {"nodes": nodes, "rels": rels}


# ---------------------------------------------------------------- entities
def _entity_index():
    rows = read("""MATCH (n) WHERE n:Recipe OR n:Ingredient OR n:Equipment OR n:Category
                   RETURN labels(n)[0] AS label, n.name AS name""")
    return [(r["label"], r["name"]) for r in rows]


def _pattern(name: str):
    forms = {name, name + "s", name + "es"}
    if name.endswith("y"):
        forms.add(name[:-1] + "ies")
    return re.compile(r"\b(" + "|".join(re.escape(f) for f in forms) + r")\b")


def find_entities(question: str) -> list[tuple[str, str]]:
    """Entities (label, name) whose name occurs in the question."""
    q = expand_question(question)
    found = [(l, n) for l, n in _entity_index() if len(n) > 2 and _pattern(n).search(q)]
    # "butter" is dropped when "peanut butter" is also found
    return [e for e in found if not any(e[1] != o[1] and e[1] in o[1] for o in found)]


# ---------------------------------------------------------------- graph queries
def recipe_facts(name: str) -> dict:
    rows = read("""
        MATCH (r:Recipe {name: $name})
        OPTIONAL MATCH (r)-[hi:HAS_INGREDIENT]->(i:Ingredient)
        WITH r, collect(DISTINCT {name: i.name, amount: hi.amount, chunks: hi.chunk_ids}) AS ings
        OPTIONAL MATCH (r)-[ne:NEEDS_EQUIPMENT]->(e:Equipment)
        WITH r, ings, collect(DISTINCT {name: e.name, chunks: ne.chunk_ids}) AS eqs
        OPTIONAL MATCH (r)-[bt:BELONGS_TO]->(c:Category)
        RETURN r.name AS recipe, ings, eqs,
               collect(DISTINCT {name: c.name, chunks: bt.chunk_ids}) AS cats
    """, name=name)
    if not rows:
        return {}
    row = rows[0]
    for key in ("ings", "eqs", "cats"):
        row[key] = [x for x in row[key] if x["name"]]
    return row


def recipes_for(label: str, name: str) -> list[dict]:
    rel = REL_BY_LABEL[label]
    return read(f"""
        MATCH (r:Recipe)-[x:{rel}]->(n:{label} {{name: $name}})
        RETURN r.name AS recipe, x.amount AS amount, x.chunk_ids AS chunks
        ORDER BY recipe
    """, name=name)


def get_chunks(ids: list[str]) -> list[dict]:
    rows = read("MATCH (c:Chunk) WHERE c.id IN $ids RETURN c.id AS id, c.source AS source, c.text AS text", ids=ids)
    order = {cid: i for i, cid in enumerate(ids)}
    return sorted(rows, key=lambda r: order[r["id"]])


# ---------------------------------------------------------------- retrieval
def _vector_chunks(query, store: VectorStore, top_k):
    out = []
    for h in store.search(query, n_results=top_k):
        meta = h["metadata"]
        out.append({"id": meta.get("chunk_id", h["id"]), "source": meta.get("source", "?"),
                    "text": h["chunk"], "distance": h["distance"], "origin": "vector"})
    return out


def _build_context(facts, chunks):
    parts = []
    if facts:
        parts.append("GRAPH FACTS (each fact lists the chunk ids that support it):\n" + "\n".join(facts))
    parts.append("TEXT CHUNKS:\n" + "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks))
    return "\n\n".join(parts)


def vector_search(query: str, store: VectorStore, top_k: int = 5) -> dict:
    """Baseline from Lab 1: vector similarity only."""
    chunks = _vector_chunks(query, store, top_k)
    return {"context": _build_context([], chunks), "facts": [], "chunks": chunks,
            "entities": [], "triples": []}


def hybrid_search(query: str, store: VectorStore, top_k: int = 5, max_recipes: int = 8,
                  max_extra_chunks: int = 6) -> dict:
    """Vector chunks -> graph facts about the recipes they mention
    + graph traversal from entities named in the question (reverse direction)."""
    chunks = _vector_chunks(query, store, top_k)
    vec_ids = [c["id"] for c in chunks]
    entities = find_entities(query)

    # recipes mentioned by retrieved chunks or named in the question
    rows = read("MATCH (c:Chunk)-[:MENTIONS]->(r:Recipe) WHERE c.id IN $ids RETURN DISTINCT r.name AS name",
                ids=vec_ids)
    recipe_names = [e[1] for e in entities if e[0] == "Recipe"]
    recipe_names += [r["name"] for r in rows if r["name"] not in recipe_names]
    recipe_names = recipe_names[:max_recipes]

    facts, triples, support = [], [], []

    def add_support(ids):
        for i in ids or []:
            if i not in support:
                support.append(i)

    # 1) forward facts about recipes
    for name in recipe_names:
        f = recipe_facts(name)
        if not f:
            continue
        ing = ", ".join(f"{i['name']}" + (f" ({i['amount']})" if i["amount"] else "") for i in f["ings"]) or "none"
        eq = ", ".join(e["name"] for e in f["eqs"]) or "none"
        cat = ", ".join(c["name"] for c in f["cats"]) or "none"
        ids = sorted({cid for x in f["ings"] + f["eqs"] + f["cats"] for cid in (x["chunks"] or [])})
        facts.append(f"- Recipe '{name}': ingredients = {ing}; equipment = {eq}; categories = {cat}. "
                     f"Sources: {', '.join(f'[{i}]' for i in ids)}")
        add_support(ids)
        triples += [(name, "HAS_INGREDIENT", i["name"]) for i in f["ings"]]
        triples += [(name, "NEEDS_EQUIPMENT", e["name"]) for e in f["eqs"]]
        triples += [(name, "BELONGS_TO", c["name"]) for c in f["cats"]]

    # 2) reverse traversal from entities in the question
    sets = []
    non_recipe = [e for e in entities if e[0] != "Recipe"]
    for label, name in non_recipe:
        found = recipes_for(label, name)
        sets.append({r["recipe"] for r in found})
        ids = sorted({cid for r in found for cid in (r["chunks"] or [])})
        items = "; ".join(f"{r['recipe']}" + (f" ({r['amount']})" if r["amount"] else "") for r in found) or "none"
        facts.append(f"- Recipes linked to {label.lower()} '{name}' ({len(found)}): {items}. "
                     f"Sources: {', '.join(f'[{i}]' for i in ids)}")
        add_support(ids)
        triples += [(r["recipe"], REL_BY_LABEL[label], name) for r in found]
    if len(non_recipe) >= 2:
        common = sorted(set.intersection(*sets))
        names = " AND ".join(f"'{n}'" for _, n in non_recipe)
        facts.append(f"- Recipes linked to ALL of {names}: {', '.join(common) if common else 'none'}")

    # 3) supporting chunks for the facts (so every fact has its text evidence)
    extra_ids = [i for i in support if i not in vec_ids][:max_extra_chunks]
    for c in get_chunks(extra_ids) if extra_ids else []:
        chunks.append({"id": c["id"], "source": c["source"], "text": c["text"],
                       "distance": None, "origin": "graph"})

    return {"context": _build_context(facts, chunks), "facts": facts, "chunks": chunks,
            "entities": entities, "triples": triples}