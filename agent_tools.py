import json
import re

from graph_retriever import read, recipe_facts, recipes_for, hybrid_search
from normalize import normalize

MAX_OBS = 3500
KINDS = {"ingredient": "Ingredient", "equipment": "Equipment", "category": "Category"}

# ------------------------------------------------------------------ scaling (pure code, no LLM)
UNI = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅛": 0.125}
NUM = r"\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?[½¼¾⅓⅔⅛]?|[½¼¾⅓⅔⅛]"


def _val(tok: str) -> float:
    tok = tok.strip()
    m = re.fullmatch(r"(\d+)\s+(\d+)/(\d+)", tok)
    if m:
        return int(m[1]) + int(m[2]) / int(m[3])
    m = re.fullmatch(r"(\d+)/(\d+)", tok)
    if m:
        return int(m[1]) / int(m[2])
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([½¼¾⅓⅔⅛])", tok)
    if m:
        return float(m[1]) + UNI[m[2]]
    if tok in UNI:
        return UNI[tok]
    return float(tok)


def _fmt(x: float) -> str:
    x = round(x, 2)
    return str(int(x)) if x == int(x) else str(x)


def scale_amount(amount: str, factor: float) -> str:
    """'1½ cup' x 2 -> '3 cup'. Amounts with no leading number are returned unchanged."""
    if not amount:
        return amount
    m = re.match(rf"\s*({NUM})(?:\s*(?:-|to)\s*({NUM}))?(.*)$", amount)
    if not m:
        return amount
    first = _fmt(_val(m.group(1)) * factor)
    second = f"-{_fmt(_val(m.group(2)) * factor)}" if m.group(2) else ""
    return f"{first}{second}{m.group(3)}"


# ------------------------------------------------------------------ helpers
def _resolve_recipe(name: str):
    n = normalize(name, "Recipe")
    if recipe_facts(n):
        return n, []
    rows = read("MATCH (r:Recipe) WHERE r.name CONTAINS $n RETURN r.name AS name ORDER BY name LIMIT 6", n=n)
    names = [r["name"] for r in rows]
    if len(names) == 1:
        return names[0], []
    return None, names


def _ids(items):
    return sorted({cid for x in items for cid in (x.get("chunks") or [])})


# ------------------------------------------------------------------ tools
def search_recipes(args, store, memory):
    query = str(args.get("query", "")).strip()
    if not query:
        return "ERROR: argument 'query' is required."
    return hybrid_search(query, store, top_k=4)["context"]


def find_recipes_by_entity(args, store, memory):
    kind = str(args.get("kind", "")).strip().lower()
    name = str(args.get("name", "")).strip()
    if kind not in KINDS or not name:
        return "ERROR: need kind (ingredient|equipment|category) and name."
    label = KINDS[kind]
    n = normalize(name, label)
    rows = recipes_for(label, n)
    if not rows:
        similar = read(f"MATCH (n:{label}) WHERE n.name CONTAINS $n RETURN n.name AS name LIMIT 8", n=n)
        hint = ", ".join(r["name"] for r in similar) or "none"
        return f"No recipes linked to {kind} '{n}'. Similar {kind} names in the graph: {hint}."
    lines = [f"Recipes linked to {kind} '{n}' ({len(rows)}):"]
    for r in rows:
        amt = f" ({r['amount']})" if r.get("amount") else ""
        src = ", ".join(f"[{c}]" for c in (r.get("chunks") or []))
        lines.append(f"- {r['recipe']}{amt} {src}")
    return "\n".join(lines)


def _format_recipe(f, factor=None):
    def amt(i):
        a = i.get("amount") or ""
        if factor is not None and a:
            return f"{i['name']}: {scale_amount(a, factor)} (original: {a})"
        return f"{i['name']}" + (f": {a}" if a else "")

    ids = _ids(f["ings"] + f["eqs"] + f["cats"])
    head = f"Recipe '{f['recipe']}'" + (f" scaled x{factor}" if factor is not None else "")
    return (head + "\nIngredients:\n" + "\n".join("- " + amt(i) for i in f["ings"])
            + "\nEquipment: " + (", ".join(e["name"] for e in f["eqs"]) or "none recorded")
            + "\nCategories: " + (", ".join(c["name"] for c in f["cats"]) or "none recorded")
            + "\nSources: " + ", ".join(f"[{i}]" for i in ids))


def get_recipe(args, store, memory):
    name = str(args.get("recipe_name", "")).strip()
    if not name:
        return "ERROR: argument 'recipe_name' is required."
    resolved, options = _resolve_recipe(name)
    if not resolved:
        return "No recipe with that name. Candidates: " + (", ".join(options) or "none")
    return _format_recipe(recipe_facts(resolved))


def scale_recipe(args, store, memory):
    name = str(args.get("recipe_name", "")).strip()
    try:
        factor = float(args.get("factor"))
    except (TypeError, ValueError):
        return "ERROR: 'factor' must be a number, e.g. 2 for double."
    if not name or not (0.1 <= factor <= 20):
        return "ERROR: need recipe_name and factor between 0.1 and 20."
    resolved, options = _resolve_recipe(name)
    if not resolved:
        return "No recipe with that name. Candidates: " + (", ".join(options) or "none")
    return _format_recipe(recipe_facts(resolved), factor) + \
        "\nNote: amounts without a leading number (e.g. 'pinch') and parts after '+' are not scaled."


def save_memory(args, store, memory):
    key, value = str(args.get("key", "")).strip(), str(args.get("value", "")).strip()
    if not key or not value:
        return "ERROR: need key and value."
    old = memory.save(key, value)
    return f"Saved {key} = '{value}'" + (f" (replaced old value '{old}')" if old else "")


def delete_memory(args, store, memory):
    key = str(args.get("key", "")).strip()
    return f"Deleted '{key}'." if key and memory.delete(key) else f"No stored fact with key '{key}'."


TOOLS = {
    "search_recipes": search_recipes,
    "find_recipes_by_entity": find_recipes_by_entity,
    "get_recipe": get_recipe,
    "scale_recipe": scale_recipe,
    "save_memory": save_memory,
    "delete_memory": delete_memory,
}

TOOL_DESCRIPTIONS = """TOOLS (call exactly one per step):
1. search_recipes - args {"query": "..."}: hybrid search (text chunks + graph facts) over the cookbooks. Use for cooking times, temperatures, methods, and anything the other tools cannot answer.
2. find_recipes_by_entity - args {"kind": "ingredient|equipment|category", "name": "..."}: fixed Cypher query on the knowledge graph. Returns ALL recipes that use an ingredient / need equipment / belong to a category, with amounts and source chunks. Prefer it for "which recipes contain/use/need X".
3. get_recipe - args {"recipe_name": "..."}: ingredients with amounts, equipment and categories of ONE recipe (graph).
4. scale_recipe - args {"recipe_name": "...", "factor": 2}: all amounts multiplied by factor, computed in code. Use for doubling, halving, "for 3 times the amount". Never do this arithmetic yourself.
5. save_memory - args {"key": "diet", "value": "vegetarian"}: store a lasting fact about the user. Same key REPLACES the old value.
6. delete_memory - args {"key": "diet"}: remove a fact that is no longer true."""


def run_tool(name, args, store, memory) -> str:
    out = TOOLS[name](args, store, memory)
    return out if len(out) <= MAX_OBS else out[:MAX_OBS] + "\n...[truncated]"