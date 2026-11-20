import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from llm_client import LLMClient
from normalize import normalize

load_dotenv()
client = LLMClient()


class IngredientRel(BaseModel):
    recipe_name: str
    ingredient_name: str
    amount: str = Field(description="Amount, e.g., '200g' or 'to taste'; empty string if unknown")

class EquipmentRel(BaseModel):
    recipe_name: str
    equipment_name: str

class CategoryRel(BaseModel):
    recipe_name: str
    category_name: str

class GraphExtraction(BaseModel):
    recipes: list[str] = Field(description="Recipe titles that appear in the text")
    contains_ingredients: list[IngredientRel]
    needs_equipment: list[EquipmentRel]
    belongs_to: list[CategoryRel]


SYSTEM_PROMPT = """You are a culinary information extractor. Extract a knowledge graph from the text chunk.
Rules:
- recipes: the recipe titles exactly as written (without serving info like '4 SERVES | 10 MINS').
- Ingredient names: base name only, lowercase, no quantities and no preparation words
  ('2 cloves garlic, crushed' -> name 'garlic', amount '2 cloves'). Put the quantity in amount.
- Optional add-ins ARE ingredients (amount: 'optional').
- Equipment: only items explicitly named in the text (blender, oven, skillet, frying pan, food processor...). Never infer.
- Categories: choose from breakfast, lunch, dinner, dessert, snack, drink, side. Omit if unclear.
- IMPORTANT: if the chunk starts in the middle of a recipe and its title is NOT visible in this chunk,
  do NOT invent a recipe name and do NOT create relations for those lines.
- Every relation must use a recipe title that is listed in recipes.
- If the chunk has no recipes, return empty lists."""


def extract_graph_from_text(text_chunk: str, chunk_id: str):
    """Returns the raw extraction as a dict, or None if the LLM failed twice."""
    for attempt in (1, 2):
        try:
            result = client.chat(
                prompt=f"Text:\n{text_chunk}",
                system=SYSTEM_PROMPT + f"\nChunk ID: {chunk_id}",
                temperature=0.0,
                schema=GraphExtraction,
            )
            parsed = getattr(result, "parsed", None)
            if isinstance(parsed, BaseModel):
                return parsed.model_dump()
            if isinstance(parsed, dict):
                return parsed
            return json.loads(result.text)
        except Exception as e:
            print(f"  LLM error for {chunk_id} (attempt {attempt}): {e}")
    return None


def new_report() -> dict:
    return {"merges": {}, "dangling": [], "dropped": [], "failed_chunks": []}


def clean_extraction(raw: dict, chunk_id: str, report: dict) -> dict:
    """Normalize names, merge spellings, fix relations that point to unknown recipes.
    Everything it changes is written to `report` (evidence for the lab write-up)."""

    def norm(name, kind):
        n = normalize(name, kind)
        if n and (name or "").lower().strip() != n:
            report["merges"].setdefault(kind, {}).setdefault(n, set()).add(name.strip())
        return n

    recipes = {norm(r, "Recipe") for r in raw.get("recipes", [])} - {""}
    ings, eqs, cats = {}, set(), set()

    def check_recipe(r, rel_type):
        if r not in recipes:
            report["dangling"].append({"chunk": chunk_id, "recipe": r, "relation": rel_type,
                                       "fix": "recipe not in recipes list; added"})
            recipes.add(r)

    for rel in raw.get("contains_ingredients", []):
        r, i = norm(rel["recipe_name"], "Recipe"), norm(rel["ingredient_name"], "Ingredient")
        if not r or not i:
            report["dropped"].append({"chunk": chunk_id, "rel": rel}); continue
        check_recipe(r, "HAS_INGREDIENT")
        amt = (rel.get("amount") or "").strip()
        if (r, i) not in ings or (amt and not ings[(r, i)]):
            ings[(r, i)] = amt
    for rel in raw.get("needs_equipment", []):
        r, e = norm(rel["recipe_name"], "Recipe"), norm(rel["equipment_name"], "Equipment")
        if not r or not e:
            report["dropped"].append({"chunk": chunk_id, "rel": rel}); continue
        check_recipe(r, "NEEDS_EQUIPMENT"); eqs.add((r, e))
    for rel in raw.get("belongs_to", []):
        r, c = norm(rel["recipe_name"], "Recipe"), norm(rel["category_name"], "Category")
        if not r or not c:
            report["dropped"].append({"chunk": chunk_id, "rel": rel}); continue
        check_recipe(r, "BELONGS_TO"); cats.add((r, c))

    return {
        "recipes": sorted(recipes),
        "ingredients": [{"recipe": r, "name": i, "amount": a} for (r, i), a in sorted(ings.items())],
        "equipment": [{"recipe": r, "name": e} for r, e in sorted(eqs)],
        "categories": [{"recipe": r, "name": c} for r, c in sorted(cats)],
    }