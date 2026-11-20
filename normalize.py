"""Name normalization: merges different spellings of one entity."""
import re

# explicit merges (keys are already lowercase + singular last word)
SYNONYMS = {
    "mayo": "mayonnaise",
    "tuna in springwater": "tuna",
    "skim milk": "milk", "fat-free milk": "milk", "low-fat milk": "milk",
    "reduced fat milk": "milk", "1% milk": "milk", "whole milk": "milk",
    "milk of choice": "milk", "skim or 1% milk": "milk",
    "skillet": "frying pan", "pan": "frying pan",
}

# if a name ENDS with this phrase, it is the same entity (per node kind)
SUFFIX_RULES = {
    "Ingredient": [
        ("peanut butter", "peanut butter"), ("olive oil", "olive oil"),
        ("yogurt", "yogurt"), ("tuna", "tuna"), ("mayonnaise", "mayonnaise"),
    ],
    "Equipment": [
        ("frying pan", "frying pan"), ("skillet", "frying pan"),
        ("saucepan", "saucepan"), ("oven", "oven"), ("blender", "blender"),
        ("microwave", "microwave"), ("waffle iron", "waffle iron"),
        ("food processor", "food processor"), ("baking tray", "baking tray"),
        ("baking dish", "baking dish"), ("wok", "wok"),
    ],
}

# words typed by users in questions -> canonical entity name
ALIASES = {"skillet": "frying pan", "mayo": "mayonnaise", "yoghurt": "yogurt"}

_MODIFIERS = {"fresh", "chopped", "diced", "sliced", "grated", "crushed", "large",
              "small", "medium", "frozen", "cooked", "shredded", "minced",
              "finely", "thinly", "coarsely"}
_KEEP_S = {"hummus", "asparagus", "oats", "molasses", "couscous", "citrus", "grits"}


def _singular(w: str) -> str:
    if w in _KEEP_S or len(w) < 4:
        return w
    if w.endswith("leaves"):
        return w[:-6] + "leaf"
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("oes"):
        return w[:-2]
    if w.endswith("ss") or w.endswith("us"):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def normalize(name: str, kind: str) -> str:
    """kind: Recipe | Ingredient | Equipment | Category"""
    n = (name or "").lower().strip().replace("yoghurt", "yogurt")
    n = re.sub(r"\(.*?\)", " ", n)            # "(makes 4)", "(greek style fish)"
    n = re.sub(r"[^\w\s%'-]", " ", n)           # punctuation -> space
    words = n.split()
    if kind in ("Ingredient", "Equipment"):
        if kind == "Ingredient":
            words = [w for w in words if w not in _MODIFIERS] or words
        if words:
            words[-1] = _singular(words[-1])
    n = " ".join(words)
    if kind in SUFFIX_RULES:
        for suffix, canon in SUFFIX_RULES[kind]:
            if n.endswith(suffix):
                n = canon
                break
    if kind in ("Ingredient", "Equipment"):
        n = SYNONYMS.get(n, n)
    return n


def expand_question(q: str) -> str:
    """Append canonical names for aliases used in a question."""
    q = q.lower()
    extra = [canon for alias, canon in ALIASES.items() if alias in q]
    return q + " " + " ".join(extra) if extra else q