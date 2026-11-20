import json, os

TESTS = "test_set.json"
CACHE = "eval_cache.json"
GRADES = "grades.json"


def _load(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def load_tests():   return _load(TESTS, [])
def save_tests(d):  _save(TESTS, d)
def load_cache():   return _load(CACHE, {})
def save_cache(d):  _save(CACHE, d)
def load_grades():  return _load(GRADES, {})
def save_grades(d): _save(GRADES, d)


def is_ok(entry) -> bool:
    return bool(entry) and not entry["answer"].startswith("ERROR")


def entry_from_result(res: dict) -> dict:
    return {"answer": res["answer"],
            "retrieved": [c["id"] for c in res["chunks"]],
            "cited": res["cited"]}