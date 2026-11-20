import csv, json, os, time
from vector_store import VectorStore
from llm_client import LLMClient
from rag import answer, MODES

CACHE_FILE = "eval_cache.json"
PAUSE = 13  # seconds between LLM calls: stays under ~6 requests/minute
GRADES = "correct | partial | wrong | no_info_but_in_sources"


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def main():
    with open("test_set.json", encoding="utf-8") as f:
        tests = json.load(f)
    store = VectorStore(name="tech_knowledge_base")
    llm = LLMClient()
    cache = load_cache()

    total = len(tests) * 4
    done = 0
    for t in tests:
        for mode in ("vector", "graph"):
            for run in (1, 2):
                done += 1
                key = f"{t['id']}|{mode}|{run}"
                if key in cache and not cache[key]["answer"].startswith("ERROR"):
                    continue
                entry = None
                for attempt in (1, 2):
                    try:
                        res = answer(t["question"], store, llm, mode=mode)
                        entry = {"answer": res["answer"],
                                 "retrieved": [c["id"] for c in res["chunks"]],
                                 "cited": res["cited"]}
                        break
                    except Exception as e:
                        entry = {"answer": f"ERROR: {e}", "retrieved": [], "cited": []}
                        print(f"  attempt {attempt} failed, waiting 40s")
                        time.sleep(40)
                cache[key] = entry
                save_cache(cache)
                print(f"[{done}/{total}] {key} -> {'OK' if not entry['answer'].startswith('ERROR') else 'ERROR'}")
                time.sleep(PAUSE)

    with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "expected", "mode", "run", "answer",
                    "retrieved_chunks", "cited", f"grade ({GRADES})", "stage_if_wrong"])
        for t in tests:
            for mode in ("vector", "graph"):
                for run in (1, 2):
                    e = cache.get(f"{t['id']}|{mode}|{run}",
                                  {"answer": "NOT RUN", "retrieved": [], "cited": []})
                    w.writerow([t["id"], t["type"], t["question"], t["expected"], MODES[mode], run,
                                e["answer"], " ".join(e["retrieved"]), " ".join(e["cited"]), "", ""])

    errors = [k for k, v in cache.items() if v["answer"].startswith("ERROR")]
    if errors:
        print(f"\n{len(errors)} answers still failed: {errors}\nRun evaluate.py again, it only retries these.")
    else:
        print("\nAll answers collected. results.csv saved, fill in the grade column manually.")


if __name__ == "__main__":
    main()