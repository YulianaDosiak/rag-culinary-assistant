import json
import os
import time


class MemoryStore:
    """Key-value facts about the user, persisted in a JSON file."""

    def __init__(self, path: str = "memory.json"):
        self.path = path

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self, data: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    @staticmethod
    def _key(key: str) -> str:
        return "_".join(key.strip().lower().split())

    def all(self) -> dict:
        return self._load()

    def save(self, key: str, value: str):
        """Insert or REPLACE a fact. Returns the old value (or None)."""
        data = self._load()
        key = self._key(key)
        old = data.get(key, {}).get("value")
        data[key] = {"value": value.strip(), "updated": time.strftime("%Y-%m-%d %H:%M")}
        self._save(data)
        return old

    def delete(self, key: str) -> bool:
        data = self._load()
        key = self._key(key)
        if key in data:
            del data[key]
            self._save(data)
            return True
        return False

    def clear(self) -> None:
        self._save({})

    def as_prompt(self) -> str:
        data = self._load()
        if not data:
            return "(nothing known yet)"
        return "\n".join(f"- {k}: {v['value']}" for k, v in data.items())