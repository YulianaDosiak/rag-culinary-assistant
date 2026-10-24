"""Week 3 — dense retrieval: store chunk embeddings, search them by similarity.

    from vector_store import VectorStore
    store = VectorStore()
    store.add(chunks)                      # embeds and stores every chunk, once
    results = store.search("a question")   # embeds the question, searches the index

This is the "ingest once, query every call" half of the retrieval pipeline from
week-03-lecture-plan.md §6: `add()` is the ingestion-time work, `search()` is what
runs on every question.

The store is PERSISTENT, and that matters more than it looks
--------------------------------------------------------------
Embeddings are written to `./chroma/` on disk and survive the process. Embed a
corpus once, with `python ingest.py`, and every notebook and script afterwards
opens the same index without paying for it again.

This is not a performance nicety. `gemini-embedding-001` allows roughly 100 texts
per minute (measured 2026-09-14), so a 3,000-chunk corpus is about half an hour of
ingest. An in-memory store would spend that again on every kernel restart.

`reset=True` clears the collection first -- which is what you want after changing
chunk size, and never want otherwise.

Chroma's default distance is NOT cosine
-----------------------------------------
Confirmed 2026-09-13: an unconfigured Chroma collection uses squared Euclidean
distance. This course teaches and measures cosine similarity, so the collection
below is created with `metadata={"hnsw:space": "cosine"}` explicitly. Without that
line, `search()`'s scores would not be the metric taught in the lecture, silently.
"""

import pathlib
from typing import Optional

import chromadb

from embedding_client import EmbeddingClient


class VectorStore:
    def __init__(
        self,
        embedder: Optional[EmbeddingClient] = None,
        name: str = "chunks",
        *,
        path: Optional[str] = None,
        reset: bool = False,
    ):
        self.embedder = embedder or EmbeddingClient()
        # Anchored to this file's folder (week3/chroma), NOT the current working
        # directory. So `ingest.py` and the notebook agree on where the store is
        # whether you run them from week3/ or from the repo root -- a cwd-relative
        # "./chroma" would put them in different places and the notebook would
        # read an empty collection. Pass `path=` to override.
        if path is None:
            path = str(pathlib.Path(__file__).resolve().parent / "chroma")
        self.path = path
        self._client = chromadb.PersistentClient(path=path)
        if reset:
            try:
                self._client.delete_collection(name)
            except Exception:
                pass  # nothing stored under that name yet
        self._collection = self._client.get_or_create_collection(
            name,
            # Explicitly NONE. We pass our own vectors to add() and search(),
            # so Chroma must not record an embedding function of its own --
            # left unset it stores "default", meaning its 384-dimension ONNX
            # model, while the vectors here are 3072 from gemini-embedding-001.
            # Nothing in this file notices, because it never asks Chroma to
            # embed anything; an outside tool opening the collection does.
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """How many chunks are already indexed. Zero means you need ingest.py."""
        return self._collection.count()

    def peek(self, n: int = 5) -> list[dict]:
        """The first n stored chunks, without searching for anything.

        `search()` answers "what is relevant to this question". This answers
        "what is actually in there" -- which is the one you want when a chunk
        boundary looks wrong, or when a retrieved chunk is not what you expected.
        """
        got = self._collection.get(limit=n, include=["documents", "metadatas"])
        metas = got.get("metadatas") or [None] * len(got["documents"])
        return [
            {"id": i, "chunk": d, "metadata": m or {}}
            for i, d, m in zip(got["ids"], got["documents"], metas)
        ]

    def export(self, limit: Optional[int] = None, offset: Optional[int] = None) -> list[dict]:
        """Stored chunks WITH their vectors, for moving into another store.

        Returns [{"id", "chunk", "metadata", "embedding"}]. Added for the Week 4
        practical, which copies chunks into Neo4j: the vectors are already paid
        for, so copying them costs nothing against the embedding quota, while
        re-embedding the same chunks would cost the full ingest again.

        Page through a large collection with `limit` and `offset`: 3,000 chunks
        of 3072 numbers each is roughly 300 MB as Python floats, all at once.
        """
        got = self._collection.get(
            limit=limit, offset=offset, include=["documents", "metadatas", "embeddings"]
        )
        metas = got.get("metadatas") or [None] * len(got["ids"])
        return [
            {"id": i, "chunk": d, "metadata": m or {}, "embedding": [float(x) for x in e]}
            for i, d, m, e in zip(got["ids"], got["documents"], metas, got["embeddings"])
        ]

    def collections(self) -> list[str]:
        """Every collection name stored at this path.

        Run this when count() is 0 but ingest.py said it succeeded: almost
        always the name here and the name passed to `ingest.py --name` differ.
        """
        return [c.name for c in self._client.list_collections()]

    def add(self, chunks: list[str], metadatas: Optional[list[dict]] = None) -> None:
        """Embed every chunk once and store it. Ingestion-time work.

        Slow and rate-limited on first run -- see the module docstring. Call it
        from `ingest.py`, not from a notebook cell you will re-run.

        `metadatas`, if given, is one dict per chunk -- e.g. its source document
        and position -- stored beside it and handed back by `search()` and
        `peek()`. This is how provenance ("which document did this come from, and
        where in it") travels with a chunk. Values must be str/int/float/bool.
        """
        if metadatas is not None and len(metadatas) != len(chunks):
            raise ValueError(f"{len(chunks)} chunks but {len(metadatas)} metadatas")
        vectors = self.embedder.embed_batch(chunks, task_type="RETRIEVAL_DOCUMENT")
        start = self._collection.count()
        ids = [str(start + i) for i in range(len(chunks))]
        self._collection.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)

    def search(self, query: str, *, n_results: int = 5) -> list[dict]:
        """Embed the question and search the index. Query-time work.

        Returns a list of {"chunk": str, "distance": float, "metadata": dict,
        "id": str} ordered nearest first, where distance is 1 − cosine similarity (0 =
        identical, larger = less similar -- see week-03-lecture-plan.md §3).

        `metadata` carries whatever `ingest.py` stored with the chunk -- its
        source document, chunk index, and word offset -- so an answer can cite
        where each retrieved passage came from. It is `{}` for chunks that were
        indexed before provenance existed; re-ingest with `--reset` to populate it.
        """
        query_vector = self.embedder.embed(query, task_type="RETRIEVAL_QUERY")
        return self.search_by_vector(query_vector, n_results=n_results)

    def search_by_vector(self, query_vector: list[float], *, n_results: int = 5) -> list[dict]:
        """`search()` for a question you have already embedded. Same return shape.

        Added for the Week 4 practical, which sends ONE query vector to both Chroma
        and Neo4j, so that the only difference between the two result lists is the
        store -- not a second embedding call.
        """
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        metas = result.get("metadatas") or [[None] * len(result["documents"][0])]
        return [
            {"chunk": doc, "distance": dist, "metadata": meta or {}, "id": i}
            for doc, dist, meta, i in zip(
                result["documents"][0], result["distances"][0], metas[0], result["ids"][0]
            )
        ]


if __name__ == "__main__":
    # Smoke test: three chunks, one clearly relevant to the question, one not.
    # Its own collection, reset each run, so it never pollutes a real corpus.
    store = VectorStore(name="smoke_test", reset=True)
    store.add(
        [
            "The model can only work with what is in its context.",
            "Everything in the context window is paid for on every call.",
            "A recipe for making bread starts with mixing flour and water.",
        ],
        metadatas=[
            {"source": "context.md", "chunk_index": 0},
            {"source": "context.md", "chunk_index": 1},
            {"source": "cookbook.md", "chunk_index": 0},
        ],
    )
    for row in store.search("what does the model see?", n_results=3):
        m = row["metadata"]
        cite = f"{m.get('source', '?')}#{m.get('chunk_index', '?')}"
        print(f"  distance={row['distance']:.4f}  [{cite}]  {row['chunk']}")
