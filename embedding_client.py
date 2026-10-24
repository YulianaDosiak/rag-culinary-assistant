"""One interface for embeddings: Gemini (default) or OpenRouter.

    from embedding_client import EmbeddingClient
    emb = EmbeddingClient()                          # reads EMBEDDING_PROVIDER from .env
    emb = EmbeddingClient(provider="openrouter")      # or choose one explicitly

    doc_vec = emb.embed("The model can only work with what is in its context.",
                         task_type="RETRIEVAL_DOCUMENT")
    query_vec = emb.embed("what does the model see?", task_type="RETRIEVAL_QUERY")
    print(EmbeddingClient.cosine(doc_vec, query_vec))

`gemini` is the default. It is the only backend here on which `task_type` does
anything, and the query/document asymmetry is Week 3 material -- a default that
silently ignored the parameter would teach a mechanism the student's own code did
not implement.

THE RATE LIMIT IS THE THING TO KNOW -- measured 2026-09-14
------------------------------------------------------------------------------
`gemini-embedding-001` allows roughly **100 texts per minute** on the free tier,
and the quota counts TEXTS, NOT REQUESTS. Batching does not raise it: a cold batch
of 100 succeeds, and 50+50 is followed immediately by 429.

This corrects an earlier reading (2026-09-13) that measured 40 single-text calls
in 11.9s and concluded the ceiling was far above generation's. It is far above in
requests and roughly comparable in items.

Consequence: a 3,000-chunk corpus is about 30 minutes of wall-clock ingest.
`embed_batch()` below therefore paces itself rather than failing -- see `_throttle`.
Embed a corpus ONCE, into a persistent store, using `ingest.py`. Do not re-embed
from a notebook cell on every kernel restart.

Lightning AI is not an option here: it has no embeddings API at all, checked three
ways on 2026-09-13 -- see the note kept in `llm_client.py`. OpenRouter does have a
real one, verified live before being wired in as the second backend.

Capability matrix -- verified 2026-09-13, rate limit re-measured 2026-09-14
------------------------------------------------------------------------------
                Gemini (default)             OpenRouter
Models          gemini-embedding-001         any of theirs; default here
                pinned (see below)            is baai/bge-m3
Batching        Requires wrapping each       Native: `input` as a list of
                text as its own Content      strings returns one vector per
                -- a flat list[str] is       string, correctly, out of the
                silently treated as ONE      box
                document's parts and
                returns ONE vector for
                many inputs (see
                llm_client.py's sibling
                note -- same trap)
Matryoshka      output_dimensionality=N,     dimensions=N, confirmed:
truncation      confirmed exact               requested length returned
task_type       CONFIRMED working: doc       ACCEPTED, CONFIRMED A NO-OP:
(query/document and query embeddings of      routing the SAME model through
asymmetry)      identical text are           OpenRouter with task_type set
                different vectors            produced cosine similarity
                                              0.999999999 between
                                              RETRIEVAL_DOCUMENT and
                                              RETRIEVAL_QUERY -- i.e.
                                              identical vectors. Code that
                                              depends on the asymmetry must
                                              use the Gemini backend.
Rate limit      ~100 texts/minute, counted   Not observed in testing
                per text not per request      (10 calls in 5s)
                (2026-09-14)

Context length -- unresolved, do not trust either figure without re-checking
------------------------------------------------------------------------------
OpenRouter's own model listing reports gemini-embedding-001 at 20K tokens and
gemini-embedding-2 at 8K. Google's model metadata, read directly from the API,
reports 2048 and 8192 respectively -- neither figure matches. Sending a ~3000-token
input to gemini-embedding-001 both directly and via OpenRouter succeeded both times
without error, which does not resolve the discrepancy: it suggests the stated
"2048" may not be a hard, enforced rejection (possibly silent truncation instead
of a clear failure), rather than confirming either source's number. Re-verify with
a controlled token count before teaching a specific input limit for this model.
"""

import collections
import math
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# The free-tier embedding quota is ~100 items per minute (measured 2026-09-14,
# reconfirmed by a 429 on 2026-09-19). The metric is named "...Requests..." but a
# batched embed_content bills one unit PER TEXT, so batching does not raise it.
#
# Crucially it is a ROLLING 60-second window, not a fixed one. An earlier version
# here reset a fixed window and burst the cap at the start of each -- two windows
# back to back then put ~2x the cap inside one rolling minute and earned a 429.
# `_throttle` now looks at the trailing 60 seconds. `_EMBED_SAFE` stays a margin
# under the ceiling, and `_embed_batch_gemini` still retries on 429 as a backstop.
_EMBED_ITEMS_PER_MIN = 100
_EMBED_SAFE = 90
_EMBED_WINDOW = 60.0


def _retry_delay_seconds(exc) -> float:
    """Seconds to wait after a 429, from the server's own 'retry in Ns' hint."""
    m = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(m.group(1)) + 1.0 if m else 50.0


class EmbeddingClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or os.environ.get("EMBEDDING_PROVIDER", "gemini")).lower()
        self.calls = 0
        self.texts_embedded = 0
        self._recent: collections.deque = collections.deque()  # (timestamp, item_count)

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openrouter":
            self._init_openrouter()
        else:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER {self.provider!r}. Use 'gemini' or 'openrouter'."
            )

    def _items_in_window(self) -> int:
        """How many items were embedded in the trailing 60 seconds."""
        cutoff = time.time() - _EMBED_WINDOW
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()
        return sum(count for _, count in self._recent)

    def _throttle(self, n: int) -> None:
        """Wait until embedding n more items stays within the rolling per-minute cap.

        The Gemini quota counts items, not requests, so batching harder does not
        help -- only waiting does. This keeps a long ingest running unattended
        rather than dying on a 429.
        """
        if self.provider != "gemini":
            return
        while self._items_in_window() + n > _EMBED_SAFE and self._recent:
            # Wait for the oldest batch to fall out of the trailing window.
            wait = _EMBED_WINDOW - (time.time() - self._recent[0][0]) + 0.5
            if wait <= 0:
                continue
            print(f"  [embedding quota: {self._items_in_window()} items in the last "
                  f"minute, waiting {wait:.0f}s]")
            time.sleep(wait)

    # ------------------------------------------------------------ gemini
    def _init_gemini(self) -> None:
        from google import genai

        # Pinned over the newer gemini-embedding-2 specifically for task_type
        # support -- see the module docstring and week-03-lecture-plan.md's
        # "Notes for the syllabus". Not a default-to-latest choice.
        self.model = "gemini-embedding-2"
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _embed_batch_gemini(self, texts, task_type, output_dimensionality):
        from google.genai import errors, types

        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=output_dimensionality)
        # `_throttle` should keep us under the quota, but honour a 429 as a
        # backstop -- other callers on the same key, or a window edge, can still
        # trip it. Wait the server's stated delay and retry rather than crash.
        for attempt in range(1, 6):
            try:
                response = self._client.models.embed_content(
                    model=self.model, contents=contents, config=config
                )
                self.calls += 1
                return [e.values for e in response.embeddings]
            except errors.ClientError as exc:
                if getattr(exc, "code", None) != 429 or attempt == 5:
                    raise
                delay = _retry_delay_seconds(exc)
                print(f"  [429 embedding quota; waiting {delay:.0f}s, retry {attempt}/4]")
                time.sleep(delay)

    # ---------------------------------------------------------- openrouter
    def _init_openrouter(self) -> None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("No OPENROUTER_API_KEY in .env  (openrouter.ai -> Keys)")
        # BGE-M3: multilingual, cheap, and the model week-03-lecture-plan.md §7
        # already names -- chosen for pedagogical alignment, not just price.
        self.model = "baai/bge-m3"
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self._url = "https://openrouter.ai/api/v1/embeddings"

    def _embed_batch_openrouter(self, texts, task_type, output_dimensionality):
        import requests

        # task_type is deliberately NOT sent: confirmed a no-op on this backend
        # even for the same underlying model (see module docstring). Sending a
        # field that provably does nothing would misrepresent what the call did.
        body = {"model": self.model, "input": texts}
        if output_dimensionality is not None:
            body["dimensions"] = output_dimensionality

        response = requests.post(self._url, headers=self._headers, json=body)
        response.raise_for_status()
        self.calls += 1
        return [row["embedding"] for row in response.json()["data"]]

    # ---------------------------------------------------------------- api
    def embed(
        self,
        text: str,
        *,
        task_type: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ) -> list[float]:
        """One text in, one vector out.

        task_type: "RETRIEVAL_DOCUMENT" for a chunk being stored, "RETRIEVAL_QUERY"
        for a question at search time. Only affects the result on the gemini
        backend -- see the module docstring's capability matrix. Accepted on both
        backends so the same call works regardless of EMBEDDING_PROVIDER.

        output_dimensionality: Matryoshka truncation, e.g. 768 or 256 instead of
        the model's native size. Confirmed exact on both backends.
        """
        return self.embed_batch([text], task_type=task_type, output_dimensionality=output_dimensionality)[0]

    def embed_batch(
        self,
        texts: list[str],
        *,
        task_type: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ) -> list[list[float]]:
        """Several texts in, one vector per text out, in the same order.

        On Gemini this paces itself against the rolling per-minute item cap, so a
        corpus-sized call completes slowly rather than raising.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_SAFE):
            window = texts[start : start + _EMBED_SAFE]
            self._throttle(len(window))

            if self.provider == "gemini":
                got = self._embed_batch_gemini(window, task_type, output_dimensionality)
            else:
                got = self._embed_batch_openrouter(window, task_type, output_dimensionality)

            if len(got) != len(window):
                raise RuntimeError(
                    f"sent {len(window)} texts, got {len(got)} vectors back -- "
                    "the batching call shape has changed; see the module docstring."
                )
            vectors.extend(got)
            if self.provider == "gemini":
                self._recent.append((time.time(), len(window)))
            self.texts_embedded += len(window)

        return vectors

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        """The metric this course uses. See week-03-lecture-plan.md §3 for the
        definition this implements: dot product over the product of magnitudes.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b)


if __name__ == "__main__":
    # Smoke test: confirms the active backend's key and API both work, and
    # demonstrates the query/document asymmetry where the backend supports it.
    emb = EmbeddingClient()
    doc = emb.embed("The model can only work with what is in its context.",
                     task_type="SEMANTIC_SIMILARITY")
    query = emb.embed("Context is everything LLM can see", task_type="SEMANTIC_SIMILARITY")
    query_ukr = emb.embed("Модель працює тільки з контекстом", task_type="SEMANTIC_SIMILARITY")
        
    unrelated = emb.embed("fox is an animal", task_type="SEMANTIC_SIMILARITY")

    print(f"embedding client ready -- provider={emb.provider}, model={emb.model}, dims={len(doc)}")
    print(f"cosine(relevant doc, query)   = {EmbeddingClient.cosine(doc, query):.4f}")
    print(f"cosine(query_ukr, query)      = {EmbeddingClient.cosine(doc, query_ukr):.4f}")
    print(f"cosine(unrelated doc, query)  = {EmbeddingClient.cosine(doc, unrelated):.4f}")
    print(f"calls made: {emb.calls}")
