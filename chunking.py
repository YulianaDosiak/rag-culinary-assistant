"""Week 3 — splitting a corpus into retrievable pieces.

    from chunking import chunk_text

A whole document embeds to one vector, which can only say "this document,
broadly, is relevant" — it cannot say which paragraph. Retrieval needs to point
at the part that answers the question, which is why the corpus is chunked
before anything is embedded or indexed.

This is a plain word-count sliding window, deliberately not sentence-aware: it
is the simplest chunker that demonstrates the actual trade-off week-03-lecture-
plan.md names — too small loses the context that gave a chunk meaning, too
large dilutes it with unrelated material, and overlap mitigates whatever is
lost at a boundary. A production chunker would respect sentence or paragraph
boundaries; this one does not, on purpose, so the boundary effect stays visible
rather than being engineered away.
"""


def chunk_text(text: str, *, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping chunks of `chunk_size` words.

    `overlap` words at the end of one chunk are repeated at the start of the
    next, so a sentence split across a boundary still appears whole in at
    least one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


if __name__ == "__main__":
    # Smoke test: a short passage, chunked small enough to show boundaries
    # and overlap clearly. Real corpora use a much larger chunk_size.
    sample = (
        "The model can only work with what is in its context. "
        "Everything in the window is paid for on every call. "
        "Retrieval is how software finds the right passage without the "
        "developer already knowing which one answers the question. "
        "A chunk that is too small loses the surrounding context that gave "
        "it meaning. A chunk that is too large dilutes relevance with "
        "unrelated material sitting in the same piece."
    )

    chunks = chunk_text(sample, chunk_size=12, overlap=4)
    print(f"{len(sample.split())} words -> {len(chunks)} chunks of up to 12 words, 4-word overlap\n")
    for i, c in enumerate(chunks):
        print(f"[{i}] {c}")
