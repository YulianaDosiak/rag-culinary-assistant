import os, re, glob, json, hashlib
from dotenv import load_dotenv
from neo4j import GraphDatabase

from chunking import chunk_text
from vector_store import VectorStore
from extract_knowledge import extract_graph_from_text, clean_extraction, new_report

load_dotenv()
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

CACHE_FILE = "extraction_cache.json"
REPORT_FILE = "extraction_report.json"
LABELS = ["Recipe", "Ingredient", "Equipment", "Category"]


def clean_text(text: str) -> str:
    text = re.sub(r"Файл\s*\d+:\s*\S+\.pdf", " ", text)   # leftover artefact from copying PDFs
    return re.sub(r"\s+", " ", text).strip()


def setup_schema(session):
    for label in LABELS:
        session.run(f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.name IS UNIQUE")
    session.run("CREATE CONSTRAINT document_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE")
    session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")


def _link(tx, rel_type, label, recipe, name, cid, amount=""):
    """MERGE both ends and the relation; accumulate chunk ids without duplicates."""
    tx.run(f"""
        MATCH (c:Chunk {{id: $cid}})
        MERGE (r:Recipe {{name: $recipe}})
        MERGE (n:{label} {{name: $name}})
        MERGE (r)-[x:{rel_type}]->(n)
        SET x.chunk_ids = CASE WHEN $cid IN coalesce(x.chunk_ids, []) THEN x.chunk_ids
                               ELSE coalesce(x.chunk_ids, []) + [$cid] END
        MERGE (c)-[:MENTIONS]->(r)
        MERGE (c)-[:MENTIONS]->(n)
    """, cid=cid, recipe=recipe, name=name)
    if amount:
        tx.run(f"""
            MATCH (:Recipe {{name: $recipe}})-[x:{rel_type}]->(:{label} {{name: $name}})
            SET x.amount = $amount
        """, recipe=recipe, name=name, amount=amount)


def import_chunk(tx, source, chunk_id, index, text, data):
    tx.run("""
        MERGE (d:Document {name: $source})
        MERGE (c:Chunk {id: $cid})
        SET c.text = $text, c.index = $index, c.source = $source
        MERGE (d)-[:HAS_CHUNK]->(c)
    """, source=source, cid=chunk_id, text=text, index=index)
    for r in data["recipes"]:
        tx.run("""
            MATCH (c:Chunk {id: $cid})
            MERGE (r:Recipe {name: $name})
            MERGE (c)-[:MENTIONS]->(r)
        """, cid=chunk_id, name=r)
    for rel in data["ingredients"]:
        _link(tx, "HAS_INGREDIENT", "Ingredient", rel["recipe"], rel["name"], chunk_id, rel["amount"])
    for rel in data["equipment"]:
        _link(tx, "NEEDS_EQUIPMENT", "Equipment", rel["recipe"], rel["name"], chunk_id)
    for rel in data["categories"]:
        _link(tx, "BELONGS_TO", "Category", rel["recipe"], rel["name"], chunk_id)


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    print("Reading and chunking documents...")
    cache = load_cache()
    report = new_report()
    chunks, metadatas, graph_items = [], [], []

    for txt_file in sorted(glob.glob("data/*.txt")):
        source = os.path.basename(txt_file)
        with open(txt_file, "r", encoding="utf-8") as f:
            text = clean_text(f.read())
        pieces = chunk_text(text, chunk_size=200, overlap=40)

        for i, piece in enumerate(pieces):
            chunk_id = f"{source}_chunk_{i}"
            chunks.append(piece)
            metadatas.append({"source": source, "chunk_index": i, "chunk_id": chunk_id})

            digest = hashlib.sha1(piece.encode("utf-8")).hexdigest()
            cached = cache.get(chunk_id)
            if cached and cached["hash"] == digest:
                raw = cached["raw"]
            else:
                print(f"Extracting {chunk_id}...")
                raw = extract_graph_from_text(piece, chunk_id)
                if raw is not None:
                    cache[chunk_id] = {"hash": digest, "raw": raw}
                    with open(CACHE_FILE, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False, indent=1)
            if raw is None:
                report["failed_chunks"].append(chunk_id)
                continue
            graph_items.append((source, chunk_id, i, piece, clean_extraction(raw, chunk_id, report)))
        print(f"Processed {source}: {len(pieces)} chunks.")

    if not chunks:
        print("No texts found in 'data' folder.")
        return
    if report["failed_chunks"]:
        print(f"WARNING: extraction failed for {report['failed_chunks']} - run ingest.py again "
              "(successful chunks are cached, so it is cheap).")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2,
                  default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"Extraction report saved to {REPORT_FILE}")

    # Everything is extracted -> only now touch the database
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        setup_schema(session)
        for source, chunk_id, idx, piece, data in graph_items:
            session.execute_write(import_chunk, source, chunk_id, idx, piece, data)
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    driver.close()
    print(f"Neo4j rebuilt: {nodes} nodes, {rels} relationships.")

    store = VectorStore(name="tech_knowledge_base", reset=True)
    print(f"Embedding {len(chunks)} chunks into ChromaDB...")
    store.add(chunks=chunks, metadatas=metadatas)
    print(f"Done! ChromaDB: {store.count()} chunks.")


if __name__ == "__main__":
    main()