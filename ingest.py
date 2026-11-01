import os
import glob
import json
import pymupdf
from dotenv import load_dotenv
from neo4j import GraphDatabase

from chunking import chunk_text
from vector_store import VectorStore
from extract_knowledge import extract_graph_from_text

load_dotenv()
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

def extract_clean_text(pdf_path):
    """Extracts and cleans text from a PDF document."""
    doc = pymupdf.open(pdf_path)
    text = "".join([page.get_text() + " " for page in doc])
    return text.replace('\n', ' ').strip()

def clear_graph(tx):
    """Clears the graph before a fresh import."""
    tx.run("MATCH (n) DETACH DELETE n")

def import_graph_data(tx, graph_data, chunk_id):
    """Writes extracted JSON data into the Neo4j graph using MERGE."""
    
    # 1. Create Nodes
    for recipe in graph_data.get("recipes", []):
        tx.run("MERGE (r:Recipe {name: $name})", name=recipe)

    for ingredient in graph_data.get("ingredients", []):
        tx.run("MERGE (i:Ingredient {name: $name})", name=ingredient)

    for eq in graph_data.get("equipment", []):
        tx.run("MERGE (e:Equipment {name: $name})", name=eq)
        
    for cat in graph_data.get("categories", []):
        tx.run("MERGE (c:Category {name: $name})", name=cat)

    # 2. Create Relationships
    for rel in graph_data.get("contains_ingredients", []):
        tx.run("""
            MATCH (r:Recipe {name: $recipe_name})
            MATCH (i:Ingredient {name: $ingredient_name})
            MERGE (r)-[rel:HAS_INGREDIENT]->(i)
            SET rel.amount = $amount, rel.chunk_id = $chunk_id
        """, recipe_name=rel["recipe_name"], 
             ingredient_name=rel["ingredient_name"],
             amount=rel.get("amount", ""), 
             chunk_id=chunk_id)

    for rel in graph_data.get("needs_equipment", []):
        tx.run("""
            MATCH (r:Recipe {name: $recipe_name})
            MATCH (e:Equipment {name: $equipment_name})
            MERGE (r)-[rel:NEEDS_EQUIPMENT]->(e)
            SET rel.chunk_id = $chunk_id
        """, recipe_name=rel["recipe_name"], 
             equipment_name=rel["equipment_name"], 
             chunk_id=chunk_id)
             
    for rel in graph_data.get("belongs_to", []):
        tx.run("""
            MATCH (r:Recipe {name: $recipe_name})
            MATCH (c:Category {name: $category_name})
            MERGE (r)-[rel:BELONGS_TO]->(c)
            SET rel.chunk_id = $chunk_id
        """, recipe_name=rel["recipe_name"], 
             category_name=rel["category_name"], 
             chunk_id=chunk_id)

def main():
    print("Starting PDF processing...")
    pdf_files = glob.glob("data/*.pdf")
    
    chunks = []
    metadatas = []
    
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    with driver.session() as session:
        session.execute_write(clear_graph)
        print("Neo4j database cleared for fresh import.")

    for pdf_file in pdf_files:
        text = extract_clean_text(pdf_file)
        pieces = chunk_text(text, chunk_size=200, overlap=40)
        
        for i, piece in enumerate(pieces):
            chunk_id = f"{os.path.basename(pdf_file)}_chunk_{i}"
            
            chunks.append(piece)
            metadatas.append({
                "source": os.path.basename(pdf_file),
                "chunk_index": i,
                "chunk_id": chunk_id
            })
            
            print(f"Extracting graph entities from {chunk_id}...")
            try:
                json_result = extract_graph_from_text(piece, chunk_id)
                graph_data = json.loads(json_result)
                
                with driver.session() as session:
                    session.execute_write(import_graph_data, graph_data, chunk_id)
            except Exception as e:
                print(f"Error extracting/importing graph for {chunk_id}: {e}")
                
        print(f"Processed {os.path.basename(pdf_file)}: {len(pieces)} chunks.")

    driver.close()

    if not chunks:
        print("No texts found for import. Check the 'data' folder.")
        return

    store = VectorStore(name="tech_knowledge_base", reset=True)
    print(f"\nImporting {len(chunks)} vectors into ChromaDB. This may take a moment...")
    store.add(chunks=chunks, metadatas=metadatas)
    print(f"Done! The ChromaDB now contains {store.count()} chunks. Neo4j graph is also populated.")

if __name__ == "__main__":
    main()