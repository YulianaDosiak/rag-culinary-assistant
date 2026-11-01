import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from vector_store import VectorStore

load_dotenv()
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
driver = GraphDatabase.driver(URI, auth=AUTH)

def get_graph_context(recipe_name: str) -> str:
    """Retrieves exact facts about a recipe from the Neo4j graph."""
    query = """
    MATCH (r:Recipe {name: $recipe_name})
    OPTIONAL MATCH (r)-[rel_i:HAS_INGREDIENT]->(i:Ingredient)
    OPTIONAL MATCH (r)-[rel_e:NEEDS_EQUIPMENT]->(e:Equipment)
    OPTIONAL MATCH (r)-[rel_c:BELONGS_TO]->(c:Category)
    RETURN r.name AS Recipe, 
           collect(DISTINCT i.name) AS Ingredients, 
           collect(DISTINCT e.name) AS Equipment,
           collect(DISTINCT c.name) AS Categories
    """
    with driver.session() as session:
        result = session.run(query, recipe_name=recipe_name).single()
        
    if not result:
        return ""
        
    ingredients = ", ".join(result["Ingredients"]) if result["Ingredients"] else "none specified"
    equipment = ", ".join(result["Equipment"]) if result["Equipment"] else "none specified"
    categories = ", ".join(result["Categories"]) if result["Categories"] else "none specified"
    
    return f"\n* Graph Knowledge for '{recipe_name}':\n  - Ingredients: {ingredients}\n  - Equipment needed: {equipment}\n  - Categories: {categories}"

def hybrid_search(query: str, store: VectorStore, top_k: int = 5) -> str:
    """Hybrid search: searches via vector chunks first, then fetches exact graph facts."""
    results = store.search(query, n_results=top_k)
    
    context_texts = []
    recipes_to_lookup = set()
    
    for row in results:
        chunk_text = row['chunk']
        source = row['metadata'].get('source', 'Unknown')
        context_texts.append(f"[{source}]: {chunk_text}")
        
        if "Peanut Butter" in chunk_text or "Overnight Oats" in chunk_text:
            recipes_to_lookup.add("Peanut Butter Overnight Oats")
        if "Borscht" in chunk_text:
            recipes_to_lookup.add("Borscht")
            
    vector_context = "\n\n".join(context_texts)
    
    graph_context = ""
    for recipe in recipes_to_lookup:
        graph_context += get_graph_context(recipe)
        
    final_context = "=== VECTOR CHUNKS ===\n" + vector_context
    if graph_context:
        final_context += "\n\n=== NEO4J GRAPH FACTS ===" + graph_context
        
    return final_context