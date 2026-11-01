import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from llm_client import LLMClient

load_dotenv()
client = LLMClient() 

class IngredientRel(BaseModel):
    recipe_name: str
    ingredient_name: str
    amount: str = Field(description="Amount, e.g., '200g' or 'to taste'")

class EquipmentRel(BaseModel):
    recipe_name: str
    equipment_name: str

class CategoryRel(BaseModel):
    recipe_name: str
    category_name: str

class GraphExtraction(BaseModel):
    recipes: list[str] = Field(description="List of recipe names in the text")
    ingredients: list[str] = Field(description="List of basic ingredients (combine synonyms)")
    equipment: list[str] = Field(description="List of kitchen equipment")
    categories: list[str] = Field(description="List of categories, e.g., Breakfast, Dinner, Dessert")
    contains_ingredients: list[IngredientRel] = Field(description="Ingredient relationships")
    needs_equipment: list[EquipmentRel] = Field(description="Equipment relationships")
    belongs_to: list[CategoryRel] = Field(description="Category relationships")

def extract_graph_from_text(text_chunk: str, chunk_id: str):
    system_prompt = f"You are a culinary expert. Analyze this text chunk (ID: {chunk_id}) and extract all recipes, ingredients, equipment, and categories, as well as the relationships between them. If there are no recipes, return empty lists."
    
    result = client.chat(
        prompt=f"Text:\n{text_chunk}",
        system=system_prompt,
        temperature=0.0,
        schema=GraphExtraction
    )
    
    if hasattr(result, 'parsed') and isinstance(result.parsed, BaseModel):
        return result.parsed.model_dump_json()
    elif hasattr(result, 'parsed') and isinstance(result.parsed, dict):
        return json.dumps(result.parsed)
        
    return result.text