// CQ1: Which recipes contain peanut butter, and how much?
MATCH (r:Recipe)-[h:HAS_INGREDIENT]->(:Ingredient {name: 'peanut butter'})
RETURN r.name, h.amount, h.chunk_ids;

// CQ2: What equipment does a recipe need?
MATCH (:Recipe {name: 'vietnamese fish cakes with vermicelli salad'})-[n:NEEDS_EQUIPMENT]->(e:Equipment)
RETURN e.name, n.chunk_ids;

// CQ3: Which recipes contain BOTH tuna and mayonnaise?
MATCH (r:Recipe)-[:HAS_INGREDIENT]->(:Ingredient {name: 'tuna'}),
      (r)-[:HAS_INGREDIENT]->(:Ingredient {name: 'mayonnaise'})
RETURN r.name;

// CQ4: Which ingredients do two recipes share?
MATCH (a:Recipe {name: 'fish tacos with tomato & jalapeno salsa'})-[:HAS_INGREDIENT]->(i:Ingredient)<-[:HAS_INGREDIENT]-(b:Recipe {name: 'lentil tacos'})
RETURN i.name;

// CQ5: Which desserts need an oven?
MATCH (r:Recipe)-[:BELONGS_TO]->(:Category {name: 'dessert'}),
      (r)-[:NEEDS_EQUIPMENT]->(:Equipment {name: 'oven'})
RETURN r.name;