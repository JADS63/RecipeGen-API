# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from src.infer import RecipeGenerator

app = FastAPI(title="RecipeGen API", version="0.1")

# Charge le modèle à l'import de l'app
generator = RecipeGenerator(model_dir="artifacts/model")

class GenerateIn(BaseModel):
    ingredients: List[str]
    max_new_tokens: Optional[int] = 180

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateIn):
    steps = generator.generate(req.ingredients, max_new_tokens=req.max_new_tokens or 180)
    return {
        "ingredients_in": req.ingredients,
        "steps": steps
    }
