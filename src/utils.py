# src/utils.py
import re
from typing import List

UNIT_RE = re.compile(r"\b\d+([.,]\d+)?\s*(g|gram|grams|kg|ml|l|cup|cups|tbsp|tsp|oz|lb|pound)s?\b", re.I)

def normalize_ingredient(s: str) -> str:
    """
    Nettoyage léger d'un ingrédient: minuscules, retrait des unités/chiffres,
    espaces compressés. Objectif: format stable pour le prompt.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = UNIT_RE.sub("", s)
    s = re.sub(r"[\d/.-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def build_prompt(ingredients: List[str]) -> str:
    """Construit le prompt: bloc Ingredients suivi du marqueur Recipe:."""
    clean = [normalize_ingredient(x) for x in ingredients if isinstance(x, str) and x.strip()]
    body = "\n".join(f"- {x}" for x in clean[:30])  # borne pour éviter les prompts énormes
    return f"Ingredients:\n{body}\n\nRecipe:\n"
