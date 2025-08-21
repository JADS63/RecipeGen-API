# src/infer.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List
from src.utils import build_prompt

class RecipeGenerator:
    def __init__(self, model_dir: str = "artifacts/model"):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()

    def generate(self, ingredients: List[str], max_new_tokens: int = 180) -> str:
        """
        Génère les étapes d'une recette à partir d'une liste d'ingrédients.
        """
        prompt = build_prompt(ingredients)
        inputs = self.tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.9,
                top_p=0.92,
                repetition_penalty=1.15,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.eos_token_id
            )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        # On retire le prompt pour garder seulement la partie "Recipe:"
        return text.split("Recipe:\n", 1)[-1].strip()

if __name__ == "__main__":
    gen = RecipeGenerator()
    print(gen.generate(["chicken", "garlic", "lemon", "olive oil"]))
