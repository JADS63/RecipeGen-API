# RecipeGen API — Générateur de recettes (IA locale)

Projet court et démonstratif : une **API locale** qui génère des **étapes de recette** à partir d’une **liste d’ingrédients**.  
Techno : Python, PyTorch, Hugging Face Transformers, FastAPI.

##  Fonctionnement (vue d'ensemble)
1) **Datasets** : téléchargement de RecipeNLG (Hugging Face).
2) **Préparation** : format `Ingredients: ... \n\nRecipe:\n ...`.
3) **Entraînement** (fine-tuning) d’un petit modèle (**distilGPT-2**) pour générer les **instructions**.
4) **Servir l’API** (FastAPI) → endpoint `/generate`.

##  Installation rapide
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
