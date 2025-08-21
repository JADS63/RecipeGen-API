# src/train.py
import argparse
import os
from typing import Any, Dict, List

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src.utils import normalize_ingredient  # fonctionne avec: python -m src.train

# ---------- helpers sur les colonnes ----------
def _to_ingredients_list(val: Any) -> List[str]:
    """
    Accepte soit une liste ['chicken','garlic',...], soit une chaîne 'chicken, garlic, ...'
    -> renvoie une liste normalisée.
    """
    items: List[str] = []
    if isinstance(val, list):
        items = [str(x) for x in val if isinstance(x, str)]
    elif isinstance(val, str):
        items = [t.strip() for t in val.split(",") if t.strip()]
    return [normalize_ingredient(x) for x in items if x]

def _steps_text(val: Any) -> str:
    """
    Accepte soit une liste d'étapes, soit une chaîne longue.
    -> renvoie un texte unique (phrases concaténées).
    """
    if isinstance(val, list):
        steps = [str(s).strip().replace("\n", " ") for s in val if isinstance(s, str) and s.strip()]
        return " ".join(steps)
    if isinstance(val, str):
        return val.strip().replace("\n", " ")
    return ""

def _format_record(rec: Dict[str, Any]) -> Dict[str, str]:
    # essaie d'abord paires classiques: ingredients / directions
    ings = rec.get("ingredients", [])
    dirs = rec.get("directions", None)
    # sinon, colonnes du dataset lite: ingredients / steps
    if dirs is None:
        dirs = rec.get("steps", None)

    ings_norm = _to_ingredients_list(ings)
    steps_joined = _steps_text(dirs)

    if len(ings_norm) < 3 or len(steps_joined) < 30:
        return {"text": ""}

    ingredients_block = "\n".join(f"- {x}" for x in ings_norm[:30])
    text = f"Ingredients:\n{ingredients_block}\n\nRecipe:\n{steps_joined}"
    return {"text": text}

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lite",
                        choices=["lite", "mbien"],
                        help="lite=m3hrdadfi/recipe_nlg_lite (recommandé), mbien=mbien/recipe_nlg (téléchargement manuel requis)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Obligatoire si --dataset mbien (répertoire contenant le fichier demandé par la carte de dataset).")
    parser.add_argument("--samples", type=int, default=50000, help="Nb d'exemples pour l'entraînement (subset).")
    parser.add_argument("--epochs", type=int, default=1, help="Nb d'époques d'entraînement.")
    parser.add_argument("--outdir", type=str, default="artifacts/model", help="Répertoire de sortie du modèle.")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Backbone pré-entraîné.")
    parser.add_argument("--max_length", type=int, default=512, help="Longueur max des séquences.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Charger dataset
    if args.dataset == "lite":
        # Dataset prêt à l'emploi (train/test) ; on prend train et on sous-échantillonne
        ds = load_dataset("m3hrdadfi/recipe_nlg_lite", split="train")
        # (colonnes: uid, name, description, link, ner, ingredients, steps)  cf. carte dataset.  # noqa
    else:
        # mbien/recipe_nlg exige un téléchargement manuel + data_dir
        if not args.data_dir:
            raise SystemExit(
                "❌ --data_dir est requis pour 'mbien'. Télécharge d'abord les données "
                "depuis le site du projet et passe le dossier via --data_dir. "
                "Sinon, utilise --dataset lite pour un démarrage immédiat."
            )
        ds = load_dataset("mbien/recipe_nlg", split="train", data_dir=args.data_dir)

    # 2) Sous-échantillonnage déterministe
    ds = ds.shuffle(seed=42).select(range(min(args.samples, len(ds))))

    # 3) Mise en forme texte
    ds = ds.map(_format_record)
    ds = ds.filter(lambda r: isinstance(r["text"], str) and len(r["text"]) > 0)

    # 4) Split train/val (≈2% de val, min 1000 quand possible)
    n = len(ds)
    ratio = max(0.02, (1000 / n) if n else 0.02)
    ratio = min(ratio, 0.2)
    split = ds.train_test_split(test_size=ratio, seed=7, shuffle=True)
    ds_train, ds_val = split["train"], split["test"]

    # 5) Tokenizer + modèle
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token  # GPT-2 n'a pas de pad_token

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    t_train = ds_train.map(tokenize, batched=True, remove_columns=ds_train.column_names)
    t_val   = ds_val.map(tokenize, batched=True, remove_columns=ds_val.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tok))

    # 6) Entraînement
    train_args = TrainingArguments(
        output_dir=os.path.join(args.outdir, "chkpts"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        fp16=False,  # True si GPU NVIDIA + torch CUDA installés
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=t_train,
        eval_dataset=t_val,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    # 7) Sauvegarde
    model.save_pretrained(args.outdir)
    tok.save_pretrained(args.outdir)
    print(f"✅ Modèle sauvegardé dans: {args.outdir}")

if __name__ == "__main__":
    main()
