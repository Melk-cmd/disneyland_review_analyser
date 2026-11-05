"""
Disneyland Review Analyser
--------------------------
Prédit la note (1 à 5 étoiles) d'un avis français sur Disneyland Paris
à partir de son texte, avec TF-IDF + régression logistique (scikit-learn).

Exemple :
    python train.py --C 1.0 --max_features 5000 --balanced 1
"""
import argparse
import time

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_URL = (
    "https://full-stack-assets.s3.eu-west-3.amazonaws.com/images/"
    "M08-DeepLearning/NLP/french_review_clean.csv"
)
TEXT_COL = "review_format_clean"   # texte déjà nettoyé (mots simplifiés, mots vides retirés)
TARGET_COL = "stars"               # note de 1 à 5
SEED = 42


# --------------------------------------------------------------------------
# 1. Paramètres de la ligne de commande
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_url", type=str, default=DATA_URL)
    parser.add_argument("--C", type=float, default=1.0,
                        help="Régularisation : petit = modèle plus simple, grand = colle davantage aux données")
    parser.add_argument("--max_features", type=int, default=5000,
                        help="Nombre maximum de mots/expressions gardés")
    parser.add_argument("--ngram_max", type=int, default=2, choices=[1, 2, 3],
                        help="1 = mots seuls, 2 = mots seuls + paires de mots, ...")
    parser.add_argument("--balanced", type=int, default=0, choices=[0, 1],
                        help="1 = donner plus de poids aux notes rares")
    return parser.parse_args()


# --------------------------------------------------------------------------
# 2. Chargement, analyse et nettoyage des données
# --------------------------------------------------------------------------
def load_and_clean(url):
    df = pd.read_csv(url)
    stats = {"rows_raw": len(df)}

    print("=== ANALYSE DES DONNÉES BRUTES ===")
    print(f"Lignes : {len(df)} | Colonnes : {list(df.columns)}")
    print("Valeurs manquantes par colonne :")
    print(df.isna().sum().to_string())
    print("Répartition des notes (%) :")
    print((df[TARGET_COL].value_counts(normalize=True).sort_index() * 100).round(1).to_string())

    # --- Nettoyage ---------------------------------------------------------
    # a) textes manquants (avis entièrement vidés par le nettoyage amont)
    df = df.dropna(subset=[TEXT_COL])
    # b) espaces en trop, textes vides
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""]
    # c) doublons : un même texte ne doit pas se retrouver à la fois dans
    #    l'entraînement et dans le test (sinon le score serait trop beau)
    n_before = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL])
    stats["duplicates_removed"] = n_before - len(df)
    # d) notes valides uniquement
    df = df[df[TARGET_COL].isin(range(1, 6))]
    df = df.reset_index(drop=True)
    stats["rows_clean"] = len(df)

    n_words = df[TEXT_COL].str.split().str.len()
    stats["words_mean"] = float(n_words.mean())
    stats["baseline_accuracy"] = float(df[TARGET_COL].value_counts(normalize=True).max())

    print("\n=== APRÈS NETTOYAGE ===")
    print(f"Lignes gardées : {stats['rows_clean']} (sur {stats['rows_raw']})")
    print(f"Doublons retirés : {stats['duplicates_removed']}")
    print(f"Mots par avis (moyenne) : {stats['words_mean']:.1f}")
    print(f"Précision d'un modèle qui répond toujours la note la plus fréquente : {stats['baseline_accuracy']:.3f}")
    return df, stats


# --------------------------------------------------------------------------
# 3. Le modèle : texte -> chiffres (TF-IDF) -> régression logistique
# --------------------------------------------------------------------------
def build_model(args):
    # TF-IDF : pour chaque avis, un nombre par mot du vocabulaire.
    # Un mot fréquent dans l'avis mais rare dans l'ensemble des avis pèse lourd.
    text_to_numbers = ColumnTransformer([
        ("tfidf",
         TfidfVectorizer(max_features=args.max_features, ngram_range=(1, args.ngram_max)),
         TEXT_COL),
    ])
    classifier = LogisticRegression(
        C=args.C,
        max_iter=1000,
        class_weight="balanced" if args.balanced else None,
        random_state=SEED,
    )
    return Pipeline([("text_to_numbers", text_to_numbers), ("classifier", classifier)])


# --------------------------------------------------------------------------
# 4. Programme principal
# --------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()

    # Pas de mlflow.set_experiment() ici : avec `mlflow run`, c'est MLflow qui crée le run.
    # On choisit l'expérience dans la commande : mlflow run ... --experiment-name <nom>
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        df, stats = load_and_clean(args.data_url)
        mlflow.log_metrics({
            "rows_clean": stats["rows_clean"],
            "duplicates_removed": stats["duplicates_removed"],
            "baseline_accuracy": stats["baseline_accuracy"],
        })

        X = df[[TEXT_COL]]
        y = df[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        print(f"\nEntraînement : {len(X_train)} avis | Test : {len(X_test)} avis")

        model = build_model(args)

        print("\ntraining model...")
        model.fit(X_train, y_train)

        # --- Évaluation sur des avis jamais vus ---------------------------
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, average="macro")
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        mlflow.log_metrics({
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "test_f1_macro": test_f1_macro,
            "gain_vs_baseline": test_accuracy - stats["baseline_accuracy"],
        })
        print(f"\nPrécision (entraînement) : {train_accuracy:.3f}")
        print(f"Précision (test)         : {test_accuracy:.3f}  (référence : {stats['baseline_accuracy']:.3f})")
        print(f"F1 macro (test)          : {test_f1_macro:.3f}")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="disneyland_review_analyser",
            signature=infer_signature(X_train, model.predict(X_train)),
        )

        total_time = time.time() - start_time
        mlflow.log_metric("total_training_time_s", total_time)

    print("...Done!")
    print(f"---Total training time: {total_time:.1f}s")
