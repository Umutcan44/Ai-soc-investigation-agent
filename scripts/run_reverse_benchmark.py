import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml.benchmark import (
    build_random_forest,
    evaluate_binary_classifier,
    prepare_xy,
)


RANDOM_SEED = 42
TEST_SIZE = 0.20

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/benchmarks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading sampled datasets...")

data_2017 = pd.read_pickle(
    DATA_DIR / "cicids2017_sample.pkl"
)

data_2018 = pd.read_pickle(
    DATA_DIR / "cicids2018_sample.pkl"
)

X_2017, y_2017 = prepare_xy(data_2017)
X_2018, y_2018 = prepare_xy(data_2018)


X_train, X_holdout, y_train, y_holdout = train_test_split(
    X_2018,
    y_2018,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_2018,
)


print("Training Random Forest on CSE-CIC-IDS2018...")
print("Training flows:", len(y_train))
print("2018 holdout flows:", len(y_holdout))
print("2017 cross-dataset flows:", len(y_2017))

model = build_random_forest(
    random_seed=RANDOM_SEED,
)

model.fit(
    X_train,
    y_train,
)


print("\nEvaluating CSE-CIC-IDS2018 holdout...")

internal_metrics = evaluate_binary_classifier(
    model,
    X_holdout,
    y_holdout,
)


print("Evaluating CICIDS2017 cross-dataset...")

cross_dataset_metrics = evaluate_binary_classifier(
    model,
    X_2017,
    y_2017,
)


results = {
    "experiment": "CSE-CIC-IDS2018_train_to_CICIDS2017_test",
    "random_seed": RANDOM_SEED,
    "training": {
        "dataset": "CSE-CIC-IDS2018",
        "sample_size": int(len(y_train)),
    },
    "cse_cic_ids_2018_holdout": internal_metrics,
    "cicids2017_cross_dataset": cross_dataset_metrics,
}


output_path = (
    OUTPUT_DIR
    / "rf_2018_to_2017.json"
)

output_path.write_text(
    json.dumps(
        results,
        indent=2,
        sort_keys=True,
    )
)


def print_metrics(title, metrics):
    print(f"\n=== {title} ===")
    print(f"Accuracy : {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall   : {metrics['recall']:.6f}")
    print(f"F1       : {metrics['f1']:.6f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.6f}")
    print(f"PR-AUC   : {metrics['pr_auc']:.6f}")
    print(
        "Confusion matrix:",
        metrics["confusion_matrix"],
    )


print_metrics(
    "CSE-CIC-IDS2018 INTERNAL HOLDOUT",
    internal_metrics,
)

print_metrics(
    "CSE-CIC-IDS2018 -> CICIDS2017",
    cross_dataset_metrics,
)

print(
    "\nResults saved to:",
    output_path,
)
