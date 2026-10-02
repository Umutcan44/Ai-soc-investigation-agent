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


train_2017, test_2017 = train_test_split(
    data_2017,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=data_2017["Label"],
)

train_2018, test_2018 = train_test_split(
    data_2018,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=data_2018["Label"],
)


combined_train = pd.concat(
    [
        train_2017,
        train_2018,
    ],
    ignore_index=True,
)

combined_train = combined_train.sample(
    frac=1.0,
    random_state=RANDOM_SEED,
).reset_index(drop=True)


X_train, y_train = prepare_xy(
    combined_train
)

X_test_2017, y_test_2017 = prepare_xy(
    test_2017
)

X_test_2018, y_test_2018 = prepare_xy(
    test_2018
)


print("Training combined Random Forest...")
print("Training flows:", len(y_train))
print("2017 holdout flows:", len(y_test_2017))
print("2018 holdout flows:", len(y_test_2018))


model = build_random_forest(
    random_seed=RANDOM_SEED,
)

model.fit(
    X_train,
    y_train,
)


print("\nEvaluating CICIDS2017 holdout...")

metrics_2017 = evaluate_binary_classifier(
    model,
    X_test_2017,
    y_test_2017,
)


print("Evaluating CSE-CIC-IDS2018 holdout...")

metrics_2018 = evaluate_binary_classifier(
    model,
    X_test_2018,
    y_test_2018,
)


results = {
    "experiment": "combined_2017_2018_training",
    "random_seed": RANDOM_SEED,
    "training": {
        "datasets": [
            "CICIDS2017",
            "CSE-CIC-IDS2018",
        ],
        "sample_size": int(len(y_train)),
        "cicids2017_training_flows": int(
            len(train_2017)
        ),
        "cse_cic_ids_2018_training_flows": int(
            len(train_2018)
        ),
    },
    "cicids2017_holdout": metrics_2017,
    "cse_cic_ids_2018_holdout": metrics_2018,
}


output_path = (
    OUTPUT_DIR
    / "rf_combined_2017_2018.json"
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
    "COMBINED MODEL -> CICIDS2017 HOLDOUT",
    metrics_2017,
)

print_metrics(
    "COMBINED MODEL -> CSE-CIC-IDS2018 HOLDOUT",
    metrics_2018,
)

print(
    "\nResults saved to:",
    output_path,
)
