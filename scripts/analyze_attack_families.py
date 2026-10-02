import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml.benchmark import (
    build_random_forest,
    prepare_xy,
)


RANDOM_SEED = 42
TEST_SIZE = 0.20

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/benchmarks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading datasets...")

data_2017 = pd.read_pickle(
    DATA_DIR / "cicids2017_sample.pkl"
)

data_2018 = pd.read_pickle(
    DATA_DIR / "cicids2018_sample.pkl"
)


X_2017, y_2017 = prepare_xy(data_2017)
X_2018, y_2018 = prepare_xy(data_2018)


X_train, _, y_train, _ = train_test_split(
    X_2017,
    y_2017,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_2017,
)


print("Training Random Forest...")

model = build_random_forest(
    random_seed=RANDOM_SEED,
)

model.fit(
    X_train,
    y_train,
)


print("Predicting CSE-CIC-IDS2018...")

predictions = model.predict(X_2018)

analysis = data_2018[
    [
        "Label",
        "original_label",
        "source_file",
    ]
].copy()

analysis["prediction"] = predictions


attack_rows = analysis[
    analysis["Label"] == "ATTACK"
].copy()


results = []

for attack_label, group in attack_rows.groupby(
    "original_label"
):
    total = len(group)

    detected = int(
        (group["prediction"] == 1).sum()
    )

    missed = int(
        (group["prediction"] == 0).sum()
    )

    detection_rate = (
        detected / total
        if total
        else 0.0
    )

    results.append({
        "attack_label": str(attack_label),
        "total": int(total),
        "detected": detected,
        "missed": missed,
        "detection_rate": float(detection_rate),
    })


results.sort(
    key=lambda item: (
        item["detection_rate"],
        -item["total"],
    )
)


output = {
    "experiment": "CICIDS2017_to_CSE-CIC-IDS2018_attack_family_analysis",
    "random_seed": RANDOM_SEED,
    "attack_sample_size": int(
        len(attack_rows)
    ),
    "families": results,
}


output_path = (
    OUTPUT_DIR
    / "rf_2017_to_2018_attack_families.json"
)

output_path.write_text(
    json.dumps(
        output,
        indent=2,
        sort_keys=True,
    )
)


print(
    "\n=== 2018 ATTACK FAMILY DETECTION ==="
)

print(
    f"{'Attack family':35} "
    f"{'Total':>8} "
    f"{'Detected':>10} "
    f"{'Missed':>8} "
    f"{'Rate':>9}"
)

print("-" * 76)

for item in results:
    print(
        f"{item['attack_label'][:35]:35} "
        f"{item['total']:8d} "
        f"{item['detected']:10d} "
        f"{item['missed']:8d} "
        f"{item['detection_rate']:8.2%}"
    )


print(
    "\nResults saved to:",
    output_path,
)
