import json
from pathlib import Path

import pandas as pd

from src.datasets.cicids2017 import CICIDS2017_FEATURES


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/benchmarks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


datasets = {
    "CICIDS2017": pd.read_pickle(
        DATA_DIR / "cicids2017_sample.pkl"
    ),
    "CSE-CIC-IDS2018": pd.read_pickle(
        DATA_DIR / "cicids2018_sample.pkl"
    ),
}


results = {}

for dataset_name, dataframe in datasets.items():
    results[dataset_name] = {}

    for label in ["BENIGN", "ATTACK"]:
        subset = dataframe[
            dataframe["Label"] == label
        ]

        feature_results = {}

        for feature in CICIDS2017_FEATURES:
            values = pd.to_numeric(
                subset[feature],
                errors="coerce",
            )

            zero_count = int(
                (values == 0).sum()
            )

            total = int(len(values))

            zero_rate = (
                zero_count / total
                if total
                else 0.0
            )

            feature_results[feature] = {
                "total": total,
                "zero_count": zero_count,
                "zero_rate": float(zero_rate),
            }

        results[dataset_name][label] = feature_results


comparison = []

for label in ["BENIGN", "ATTACK"]:
    for feature in CICIDS2017_FEATURES:
        rate_2017 = results[
            "CICIDS2017"
        ][label][feature]["zero_rate"]

        rate_2018 = results[
            "CSE-CIC-IDS2018"
        ][label][feature]["zero_rate"]

        comparison.append({
            "label": label,
            "feature": feature,
            "zero_rate_2017": rate_2017,
            "zero_rate_2018": rate_2018,
            "absolute_gap": abs(
                rate_2018 - rate_2017
            ),
        })


comparison.sort(
    key=lambda item: item["absolute_gap"],
    reverse=True,
)


output = {
    "method": (
        "Descriptive zero-value rate comparison "
        "using balanced 100k-per-class samples."
    ),
    "zero_rates": results,
    "comparison": comparison,
}


output_path = (
    OUTPUT_DIR
    / "feature_zero_rate_shift.json"
)

output_path.write_text(
    json.dumps(
        output,
        indent=2,
        sort_keys=True,
    )
)


print("\n=== LARGEST ZERO-RATE SHIFTS ===")

for item in comparison:
    print(
        f"{item['label']:6} | "
        f"{item['feature']:28} | "
        f"2017={item['zero_rate_2017']:7.2%} | "
        f"2018={item['zero_rate_2018']:7.2%} | "
        f"gap={item['absolute_gap']:7.2%}"
    )


print(
    "\nResults saved to:",
    output_path,
)
