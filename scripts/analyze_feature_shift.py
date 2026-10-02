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


def summarize(dataframe):
    result = {}

    for label in ["BENIGN", "ATTACK"]:
        subset = dataframe[
            dataframe["Label"] == label
        ]

        result[label] = {}

        for feature in CICIDS2017_FEATURES:
            values = pd.to_numeric(
                subset[feature],
                errors="coerce",
            )

            result[label][feature] = {
                "median": float(values.median()),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
            }

    return result


summary = {
    name: summarize(dataframe)
    for name, dataframe in datasets.items()
}


comparison = []

for label in ["BENIGN", "ATTACK"]:
    for feature in CICIDS2017_FEATURES:
        value_2017 = summary[
            "CICIDS2017"
        ][label][feature]["median"]

        value_2018 = summary[
            "CSE-CIC-IDS2018"
        ][label][feature]["median"]

        denominator = max(
            abs(value_2017),
            abs(value_2018),
            1.0,
        )

        normalized_median_gap = (
            abs(value_2018 - value_2017)
            / denominator
        )

        comparison.append({
            "label": label,
            "feature": feature,
            "median_2017": value_2017,
            "median_2018": value_2018,
            "normalized_median_gap": float(
                normalized_median_gap
            ),
        })


comparison.sort(
    key=lambda item: item[
        "normalized_median_gap"
    ],
    reverse=True,
)


output = {
    "method": (
        "Descriptive comparison of sampled feature "
        "distributions. normalized_median_gap is a "
        "project-specific descriptive statistic, not "
        "a formal statistical distance metric."
    ),
    "summary": summary,
    "comparison": comparison,
}


output_path = (
    OUTPUT_DIR
    / "feature_distribution_shift.json"
)

output_path.write_text(
    json.dumps(
        output,
        indent=2,
        sort_keys=True,
    )
)


print(
    "\n=== LARGEST FEATURE MEDIAN SHIFTS ==="
)

for item in comparison:
    print(
        f"{item['label']:6} | "
        f"{item['feature']:28} | "
        f"2017={item['median_2017']:.3f} | "
        f"2018={item['median_2018']:.3f} | "
        f"gap={item['normalized_median_gap']:.3f}"
    )

print(
    "\nResults saved to:",
    output_path,
)
