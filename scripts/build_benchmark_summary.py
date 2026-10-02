import json
from pathlib import Path


RESULTS_DIR = Path("results/benchmarks")

forward = json.loads(
    (RESULTS_DIR / "rf_2017_to_2018.json").read_text()
)

reverse = json.loads(
    (RESULTS_DIR / "rf_2018_to_2017.json").read_text()
)

combined = json.loads(
    (RESULTS_DIR / "rf_combined_2017_2018.json").read_text()
)


experiments = [
    {
        "training": "CICIDS2017",
        "evaluation": "CICIDS2017 holdout",
        "evaluation_type": "internal_holdout",
        **forward["cicids2017_holdout"],
    },
    {
        "training": "CICIDS2017",
        "evaluation": "CSE-CIC-IDS2018",
        "evaluation_type": "cross_dataset",
        **forward["cse_cic_ids_2018_cross_dataset"],
    },
    {
        "training": "CSE-CIC-IDS2018",
        "evaluation": "CSE-CIC-IDS2018 holdout",
        "evaluation_type": "internal_holdout",
        **reverse["cse_cic_ids_2018_holdout"],
    },
    {
        "training": "CSE-CIC-IDS2018",
        "evaluation": "CICIDS2017",
        "evaluation_type": "cross_dataset",
        **reverse["cicids2017_cross_dataset"],
    },
    {
        "training": "CICIDS2017 + CSE-CIC-IDS2018",
        "evaluation": "CICIDS2017 holdout",
        "evaluation_type": "multi_domain_holdout",
        **combined["cicids2017_holdout"],
    },
    {
        "training": "CICIDS2017 + CSE-CIC-IDS2018",
        "evaluation": "CSE-CIC-IDS2018 holdout",
        "evaluation_type": "multi_domain_holdout",
        **combined["cse_cic_ids_2018_holdout"],
    },
]


summary = {
    "model": "RandomForestClassifier",
    "features": 11,
    "sampling": {
        "strategy": "balanced binary reservoir sample",
        "per_dataset": {
            "BENIGN": 100000,
            "ATTACK": 100000,
        },
        "random_seed": 42,
    },
    "experiments": experiments,
    "interpretation": {
        "cross_dataset_finding": (
            "Internal holdout performance substantially exceeds "
            "cross-dataset performance, indicating dataset/domain shift."
        ),
        "multi_domain_finding": (
            "Training on both known datasets produces strong holdout "
            "performance on both represented domains."
        ),
        "limitation": (
            "Multi-domain holdout performance does not demonstrate "
            "generalization to a third unseen dataset or production traffic."
        ),
    },
}


output_path = RESULTS_DIR / "benchmark_summary.json"

output_path.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
)


print("\n=== BENCHMARK SUMMARY ===")

print(
    f"{'Training':32} "
    f"{'Evaluation':28} "
    f"{'F1':>8} "
    f"{'Recall':>8} "
    f"{'ROC-AUC':>8}"
)

print("-" * 90)

for item in experiments:
    print(
        f"{item['training'][:32]:32} "
        f"{item['evaluation'][:28]:28} "
        f"{item['f1']:8.4f} "
        f"{item['recall']:8.4f} "
        f"{item['roc_auc']:8.4f}"
    )

print(
    "\nSaved to:",
    output_path,
)
