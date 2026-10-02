from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.datasets.cicids2017 import CICIDS2017_FEATURES


def profile_dataset(
    chunks,
    dataset_name: str,
    output_path: str | Path,
) -> dict:
    total_rows = 0
    binary_labels = Counter()
    original_labels = Counter()
    source_files = Counter()

    nan_counts = Counter()
    non_numeric_counts = Counter()
    positive_inf_counts = Counter()
    negative_inf_counts = Counter()

    for index, chunk in enumerate(chunks, start=1):
        total_rows += len(chunk)

        binary_labels.update(
            chunk["Label"].value_counts().to_dict()
        )

        original_labels.update(
            chunk["original_label"].value_counts().to_dict()
        )

        source_files.update(
            chunk["source_file"].value_counts().to_dict()
        )

        for column in CICIDS2017_FEATURES:
            raw = chunk[column]

            original_nan = raw.isna()

            numeric = pd.to_numeric(
                raw,
                errors="coerce",
            )

            non_numeric = (
                numeric.isna()
                & ~original_nan
            )

            array = numeric.to_numpy(
                dtype=float,
                na_value=np.nan,
            )

            nan_counts[column] += int(
                original_nan.sum()
            )

            non_numeric_counts[column] += int(
                non_numeric.sum()
            )

            positive_inf_counts[column] += int(
                np.isposinf(array).sum()
            )

            negative_inf_counts[column] += int(
                np.isneginf(array).sum()
            )

        if index % 10 == 0:
            print(
                f"[{dataset_name}] "
                f"Processed {total_rows:,} flows"
            )

    profile = {
        "dataset": dataset_name,
        "total_flows": total_rows,
        "binary_labels": dict(binary_labels),
        "original_labels": dict(original_labels),
        "source_files": dict(source_files),
        "features": {},
    }

    for column in CICIDS2017_FEATURES:
        profile["features"][column] = {
            "nan": nan_counts[column],
            "non_numeric": non_numeric_counts[column],
            "positive_inf": positive_inf_counts[column],
            "negative_inf": negative_inf_counts[column],
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            profile,
            indent=2,
            sort_keys=True,
        )
    )

    return profile
