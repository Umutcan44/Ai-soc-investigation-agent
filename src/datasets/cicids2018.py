from pathlib import Path
from collections.abc import Iterator

import pandas as pd

from src.datasets.cicids2017 import (
    CICIDS2017_FEATURES,
    normalize_label,
)


CICIDS2018_COLUMN_MAP = {
    "Dst Port": "Destination Port",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "Pkt Size Avg": "Average Packet Size",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
}

LABEL_COLUMN = "Label"


def normalize_columns(columns) -> list[str]:
    return [str(column).strip() for column in columns]


def iter_cicids2018(
    data_dir: str | Path,
    chunksize: int = 25_000,
) -> Iterator[pd.DataFrame]:

    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}"
        )

    required_2018_columns = (
        list(CICIDS2018_COLUMN_MAP.keys())
        + [LABEL_COLUMN]
    )

    for csv_file in csv_files:
        reader = pd.read_csv(
            csv_file,
            chunksize=chunksize,
            low_memory=False,
        )

        try:
            for chunk in reader:
                chunk.columns = normalize_columns(chunk.columns)

                missing = [
                    column
                    for column in required_2018_columns
                    if column not in chunk.columns
                ]

                if missing:
                    raise ValueError(
                        f"{csv_file.name} is missing columns: {missing}"
                    )

                selected = chunk[
                    required_2018_columns
                ].copy()

                selected = selected.rename(
                    columns=CICIDS2018_COLUMN_MAP
                )

                selected["original_label"] = (
                    selected["Label"]
                    .astype(str)
                    .str.strip()
                )

                selected["Label"] = (
                    selected["original_label"]
                    .map(normalize_label)
                )

                selected["dataset"] = "CSE-CIC-IDS2018"
                selected["source_file"] = csv_file.name

                output_columns = (
                    CICIDS2017_FEATURES
                    + [
                        "Label",
                        "original_label",
                        "dataset",
                        "source_file",
                    ]
                )

                yield selected[output_columns]
        finally:
            reader.close()
