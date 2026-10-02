from pathlib import Path
from collections.abc import Iterator

import pandas as pd


CICIDS2017_FEATURES = [
    "Destination Port",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Average Packet Size",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Avg Bwd Segment Size",
    "Subflow Fwd Bytes",
]

LABEL_COLUMN = "Label"


def normalize_columns(columns) -> list[str]:
    return [str(column).strip() for column in columns]


def normalize_label(label: str) -> str:
    label = str(label).strip()
    return "BENIGN" if label.upper() == "BENIGN" else "ATTACK"


def iter_cicids2017(
    data_dir: str | Path,
    chunksize: int = 25_000,
) -> Iterator[pd.DataFrame]:

    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}"
        )

    required_columns = CICIDS2017_FEATURES + [LABEL_COLUMN]

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
                    for column in required_columns
                    if column not in chunk.columns
                ]

                if missing:
                    raise ValueError(
                        f"{csv_file.name} is missing columns: {missing}"
                    )

                output = chunk[required_columns].copy()

                output["original_label"] = (
                    output["Label"]
                    .astype(str)
                    .str.strip()
                )

                output["Label"] = (
                    output["original_label"]
                    .map(normalize_label)
                )

                output["dataset"] = "CICIDS2017"
                output["source_file"] = csv_file.name

                yield output
        finally:
            reader.close()
