import random

import pandas as pd

from src.datasets.cicids2017 import CICIDS2017_FEATURES


def reservoir_sample_binary(
    chunks,
    benign_size: int,
    attack_size: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Memory-efficient reservoir sampling for binary network-flow datasets.

    Maintains independent reservoirs for BENIGN and ATTACK so the caller
    can control class balance without loading the full dataset into memory.
    """
    rng = random.Random(random_seed)

    target_sizes = {
        "BENIGN": benign_size,
        "ATTACK": attack_size,
    }

    reservoirs = {
        "BENIGN": [],
        "ATTACK": [],
    }

    seen = {
        "BENIGN": 0,
        "ATTACK": 0,
    }

    columns = CICIDS2017_FEATURES + [
        "Label",
        "original_label",
        "dataset",
        "source_file",
    ]

    for chunk in chunks:
        for row in chunk[columns].itertuples(
            index=False,
            name=None,
        ):
            label = row[len(CICIDS2017_FEATURES)]

            if label not in target_sizes:
                continue

            seen[label] += 1
            reservoir = reservoirs[label]
            target_size = target_sizes[label]

            if len(reservoir) < target_size:
                reservoir.append(row)
                continue

            replacement_index = rng.randrange(seen[label])

            if replacement_index < target_size:
                reservoir[replacement_index] = row

    rows = (
        reservoirs["BENIGN"]
        + reservoirs["ATTACK"]
    )

    result = pd.DataFrame(
        rows,
        columns=columns,
    )

    return result.sample(
        frac=1.0,
        random_state=random_seed,
    ).reset_index(drop=True)
