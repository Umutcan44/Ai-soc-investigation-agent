from pathlib import Path

from src.datasets.cicids2017 import iter_cicids2017
from src.datasets.cicids2018 import iter_cicids2018
from src.ml.sampler import reservoir_sample_binary


OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PER_CLASS = 100_000
RANDOM_SEED = 42


print("Building CICIDS2017 sample...")

sample_2017 = reservoir_sample_binary(
    chunks=iter_cicids2017(
        "/home/kali/Desktop/input/MachineLearningCVE",
        chunksize=100_000,
    ),
    benign_size=SAMPLE_PER_CLASS,
    attack_size=SAMPLE_PER_CLASS,
    random_seed=RANDOM_SEED,
)

sample_2017.to_pickle(
    OUTPUT_DIR / "cicids2017_sample.pkl"
)

print(
    "CICIDS2017:",
    sample_2017["Label"].value_counts().to_dict(),
)


print("\nBuilding CSE-CIC-IDS2018 sample...")

sample_2018 = reservoir_sample_binary(
    chunks=iter_cicids2018(
        "data/raw/cse-cic-ids2018",
        chunksize=100_000,
    ),
    benign_size=SAMPLE_PER_CLASS,
    attack_size=SAMPLE_PER_CLASS,
    random_seed=RANDOM_SEED,
)

sample_2018.to_pickle(
    OUTPUT_DIR / "cicids2018_sample.pkl"
)

print(
    "CSE-CIC-IDS2018:",
    sample_2018["Label"].value_counts().to_dict(),
)


print("\nSamples saved successfully.")
