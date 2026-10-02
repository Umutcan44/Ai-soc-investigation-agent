import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets.cicids2017 import CICIDS2017_FEATURES
from src.ml.benchmark import (
    build_random_forest,
    prepare_xy,
)


RANDOM_SEED = 42
TEST_SIZE = 0.20

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


print("Loading sampled datasets...")

data_2017 = pd.read_pickle(
    DATA_DIR / "cicids2017_sample.pkl"
)

data_2018 = pd.read_pickle(
    DATA_DIR / "cicids2018_sample.pkl"
)


train_2017, _ = train_test_split(
    data_2017,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=data_2017["Label"],
)

train_2018, _ = train_test_split(
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


print(
    "Training combined model on",
    f"{len(y_train):,}",
    "flows...",
)

model = build_random_forest(
    random_seed=RANDOM_SEED,
)

model.fit(
    X_train,
    y_train,
)


model_path = (
    MODEL_DIR
    / "rf_combined_cicids2017_2018.joblib"
)

joblib.dump(
    model,
    model_path,
)


metadata = {
    "model_type": "RandomForestClassifier",
    "model_file": model_path.name,
    "training_datasets": [
        "CICIDS2017",
        "CSE-CIC-IDS2018",
    ],
    "training_flows": int(len(y_train)),
    "training_class_counts": {
        "BENIGN": int((y_train == 0).sum()),
        "ATTACK": int((y_train == 1).sum()),
    },
    "features": CICIDS2017_FEATURES,
    "feature_count": len(CICIDS2017_FEATURES),
    "random_seed": RANDOM_SEED,
    "sampling": (
        "100k BENIGN + 100k ATTACK reservoir sample "
        "per dataset before 80/20 split"
    ),
    "label_mapping": {
        "BENIGN": 0,
        "ATTACK": 1,
    },
    "intended_use": (
        "Research and portfolio SOC investigation pipeline."
    ),
    "limitations": [
        (
            "Training data is derived from CICIDS2017 and "
            "CSE-CIC-IDS2018 benchmark datasets."
        ),
        (
            "Performance on production network traffic "
            "has not been validated."
        ),
        (
            "Multi-domain holdout results do not establish "
            "generalization to unseen network environments."
        ),
    ],
}


metadata_path = (
    MODEL_DIR
    / "rf_combined_cicids2017_2018.metadata.json"
)

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2,
    )
)


print("Model saved to:", model_path)
print("Metadata saved to:", metadata_path)
print(
    "Training classes:",
    metadata["training_class_counts"],
)
