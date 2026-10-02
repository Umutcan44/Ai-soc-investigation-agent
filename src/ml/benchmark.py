from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.datasets.cicids2017 import CICIDS2017_FEATURES


def prepare_xy(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Convert the canonical feature schema into numeric model input.

    BENIGN -> 0
    ATTACK -> 1
    """
    X = dataframe[CICIDS2017_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError(
            "Model input contains NaN or non-numeric feature values."
        )

    values = X.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Model input contains infinite feature values."
        )

    y = (
        dataframe["Label"]
        .map({
            "BENIGN": 0,
            "ATTACK": 1,
        })
    )

    if y.isna().any():
        raise ValueError(
            "Unexpected binary label found."
        )

    return X, y.to_numpy(dtype=np.int8)


def build_random_forest(
    random_seed: int = 42,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        random_state=random_seed,
        n_jobs=1,
        class_weight=None,
    )


def evaluate_binary_classifier(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                y,
                probabilities,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "sample_size": int(len(y)),
    }
