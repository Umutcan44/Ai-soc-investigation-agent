import unittest

import numpy as np
import pandas as pd

from src.datasets.cicids2017 import CICIDS2017_FEATURES
from src.ml.benchmark import (
    build_random_forest,
    evaluate_binary_classifier,
    prepare_xy,
)


class TestBenchmark(unittest.TestCase):

    def make_dataset(self):
        rows = []

        for i in range(20):
            row = {
                feature: float(i)
                for feature in CICIDS2017_FEATURES
            }
            row["Label"] = "BENIGN"
            rows.append(row)

        for i in range(20):
            row = {
                feature: float(100 + i)
                for feature in CICIDS2017_FEATURES
            }
            row["Label"] = "ATTACK"
            rows.append(row)

        return pd.DataFrame(rows)

    def test_prepare_xy(self):
        dataframe = self.make_dataset()

        X, y = prepare_xy(dataframe)

        self.assertEqual(
            list(X.columns),
            CICIDS2017_FEATURES,
        )
        self.assertEqual(X.shape, (40, 11))
        self.assertEqual(len(y), 40)
        self.assertEqual(set(y.tolist()), {0, 1})

    def test_invalid_numeric_value_rejected(self):
        dataframe = self.make_dataset()

        feature = CICIDS2017_FEATURES[0]
        dataframe[feature] = dataframe[feature].astype(object)
        dataframe.loc[0, feature] = "invalid"

        with self.assertRaises(ValueError):
            prepare_xy(dataframe)

    def test_evaluation_metrics(self):
        dataframe = self.make_dataset()
        X, y = prepare_xy(dataframe)

        model = build_random_forest(
            random_seed=42,
        )
        model.fit(X, y)

        metrics = evaluate_binary_classifier(
            model,
            X,
            y,
        )

        expected_keys = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "confusion_matrix",
            "sample_size",
        }

        self.assertEqual(
            set(metrics.keys()),
            expected_keys,
        )

        self.assertEqual(
            metrics["sample_size"],
            40,
        )

        self.assertEqual(
            np.array(
                metrics["confusion_matrix"]
            ).shape,
            (2, 2),
        )


if __name__ == "__main__":
    unittest.main()
