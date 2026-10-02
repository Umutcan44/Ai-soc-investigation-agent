import unittest

import pandas as pd

from src.datasets.cicids2017 import CICIDS2017_FEATURES
from src.ml.sampler import reservoir_sample_binary


class TestReservoirSampler(unittest.TestCase):

    def make_chunk(self, benign_count=20, attack_count=10):
        rows = []

        for i in range(benign_count):
            row = {
                feature: float(i)
                for feature in CICIDS2017_FEATURES
            }
            row.update({
                "Label": "BENIGN",
                "original_label": "BENIGN",
                "dataset": "TEST",
                "source_file": "test.csv",
            })
            rows.append(row)

        for i in range(attack_count):
            row = {
                feature: float(1000 + i)
                for feature in CICIDS2017_FEATURES
            }
            row.update({
                "Label": "ATTACK",
                "original_label": "DDoS",
                "dataset": "TEST",
                "source_file": "test.csv",
            })
            rows.append(row)

        return pd.DataFrame(rows)

    def test_requested_class_sizes(self):
        chunk = self.make_chunk()

        sample = reservoir_sample_binary(
            chunks=[chunk],
            benign_size=5,
            attack_size=4,
            random_seed=42,
        )

        counts = sample["Label"].value_counts().to_dict()

        self.assertEqual(len(sample), 9)
        self.assertEqual(counts["BENIGN"], 5)
        self.assertEqual(counts["ATTACK"], 4)

    def test_reproducible_with_same_seed(self):
        chunk = self.make_chunk()

        sample_a = reservoir_sample_binary(
            chunks=[chunk],
            benign_size=5,
            attack_size=4,
            random_seed=42,
        )

        sample_b = reservoir_sample_binary(
            chunks=[chunk],
            benign_size=5,
            attack_size=4,
            random_seed=42,
        )

        pd.testing.assert_frame_equal(
            sample_a,
            sample_b,
        )

    def test_canonical_features_preserved(self):
        chunk = self.make_chunk()

        sample = reservoir_sample_binary(
            chunks=[chunk],
            benign_size=3,
            attack_size=3,
            random_seed=42,
        )

        self.assertEqual(
            list(sample.columns[:len(CICIDS2017_FEATURES)]),
            CICIDS2017_FEATURES,
        )


if __name__ == "__main__":
    unittest.main()
