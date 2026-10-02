import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.cicids2017 import CICIDS2017_FEATURES
from src.profiling.dataset_profiler import profile_dataset


class TestDatasetProfiler(unittest.TestCase):

    def test_profile_dataset(self):
        rows = []

        for label, original_label in [
            ("BENIGN", "Benign"),
            ("ATTACK", "DDoS"),
        ]:
            row = {
                feature: 1.0
                for feature in CICIDS2017_FEATURES
            }

            row["Label"] = label
            row["original_label"] = original_label
            row["source_file"] = "sample.csv"

            rows.append(row)

        rows[0]["Packet Length Mean"] = np.nan
        rows[1]["Packet Length Std"] = np.inf
        rows[1]["Packet Length Variance"] = -np.inf

        chunk = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "profile.json"

            profile = profile_dataset(
                chunks=[chunk],
                dataset_name="TEST",
                output_path=output,
            )

            self.assertEqual(profile["total_flows"], 2)
            self.assertEqual(
                profile["binary_labels"]["BENIGN"],
                1,
            )
            self.assertEqual(
                profile["binary_labels"]["ATTACK"],
                1,
            )

            self.assertEqual(
                profile["features"]["Packet Length Mean"]["nan"],
                1,
            )

            self.assertEqual(
                profile["features"]["Packet Length Std"]["positive_inf"],
                1,
            )

            self.assertEqual(
                profile["features"]["Packet Length Variance"]["negative_inf"],
                1,
            )

            self.assertTrue(output.exists())

            saved = json.loads(output.read_text())

            self.assertEqual(
                saved["total_flows"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
