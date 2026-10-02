import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.datasets.cicids2017 import (
    CICIDS2017_FEATURES,
    iter_cicids2017,
    normalize_label,
)


class TestCICIDS2017Adapter(unittest.TestCase):

    def test_label_normalization(self):
        self.assertEqual(normalize_label("BENIGN"), "BENIGN")
        self.assertEqual(normalize_label(" BENIGN "), "BENIGN")
        self.assertEqual(normalize_label("DDoS"), "ATTACK")
        self.assertEqual(normalize_label("PortScan"), "ATTACK")

    def test_adapter_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            row = {feature: 1 for feature in CICIDS2017_FEATURES}
            row["Label"] = "DDoS"

            csv_path = Path(tmpdir) / "sample.csv"
            pd.DataFrame([row]).to_csv(csv_path, index=False)

            chunk = next(
                iter_cicids2017(tmpdir, chunksize=1)
            )

            self.assertEqual(chunk.shape, (1, 15))
            self.assertEqual(chunk.iloc[0]["Label"], "ATTACK")
            self.assertEqual(chunk.iloc[0]["original_label"], "DDoS")
            self.assertEqual(
                chunk.iloc[0]["dataset"],
                "CICIDS2017",
            )
            self.assertEqual(
                chunk.iloc[0]["source_file"],
                "sample.csv",
            )


if __name__ == "__main__":
    unittest.main()
