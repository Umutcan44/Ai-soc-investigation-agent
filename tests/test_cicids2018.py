import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.datasets.cicids2017 import CICIDS2017_FEATURES
from src.datasets.cicids2018 import (
    CICIDS2018_COLUMN_MAP,
    iter_cicids2018,
)


class TestCICIDS2018Adapter(unittest.TestCase):

    def test_adapter_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            row = {
                source_column: 1
                for source_column in CICIDS2018_COLUMN_MAP
            }
            row["Label"] = "Bot"

            csv_path = Path(tmpdir) / "sample2018.csv"
            pd.DataFrame([row]).to_csv(csv_path, index=False)

            chunk = next(
                iter_cicids2018(
                    tmpdir,
                    chunksize=1,
                )
            )

            self.assertEqual(chunk.shape, (1, 15))
            self.assertEqual(
                list(chunk.columns[:11]),
                CICIDS2017_FEATURES,
            )
            self.assertEqual(
                chunk.iloc[0]["Label"],
                "ATTACK",
            )
            self.assertEqual(
                chunk.iloc[0]["original_label"],
                "Bot",
            )
            self.assertEqual(
                chunk.iloc[0]["dataset"],
                "CSE-CIC-IDS2018",
            )
            self.assertEqual(
                chunk.iloc[0]["source_file"],
                "sample2018.csv",
            )

    def test_common_feature_schema(self):
        mapped_features = list(
            CICIDS2018_COLUMN_MAP.values()
        )

        self.assertEqual(
            mapped_features,
            CICIDS2017_FEATURES,
        )


if __name__ == "__main__":
    unittest.main()
