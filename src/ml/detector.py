from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd

from src.datasets.cicids2017 import CICIDS2017_FEATURES
from src.schemas import SecurityEvent


DEFAULT_MODEL_PATH = Path(
    "model/rf_combined_cicids2017_2018.joblib"
)


class NetworkAnomalyDetector:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

    def predict_event(
        self,
        features: dict,
        *,
        source: str = "other",
        source_ip: str | None = None,
        destination_ip: str | None = None,
        source_port: int | None = None,
        destination_port: int | None = None,
        protocol: str | None = None,
        evidence: dict | None = None,
    ) -> SecurityEvent:

        missing = [
            feature
            for feature in CICIDS2017_FEATURES
            if feature not in features
        ]

        if missing:
            raise ValueError(
                f"Missing model features: {missing}"
            )

        row = pd.DataFrame(
            [
                {
                    feature: features[feature]
                    for feature in CICIDS2017_FEATURES
                }
            ]
        )

        prediction_value = int(
            self.model.predict(row)[0]
        )

        probabilities = self.model.predict_proba(row)[0]

        classes = list(self.model.classes_)
        predicted_index = classes.index(prediction_value)

        confidence = float(
            probabilities[predicted_index]
        )

        prediction = (
            "ATTACK"
            if prediction_value == 1
            else "BENIGN"
        )

        model_evidence = {
            "model": "rf_combined_cicids2017_2018",
            "model_prediction": prediction_value,
            "model_confidence": confidence,
        }

        if evidence:
            model_evidence.update(evidence)

        return SecurityEvent(
            event_id=f"rf-{uuid4()}",
            source=source,
            event_type="network_anomaly",
            source_ip=source_ip,
            destination_ip=destination_ip,
            source_port=source_port,
            destination_port=destination_port,
            protocol=protocol,
            prediction=prediction,
            confidence=confidence,
            evidence=model_evidence,
        )
