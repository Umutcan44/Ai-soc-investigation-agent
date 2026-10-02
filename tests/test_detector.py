import pytest

from src.datasets.cicids2017 import CICIDS2017_FEATURES
from src.ml.detector import NetworkAnomalyDetector


class FakeModel:
    classes_ = [0, 1]

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.03, 0.97]]


def build_features():
    return {
        feature: 1.0
        for feature in CICIDS2017_FEATURES
    }


def build_detector():
    detector = NetworkAnomalyDetector.__new__(
        NetworkAnomalyDetector
    )
    detector.model = FakeModel()
    return detector


def test_attack_prediction_creates_security_event():
    detector = build_detector()

    event = detector.predict_event(
        build_features(),
        source="cicids2017",
        source_ip="192.168.1.10",
        destination_ip="10.0.0.5",
        destination_port=8080,
        protocol="TCP",
    )

    assert event.prediction == "ATTACK"
    assert event.confidence == pytest.approx(0.97)
    assert event.source == "cicids2017"
    assert event.destination_port == 8080
    assert event.evidence["model_prediction"] == 1


def test_confidence_is_valid_probability():
    detector = build_detector()

    event = detector.predict_event(
        build_features()
    )

    assert 0.0 <= event.confidence <= 1.0


def test_missing_feature_is_rejected():
    detector = build_detector()

    features = build_features()
    features.pop(CICIDS2017_FEATURES[0])

    with pytest.raises(
        ValueError,
        match="Missing model features",
    ):
        detector.predict_event(features)
