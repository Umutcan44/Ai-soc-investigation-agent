import pandas as pd

from src.investigator import investigate_event
from src.ml.detector import NetworkAnomalyDetector


DATASET = "data/processed/cicids2018_sample.pkl"


print("Loading demonstration flow...")

data = pd.read_pickle(DATASET)

attack_row = data[
    data["Label"] == "ATTACK"
].iloc[0]


features = {
    "Destination Port": attack_row["Destination Port"],
    "Max Packet Length": attack_row["Max Packet Length"],
    "Packet Length Mean": attack_row["Packet Length Mean"],
    "Packet Length Std": attack_row["Packet Length Std"],
    "Packet Length Variance": attack_row["Packet Length Variance"],
    "Average Packet Size": attack_row["Average Packet Size"],
    "Bwd Packet Length Max": attack_row["Bwd Packet Length Max"],
    "Bwd Packet Length Mean": attack_row["Bwd Packet Length Mean"],
    "Bwd Packet Length Std": attack_row["Bwd Packet Length Std"],
    "Avg Bwd Segment Size": attack_row["Avg Bwd Segment Size"],
    "Subflow Fwd Bytes": attack_row["Subflow Fwd Bytes"],
}


detector = NetworkAnomalyDetector()

event = detector.predict_event(
    features,
    source="cse-cic-ids2018",
    destination_port=int(
        attack_row["Destination Port"]
    ),
    evidence={
        "dataset": attack_row["dataset"],
        "dataset_label": attack_row["original_label"],
        "source_file": attack_row["source_file"],
    },
)


print("\n=== 1. ML SECURITY EVENT ===")
print(
    event.model_dump_json(
        indent=2,
    )
)


report = investigate_event(event)


print("\n=== 2. INVESTIGATION REPORT ===")
print(
    report.model_dump_json(
        indent=2,
    )
)


print("\n=== PIPELINE SUMMARY ===")
print("Prediction :", event.prediction)
print("Confidence :", f"{event.confidence:.4f}")
print("Severity   :", report.severity)
print("Triage     :", report.triage_status)
print(
    "MITRE candidates:",
    len(report.mitre_candidates),
)

print("\n=== 3. AI SOC ANALYST ===")

from src.providers.openai_provider import OpenAIProvider

provider = OpenAIProvider()

analysis = provider.analyze(report)

print(analysis)
