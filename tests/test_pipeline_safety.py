from src.investigator import investigate_event
from src.schemas import SecurityEvent


def test_high_confidence_attack_requires_review():
    event = SecurityEvent(
        event_id="test-high-attack",
        source="other",
        destination_port=8080,
        prediction="ATTACK",
        confidence=0.99,
    )

    report = investigate_event(event)

    assert report.severity == "high"
    assert report.triage_status == "requires_review"


def test_anomaly_does_not_invent_mitre_technique():
    event = SecurityEvent(
        event_id="test-no-mitre",
        source="other",
        destination_port=8080,
        prediction="ATTACK",
        confidence=0.99,
        evidence={
            "dataset_label": "Bot",
        },
    )

    report = investigate_event(event)

    assert report.mitre_candidates == []


def test_ssh_port_alone_does_not_imply_brute_force():
    event = SecurityEvent(
        event_id="test-ssh-only",
        source="other",
        destination_port=22,
        protocol="TCP",
        prediction="ATTACK",
        confidence=0.99,
    )

    report = investigate_event(event)

    assert report.mitre_candidates == []
