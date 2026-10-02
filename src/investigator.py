from src.schemas import SecurityEvent
from src.ioc_extractor import extract_iocs
from src.enrichment import enrich_event


def investigate_event(event: SecurityEvent) -> dict:
    iocs = extract_iocs(event)
    context = enrich_event(event)

    findings = []
    recommended_actions = []

    if event.prediction == "ATTACK":
        findings.append(
            "The upstream detection system classified the network event as anomalous."
        )

        recommended_actions.extend([
            "Review related network telemetry.",
            "Check the source host for additional suspicious activity.",
            "Correlate the event with authentication and endpoint logs.",
        ])

    if context["destination_service"] == "SSH":
        findings.append(
            "The destination service is commonly associated with SSH."
        )

        recommended_actions.extend([
            "Review SSH authentication logs.",
            "Check for repeated failed login attempts.",
            "Determine whether the source host is authorized to access SSH.",
        ])

    return {
        "event_id": event.event_id,
        "prediction": event.prediction,
        "confidence": event.confidence,
        "iocs": [
            {
                "type": ioc.type,
                "value": ioc.value,
                "role": ioc.role,
            }
            for ioc in iocs
        ],
        "context": context,
        "findings": findings,
        "recommended_actions": recommended_actions,
    }
