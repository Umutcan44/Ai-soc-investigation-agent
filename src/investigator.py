from src.schemas import SecurityEvent, InvestigationReport
from src.ioc_extractor import extract_iocs
from src.enrichment import enrich_event
from src.mitre_mapper import map_mitre_candidates


def determine_triage_status(event: SecurityEvent) -> str:
    if event.prediction == "ATTACK":
        return "requires_review"

    if event.prediction == "BENIGN":
        return "no_immediate_action"

    return "insufficient_evidence"


def determine_severity(event: SecurityEvent) -> str:
    if event.prediction == "ATTACK":
        if event.confidence is not None and event.confidence >= 0.90:
            return "high"

        return "medium"

    if event.prediction == "UNKNOWN":
        return "low"

    return "informational"


def investigate_event(event: SecurityEvent) -> InvestigationReport:
    iocs = extract_iocs(event)
    context = enrich_event(event)
    mitre_candidates = map_mitre_candidates(event, context)

    facts = []
    recommended_actions = []

    if event.prediction == "ATTACK":
        facts.append(
            "The upstream detection system classified the network event as anomalous."
        )

        recommended_actions.extend([
            "Review related network telemetry.",
            "Check the source host for additional suspicious activity.",
            "Correlate the event with authentication and endpoint logs.",
        ])

    if context["destination_service"] == "SSH":
        facts.append(
            "The destination service is commonly associated with SSH."
        )

        recommended_actions.extend([
            "Review SSH authentication logs.",
            "Check for repeated failed login attempts.",
            "Determine whether the source host is authorized to access SSH.",
        ])

    if iocs:
        facts.append(
            f"{len(iocs)} observable network indicators were extracted from the event."
        )

    return InvestigationReport(
        event_id=event.event_id,
        triage_status=determine_triage_status(event),
        severity=determine_severity(event),
        confidence=event.confidence,
        facts=facts,
        mitre_candidates=mitre_candidates,
        recommended_actions=recommended_actions,
    )
