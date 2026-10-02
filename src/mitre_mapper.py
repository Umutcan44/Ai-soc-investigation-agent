from src.schemas import SecurityEvent


def map_mitre_candidates(
    event: SecurityEvent,
    context: dict,
) -> list[dict]:
    """
    Generate candidate MITRE ATT&CK techniques only when
    sufficient behavioral evidence is available.

    Network ports or services alone are not considered
    sufficient evidence for an ATT&CK technique.
    """

    candidates = []

    evidence = event.evidence or {}

    failed_logins = evidence.get("failed_login_count", 0)

    # Candidate: Brute Force
    # Requires behavioral evidence, not merely SSH traffic.
    if (
        context.get("destination_service") == "SSH"
        and failed_logins >= 5
    ):
        candidates.append(
            {
                "technique_id": "T1110",
                "technique_name": "Brute Force",
                "confidence": "medium",
                "evidence": [
                    f"{failed_logins} failed login attempts observed.",
                    "Destination service identified as SSH.",
                ],
                "status": "candidate",
            }
        )

    return candidates
