from src.schemas import SecurityEvent


PORT_SERVICES = {
    20: "FTP Data",
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    135: "MS RPC",
    139: "NetBIOS",
    143: "IMAP",
    389: "LDAP",
    443: "HTTPS",
    445: "SMB",
    3389: "RDP",
}


def enrich_event(event: SecurityEvent) -> dict:
    context = {
        "source": event.source,
        "prediction": event.prediction,
        "protocol": event.protocol.upper() if event.protocol else None,
        "destination_service": None,
        "observations": [],
    }

    if event.destination_port is not None:
        service = PORT_SERVICES.get(event.destination_port)

        if service:
            context["destination_service"] = service
            context["observations"].append(
                f"Destination port {event.destination_port} is commonly associated with {service}."
            )
        else:
            context["observations"].append(
                f"Destination port {event.destination_port} has no service mapping in the local enrichment table."
            )

    if event.prediction == "ATTACK":
        context["observations"].append(
            "The upstream detector classified this event as ATTACK."
        )

    return context
