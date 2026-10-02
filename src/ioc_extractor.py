from dataclasses import dataclass

from src.schemas import SecurityEvent


@dataclass
class IOC:
    type: str
    value: str
    role: str


def extract_iocs(event: SecurityEvent) -> list[IOC]:
    iocs: list[IOC] = []

    if event.source_ip:
        iocs.append(
            IOC(
                type="ip",
                value=event.source_ip,
                role="source",
            )
        )

    if event.destination_ip:
        iocs.append(
            IOC(
                type="ip",
                value=event.destination_ip,
                role="destination",
            )
        )

    if event.source_port is not None:
        iocs.append(
            IOC(
                type="port",
                value=str(event.source_port),
                role="source",
            )
        )

    if event.destination_port is not None:
        iocs.append(
            IOC(
                type="port",
                value=str(event.destination_port),
                role="destination",
            )
        )

    if event.protocol:
        iocs.append(
            IOC(
                type="protocol",
                value=event.protocol.upper(),
                role="network",
            )
        )

    return iocs
