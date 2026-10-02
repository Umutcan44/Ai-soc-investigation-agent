import unittest

from src.schemas import SecurityEvent
from src.ioc_extractor import extract_iocs
from src.enrichment import enrich_event
from src.investigator import investigate_event


class TestSOCPipeline(unittest.TestCase):

    def setUp(self):
        self.event = SecurityEvent(
            event_id="test-001",
            source="cicids2017",
            source_ip="192.168.1.25",
            destination_ip="10.0.0.8",
            source_port=51544,
            destination_port=22,
            protocol="TCP",
            prediction="ATTACK",
            confidence=0.91,
        )

    def test_security_event(self):
        self.assertEqual(self.event.prediction, "ATTACK")
        self.assertEqual(self.event.destination_port, 22)

    def test_ioc_extraction(self):
        iocs = extract_iocs(self.event)

        values = [ioc.value for ioc in iocs]

        self.assertIn("192.168.1.25", values)
        self.assertIn("10.0.0.8", values)
        self.assertIn("22", values)
        self.assertIn("TCP", values)

    def test_enrichment(self):
        context = enrich_event(self.event)

        self.assertEqual(
            context["destination_service"],
            "SSH",
        )

    def test_investigation(self):
        report = investigate_event(self.event)

        self.assertEqual(report["event_id"], "test-001")
        self.assertEqual(report["prediction"], "ATTACK")
        self.assertGreater(len(report["findings"]), 0)
        self.assertGreater(len(report["recommended_actions"]), 0)


if __name__ == "__main__":
    unittest.main()
