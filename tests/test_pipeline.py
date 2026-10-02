import unittest

from src.schemas import SecurityEvent, InvestigationReport
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
        self.assertIn("51544", values)
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

        self.assertIsInstance(report, InvestigationReport)
        self.assertEqual(report.event_id, "test-001")
        self.assertEqual(report.triage_status, "requires_review")
        self.assertEqual(report.severity, "high")
        self.assertEqual(report.confidence, 0.91)
        self.assertGreater(len(report.facts), 0)
        self.assertGreater(len(report.recommended_actions), 0)

    def test_mitre_not_assigned_without_behavioral_evidence(self):
        report = investigate_event(self.event)

        self.assertEqual(
            report.mitre_candidates,
            [],
        )

    def test_mitre_brute_force_candidate_with_failed_logins(self):
        event = SecurityEvent(
            event_id="mitre-test-002",
            source="cicids2017",
            source_ip="192.168.1.25",
            destination_ip="10.0.0.8",
            destination_port=22,
            protocol="TCP",
            prediction="ATTACK",
            confidence=0.91,
            evidence={
                "failed_login_count": 8
            },
        )

        report = investigate_event(event)

        self.assertEqual(
            len(report.mitre_candidates),
            1,
        )

        candidate = report.mitre_candidates[0]

        self.assertEqual(candidate["technique_id"], "T1110")
        self.assertEqual(candidate["technique_name"], "Brute Force")
        self.assertEqual(candidate["status"], "candidate")
        self.assertEqual(candidate["confidence"], "medium")


if __name__ == "__main__":
    unittest.main()

class TestAIAgent(unittest.TestCase):

    def test_ai_prompt_uses_structured_evidence(self):
        from src.ai_agent import SYSTEM_PROMPT, build_investigation_prompt

        event = SecurityEvent(
            event_id="ai-test-001",
            source="cicids2017",
            destination_port=22,
            protocol="TCP",
            prediction="ATTACK",
            confidence=0.91,
            evidence={"failed_login_count": 8},
        )

        report = investigate_event(event)
        prompt = build_investigation_prompt(report)

        self.assertIn("ai-test-001", prompt)
        self.assertIn("T1110", prompt)
        self.assertIn("Brute Force", prompt)
        self.assertIn("candidate", prompt)
        self.assertIn("Do not introduce facts", prompt)

        self.assertIn(
            "Do not invent indicators",
            SYSTEM_PROMPT,
        )
