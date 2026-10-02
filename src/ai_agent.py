from src.schemas import InvestigationReport


SYSTEM_PROMPT = """
You are an AI assistant supporting a Security Operations Center analyst.

Analyze only the evidence contained in the investigation report.

Rules:
- Do not invent indicators, attack techniques, malware, or attacker intent.
- Distinguish confirmed facts from hypotheses.
- Treat MITRE ATT&CK mappings marked as candidate as unconfirmed.
- Do not claim that a candidate technique definitely occurred.
- Keep recommendations actionable for a SOC analyst.
- State when available evidence is insufficient.
""".strip()


def build_investigation_prompt(report: InvestigationReport) -> str:
    report_json = report.model_dump_json(indent=2)

    return f"""
Review the following structured SOC investigation report.

INVESTIGATION REPORT:
{report_json}

Produce the following sections:

1. Executive Summary
2. Confirmed Facts
3. MITRE ATT&CK Candidates
4. Analyst Assessment
5. Recommended Next Actions

Do not introduce facts that are not supported by the report.
""".strip()
