import os

from openai import OpenAI

from src.ai_agent import SYSTEM_PROMPT, build_investigation_prompt
from src.schemas import InvestigationReport


DEFAULT_MODEL = "gpt-5.6-luna"


class OpenAIProvider:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv(
            "OPENAI_MODEL",
            DEFAULT_MODEL,
        )

        self.client = OpenAI()

    def analyze(self, report: InvestigationReport) -> str:
        prompt = build_investigation_prompt(report)

        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
        )

        return response.output_text
