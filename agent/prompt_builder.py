from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import Question
from .utils import load_yaml, resolve_path


class PromptBuilder:
    def __init__(self, config: dict[str, Any]):
        prompts_path = resolve_path(config.get("prompts_path", "config/prompts.yaml"))
        prompts = load_yaml(prompts_path)
        self.system_prompt = prompts["system"]
        self.answer_template = prompts["answer_prompt"]
        self.domain_guidance = prompts.get("domain_guidance", {})

    def build_messages(self, question: Question, evidence_context: str) -> list[dict[str, str]]:
        options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
        user_prompt = self.answer_template.format(
            qid=question.qid,
            domain=question.domain,
            answer_format=question.answer_format,
            question_type=question.type,
            question=question.question,
            options=options,
            evidence_context=evidence_context or "未检索到有效证据。",
            domain_guidance=self.domain_guidance.get(question.domain, self.domain_guidance.get("default", "")),
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
