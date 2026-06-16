from __future__ import annotations

from pathlib import Path
from typing import Any

from .question_analyzer import QuestionAnalysis
from .schema import Question
from .utils import load_yaml, resolve_path


class PromptBuilder:
    def __init__(self, config: dict[str, Any]):
        prompts_path = resolve_path(config.get("prompts_path", "config/prompts.yaml"))
        prompts = load_yaml(prompts_path)
        self.system_prompt = prompts.get("system", "")
        self.answer_template = prompts["answer_prompt"]
        self.question_templates = prompts.get("question_templates", {})
        self.domain_guidance = prompts.get("domain_guidance", {})
        self.question_guidance = prompts.get("question_guidance", {})

    def build_messages(
        self,
        question: Question,
        evidence_context: str,
        analysis: QuestionAnalysis | None = None,
        calculation_context: str = "",
    ) -> list[dict[str, str]]:
        options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
        question_kind = analysis.kind if analysis else "fact_lookup"
        template = self.question_templates.get(question_kind, self.answer_template)
        user_prompt = template.format(
            qid=question.qid,
            domain=question.domain,
            answer_format=question.answer_format,
            question_type=question.type,
            question_kind=question_kind,
            question=question.question,
            options=options,
            evidence_context=evidence_context or "未检索到有效证据。",
            analysis_hint=analysis.to_prompt_hint() if analysis else "",
            calculation_context=calculation_context,
            question_guidance=self.question_guidance.get(question_kind, self.question_guidance.get("default", "")),
            domain_guidance=self.domain_guidance.get(question.domain, self.domain_guidance.get("default", "")),
        )
        messages = []
        if self.system_prompt.strip():
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages
