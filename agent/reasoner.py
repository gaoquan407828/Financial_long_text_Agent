from __future__ import annotations

import logging
import json
from typing import Any

from .answer_parser import (
    extract_answer_from_text,
    extract_json_object,
    normalize_answer,
    parse_option_judgements,
)
from .calculator import CalculationSolver
from .prompt_builder import PromptBuilder
from .question_analyzer import QuestionAnalyzer
from .qwen_client import QwenClient, QwenClientError
from .schema import AnswerResult, EvidenceItem, OptionJudgement, Question, TokenUsage


class Reasoner:
    def __init__(
        self,
        client: QwenClient,
        prompt_builder: PromptBuilder,
        config: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config or {}
        self.client = client
        self.prompt_builder = prompt_builder
        self.analyzer = QuestionAnalyzer(self.config)
        self.calculator = CalculationSolver(self.config, logger)
        self.second_pass_enabled = bool(self.config.get("whether_enable_second_pass", False))
        self.second_pass_min_confidence = float(self.config.get("second_pass_min_confidence", 0.68))
        self.second_pass_formats = set(self.config.get("second_pass_answer_formats", ["multi", "tf"]))
        self.second_pass_kinds = set(self.config.get("second_pass_question_kinds", ["calculation", "comparison"]))
        self.logger = logger or logging.getLogger(__name__)

    def answer(
        self,
        question: Question,
        compact_context: str,
        evidence_items: list[EvidenceItem],
        used_doc_ids: list[str],
    ) -> AnswerResult:
        raw_output = ""
        usage = TokenUsage()
        error: str | None = None
        judgements: dict[str, OptionJudgement] = {}
        confidence = 0.0
        try:
            analysis = self.analyzer.analyze(question)
            calculation = self.calculator.solve(question, analysis, evidence_items)
            calculation_context = calculation.to_context()

            if self.calculator.can_direct_answer(question, calculation):
                answer = normalize_answer(calculation.answer, question, judgements)
                judgements = self._judgements_from_direct_answer(question, answer, calculation_context)
                raw_output = json.dumps(
                    {
                        "qid": question.qid,
                        "answer": answer,
                        "judgements": {key: value.verdict for key, value in judgements.items()},
                        "confidence": calculation.confidence,
                        "source": "code_calculation",
                    },
                    ensure_ascii=False,
                )
                confidence = calculation.confidence
                return self._build_result(
                    question=question,
                    answer=answer,
                    judgements=judgements,
                    raw_output=raw_output,
                    usage=usage,
                    evidence_items=evidence_items,
                    used_doc_ids=used_doc_ids,
                    confidence=confidence,
                    error=None,
                )

            messages = self.prompt_builder.build_messages(question, compact_context, analysis, calculation_context)
            chat_result = self.client.chat(messages, qid=question.qid)
            raw_output = chat_result.content
            usage = chat_result.usage
            data = extract_json_object(raw_output)
            judgements = parse_option_judgements(data)
            answer = normalize_answer((data or {}).get("answer") if data else None, question, judgements)
            confidence = float((data or {}).get("confidence", 0.0) or 0.0) if data else 0.0
            if not answer:
                answer = extract_answer_from_text(raw_output, question)
            if self._should_second_pass(question, analysis, confidence, answer):
                second = self._second_pass(
                    question=question,
                    compact_context=compact_context,
                    analysis=analysis,
                    calculation_context=calculation_context,
                    first_answer=answer,
                    first_raw_output=raw_output,
                )
                if second is not None:
                    answer, second_judgements, second_raw, second_usage, second_confidence = second
                    usage.add(second_usage)
                    judgements = second_judgements or judgements
                    raw_output = json.dumps(
                        {
                            "first_output": raw_output,
                            "second_output": second_raw,
                            "final_answer": answer,
                        },
                        ensure_ascii=False,
                    )
                    confidence = second_confidence
        except QwenClientError:
            raise
        except Exception as exc:
            error = str(exc)
            self.logger.exception("reasoning failed for %s: %s", question.qid, exc)
            answer = normalize_answer(None, question, judgements)

        return self._build_result(
            question=question,
            answer=answer,
            judgements=judgements,
            raw_output=raw_output,
            usage=usage,
            evidence_items=evidence_items,
            used_doc_ids=used_doc_ids,
            confidence=confidence,
            error=error,
        )

    def _should_second_pass(
        self,
        question: Question,
        analysis: Any,
        confidence: float,
        answer: str,
    ) -> bool:
        if not self.second_pass_enabled:
            return False
        if not answer:
            return True
        if question.answer_format in self.second_pass_formats:
            return True
        if getattr(analysis, "kind", "") in self.second_pass_kinds:
            return True
        return confidence > 0 and confidence < self.second_pass_min_confidence

    def _second_pass(
        self,
        question: Question,
        compact_context: str,
        analysis: Any,
        calculation_context: str,
        first_answer: str,
        first_raw_output: str,
    ) -> tuple[str, dict[str, OptionJudgement], str, TokenUsage, float] | None:
        try:
            messages = self.prompt_builder.build_validation_messages(
                question=question,
                evidence_context=compact_context,
                first_answer=first_answer,
                raw_output=first_raw_output,
                analysis=analysis,
                calculation_context=calculation_context,
            )
            chat_result = self.client.chat(messages, qid=f"{question.qid}#verify")
            data = extract_json_object(chat_result.content)
            judgements = parse_option_judgements(data)
            answer = normalize_answer((data or {}).get("answer") if data else None, question, judgements)
            if not answer:
                answer = extract_answer_from_text(chat_result.content, question)
            confidence = float((data or {}).get("confidence", 0.0) or 0.0) if data else 0.0
            return answer, judgements, chat_result.content, chat_result.usage, confidence
        except QwenClientError:
            raise
        except Exception as exc:
            self.logger.warning("second pass failed for %s: %s", question.qid, exc)
            return None

    def _build_result(
        self,
        question: Question,
        answer: str,
        judgements: dict[str, OptionJudgement],
        raw_output: str,
        usage: TokenUsage,
        evidence_items: list[EvidenceItem],
        used_doc_ids: list[str],
        confidence: float,
        error: str | None,
    ) -> AnswerResult:
        for key in question.options:
            judgements.setdefault(key, OptionJudgement(verdict=(key in answer), reason="fallback/default", evidence_ids=[]))

        return AnswerResult(
            qid=question.qid,
            answer=answer,
            option_judgements=judgements,
            raw_model_output=raw_output,
            token_usage=usage,
            evidence_chunks=evidence_items,
            used_doc_ids=used_doc_ids,
            confidence=confidence,
            error=error,
        )

    def _judgements_from_direct_answer(
        self,
        question: Question,
        answer: str,
        reason: str,
    ) -> dict[str, OptionJudgement]:
        judgements: dict[str, OptionJudgement] = {}
        for key in question.options:
            judgements[key] = OptionJudgement(
                verdict=(key in answer),
                reason=reason[:400],
                evidence_ids=[],
            )
        return judgements
