from __future__ import annotations

import logging
from typing import Any

from .answer_parser import (
    extract_answer_from_text,
    extract_json_object,
    normalize_answer,
    parse_option_judgements,
)
from .prompt_builder import PromptBuilder
from .qwen_client import QwenClient, QwenClientError
from .schema import AnswerResult, EvidenceItem, OptionJudgement, Question, TokenUsage


class Reasoner:
    def __init__(
        self,
        client: QwenClient,
        prompt_builder: PromptBuilder,
        logger: logging.Logger | None = None,
    ):
        self.client = client
        self.prompt_builder = prompt_builder
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
            messages = self.prompt_builder.build_messages(question, compact_context)
            chat_result = self.client.chat(messages, qid=question.qid)
            raw_output = chat_result.content
            usage = chat_result.usage
            data = extract_json_object(raw_output)
            judgements = parse_option_judgements(data)
            answer = normalize_answer((data or {}).get("answer") if data else None, question, judgements)
            confidence = float((data or {}).get("confidence", 0.0) or 0.0) if data else 0.0
            if not answer:
                answer = extract_answer_from_text(raw_output, question)
        except QwenClientError:
            raise
        except Exception as exc:
            error = str(exc)
            self.logger.exception("reasoning failed for %s: %s", question.qid, exc)
            answer = normalize_answer(None, question, judgements)

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
