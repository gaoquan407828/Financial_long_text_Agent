from __future__ import annotations

import json
import re
from typing import Any

from .schema import OptionJudgement, Question


VALID = set("ABCD")


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def normalize_answer(answer: str | None, question: Question, judgements: dict[str, OptionJudgement] | None = None) -> str:
    answer = (answer or "").upper()
    letters = [ch for ch in answer if ch in VALID]
    option_keys = set(question.options.keys()) or VALID
    if question.answer_format in {"mcq", "tf"}:
        for ch in letters:
            if ch in option_keys:
                return ch
        return fallback_answer(question, judgements)
    normalized = "".join(sorted({ch for ch in letters if ch in option_keys}))
    if normalized:
        return normalized
    return fallback_answer(question, judgements)


def fallback_answer(question: Question, judgements: dict[str, OptionJudgement] | None = None) -> str:
    if judgements:
        true_letters = sorted(k for k, v in judgements.items() if v.verdict and k in question.options)
        if question.answer_format == "multi" and true_letters:
            return "".join(true_letters)
        if question.answer_format in {"mcq", "tf"} and true_letters:
            return true_letters[0]
    if question.answer_format == "tf":
        return "A" if "A" in question.options else next(iter(question.options or {"A": ""}))
    if question.answer_format == "mcq":
        return next(iter(sorted(question.options.keys()) or ["A"]))
    return next(iter(sorted(question.options.keys()) or ["A"]))


def parse_option_judgements(data: dict[str, Any] | None) -> dict[str, OptionJudgement]:
    if not data:
        return {}
    raw = data.get("option_judgements") or {}
    parsed: dict[str, OptionJudgement] = {}
    for key, value in raw.items():
        key = str(key).upper()
        if key not in VALID:
            continue
        if isinstance(value, dict):
            verdict = bool(value.get("verdict", False))
            reason = str(value.get("reason", ""))
            evidence_ids = [str(x) for x in value.get("evidence_ids", [])]
        else:
            verdict = bool(value)
            reason = ""
            evidence_ids = []
        parsed[key] = OptionJudgement(verdict=verdict, reason=reason, evidence_ids=evidence_ids)
    return parsed


def extract_answer_from_text(text: str, question: Question) -> str:
    match = re.search(r'"answer"\s*:\s*"([A-D]+)"', text, flags=re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1), question)
    match = re.search(r"(?:答案|answer)\s*[:：]?\s*([A-D]{1,4})", text, flags=re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1), question)
    return normalize_answer(text, question)
