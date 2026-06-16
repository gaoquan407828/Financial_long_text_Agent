from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    domain: str
    title: str
    path: str
    source_type: str
    text: str
    pages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(**data)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    domain: str
    title: str
    page_start: int
    page_end: int
    section_path: list[str]
    text: str
    keywords: list[str]
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass(slots=True)
class Question:
    qid: str
    domain: str
    split: str
    question: str
    options: dict[str, str]
    answer_format: str
    type: str
    doc_ids: list[str] = field(default_factory=list)
    answer: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Question":
        options = {str(k).upper(): str(v) for k, v in data.get("options", {}).items()}
        return cls(
            qid=str(data["qid"]),
            domain=str(data.get("domain", "")),
            split=str(data.get("split", "")),
            question=str(data.get("question", "")),
            options=options,
            answer_format=str(data.get("answer_format", "multi")).lower(),
            type=str(data.get("type", "")),
            doc_ids=[str(x) for x in data.get("doc_ids") or []],
            answer=data.get("answer"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TokenUsage":
        data = data or {}
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
        )


@dataclass(slots=True)
class OptionJudgement:
    verdict: bool
    reason: str = ""
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    chunk_id: str
    doc_id: str
    domain: str
    title: str
    page_start: int
    page_end: int
    section_path: list[str]
    text: str
    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    option: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceBundle:
    question_id: str
    items: list[EvidenceItem]
    used_doc_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerResult:
    qid: str
    answer: str
    option_judgements: dict[str, OptionJudgement]
    raw_model_output: str
    token_usage: TokenUsage
    evidence_chunks: list[EvidenceItem]
    used_doc_ids: list[str]
    confidence: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["option_judgements"] = {
            k: v.to_dict() if isinstance(v, OptionJudgement) else v
            for k, v in self.option_judgements.items()
        }
        data["token_usage"] = self.token_usage.to_dict()
        data["evidence_chunks"] = [item.to_dict() for item in self.evidence_chunks]
        return data
