from __future__ import annotations

from collections import defaultdict

from .schema import TokenUsage


class TokenTracker:
    def __init__(self) -> None:
        self.total = TokenUsage()
        self.by_qid: dict[str, TokenUsage] = defaultdict(TokenUsage)

    def record(self, qid: str, usage: TokenUsage) -> None:
        self.total.add(usage)
        self.by_qid[qid].add(usage)

    def get(self, qid: str) -> TokenUsage:
        return self.by_qid.get(qid, TokenUsage())

    def summary(self) -> TokenUsage:
        return self.total
