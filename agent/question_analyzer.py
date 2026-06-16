from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .schema import Question


CALC_RE = re.compile(
    r"(计算|算出|多少|几倍|占比|比重|比例|同比|环比|增长率|增速|增长了|下降了|减少了|增加了|"
    r"差额|相差|合计|总计|总额|平均|复合增长|CAGR|倍数)"
)
COMPARE_RE = re.compile(r"(高于|低于|超过|不超过|大于|小于|最多|最少|最高|最低|排名|相比|比较|是否超过)")
BOOLEAN_RE = re.compile(r"(是否|下列.*正确|下列.*错误|正确的是|错误的是|符合|不符合|属于|不属于)")
NUMERIC_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|亿元|万元|元|倍|日|天|个月|年)?")


@dataclass(slots=True)
class QuestionAnalysis:
    kind: str
    reasons: list[str] = field(default_factory=list)
    requires_calculation: bool = False
    requires_comparison: bool = False
    option_numeric_count: int = 0
    tags: list[str] = field(default_factory=list)

    def to_prompt_hint(self) -> str:
        tags = "、".join(self.tags) if self.tags else "无"
        reasons = "；".join(self.reasons) if self.reasons else "无"
        return (
            f"题型识别：{self.kind}\n"
            f"需要代码计算：{'是' if self.requires_calculation else '否'}\n"
            f"需要跨值比较：{'是' if self.requires_comparison else '否'}\n"
            f"选项数值数量：{self.option_numeric_count}\n"
            f"标签：{tags}\n"
            f"识别依据：{reasons}"
        )


class QuestionAnalyzer:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def analyze(self, question: Question) -> QuestionAnalysis:
        text = self._question_text(question)
        reasons: list[str] = []
        tags: list[str] = []
        option_numeric_count = sum(1 for value in question.options.values() if NUMERIC_RE.search(value))

        has_calc = bool(CALC_RE.search(text)) or option_numeric_count >= 2 and question.domain == "financial_reports"
        has_compare = bool(COMPARE_RE.search(text))
        has_boolean = bool(BOOLEAN_RE.search(text))

        if has_calc:
            reasons.append("题干或选项包含计算/比例/增长类词")
            tags.append("numeric")
        if has_compare:
            reasons.append("题干或选项包含比较类词")
            tags.append("compare")
        if has_boolean:
            reasons.append("题干或选项包含判断类表述")
            tags.append("judgement")
        if option_numeric_count:
            reasons.append(f"选项中含 {option_numeric_count} 个数值")

        if has_calc:
            kind = "calculation"
        elif has_compare:
            kind = "comparison"
        elif question.answer_format in {"tf"} or has_boolean:
            kind = "judgement"
        else:
            kind = "fact_lookup"

        if question.domain:
            tags.append(question.domain)
        if question.answer_format:
            tags.append(question.answer_format)

        return QuestionAnalysis(
            kind=kind,
            reasons=reasons,
            requires_calculation=has_calc,
            requires_comparison=has_compare or has_calc,
            option_numeric_count=option_numeric_count,
            tags=list(dict.fromkeys(tags)),
        )

    def _question_text(self, question: Question) -> str:
        options = "\n".join(question.options.values())
        return f"{question.question}\n{options}\n{question.type}"
