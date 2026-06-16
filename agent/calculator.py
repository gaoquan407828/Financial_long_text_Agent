from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .question_analyzer import QuestionAnalysis
from .schema import EvidenceItem, Question


NUMBER_RE = re.compile(
    r"(?P<num>-?\d+(?:,\d{3})*(?:\.\d+)?)(?P<unit>\s*(?:%|亿元|万元|元|倍|日|天|个月|年|个工作日))?"
)
YEAR_VALUE_RE = re.compile(
    r"(?P<year>20\d{2}|19\d{2})年?.{0,36}?"
    r"(?P<num>-?\d+(?:,\d{3})*(?:\.\d+)?)(?P<unit>\s*(?:%|亿元|万元|元|倍))"
)
VALUE_YEAR_RE = re.compile(
    r"(?P<num>-?\d+(?:,\d{3})*(?:\.\d+)?)(?P<unit>\s*(?:%|亿元|万元|元|倍)).{0,36}?"
    r"(?P<year>20\d{2}|19\d{2})年?"
)
METRIC_RE = re.compile(
    r"(营业收入|营业总收入|归母净利润|净利润|扣非净利润|经营活动现金流量净额|经营现金流|研发投入|研发费用|"
    r"资产总额|负债总额|所有者权益|发行规模|募集资金|现金分红|分红金额|回购金额)"
)


@dataclass(slots=True)
class NumericValue:
    value: float
    unit: str
    raw: str
    year: int | None = None
    metric: str = ""
    evidence_id: str = ""

    def normalized(self) -> float:
        if self.unit == "亿元":
            return self.value * 100000000
        if self.unit == "万元":
            return self.value * 10000
        return self.value

    def display(self) -> str:
        year = f"{self.year}年 " if self.year else ""
        metric = f"{self.metric} " if self.metric else ""
        return f"{self.evidence_id} {year}{metric}{self.raw}".strip()


@dataclass(slots=True)
class CalculationResult:
    enabled: bool = False
    kind: str = ""
    answer: str = ""
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    values: list[NumericValue] = field(default_factory=list)

    @property
    def has_signal(self) -> bool:
        return bool(self.notes or self.answer)

    def to_context(self) -> str:
        if not self.has_signal:
            return ""
        lines = ["[代码计算提示]"]
        if self.kind:
            lines.append(f"计算类型：{self.kind}")
        if self.answer:
            lines.append(f"代码候选答案：{self.answer}，置信度：{self.confidence:.2f}")
        if self.notes:
            lines.append("计算过程：")
            lines.extend(f"- {note}" for note in self.notes[:8])
        if self.values:
            lines.append("数值候选：")
            lines.extend(f"- {value.display()}" for value in self.values[:16])
        lines.append("注意：代码提示只用于数值核验，最终仍需和证据原文逐项比对。")
        return "\n".join(lines)


class CalculationSolver:
    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None):
        calc_config = config.get("calculation", {})
        self.enabled = bool(calc_config.get("enabled", True))
        self.direct_answer_enabled = bool(calc_config.get("direct_answer_enabled", False))
        self.direct_answer_min_confidence = float(calc_config.get("direct_answer_min_confidence", 0.88))
        self.logger = logger or logging.getLogger(__name__)

    def solve(
        self,
        question: Question,
        analysis: QuestionAnalysis,
        evidence_items: list[EvidenceItem],
    ) -> CalculationResult:
        if not self.enabled or not analysis.requires_calculation:
            return CalculationResult(enabled=self.enabled)

        kind = self._calculation_kind(question)
        values = self._extract_evidence_values(evidence_items)
        option_values = self._extract_option_values(question)
        result = CalculationResult(enabled=True, kind=kind, values=values)
        result.notes.append(f"选项数值：{self._format_option_values(option_values)}")

        if kind == "yoy":
            self._solve_yoy(question, values, option_values, result)
        elif kind == "ratio":
            self._solve_ratio(question, values, option_values, result)
        elif kind == "difference":
            self._solve_difference(question, values, option_values, result)
        elif kind == "sum":
            self._solve_sum(question, values, option_values, result)
        else:
            result.notes.append("未识别到稳定公式，保留数值候选供模型核验。")

        if result.answer and result.confidence < self.direct_answer_min_confidence:
            result.notes.append("代码候选未达到直答阈值，仅作为提示。")
        return result

    def can_direct_answer(self, question: Question, result: CalculationResult) -> bool:
        return (
            self.direct_answer_enabled
            and question.answer_format in {"mcq", "tf"}
            and bool(result.answer)
            and result.confidence >= self.direct_answer_min_confidence
        )

    def _calculation_kind(self, question: Question) -> str:
        text = f"{question.question}\n{' '.join(question.options.values())}"
        if any(keyword in text for keyword in ["同比", "增长率", "增速", "复合增长", "CAGR"]) or re.search(r"较.*(增长|下降)", text):
            return "yoy"
        if any(keyword in text for keyword in ["占比", "比重", "比例"]) or re.search(r"占.*%", text):
            return "ratio"
        if any(keyword in text for keyword in ["差额", "相差", "增加了", "减少了", "下降了", "高出", "低于"]):
            return "difference"
        if any(keyword in text for keyword in ["合计", "总计", "总额", "之和"]):
            return "sum"
        return "unknown"

    def _extract_evidence_values(self, evidence_items: list[EvidenceItem]) -> list[NumericValue]:
        values: list[NumericValue] = []
        for item in evidence_items:
            text = item.text
            for match in YEAR_VALUE_RE.finditer(text):
                values.append(self._numeric_from_match(match, item.evidence_id, text))
            for match in VALUE_YEAR_RE.finditer(text):
                values.append(self._numeric_from_match(match, item.evidence_id, text))
            if len(values) >= 80:
                break
        return self._dedup_values(values)

    def _numeric_from_match(self, match: re.Match[str], evidence_id: str, text: str) -> NumericValue:
        value = self._to_float(match.group("num"))
        unit = (match.group("unit") or "").strip()
        year = int(match.group("year")) if match.groupdict().get("year") else None
        start = max(0, match.start() - 32)
        prefix = text[start : match.start()]
        metric_match = list(METRIC_RE.finditer(prefix))
        metric = metric_match[-1].group(0) if metric_match else ""
        raw = f"{match.group('num')}{unit}"
        return NumericValue(value=value, unit=unit, raw=raw, year=year, metric=metric, evidence_id=evidence_id)

    def _extract_option_values(self, question: Question) -> dict[str, list[NumericValue]]:
        option_values: dict[str, list[NumericValue]] = {}
        for key, text in question.options.items():
            values: list[NumericValue] = []
            for match in NUMBER_RE.finditer(text):
                unit = (match.group("unit") or "").strip()
                if not unit and len(match.group("num")) <= 1:
                    continue
                values.append(
                    NumericValue(
                        value=self._to_float(match.group("num")),
                        unit=unit,
                        raw=f"{match.group('num')}{unit}",
                    )
                )
            option_values[key] = values
        return option_values

    def _solve_yoy(
        self,
        question: Question,
        values: list[NumericValue],
        option_values: dict[str, list[NumericValue]],
        result: CalculationResult,
    ) -> None:
        same_metric = self._best_year_pair(question, values)
        if not same_metric:
            result.notes.append("未找到两个年份的同指标数值，无法稳定计算同比。")
            return
        old, new = same_metric
        if abs(old.normalized()) < 1e-9:
            result.notes.append("基期数值接近 0，跳过同比计算。")
            return
        pct = (new.normalized() - old.normalized()) / abs(old.normalized()) * 100
        result.notes.append(f"同比公式：({new.display()} - {old.display()}) / {old.display()} = {pct:.4f}%")
        self._match_numeric_option(option_values, pct, "%", result)

    def _solve_ratio(
        self,
        question: Question,
        values: list[NumericValue],
        option_values: dict[str, list[NumericValue]],
        result: CalculationResult,
    ) -> None:
        candidates = [v for v in values if v.unit in {"亿元", "万元", "元", "%", "倍"}]
        if len(candidates) < 2:
            result.notes.append("数值候选少于 2 个，无法稳定计算占比。")
            return
        numerator, denominator = candidates[0], candidates[1]
        if abs(denominator.normalized()) < 1e-9:
            result.notes.append("分母接近 0，跳过占比计算。")
            return
        pct = numerator.normalized() / denominator.normalized() * 100
        result.notes.append(f"占比试算：{numerator.display()} / {denominator.display()} = {pct:.4f}%")
        self._match_numeric_option(option_values, pct, "%", result)

    def _solve_difference(
        self,
        question: Question,
        values: list[NumericValue],
        option_values: dict[str, list[NumericValue]],
        result: CalculationResult,
    ) -> None:
        same_metric = self._best_year_pair(question, values)
        if not same_metric:
            result.notes.append("未找到两个可比数值，无法稳定计算差额。")
            return
        old, new = same_metric
        diff = new.normalized() - old.normalized()
        unit = new.unit if new.unit in {"亿元", "万元", "元"} else ""
        display_value = self._from_yuan(diff, unit) if unit else diff
        result.notes.append(f"差额试算：{new.display()} - {old.display()} = {display_value:.4f}{unit}")
        self._match_numeric_option(option_values, display_value, unit, result)

    def _solve_sum(
        self,
        question: Question,
        values: list[NumericValue],
        option_values: dict[str, list[NumericValue]],
        result: CalculationResult,
    ) -> None:
        candidates = [v for v in values[:6] if v.unit in {"亿元", "万元", "元"}]
        if len(candidates) < 2:
            result.notes.append("数值候选少于 2 个，无法稳定计算合计。")
            return
        total_yuan = sum(v.normalized() for v in candidates)
        unit = candidates[0].unit
        total = self._from_yuan(total_yuan, unit)
        result.notes.append(f"合计试算：前 {len(candidates)} 个金额候选求和 = {total:.4f}{unit}")
        self._match_numeric_option(option_values, total, unit, result)

    def _best_year_pair(self, question: Question, values: list[NumericValue]) -> tuple[NumericValue, NumericValue] | None:
        metric_hint = self._metric_hint(question)
        candidates = [v for v in values if v.year and v.unit in {"亿元", "万元", "元", "%", "倍"}]
        if metric_hint:
            metric_candidates = [v for v in candidates if metric_hint in v.metric or v.metric in metric_hint]
            if len(metric_candidates) >= 2:
                candidates = metric_candidates
        if len(candidates) < 2:
            return None
        by_metric: dict[str, list[NumericValue]] = {}
        for value in candidates:
            by_metric.setdefault(value.metric or "_", []).append(value)
        groups = sorted(by_metric.values(), key=len, reverse=True)
        selected = sorted(groups[0], key=lambda v: (v.year or 0, v.evidence_id))
        return selected[0], selected[-1]

    def _metric_hint(self, question: Question) -> str:
        text = f"{question.question}\n{' '.join(question.options.values())}"
        match = METRIC_RE.search(text)
        return match.group(0) if match else ""

    def _match_numeric_option(
        self,
        option_values: dict[str, list[NumericValue]],
        target: float,
        unit: str,
        result: CalculationResult,
    ) -> None:
        best_key = ""
        best_error = math.inf
        for key, values in option_values.items():
            for value in values:
                candidate = value.value
                if unit and value.unit and unit != value.unit:
                    if unit in {"亿元", "万元", "元"} and value.unit in {"亿元", "万元", "元"}:
                        candidate = self._from_yuan(value.normalized(), unit)
                    elif unit == "%" and value.unit != "%":
                        continue
                error = abs(candidate - target)
                if error < best_error:
                    best_key = key
                    best_error = error
        if not best_key:
            result.notes.append("未能把计算结果匹配到选项数值。")
            return
        scale = max(abs(target), 1.0)
        confidence = max(0.0, 1.0 - best_error / scale)
        if unit == "%":
            confidence = max(0.0, 1.0 - best_error / 3.0)
        result.answer = best_key
        result.confidence = round(min(0.99, confidence), 4)
        result.notes.append(f"最接近选项：{best_key}，误差 {best_error:.4f}{unit}")

    def _format_option_values(self, option_values: dict[str, list[NumericValue]]) -> str:
        parts: list[str] = []
        for key, values in sorted(option_values.items()):
            raw = ", ".join(v.raw for v in values) or "无"
            parts.append(f"{key}={raw}")
        return "；".join(parts)

    def _dedup_values(self, values: list[NumericValue]) -> list[NumericValue]:
        seen: set[tuple[float, str, int | None, str, str]] = set()
        deduped: list[NumericValue] = []
        for value in values:
            key = (round(value.value, 6), value.unit, value.year, value.metric, value.evidence_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        return deduped

    def _to_float(self, value: str) -> float:
        return float(value.replace(",", ""))

    def _from_yuan(self, value: float, unit: str) -> float:
        if unit == "亿元":
            return value / 100000000
        if unit == "万元":
            return value / 10000
        return value
