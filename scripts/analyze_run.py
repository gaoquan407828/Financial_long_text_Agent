from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.answer_parser import normalize_answer
from agent.document_loader import load_questions
from agent.question_analyzer import QuestionAnalyzer
from agent.utils import load_json, load_yaml, resolve_path, setup_logging, write_json


def load_answer_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {row["qid"]: row.get("answer", "") for row in reader if row.get("qid") and row["qid"] != "summary"}


def load_evidence(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_json(path)
    if not isinstance(rows, list):
        return {}
    return {str(row.get("qid", "")): row for row in rows if row.get("qid")}


def pct(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze run outputs by domain, question kind, answer format, and evidence health.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--questions", default=None, help="Question JSON or directory. Defaults to config questions_dir.")
    parser.add_argument("--answers", default="outputs/answer.csv")
    parser.add_argument("--evidence", default="outputs/evidence.json")
    parser.add_argument("--output", default="outputs/run_analysis.json")
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    logger = setup_logging(resolve_path(config["log_dir"]), name="analyze_run")
    questions_path = resolve_path(args.questions or config["questions_dir"])
    questions = load_questions(questions_path)
    answers = load_answer_csv(resolve_path(args.answers))
    evidence_by_qid = load_evidence(resolve_path(args.evidence))
    analyzer = QuestionAnalyzer(config)

    breakdown: dict[str, Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, Any]] = []
    has_gold = any(q.answer is not None for q in questions)
    for question in questions:
        analysis = analyzer.analyze(question)
        pred = normalize_answer(answers.get(question.qid, ""), question)
        gold = normalize_answer(question.answer, question) if question.answer is not None else ""
        evidence = evidence_by_qid.get(question.qid, {})
        evidence_chunks = evidence.get("evidence_chunks") or []
        used_doc_ids = evidence.get("used_doc_ids") or []
        error = evidence.get("error")
        row = {
            "qid": question.qid,
            "domain": question.domain,
            "answer_format": question.answer_format,
            "question_type": question.type,
            "question_kind": analysis.kind,
            "requires_calculation": analysis.requires_calculation,
            "pred": pred,
            "gold": gold,
            "correct": (pred == gold) if gold else None,
            "evidence_count": len(evidence_chunks),
            "used_doc_count": len(used_doc_ids),
            "has_error": bool(error),
            "error": error,
        }
        rows.append(row)
        keys = [
            "overall",
            f"domain:{question.domain}",
            f"format:{question.answer_format}",
            f"kind:{analysis.kind}",
            f"calc:{analysis.requires_calculation}",
        ]
        for key in keys:
            breakdown[key]["total"] += 1
            if row["has_error"]:
                breakdown[key]["errors"] += 1
            if row["evidence_count"] == 0:
                breakdown[key]["no_evidence"] += 1
            if gold:
                breakdown[key]["with_gold"] += 1
                breakdown[key]["correct"] += int(pred == gold)

    report = {
        "question_count": len(rows),
        "has_gold": has_gold,
        "breakdown": {
            key: {
                **dict(counter),
                "accuracy": pct(counter.get("correct", 0), counter.get("with_gold", 0)),
                "error_rate": pct(counter.get("errors", 0), counter.get("total", 0)),
                "no_evidence_rate": pct(counter.get("no_evidence", 0), counter.get("total", 0)),
            }
            for key, counter in sorted(breakdown.items())
        },
        "rows": rows,
        "suspects": [
            row for row in rows
            if row["has_error"] or row["evidence_count"] == 0 or (row["correct"] is False)
        ],
    }
    write_json(resolve_path(args.output), report)
    logger.info("analysis written to %s", args.output)


if __name__ == "__main__":
    main()
