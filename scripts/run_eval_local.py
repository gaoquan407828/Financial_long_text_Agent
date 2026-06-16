from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.answer_parser import normalize_answer
from agent.document_loader import load_questions
from agent.utils import load_yaml, resolve_path, setup_logging, write_json


def load_answers(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {row["qid"]: row["answer"] for row in reader if row["qid"] != "summary"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answer.csv locally when ground truth answers are available.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--questions", required=True, help="Question JSON containing answer fields.")
    parser.add_argument("--answers", default="outputs/answer.csv")
    parser.add_argument("--output", default="outputs/local_eval.json")
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    logger = setup_logging(resolve_path(config["log_dir"]), name="eval_local")
    questions = load_questions(resolve_path(args.questions))
    predictions = load_answers(resolve_path(args.answers))
    rows = []
    stats = defaultdict(lambda: [0, 0])

    for question in questions:
        if question.answer is None:
            continue
        pred = normalize_answer(predictions.get(question.qid, ""), question)
        gold = normalize_answer(question.answer, question)
        correct = pred == gold
        rows.append({"qid": question.qid, "pred": pred, "gold": gold, "correct": correct})
        for key in ["overall", f"domain:{question.domain}", f"format:{question.answer_format}", f"type:{question.type}"]:
            stats[key][1] += 1
            stats[key][0] += int(correct)

    if not rows:
        logger.warning("No ground-truth answer fields found in %s", args.questions)
    report = {
        "count": len(rows),
        "accuracy": (sum(1 for r in rows if r["correct"]) / len(rows)) if rows else None,
        "breakdown": {
            key: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1] if v[1] else None}
            for key, v in sorted(stats.items())
        },
        "errors": [r for r in rows if not r["correct"]],
    }
    write_json(resolve_path(args.output), report)
    logger.info("local eval written to %s", args.output)


if __name__ == "__main__":
    main()
