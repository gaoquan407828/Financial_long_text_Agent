from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from .document_loader import load_questions
from .lexical_index import LexicalIndex
from .memory import MemoryManager
from .prompt_builder import PromptBuilder
from .qwen_client import QwenClient
from .reasoner import Reasoner
from .retriever import Retriever
from .schema import AnswerResult, TokenUsage
from .token_tracker import TokenTracker
from .utils import ensure_dir, resolve_path, setup_logging, write_json


class Pipeline:
    def __init__(self, config: dict[str, Any], dry_run: bool = False, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or setup_logging(resolve_path(config["log_dir"]), name="pipeline")
        self.tracker = TokenTracker()
        index_path = resolve_path(config["index_dir"]) / "lexical_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}. Run scripts/build_index.py first.")
        self.index = LexicalIndex.load(index_path, logger=self.logger)
        self.retriever = Retriever(self.index, config, self.logger)
        self.memory = MemoryManager(config, self.logger)
        self.prompt_builder = PromptBuilder(config)
        self.client = QwenClient(config, tracker=self.tracker, logger=self.logger, dry_run=dry_run)
        self.client.validate_configuration()
        self.reasoner = Reasoner(self.client, self.prompt_builder, config, self.logger)

    def run(self, questions_path: Path, limit: int | None = None, output_dir: Path | None = None) -> list[AnswerResult]:
        questions = load_questions(questions_path)
        if limit:
            questions = questions[:limit]
        self.logger.info("loaded %s questions from %s", len(questions), questions_path)
        results: list[AnswerResult] = []
        for idx, question in enumerate(questions, start=1):
            self.logger.info("answering %s (%s/%s)", question.qid, idx, len(questions))
            bundle = self.retriever.retrieve(question)
            compact_context, evidence_items = self.memory.compact(question, bundle)
            result = self.reasoner.answer(question, compact_context, evidence_items, bundle.used_doc_ids)
            results.append(result)
            self._write_question_log(result)

        out_dir = ensure_dir(output_dir or resolve_path(self.config["output_dir"]))
        self.write_outputs(results, out_dir)
        return results

    def write_outputs(self, results: list[AnswerResult], output_dir: Path) -> None:
        answer_path = output_dir / "answer.csv"
        evidence_path = output_dir / "evidence.json"
        summary_path = output_dir / "run_summary.json"
        total = TokenUsage()
        for result in results:
            total.add(result.token_usage)
        with answer_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
            writer.writeheader()
            writer.writerow(
                {
                    "qid": "summary",
                    "answer": "",
                    "prompt_tokens": total.prompt_tokens,
                    "completion_tokens": total.completion_tokens,
                    "total_tokens": total.total_tokens,
                }
            )
            for result in results:
                writer.writerow(
                    {
                        "qid": result.qid,
                        "answer": result.answer,
                        "prompt_tokens": result.token_usage.prompt_tokens,
                        "completion_tokens": result.token_usage.completion_tokens,
                        "total_tokens": result.token_usage.total_tokens,
                    }
                )
        write_json(evidence_path, [result.to_dict() for result in results])
        write_json(
            summary_path,
            {
                "question_count": len(results),
                "token_usage": total.to_dict(),
                "answer_csv": str(answer_path),
                "evidence_json": str(evidence_path),
            },
        )
        self.logger.info("wrote %s and %s", answer_path, evidence_path)

    def _write_question_log(self, result: AnswerResult) -> None:
        log_dir = ensure_dir(resolve_path(self.config["log_dir"]) / "questions")
        write_json(log_dir / f"{result.qid}.json", result.to_dict())
