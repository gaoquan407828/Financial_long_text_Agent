from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.pipeline import Pipeline
from agent.utils import load_yaml, resolve_path, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the financial long-text QA Agent.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--questions", default="dataset/questions/group_a")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run without Qwen API using a mock response for smoke tests.")
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    logger = setup_logging(resolve_path(config["log_dir"]), name="run_answer")
    pipeline = Pipeline(config, dry_run=args.dry_run, logger=logger)
    output_dir = resolve_path(args.output_dir) if args.output_dir else None
    pipeline.run(resolve_path(args.questions), limit=args.limit, output_dir=output_dir)


if __name__ == "__main__":
    main()
