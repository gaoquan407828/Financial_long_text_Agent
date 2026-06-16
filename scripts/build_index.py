from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.lexical_index import build_and_save_index
from agent.utils import load_yaml, resolve_path, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lexical BM25-style index from processed chunks.")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    logger = setup_logging(resolve_path(config["log_dir"]), name="build_index")
    index_path = build_and_save_index(config, logger)
    logger.info("index written to %s", index_path)


if __name__ == "__main__":
    main()
