from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.preprocessor import Preprocessor
from agent.utils import load_yaml, resolve_path, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw financial documents into JSON documents and chunks.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--domain", action="append", help="Only process one domain. Can be passed multiple times.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    logger = setup_logging(resolve_path(config["log_dir"]), name="preprocess")
    manifest = Preprocessor(config, logger).run(domains=set(args.domain or []) or None, limit=args.limit, force=args.force)
    logger.info("processed %s documents", manifest["document_count"])


if __name__ == "__main__":
    main()
