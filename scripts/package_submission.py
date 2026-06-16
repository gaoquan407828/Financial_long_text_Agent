from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.utils import ensure_dir, resolve_path


DEFAULT_ITEMS = [
    "outputs/answer.csv",
    "outputs/evidence.json",
    "processed_data",
    "agent",
    "scripts",
    "config",
    "logs",
    "requirements.txt",
    "pyproject.toml",
    ".env.example",
    "readme.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package reproducible submission files.")
    parser.add_argument("--output", default="outputs/submission.zip")
    parser.add_argument("--include-logs", action="store_true")
    args = parser.parse_args()

    output = resolve_path(args.output)
    ensure_dir(output.parent)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in DEFAULT_ITEMS:
            path = resolve_path(item)
            if not path.exists():
                continue
            if path.name == "logs" and not args.include_logs:
                continue
            if path.is_file():
                zf.write(path, path.relative_to(resolve_path(".")))
            else:
                for child in path.rglob("*"):
                    if child.is_file():
                        zf.write(child, child.relative_to(resolve_path(".")))
    print(f"submission package written to {output}")


if __name__ == "__main__":
    main()
