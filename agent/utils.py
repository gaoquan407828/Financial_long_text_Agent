from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

import yaml


ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)(?::([^}]*))?}$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or project_root()) / p


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        match = ENV_PATTERN.match(value.strip())
        if match:
            name, default = match.groups()
            return os.getenv(name, default or "")
        return value
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env(yaml.safe_load(f) or {})


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(*parts: Any, length: int = 12) -> str:
    raw = "::".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:length]


def setup_logging(log_dir: str | Path, name: str = "agent", level: int = logging.INFO) -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def command_exists(command: str) -> bool:
    try:
        subprocess.run(
            [command, "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except Exception:
        pass
    from shutil import which

    return which(command) is not None


def run_command(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def compact_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def script_bootstrap() -> Path:
    root = project_root()
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
