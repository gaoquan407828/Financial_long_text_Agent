from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from agent.utils import load_yaml, resolve_path


def mask(value: str | None) -> str:
    if not value:
        return "MISSING"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local environment for running Qwen inference.")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    load_dotenv()
    config = load_yaml(resolve_path(args.config))
    index_path = resolve_path(config["index_dir"]) / "lexical_index.json"
    api_key_env = str(config.get("api_key_env", "QWEN_API_KEY"))
    if api_key_env.lower().startswith(("sk-", "sk_", "dashscope", "bearer ")):
        print("Status: NOT READY - config api_key_env should be QWEN_API_KEY, not the real API key.")
        raise SystemExit(1)
    api_key = os.getenv(api_key_env) or os.getenv("DASHSCOPE_API_KEY")
    api_base = os.getenv("QWEN_API_BASE") or str(config.get("api_base", ""))
    model = os.getenv("QWEN_MODEL") or str(config.get("model_name", ""))

    print(f"Python: {sys.version.split()[0]}")
    print(f"Config: {resolve_path(args.config)}")
    print(f"Index: {'OK' if index_path.exists() else 'MISSING'} ({index_path})")
    print(f"API key ({api_key_env}/DASHSCOPE_API_KEY): {mask(api_key)}")
    print(f"API base: {api_base or 'MISSING'}")
    print(f"Model: {model or 'MISSING'}")
    if not api_key:
        print("Status: NOT READY - set QWEN_API_KEY in PowerShell or .env before formal inference.")
        raise SystemExit(1)
    if not api_base or not model:
        print("Status: NOT READY - API base or model is missing.")
        raise SystemExit(1)
    print("Status: READY")


if __name__ == "__main__":
    main()
