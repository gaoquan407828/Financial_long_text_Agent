from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

from .schema import TokenUsage
from .token_tracker import TokenTracker


@dataclass(slots=True)
class ChatResult:
    content: str
    usage: TokenUsage
    raw: dict[str, Any]


class QwenClientError(RuntimeError):
    """Base class for Qwen client failures."""


class QwenConfigError(QwenClientError):
    """Raised when API configuration is missing or invalid."""


class QwenAPIError(QwenClientError):
    """Raised when Qwen API calls fail after retries."""


class QwenClient:
    def __init__(
        self,
        config: dict[str, Any],
        tracker: TokenTracker,
        logger: logging.Logger | None = None,
        dry_run: bool = False,
    ):
        load_dotenv()
        self.config = config
        self.tracker = tracker
        self.logger = logger or logging.getLogger(__name__)
        self.model = os.getenv("QWEN_MODEL", str(config.get("model_name", "qwen3.6-plus")))
        self.api_base = os.getenv("QWEN_API_BASE", str(config.get("api_base", ""))).rstrip("/")
        key_env = str(config.get("api_key_env", "QWEN_API_KEY"))
        if key_env.lower().startswith(("sk-", "sk_", "dashscope", "bearer ")):
            raise QwenConfigError(
                "config api_key_env should be an environment variable name, not the API key itself. "
                "Set api_key_env: QWEN_API_KEY and put the real key in .env or PowerShell env."
            )
        self.api_key = os.getenv(key_env) or os.getenv("DASHSCOPE_API_KEY")
        self.max_retries = int(config.get("max_retries", 3))
        self.timeout = int(config.get("timeout_seconds", 120))
        self.temperature = float(config.get("temperature", 0))
        self.dry_run = dry_run

    def validate_configuration(self) -> None:
        if self.dry_run:
            return
        if not self.api_key:
            raise QwenConfigError(
                "Qwen API key is missing. Set QWEN_API_KEY in the environment or .env. "
                "Use --dry-run only for smoke tests."
            )
        if not self.api_base:
            raise QwenConfigError("Qwen API base is missing. Set QWEN_API_BASE or config api_base.")
        if not self.model:
            raise QwenConfigError("Qwen model is missing. Set QWEN_MODEL or config model_name.")

    def chat(self, messages: list[dict[str, str]], qid: str) -> ChatResult:
        if self.dry_run:
            result = self._mock_response(messages)
            self.tracker.record(qid, result.usage)
            return result
        self.validate_configuration()

        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code >= 400:
                    message = f"HTTP {response.status_code}: {response.text[:500]}"
                    if response.status_code in {401, 403, 404}:
                        raise QwenConfigError(message)
                    raise QwenAPIError(message)
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                usage = TokenUsage.from_dict(data.get("usage"))
                self.tracker.record(qid, usage)
                return ChatResult(content=content, usage=usage, raw=data)
            except QwenConfigError:
                raise
            except Exception as exc:
                last_error = exc
                self.logger.warning("Qwen call failed for %s attempt %s/%s: %s", qid, attempt, self.max_retries, exc)
                time.sleep(min(2 * attempt, 8))
        raise QwenAPIError(f"Qwen call failed after retries: {last_error}")

    def _mock_response(self, messages: list[dict[str, str]]) -> ChatResult:
        user = messages[-1]["content"]
        answer_format = "multi"
        if "answer_format: mcq" in user:
            answer_format = "mcq"
        elif "answer_format: tf" in user:
            answer_format = "tf"
        answer = "A" if answer_format in {"mcq", "tf"} else "A"
        payload = {
            "qid": self._extract_qid(user),
            "option_judgements": {
                "A": {"verdict": True, "reason": "dry-run mock", "evidence_ids": []},
                "B": {"verdict": False, "reason": "dry-run mock", "evidence_ids": []},
                "C": {"verdict": False, "reason": "dry-run mock", "evidence_ids": []},
                "D": {"verdict": False, "reason": "dry-run mock", "evidence_ids": []},
            },
            "answer": answer,
            "confidence": 0.0,
        }
        content = json.dumps(payload, ensure_ascii=False)
        approx_prompt = sum(len(m["content"]) for m in messages) // 4
        usage = TokenUsage(prompt_tokens=approx_prompt, completion_tokens=len(content) // 4, total_tokens=approx_prompt + len(content) // 4)
        return ChatResult(content=content, usage=usage, raw={"mock": True, "choices": [{"message": {"content": content}}]})

    def _extract_qid(self, prompt: str) -> str:
        for line in prompt.splitlines():
            if line.startswith("qid:"):
                return line.split(":", 1)[1].strip()
        return ""
