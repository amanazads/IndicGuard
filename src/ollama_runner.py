"""
Ollama local model runner.
Calls the Ollama REST API (http://localhost:11434).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import requests

from src.models import ModelConfig, ModelResponse, ModelRunner


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from models with reasoning tokens."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


class OllamaRunner(ModelRunner):
    """Runs inference against a locally-hosted Ollama model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    def generate(
        self,
        system_prompt: str,
        conversation: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        messages = [{"role": "system", "content": system_prompt}] + conversation

        options = {**self.config.options, **kwargs}
        # Explicitly ensure thinking is disabled if configured
        if not self.config.thinking:
            options["thinking"] = False

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        max_retries = kwargs.pop("max_retries", 3)
        timeout = kwargs.pop("timeout", 180)
        last_error = ""

        for attempt in range(max_retries):
            start = time.perf_counter()
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=timeout,
                )
                elapsed = time.perf_counter() - start

                if resp.status_code == 404:
                    return ModelResponse(
                        model_name=self.config.name,
                        text="",
                        latency_seconds=elapsed,
                        error=(
                            f"Model '{self.config.model}' not installed in Ollama. "
                            f"Run: ollama pull {self.config.model}"
                        ),
                    )
                elif resp.status_code != 200:
                    return ModelResponse(
                        model_name=self.config.name,
                        text="",
                        latency_seconds=elapsed,
                        error=f"Ollama HTTP {resp.status_code}: {resp.text[:200]}",
                    )

                data = resp.json()
                message = data.get("message", {})
                raw_text = message.get("content", "")
                if not raw_text and data.get("response"):
                    raw_text = data.get("response", "")

                # If content is still empty and thinking is available
                if not raw_text and message.get("thinking"):
                    thinking_text = message.get("thinking", "")
                    if self.config.thinking:
                        raw_text = thinking_text
                    else:
                        # Look for drafted response or return clean text
                        draft_match = re.search(r'(?:Suggested Response|Draft Response|Response|Final Response|Output):\s*["\']?(.*)', thinking_text, re.IGNORECASE | re.DOTALL)
                        if draft_match:
                            raw_text = draft_match.group(1).strip().strip('"\'')
                        else:
                            raw_text = _strip_thinking(thinking_text)

                text = _strip_thinking(raw_text) if not self.config.thinking else raw_text
                usage = data.get("usage", {})

                return ModelResponse(
                    model_name=self.config.name,
                    text=text,
                    latency_seconds=elapsed,
                    prompt_tokens=usage.get("prompt_tokens") or data.get("prompt_eval_count"),
                    completion_tokens=usage.get("completion_tokens") or data.get("eval_count"),
                )

            except requests.exceptions.ConnectionError:
                elapsed = time.perf_counter() - start
                return ModelResponse(
                    model_name=self.config.name,
                    text="",
                    latency_seconds=elapsed,
                    error=(
                        f"Cannot connect to Ollama at {self.base_url}. "
                        f"Start Ollama and pull '{self.config.model}': "
                        f"  ollama pull {self.config.model}"
                    ),
                )
            except (requests.exceptions.Timeout, requests.exceptions.RequestException) as exc:
                elapsed = time.perf_counter() - start
                last_error = str(exc)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return ModelResponse(
                    model_name=self.config.name,
                    text="",
                    latency_seconds=elapsed,
                    error=f"Ollama request failed after {max_retries} attempts: {last_error}",
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start
                return ModelResponse(
                    model_name=self.config.name,
                    text="",
                    latency_seconds=elapsed,
                    error=str(exc),
                )

        return ModelResponse(
            model_name=self.config.name,
            text="",
            latency_seconds=0.0,
            error=f"Ollama call failed: {last_error}",
        )
