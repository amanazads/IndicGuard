"""
Abstract model runner interface and registry.
All concrete providers implement ModelRunner.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str
    provider: str
    model: str
    description: str = ""
    thinking: bool = False
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    model_name: str
    text: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class ModelRunner(ABC):
    """Abstract base for all model providers."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        conversation: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate a response given a system prompt and conversation history.

        Args:
            system_prompt: The collections agent system prompt.
            conversation: List of {"role": "user"|"assistant", "content": "..."}.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ModelResponse with text, latency, and token info.
        """

    @property
    def name(self) -> str:
        return self.config.name


def load_model_configs(config_path: str = "config/models.yaml") -> list[ModelConfig]:
    """Load model configurations from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with open(config_path) as f:
        data = yaml.safe_load(f)
    configs = []
    for m in data.get("models", []):
        configs.append(
            ModelConfig(
                name=m["name"],
                provider=m["provider"],
                model=m["model"],
                description=m.get("description", ""),
                thinking=m.get("thinking", False),
                options=m.get("options", {}),
            )
        )
    return configs


def get_runner(config: ModelConfig) -> ModelRunner:
    """Factory: returns the appropriate ModelRunner for a given config."""
    from src.ollama_runner import OllamaRunner
    from src.api_runner import OpenAIRunner, GeminiRunner

    provider = config.provider.lower()
    if provider == "ollama":
        return OllamaRunner(config)
    elif provider == "openai":
        return OpenAIRunner(config)
    elif provider == "gemini":
        return GeminiRunner(config)
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Supported: ollama, openai, gemini. "
            "Add a new runner in src/api_runner.py to extend."
        )


def get_benchmark_config(config_path: str = "config/models.yaml") -> dict:
    """Load the benchmark section from models.yaml."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data.get("benchmark", {})


def load_judge_config(config_path: str = "config/models.yaml") -> ModelConfig | None:
    """
    Load the LLM-judge configuration from the `judge:` section of models.yaml,
    if present. Returns None if the file or section is missing, so callers can
    fall back to their own default.
    """
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        data = yaml.safe_load(f)
    j = data.get("judge")
    if not j:
        return None
    options = dict(j.get("options", {}))
    if "temperature" in j:
        options.setdefault("temperature", j["temperature"])
    if "max_output_tokens" in j:
        options.setdefault("max_output_tokens", j["max_output_tokens"])
    return ModelConfig(
        name=j.get("name", "judge_local"),
        provider=j["provider"],
        model=j["model"],
        description=j.get("description", "Configured LLM judge"),
        thinking=j.get("thinking", False),
        options=options,
    )
