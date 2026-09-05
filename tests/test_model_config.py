"""Tests for model configuration loading, runner factories, and error handling."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    ModelConfig,
    ModelResponse,
    get_benchmark_config,
    get_runner,
    load_model_configs,
)
from src.ollama_runner import OllamaRunner
from src.api_runner import GeminiRunner


class TestModelConfiguration:
    def test_load_models_yaml(self):
        configs = load_model_configs("config/models.yaml")
        assert len(configs) >= 2
        names = [c.name for c in configs]
        assert "gemini_baseline" in names

    def test_thinking_disabled_in_configs(self):
        configs = load_model_configs("config/models.yaml")
        for c in configs:
            if c.provider == "ollama":
                assert c.thinking is False, f"Model {c.name} should have thinking=False"

    def test_missing_config_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_model_configs("config/non_existent.yaml")

    def test_get_benchmark_config(self):
        bcfg = get_benchmark_config("config/models.yaml")
        assert "default_lender" in bcfg
        assert "default_name" in bcfg
        assert "timeout_seconds" in bcfg


class TestRunnerFactory:
    def test_ollama_runner_creation(self):
        cfg = ModelConfig(
            name="qwen_test",
            provider="ollama",
            model="qwen3.5:4b",
            thinking=False,
        )
        runner = get_runner(cfg)
        assert isinstance(runner, OllamaRunner)
        assert runner.name == "qwen_test"

    def test_gemini_runner_creation(self):
        cfg = ModelConfig(
            name="gemini_test",
            provider="gemini",
            model="gemini-flash-latest",
        )
        runner = get_runner(cfg)
        assert isinstance(runner, GeminiRunner)
        assert runner.name == "gemini_test"

    def test_unsupported_provider_raises_error(self):
        cfg = ModelConfig(
            name="invalid_test",
            provider="unsupported_provider",
            model="dummy",
        )
        with pytest.raises(ValueError, match="Unknown provider"):
            get_runner(cfg)


class TestRunnerErrorHandling:
    def test_ollama_unreachable_returns_structured_error(self):
        cfg = ModelConfig(
            name="qwen_offline",
            provider="ollama",
            model="qwen3.5:4b",
        )
        runner = OllamaRunner(cfg)
        # Point to unreachable port
        runner.base_url = "http://localhost:59999"
        resp = runner.generate(
            system_prompt="Test prompt",
            conversation=[{"role": "user", "content": "Hello"}],
        )
        assert resp.error is not None
        assert "Cannot connect to Ollama" in resp.error or "Connection" in resp.error
        assert resp.text == ""

    def test_gemini_missing_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg = ModelConfig(
            name="gemini_no_key",
            provider="gemini",
            model="gemini-flash-latest",
        )
        runner = GeminiRunner(cfg)
        resp = runner.generate(
            system_prompt="Test",
            conversation=[{"role": "user", "content": "Hello"}],
        )
        assert resp.error is not None
        assert "GEMINI_API_KEY not set" in resp.error
