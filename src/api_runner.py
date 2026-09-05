"""
Hosted API runners: OpenAI and Gemini.
Add new providers here by subclassing ModelRunner.
"""

from __future__ import annotations

import os
import time
from typing import Any
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from src.models import ModelConfig, ModelResponse, ModelRunner


class OpenAIRunner(ModelRunner):
    """Calls the OpenAI Chat Completions API."""

    def generate(
        self,
        system_prompt: str,
        conversation: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        try:
            import openai
        except ImportError:
            return ModelResponse(
                model_name=self.config.name,
                text="",
                latency_seconds=0,
                error="openai package not installed. Run: pip install openai",
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return ModelResponse(
                model_name=self.config.name,
                text="",
                latency_seconds=0,
                error="OPENAI_API_KEY not set in environment / .env file.",
            )

        client = openai.OpenAI(api_key=api_key)
        messages = [{"role": "system", "content": system_prompt}] + conversation

        opts = {**self.config.options, **kwargs}

        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=opts.get("temperature", 0.7),
                max_tokens=opts.get("max_tokens", 512),
            )
            elapsed = time.perf_counter() - start
            text = response.choices[0].message.content or ""
            usage = response.usage
            return ModelResponse(
                model_name=self.config.name,
                text=text,
                latency_seconds=elapsed,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return ModelResponse(
                model_name=self.config.name,
                text="",
                latency_seconds=elapsed,
                error=str(exc),
            )


class GeminiRunner(ModelRunner):
    """Calls the Google Gemini API using the google-genai SDK."""

    def generate(
        self,
        system_prompt: str,
        conversation: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return ModelResponse(
                model_name=self.config.name,
                text="",
                latency_seconds=0,
                error="google-genai package not installed. Run: pip install google-genai",
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return ModelResponse(
                model_name=self.config.name,
                text="",
                latency_seconds=0,
                error="GEMINI_API_KEY not set in environment / .env file.",
            )

        opts = {**self.config.options, **kwargs}
        client = genai.Client(api_key=api_key)

        # Build contents list (user/model turns)
        contents = []
        for msg in conversation:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=opts.get("temperature", 0.7),
            max_output_tokens=opts.get("max_output_tokens", 512),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        max_retries = kwargs.pop("max_retries", 4)
        last_error = ""

        for attempt in range(max_retries):
            start = time.perf_counter()
            try:
                model_name = self.config.model.replace("models/", "")

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                elapsed = time.perf_counter() - start
                text = response.text or ""

                prompt_tokens = None
                completion_tokens = None
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
                    completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

                return ModelResponse(
                    model_name=self.config.name,
                    text=text,
                    latency_seconds=elapsed,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start
                err_str = str(exc)
                last_error = err_str
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    time.sleep(2 ** attempt + 1)
                    continue
                return ModelResponse(
                    model_name=self.config.name,
                    text="",
                    latency_seconds=elapsed,
                    error=err_str,
                )

        return ModelResponse(
            model_name=self.config.name,
            text="",
            latency_seconds=0.0,
            error=f"Gemini call failed after {max_retries} attempts: {last_error}",
        )
