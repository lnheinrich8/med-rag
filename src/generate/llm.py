"""Local LLM wrapper (Qwen2.5 via Ollama).

We talk to a local Ollama server over its HTTP API rather than loading the model
in-process: Ollama ships prebuilt CUDA + runtime CPU dispatch, so it runs on the
GPU without the compile/instruction-set headaches of in-process llama.cpp. The
model is registered once from the downloaded GGUF via a Modelfile (see README).

``complete()`` runs one chat turn and returns the text plus token usage and
wall-clock latency (the eval harness records both). Ollama applies the model's
chat template, so we pass plain messages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cached_property

from ..config import GenerationConfig


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LLM:
    """Thin client over a local Ollama server exposing one chat-completion call."""

    def __init__(self, cfg: GenerationConfig) -> None:
        self.cfg = cfg

    @cached_property
    def _client(self):
        from ollama import Client

        return Client(host=self.cfg.ollama_host)

    def complete(self, messages: list[dict]) -> Completion:
        """Run one chat completion and return text + token usage + latency."""
        start = time.perf_counter()
        try:
            resp = self._client.chat(
                model=self.cfg.model,
                messages=messages,
                options={
                    "temperature": self.cfg.temperature,
                    "top_p": self.cfg.top_p,
                    "num_predict": self.cfg.max_tokens,
                    "num_ctx": self.cfg.n_ctx,
                },
                keep_alive=self.cfg.keep_alive,
            )
        except ConnectionError as exc:  # ollama server not running
            raise RuntimeError(
                f"Could not reach Ollama at {self.cfg.ollama_host}. Is it running? "
                "Start it with `ollama serve` (or the systemd service)."
            ) from exc
        latency = time.perf_counter() - start

        return Completion(
            text=resp.message.content.strip(),
            prompt_tokens=resp.prompt_eval_count or 0,
            completion_tokens=resp.eval_count or 0,
            latency_s=latency,
        )
