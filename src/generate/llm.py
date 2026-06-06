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

import httpx

from ..config import GenerationConfig

# Ollama occasionally drops a connection mid-run (transient server disconnect,
# read timeout, momentary unavailability). These surface as httpx errors and are
# almost always recoverable on a retry — the model stays loaded. A persistent
# failure (server actually down) exhausts the retries and raises.
_TRANSIENT_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
    ConnectionError,
)


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LLM:
    """Thin client over a local Ollama server exposing one chat-completion call."""

    def __init__(self, cfg: GenerationConfig, retries: int = 3) -> None:
        self.cfg = cfg
        self.retries = retries

    @cached_property
    def _client(self):
        from ollama import Client

        return Client(host=self.cfg.ollama_host)

    def complete(self, messages: list[dict]) -> Completion:
        """Run one chat completion (retrying transient errors) and return text + usage."""
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
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
            except _TRANSIENT_ERRORS as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))  # 1s, 2s, 4s backoff
                    continue
                raise RuntimeError(
                    f"Ollama at {self.cfg.ollama_host} failed after {self.retries + 1} "
                    f"attempts ({type(exc).__name__}: {exc}). Is `ollama serve` healthy?"
                ) from exc
            latency = time.perf_counter() - start
            return Completion(
                text=resp.message.content.strip(),
                prompt_tokens=resp.prompt_eval_count or 0,
                completion_tokens=resp.eval_count or 0,
                latency_s=latency,
            )
        raise RuntimeError(str(last_exc))  # unreachable; for type-checkers
