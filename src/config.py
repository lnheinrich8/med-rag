"""Configuration: runtime Settings (env) and ExperimentConfig (YAML).

Two distinct concerns are kept separate on purpose:

* ``Settings`` — environment/runtime wiring (DB URL, paths, device). Loaded from
  ``.env`` / process env. Stable across experiments.
* ``ExperimentConfig`` — the swappable pipeline knobs (chunker, embedder,
  retrieval mode, reranker, generation). Loaded from a YAML file under
  ``configs/``. This is what the eval harness sweeps to run ablations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Environment/runtime configuration. Override via .env or MEDRAG_* env vars."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="MEDRAG_",
        extra="ignore",
    )

    database_url: str = "postgresql://luke@localhost:5432/medrag"
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    models_dir: Path = PROJECT_ROOT / "models"
    hf_cache_dir: Path = PROJECT_ROOT / "hf_cache"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # "auto" resolves to cuda if available else cpu (see embed/generate layers).
    device: Literal["auto", "cpu", "cuda"] = "auto"


# ---------------------------------------------------------------------------
# Experiment configuration (the swappable pipeline)
# ---------------------------------------------------------------------------


class ChunkConfig(BaseModel):
    strategy: Literal["fixed", "recursive", "section", "semantic"] = "recursive"
    chunk_size: int = 512  # target tokens per chunk
    chunk_overlap: int = 64


class EmbedConfig(BaseModel):
    model: str = "BAAI/bge-base-en-v1.5"
    dim: int = 768  # must match the VECTOR(...) column in migrations/001_init.sql
    normalize: bool = True


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "sparse", "hybrid"] = "dense"
    top_k: int = 20  # candidates pulled before reranking
    rrf_k: int = 60  # reciprocal-rank-fusion constant for hybrid


class RerankConfig(BaseModel):
    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"
    top_n: int = 5  # kept after reranking, fed to the LLM


class GenerationConfig(BaseModel):
    # Empty until Step 3. A GGUF path on disk or an HF repo id.
    model_path: str = ""
    max_tokens: int = 512
    temperature: float = 0.1
    context_chunks: int = 5


class ExperimentConfig(BaseModel):
    """One reproducible pipeline configuration; the unit the eval harness sweeps."""

    name: str = "default"
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)


settings = Settings()
