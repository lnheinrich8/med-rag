"""Embedding model wrapper (sentence-transformers).

Two asymmetries this class hides from the rest of the pipeline:

* bge-base wants an *instruction prefix* prepended to **queries** but not to the
  passages it stores. PubMedBERT wants neither. We key the prefix off the model
  name so callers just say "embed a query" vs "embed passages".
* We L2-normalize vectors so pgvector's cosine distance (`<=>`) ranks correctly
  and the stored HNSW index (vector_cosine_ops) matches.

The model is loaded lazily (first encode) so importing this module stays cheap.
"""

from __future__ import annotations

from functools import cached_property, lru_cache

from ..config import EmbedConfig, settings

# Models that need an instruction prefix on queries only (passages get none).
_QUERY_INSTRUCTIONS = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' when a GPU is present, else 'cpu'."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str):
    """Load (and cache) one SentenceTransformer per (model, device).

    Caching is process-wide and deliberate: callers construct a fresh ``Embedder``
    per query (e.g. ``dense_search``), so without this the eval loop would reload
    the model onto the GPU once *per question* and exhaust VRAM. Keyed on the
    plain string args so the cache actually hits.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


class Embedder:
    """Lazily-loaded sentence-transformers model with query/passage asymmetry."""

    def __init__(self, cfg: EmbedConfig) -> None:
        self.cfg = cfg

    @cached_property
    def _model(self):
        return _load_model(self.cfg.model, _resolve_device(settings.device))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts for storage (no instruction prefix)."""
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self.cfg.normalize,
            batch_size=32,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (prepends the model's instruction prefix)."""
        prefix = _QUERY_INSTRUCTIONS.get(self.cfg.model, "")
        vector = self._model.encode(prefix + text, normalize_embeddings=self.cfg.normalize)
        return vector.tolist()
