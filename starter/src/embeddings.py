"""Local multilingual sentence embeddings using Sentence Transformers."""
import math
from typing import Protocol

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input document, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with stored document vectors."""
        ...


class SentenceTransformerEmbedder:
    """Create normalized document and query vectors with one local model."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = self._model.get_sentence_embedding_dimension
        dimensions = get_dimension()
        if dimensions is None or dimensions <= 0:
            raise RuntimeError(
                f"Model {model_name!r} did not report a valid embedding dimension."
            )
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one normalized vector per document, preserving input order."""
        return self._encode(texts)

    def embed_query(self, question: str) -> list[float]:
        """Embed a query with the same model and dimensions as documents."""
        vectors = self._encode([question])
        return vectors[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        encoded = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if len(encoded) != len(texts):
            raise RuntimeError(
                f"Model returned {len(encoded)} vectors for {len(texts)} inputs."
            )

        vectors = [[float(value) for value in vector] for vector in encoded]
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise RuntimeError(
                    f"Model returned a vector with {len(vector)} dimensions; "
                    f"expected {self.dimensions}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError("Model returned a non-finite embedding value.")
        return vectors
