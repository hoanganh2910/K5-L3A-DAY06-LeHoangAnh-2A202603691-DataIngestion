"""Smoke-test the configured local embedding model."""
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder


def main() -> None:
    load_dotenv()
    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    embedder = SentenceTransformerEmbedder(model_name)

    documents = [
        "Hội đồng nhân dân quyết định các vấn đề của địa phương.",
        "Ủy ban nhân dân tổ chức thi hành nghị quyết của Hội đồng nhân dân.",
    ]
    document_vectors = embedder.embed_documents(documents)
    query_vector = embedder.embed_query("Cơ quan nào tổ chức thi hành nghị quyết?")

    if len(document_vectors) != len(documents):
        raise RuntimeError("Embedding model did not return one vector per document.")
    if any(len(vector) != embedder.dimensions for vector in document_vectors):
        raise RuntimeError("A document vector has an unexpected dimension.")
    if len(query_vector) != embedder.dimensions:
        raise RuntimeError("The query vector has an unexpected dimension.")
    if not all(
        math.isfinite(value)
        for vector in [*document_vectors, query_vector]
        for value in vector
    ):
        raise RuntimeError("Embedding model returned non-finite values.")

    print("Embedding smoke test OK")
    print(f"Model: {embedder.model_name}")
    print(f"Dimensions: {embedder.dimensions}")
    print(f"Document vectors: {len(document_vectors)}")
    print(f"Query vector dimensions: {len(query_vector)}")


if __name__ == "__main__":
    main()
