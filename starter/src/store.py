"""Persistent vector storage and cosine-similarity search with pgvector."""
import math
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk


_TABLE_NAME = "ingestion_chunks"
_MAX_VECTOR_DIMENSIONS = 16_000


class PgVectorStore:
    """Store chunks and vectors in PostgreSQL, scoped by retrieval strategy."""

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url.strip():
            raise ValueError("database_url must not be empty.")
        if (
            not isinstance(dimensions, int)
            or isinstance(dimensions, bool)
            or not 1 <= dimensions <= _MAX_VECTOR_DIMENSIONS
        ):
            raise ValueError(
                f"dimensions must be an integer from 1 to {_MAX_VECTOR_DIMENSIONS}."
            )

        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """Create the pgvector extension and persistent chunk table if absent."""
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL CHECK (page > 0),
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding VECTOR({self.dimensions}) NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Replace each supplied document's chunks atomically, without duplicates."""
        self._validate_strategy(strategy)
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Received {len(chunks)} chunks but {len(vectors)} vectors."
            )
        if not chunks:
            return

        rows: list[tuple[Any, ...]] = []
        document_ids: set[str] = set()
        chunk_ids: set[str] = set()
        for chunk, vector in zip(chunks, vectors):
            if not chunk.document_id.strip():
                raise ValueError("Every chunk must have a document_id.")
            if not chunk.chunk_id.strip():
                raise ValueError("Every chunk must have a chunk_id.")
            if chunk.chunk_id in chunk_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk.chunk_id!r}.")
            if not chunk.source.strip():
                raise ValueError(f"Chunk {chunk.chunk_id!r} has an empty source.")
            if chunk.page <= 0:
                raise ValueError(f"Chunk {chunk.chunk_id!r} has an invalid page.")
            if not chunk.text.strip():
                raise ValueError(f"Chunk {chunk.chunk_id!r} has empty text.")
            if not isinstance(chunk.metadata, dict):
                raise ValueError(f"Chunk {chunk.chunk_id!r} metadata must be a dict.")

            chunk_ids.add(chunk.chunk_id)
            document_ids.add(chunk.document_id)
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    Jsonb(chunk.metadata),
                    self._format_vector(vector),
                )
            )

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {_TABLE_NAME} "
                    "WHERE strategy = %s AND document_id = ANY(%s)",
                    (strategy, list(document_ids)),
                )
                cursor.executemany(
                    f"""
                    INSERT INTO {_TABLE_NAME} (
                        strategy, chunk_id, document_id, source, page,
                        text, metadata, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source = EXCLUDED.source,
                        page = EXCLUDED.page,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                    """,
                    rows,
                )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks, optionally filtering by JSON metadata values."""
        self._validate_strategy(strategy)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        vector = self._format_vector(query_vector)
        filter_values = filters or {}
        if not isinstance(filter_values, dict):
            raise ValueError("filters must be a dictionary.")

        conditions = ["strategy = %s"]
        parameters: list[Any] = [vector, strategy]
        for key, value in filter_values.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Metadata filter keys must be non-empty strings.")
            conditions.append("metadata @> %s")
            parameters.append(Jsonb({key: value}))
        parameters.extend([vector, top_k])

        query = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {_TABLE_NAME}
            WHERE {" AND ".join(conditions)}
            ORDER BY embedding <=> %s::vector ASC, chunk_id ASC
            LIMIT %s
        """
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, parameters)
                rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]

    def _format_vector(self, vector: list[float]) -> str:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Vector has {len(vector)} dimensions; expected {self.dimensions}."
            )
        values: list[str] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Vector values must be finite numbers.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError("Vector values must be finite numbers.")
            values.append(repr(numeric_value))
        return f"[{','.join(values)}]"

    @staticmethod
    def _validate_strategy(strategy: str) -> None:
        if not strategy.strip():
            raise ValueError("strategy must not be empty.")
