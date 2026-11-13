"""Exercise pgvector persistence, metadata filtering, and document replacement."""
import os
import sys
import uuid
from pathlib import Path

import psycopg
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import by_paragraph
from src.embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder
from src.models import Chunk, DocumentPage
from src.store import PgVectorStore


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from starter/.env.")

    embedder = SentenceTransformerEmbedder(
        os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    )
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()

    strategy = f"cp5-smoke-{uuid.uuid4().hex}"
    page = DocumentPage(
        document_id=f"cp5-smoke-{uuid.uuid4().hex}",
        source="cp5-smoke-test",
        page=1,
        text=(
            "Hội đồng nhân dân quyết định ngân sách địa phương.\n\n"
            "Ủy ban nhân dân tổ chức thi hành nghị quyết."
        ),
        metadata={"chapter": "chapter_IV"},
    )
    chunks = by_paragraph(page, max_size=200)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])

    try:
        store.upsert(strategy, chunks, vectors)
        store.upsert(strategy, chunks, vectors)
        ingested = store.search(strategy, vectors[0], top_k=10)
        if len(ingested) != len(chunks) or len(
            {result["chunk_id"] for result in ingested}
        ) != len(chunks):
            raise RuntimeError("Re-ingestion created duplicate chunks.")

        filtered_results = store.search(
            strategy,
            embedder.embed_query("Cơ quan nào quyết định ngân sách địa phương?"),
            top_k=3,
            filters={"chapter": "chapter_IV"},
        )
        if not filtered_results or any(
            result["metadata"].get("chapter") != "chapter_IV"
            for result in filtered_results
        ):
            raise RuntimeError("Metadata-filtered search returned unexpected results.")

        replacement = Chunk(
            chunk_id=f"{page.document_id}:replacement",
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text="Hội đồng nhân dân quyết định ngân sách địa phương.",
            metadata=page.metadata.copy(),
        )
        replacement_vector = embedder.embed_documents([replacement.text])
        store.upsert(strategy, [replacement], replacement_vector)
        remaining = store.search(
            strategy,
            replacement_vector[0],
            top_k=10,
        )
        if len(remaining) != 1 or remaining[0]["chunk_id"] != replacement.chunk_id:
            raise RuntimeError("Re-ingestion did not remove obsolete document chunks.")

        print("pgvector store smoke test OK")
        print(f"Model: {embedder.model_name}")
        print(f"Dimensions: {embedder.dimensions}")
        print("Verified: persistence, no duplicate re-ingestion, metadata filter,")
        print("and removal of obsolete chunks on document replacement.")
    finally:
        with psycopg.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM ingestion_chunks WHERE strategy = %s",
                    (strategy,),
                )


if __name__ == "__main__":
    main()
