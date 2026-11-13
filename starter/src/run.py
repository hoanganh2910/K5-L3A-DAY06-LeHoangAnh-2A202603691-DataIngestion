"""Run the three retrieval strategies and save traces and audit templates."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    """Run A (fixed), B (paragraph), and C (paragraph with metadata filters)."""
    run_started = time.perf_counter()
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from starter/.env.")

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_benchmark.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )
    manifest = _read_json(DATA_DIR / "manifest.json")
    benchmark = _read_json(DATA_DIR / "benchmark.json")
    pages = _load_corpus(manifest)
    questions = benchmark["questions"]

    chunking_started = time.perf_counter()
    fixed_chunks = [
        chunk
        for page in pages
        for chunk in fixed_size(page, size=500, overlap=50)
    ]
    fixed_chunking_seconds = time.perf_counter() - chunking_started
    chunking_started = time.perf_counter()
    paragraph_chunks = [
        chunk
        for page in pages
        for chunk in by_paragraph(page, max_size=500)
    ]
    paragraph_chunking_seconds = time.perf_counter() - chunking_started

    strategies: list[tuple[str, list[Chunk], bool]] = [
        ("A_fixed_500_50", fixed_chunks, False),
        ("B_paragraph_500", paragraph_chunks, False),
        ("C_paragraph_500_filtered", paragraph_chunks, True),
    ]

    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    embedder = SentenceTransformerEmbedder(model_name)
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()

    resources: dict[str, dict[str, Any]] = {}
    paragraph_vectors: list[list[float]] | None = None
    for strategy, chunks, is_paragraph_strategy in strategies:
        if strategy == "C_paragraph_500_filtered":
            if paragraph_vectors is None:
                raise RuntimeError("Paragraph vectors were not produced for strategy C.")
            vectors = paragraph_vectors
            embedding_seconds = 0.0
        else:
            embedding_started = time.perf_counter()
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
            embedding_seconds = time.perf_counter() - embedding_started
            if strategy == "B_paragraph_500":
                paragraph_vectors = vectors

        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"{strategy} produced {len(vectors)} vectors for {len(chunks)} chunks."
            )

        upsert_started = time.perf_counter()
        store.upsert(strategy, chunks, vectors)
        upsert_seconds = time.perf_counter() - upsert_started
        resources[strategy] = {
            "chunk_count": len(chunks),
            "total_chunk_characters": sum(len(chunk.text) for chunk in chunks),
            "average_chunk_characters": (
                sum(len(chunk.text) for chunk in chunks) / len(chunks)
                if chunks
                else 0.0
            ),
            "chunking_seconds": (
                fixed_chunking_seconds
                if strategy == "A_fixed_500_50"
                else paragraph_chunking_seconds
            ),
            "document_embedding_seconds": embedding_seconds,
            "store_upsert_seconds": upsert_seconds,
            "query_embedding_seconds": 0.0,
            "vector_search_seconds": 0.0,
            "estimated_api_cost_usd": 0.0,
        }

    query_embedding_started = time.perf_counter()
    query_vectors = [
        embedder.embed_query(question["question"])
        for question in questions
    ]
    query_embedding_seconds = time.perf_counter() - query_embedding_started
    for resource in resources.values():
        resource["query_embedding_seconds"] = query_embedding_seconds

    traces: list[dict[str, Any]] = []
    for strategy, _, apply_filters in strategies:
        for question, query_vector in zip(questions, query_vectors):
            search_started = time.perf_counter()
            hits = store.search(
                strategy=strategy,
                query_vector=query_vector,
                top_k=3,
                filters=question.get("filter") if apply_filters else None,
            )
            search_seconds = time.perf_counter() - search_started
            resources[strategy]["vector_search_seconds"] += search_seconds

            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            trace["top_k"] = 3
            trace["vector_search_seconds"] = search_seconds
            traces.append(trace)

    output_metadata = {
        "dataset_id": benchmark["dataset_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "vector_store": "PostgreSQL + pgvector",
        "similarity": "cosine",
        "search_mode": "exact",
        "strategies": {
            "A_fixed_500_50": {"chunker": "fixed_size", "size": 500, "overlap": 50},
            "B_paragraph_500": {"chunker": "by_paragraph", "max_size": 500},
            "C_paragraph_500_filtered": {
                "chunker": "by_paragraph",
                "max_size": 500,
                "filter": "question-provided metadata only",
            },
        },
    }
    summary = summarize(traces)
    manual_audit = _build_manual_audit(
        pages,
        strategies,
        traces,
        questions,
    )
    results = {
        "run": output_metadata,
        "metrics": summary,
        "resources": resources,
        "total_run_seconds": time.perf_counter() - run_started,
        "cost_note": "API cost is 0 USD; embeddings run locally. Hardware/electricity excluded.",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / "retrieval_traces.json", {"run": output_metadata, "traces": traces})
    _write_json(RESULTS_DIR / "summary.json", results)
    _write_json(RESULTS_DIR / "manual_audit_template.json", manual_audit)

    print("Retrieval pipeline completed")
    print(f"Model: {embedder.model_name} ({embedder.dimensions} dimensions)")
    for strategy, splits in summary.items():
        print(f"\n{strategy} | chunks={resources[strategy]['chunk_count']}")
        for split in ("dev", "test"):
            metrics = splits[split]
            print(
                f"  {split}: Hit@1 {metrics['hit_at_1']['hits']}/"
                f"{metrics['hit_at_1']['total']}; Evidence Hit@3 "
                f"{metrics['evidence_hit_at_3']['hits']}/"
                f"{metrics['evidence_hit_at_3']['total']}"
            )
    print(f"\nSaved traces, summary, and manual audit template under {RESULTS_DIR}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_corpus(manifest: dict[str, Any]) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    for document in manifest.get("documents", []):
        pdf_path = PROJECT_ROOT / document["local_path"]
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Corpus PDF does not exist: {pdf_path}")
        pages.extend(
            load_pdf(
                pdf_path,
                document_id=document["source_id"],
                source=pdf_path.name,
            )
        )

    extractable_pages = [page for page in pages if page.text.strip()]
    if len(extractable_pages) < 30:
        raise ValueError(
            f"Need at least 30 extractable content pages; found {len(extractable_pages)}."
        )
    return extractable_pages


def _build_manual_audit(
    pages: list[DocumentPage],
    strategies: list[tuple[str, list[Chunk], bool]],
    traces: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    coherence_samples: list[dict[str, Any]] = []
    selected_pages = pages[:10]
    for page in selected_pages:
        strategy_samples: dict[str, Any] = {}
        for strategy, chunks, _ in strategies:
            page_chunks = [
                chunk
                for chunk in chunks
                if chunk.document_id == page.document_id and chunk.page == page.page
            ]
            if not page_chunks:
                raise RuntimeError(
                    f"{strategy} produced no chunks for page {page.page}."
                )
            strategy_samples[strategy] = {
                "chunk_id": page_chunks[0].chunk_id,
                "text": page_chunks[0].text,
                "score_0_to_2": None,
                "reason": "",
            }
        coherence_samples.append(
            {
                "document_id": page.document_id,
                "source": page.source,
                "page": page.page,
                "strategies": strategy_samples,
            }
        )

    expected_answers = {
        question["id"]: question["expected_answer"]
        for question in questions
    }
    test_grounding = [
        {
            "strategy": trace["strategy"],
            "question_id": trace["question_id"],
            "question": trace["question"],
            "expected_answer": expected_answers[trace["question_id"]],
            "hits": trace["hits"],
            "score_0_to_2": None,
            "reason": "",
        }
        for trace in traces
        if trace["split"] == "test"
    ]
    return {
        "instructions": (
            "Fill score_0_to_2 and reason manually. Do not treat automatic evidence "
            "matches as human coherence or grounding judgments."
        ),
        "coherence_samples": coherence_samples,
        "grounding_samples": test_grounding,
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
