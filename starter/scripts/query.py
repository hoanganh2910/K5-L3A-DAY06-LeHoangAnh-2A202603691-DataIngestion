"""Retrieve relevant source chunks for a free-form question."""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder
from src.store import PgVectorStore


STRATEGIES = (
    "A_fixed_500_50",
    "B_paragraph_500",
    "C_paragraph_500_filtered",
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Embed a question and retrieve source passages from pgvector."
    )
    parser.add_argument("question", help="Question to search for; put it in quotes.")
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="C_paragraph_500_filtered",
        help="Stored strategy to search (default: C with optional chapter filter).",
    )
    parser.add_argument(
        "--chapter",
        help="Optional trusted chapter filter, for example chapter_V.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of passages to return.")
    args = parser.parse_args()

    if not args.question.strip():
        parser.error("question must not be empty.")
    if args.chapter and args.strategy != "C_paragraph_500_filtered":
        parser.error("--chapter is supported only with C_paragraph_500_filtered.")

    manifest = _read_json(PROJECT_ROOT / "data" / "manifest.json")
    allowed_chapters = {
        value
        for document in manifest["documents"]
        for value in document["filter_metadata"]["values"]
    }
    if args.chapter and args.chapter not in allowed_chapters:
        parser.error(
            f"unknown chapter {args.chapter!r}; choose from "
            + ", ".join(sorted(allowed_chapters))
        )

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from starter/.env.")
    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    previous_run = _read_json(PROJECT_ROOT / "results" / "summary.json")
    indexed_model = previous_run["run"]["embedding_model"]
    if model_name != indexed_model:
        raise RuntimeError(
            f"Configured model {model_name!r} differs from indexed model "
            f"{indexed_model!r}. Run `python -m src.run` again before querying."
        )

    embedder = SentenceTransformerEmbedder(model_name)
    if embedder.dimensions != previous_run["run"]["embedding_dimensions"]:
        raise RuntimeError(
            "Configured model dimensions differ from the indexed vectors. "
            "Run `python -m src.run` again before querying."
        )

    query_vector = embedder.embed_query(args.question)
    store = PgVectorStore(database_url, embedder.dimensions)
    filters = {"chapter": args.chapter} if args.chapter else None
    hits = store.search(
        strategy=args.strategy,
        query_vector=query_vector,
        top_k=args.top_k,
        filters=filters,
    )

    print(f"Strategy: {args.strategy}")
    print(f"Question: {args.question}")
    print(f"Filter: {filters or 'none'}")
    if not hits:
        print("No matching chunks found.")
        return

    for rank, hit in enumerate(hits, start=1):
        print(
            f"\n[{rank}] cosine score={hit['score']:.4f} | "
            f"source={hit['source']} | page={hit['page']} | "
            f"document_id={hit['document_id']}"
        )
        print(f"metadata={json.dumps(hit['metadata'], ensure_ascii=False)}")
        print(hit["text"])
    print("\nThese are retrieved source passages, not a generated answer.")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
