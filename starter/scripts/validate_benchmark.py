"""Validate corpus metadata and benchmark evidence before retrieval experiments."""
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import load_pdf


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _page_range(value: str) -> tuple[int, int]:
    start, end = value.split("-", maxsplit=1)
    return int(start), int(end)


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    manifest = _read_json(data_dir / "manifest.json")
    benchmark = _read_json(data_dir / "benchmark.json")
    documents = manifest["documents"]
    if len(documents) != 1:
        raise ValueError(f"Expected one manifest document, found {len(documents)}.")

    document = documents[0]
    pdf_path = PROJECT_ROOT / document["local_path"]
    pages = load_pdf(pdf_path, document["source_id"], pdf_path.name)
    errors: list[str] = []

    if len(pages) != document["physical_page_count"]:
        errors.append(
            f"PDF page count is {len(pages)}, expected {document['physical_page_count']}."
        )
    extractable_pages = sum(bool(page.text) for page in pages)
    if extractable_pages != document["extractable_page_count"]:
        errors.append(
            f"Extractable page count is {extractable_pages}, "
            f"expected {document['extractable_page_count']}."
        )
    if document["content_page_count"] < 30:
        errors.append("Corpus has fewer than 30 content pages.")

    questions = benchmark["questions"]
    ids = [question["id"] for question in questions]
    if len(questions) != 20:
        errors.append(f"Found {len(questions)} questions; expected 20.")
    if len(set(ids)) != len(ids):
        errors.append("Question IDs must be unique.")

    split_counts = {
        split: sum(question["split"] == split for question in questions)
        for split in ("dev", "test")
    }
    if split_counts != {"dev": 15, "test": 5}:
        errors.append(f"Expected 15 dev and 5 test questions; found {split_counts}.")

    metadata = document["filter_metadata"]
    filter_field = metadata["field"]
    page_ranges = {
        value: _page_range(page_range)
        for value, page_range in metadata["page_ranges"].items()
    }
    filtered_questions = 0
    filtered_test_questions = 0

    for question in questions:
        page_number = question["gold_page"]
        if not 1 <= page_number <= len(pages):
            errors.append(f"{question['id']}: gold page {page_number} is outside the PDF.")
            continue
        if question["gold_source_id"] != document["source_id"]:
            errors.append(f"{question['id']}: gold source does not match the manifest.")
        evidence = _normalize_whitespace(question["evidence_quote"])
        page_text = _normalize_whitespace(pages[page_number - 1].text)
        if not evidence or evidence not in page_text:
            errors.append(
                f"{question['id']}: evidence quote was not found on gold page {page_number}."
            )

        filters = question.get("filter")
        if filters:
            filtered_questions += 1
            filtered_test_questions += question["split"] == "test"
            if set(filters) != {filter_field}:
                errors.append(
                    f"{question['id']}: filter must use only {filter_field!r}."
                )
                continue
            filter_value = filters[filter_field]
            page_range = page_ranges.get(filter_value)
            if page_range is None:
                errors.append(
                    f"{question['id']}: unknown {filter_field} value {filter_value!r}."
                )
            elif not page_range[0] <= page_number <= page_range[1]:
                errors.append(
                    f"{question['id']}: filter {filter_value!r} does not match "
                    f"gold page {page_number}."
                )

    if filtered_questions < 3 or filtered_test_questions < 1:
        errors.append(
            "Need at least 3 metadata-filter questions, including at least 1 test question."
        )
    if errors:
        raise SystemExit("\n".join(errors))

    print("Benchmark validation OK")
    print(
        f"Corpus: {len(pages)} pages, {extractable_pages} with extractable text"
    )
    print(
        f"Questions: {len(questions)} "
        f"(dev={split_counts['dev']}, test={split_counts['test']})"
    )
    print(
        f"Metadata filters: {filtered_questions} "
        f"(test={filtered_test_questions}); evidence verified for all questions"
    )


if __name__ == "__main__":
    main()
