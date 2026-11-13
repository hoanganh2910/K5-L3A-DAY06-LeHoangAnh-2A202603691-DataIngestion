"""Exact evidence matching and retrieval metric aggregation."""
import re
import unicodedata
from collections import defaultdict
from typing import Any


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Annotate top-k hits and calculate top-1 and top-3 evidence retrieval."""
    gold_evidence = _gold_evidence(question)
    annotated_hits: list[dict[str, Any]] = []

    for hit in hits:
        normalized_text = _normalize(hit.get("text", ""))
        matched_evidence = [
            item
            for item in gold_evidence
            if hit.get("document_id") == item["source_id"]
            and hit.get("page") == item["page"]
            and _normalize(item["evidence_quote"]) in normalized_text
        ]
        annotated_hit = dict(hit)
        annotated_hit["evidence_match"] = bool(matched_evidence)
        annotated_hit["matched_gold_evidence"] = matched_evidence
        annotated_hits.append(annotated_hit)

    top_three = annotated_hits[:3]
    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "filter": question.get("filter"),
        "gold_evidence": gold_evidence,
        "hits": annotated_hits,
        "hit_at_1": bool(annotated_hits and annotated_hits[0]["evidence_match"]),
        "evidence_hit_at_3": any(hit["evidence_match"] for hit in top_three),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for trace in traces:
        strategy = trace.get("strategy")
        split = trace.get("split")
        if not strategy or split not in {"dev", "test"}:
            raise ValueError("Every trace must include a strategy and dev/test split.")
        grouped[strategy][split].append(trace)

    summary: dict[str, Any] = {}
    for strategy in sorted(grouped):
        summary[strategy] = {}
        for split in ("dev", "test"):
            split_traces = grouped[strategy].get(split, [])
            total = len(split_traces)
            hit_at_1 = sum(bool(trace["hit_at_1"]) for trace in split_traces)
            hit_at_3 = sum(
                bool(trace["evidence_hit_at_3"]) for trace in split_traces
            )
            summary[strategy][split] = {
                "question_count": total,
                "hit_at_1": {
                    "hits": hit_at_1,
                    "total": total,
                    "rate": hit_at_1 / total if total else None,
                },
                "evidence_hit_at_3": {
                    "hits": hit_at_3,
                    "total": total,
                    "rate": hit_at_3 / total if total else None,
                },
            }
    return summary


def _gold_evidence(question: dict[str, Any]) -> list[dict[str, Any]]:
    alternatives = question.get("gold_evidence")
    if alternatives is not None:
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError(f"{question.get('id', '<unknown>')}: gold_evidence must be a non-empty list.")
        evidence = alternatives
    else:
        evidence = [
            {
                "source_id": question.get("gold_source_id"),
                "page": question.get("gold_page"),
                "evidence_quote": question.get("evidence_quote"),
            }
        ]

    for item in evidence:
        if (
            not isinstance(item, dict)
            or not item.get("source_id")
            or not isinstance(item.get("page"), int)
            or not item.get("evidence_quote")
        ):
            raise ValueError(
                f"{question.get('id', '<unknown>')}: invalid gold evidence entry."
            )
    return evidence


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip().casefold()
