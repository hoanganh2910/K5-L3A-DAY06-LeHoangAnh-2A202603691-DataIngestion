"""Split document pages into fixed-size or paragraph-aware chunks."""
import re

from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into character windows with a configurable overlap."""
    if size <= 0:
        raise ValueError("size must be positive.")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size.")

    text = page.text
    if not text.strip():
        return []

    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end]
        chunks.append(
            Chunk(
                chunk_id=(
                    f"{page.document_id}:p{page.page}:fixed:{start}-{end}"
                ),
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=chunk_text,
                metadata=page.metadata.copy(),
            )
        )
        if end == len(text):
            break
        start += step

    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs up to max_size and split oversized paragraphs."""
    if max_size <= 0:
        raise ValueError("max_size must be positive.")
    if not page.text.strip():
        return []

    paragraphs = [
        re.sub(r"\s*\n\s*", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", page.text.strip())
        if paragraph.strip()
    ]
    chunk_texts: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            if current:
                chunk_texts.append(current)
                current = ""
            chunk_texts.extend(_split_long_paragraph(paragraph, max_size))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_size:
            current = candidate
        else:
            chunk_texts.append(current)
            current = paragraph

    if current:
        chunk_texts.append(current)

    return [
        Chunk(
            chunk_id=f"{page.document_id}:p{page.page}:paragraph:{index}",
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=text,
            metadata=page.metadata.copy(),
        )
        for index, text in enumerate(chunk_texts, start=1)
    ]


def _split_long_paragraph(paragraph: str, max_size: int) -> list[str]:
    """Split an oversized paragraph at word boundaries where possible."""
    chunks: list[str] = []
    current = ""

    for word in paragraph.split():
        if len(word) > max_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                word[start : start + max_size]
                for start in range(0, len(word), max_size)
            )
        elif not current:
            current = word
        elif len(current) + 1 + len(word) <= max_size:
            current = f"{current} {word}"
        else:
            chunks.append(current)
            current = word

    if current:
        chunks.append(current)
    return chunks
