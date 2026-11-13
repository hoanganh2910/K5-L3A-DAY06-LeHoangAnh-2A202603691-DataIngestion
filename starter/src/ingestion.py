"""Read and normalize source data."""
import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


logger = logging.getLogger(__name__)
_CHAPTER_HEADING = re.compile(
    r"^Chương\s+(VII|VI|IV|V|III|II|I)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page and retain its source and page number.

    Pages without extractable text are kept with empty text and logged. OCR,
    when needed, is a separate step.
    """
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    current_chapter: str | None = None

    for page_number, pdf_page in enumerate(reader.pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")
        chapter_heading = _CHAPTER_HEADING.search(text)
        if chapter_heading:
            current_chapter = f"chapter_{chapter_heading.group(1).upper()}"
        if not text:
            logger.warning(
                "PDF page has no extractable text: document_id=%s source=%s page=%d",
                document_id,
                source,
                page_number,
            )
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={"chapter": current_chapter} if current_chapter else {},
            )
        )

    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph breaks."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        re.sub(r"[^\S\n]+", " ", line).strip()
        for line in normalized.split("\n")
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
