import unittest
from pathlib import Path

from src.chunking import by_paragraph, fixed_size
from src.ingestion import load_pdf
from src.models import DocumentPage


class ChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.page = DocumentPage(
            document_id="law",
            source="law.pdf",
            page=7,
            text="one two three four five",
            metadata={"chapter": "chapter_IV"},
        )

    def test_fixed_size_chunks_overlap_without_redundant_tail(self) -> None:
        page = DocumentPage("doc", "source.pdf", 1, "abcdefghijklmnopqrst")

        chunks = fixed_size(page, size=10, overlap=4)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["abcdefghij", "ghijklmnop", "mnopqrst"],
        )
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), 3)

    def test_fixed_size_copies_traceability_and_metadata(self) -> None:
        chunks = fixed_size(self.page, size=10, overlap=2)

        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk.document_id, "law")
            self.assertEqual(chunk.source, "law.pdf")
            self.assertEqual(chunk.page, 7)
            self.assertEqual(chunk.metadata, {"chapter": "chapter_IV"})
            self.assertIsNot(chunk.metadata, self.page.metadata)

    def test_paragraph_chunking_groups_and_splits_at_word_boundaries(self) -> None:
        page = DocumentPage(
            "doc",
            "source.pdf",
            1,
            "alpha beta gamma delta\n\none two",
        )

        chunks = by_paragraph(page, max_size=10)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["alpha beta", "gamma", "delta", "one two"],
        )
        self.assertTrue(all(len(chunk.text) <= 10 for chunk in chunks))

    def test_paragraph_chunking_splits_words_longer_than_limit(self) -> None:
        page = DocumentPage("doc", "source.pdf", 1, "abcdefghijk")

        chunks = by_paragraph(page, max_size=4)

        self.assertEqual("".join(chunk.text for chunk in chunks), "abcdefghijk")
        self.assertTrue(all(len(chunk.text) <= 4 for chunk in chunks))

    def test_empty_pages_and_invalid_sizes(self) -> None:
        empty_page = DocumentPage("doc", "source.pdf", 1, " \n ")

        self.assertEqual(fixed_size(empty_page, size=10, overlap=2), [])
        self.assertEqual(by_paragraph(empty_page, max_size=10), [])
        with self.assertRaises(ValueError):
            fixed_size(self.page, size=0, overlap=0)
        with self.assertRaises(ValueError):
            fixed_size(self.page, size=5, overlap=5)
        with self.assertRaises(ValueError):
            by_paragraph(self.page, max_size=0)


class ChapterMetadataTests(unittest.TestCase):
    def test_pdf_pages_retain_chapter_metadata(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        pdf_path = (
            project_root
            / "data"
            / "luat_to_chuc_chinh_quyen_dia_phuong_72_2025.pdf"
        )
        pages = load_pdf(pdf_path, "law-72-2025", "law.pdf")
        chapter_ranges = {
            "chapter_I": (1, 4),
            "chapter_II": (5, 7),
            "chapter_III": (8, 13),
            "chapter_IV": (14, 32),
            "chapter_V": (33, 49),
            "chapter_VI": (50, 54),
            "chapter_VII": (55, 61),
        }

        self.assertEqual(len(pages), 61)
        for page in pages:
            expected_chapter = next(
                chapter
                for chapter, (start, end) in chapter_ranges.items()
                if start <= page.page <= end
            )
            self.assertEqual(page.metadata["chapter"], expected_chapter)


if __name__ == "__main__":
    unittest.main()
