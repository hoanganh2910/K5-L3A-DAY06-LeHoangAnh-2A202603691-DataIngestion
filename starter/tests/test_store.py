import unittest
from unittest.mock import MagicMock, patch

from src.models import Chunk
from src.store import PgVectorStore


class PgVectorStoreTests(unittest.TestCase):
    def test_requires_valid_database_url_and_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            PgVectorStore(" ", 3)
        with self.assertRaises(ValueError):
            PgVectorStore("postgresql://example", 0)
        with self.assertRaises(ValueError):
            PgVectorStore("postgresql://example", 16_001)
        with self.assertRaises(ValueError):
            PgVectorStore("postgresql://example", True)

    def test_upsert_checks_vector_count_before_connecting(self) -> None:
        store = PgVectorStore("postgresql://example", 2)
        chunk = Chunk("chunk-1", "doc-1", "law.pdf", 1, "Some text.")

        with patch("src.store.psycopg.connect") as connect:
            with self.assertRaises(ValueError):
                store.upsert("strategy-a", [chunk], [])

        connect.assert_not_called()

    def test_upsert_rejects_duplicate_chunk_ids_before_connecting(self) -> None:
        store = PgVectorStore("postgresql://example", 2)
        chunks = [
            Chunk("same-id", "doc-1", "law.pdf", 1, "First."),
            Chunk("same-id", "doc-1", "law.pdf", 1, "Duplicate."),
        ]

        with patch("src.store.psycopg.connect") as connect:
            with self.assertRaisesRegex(ValueError, "Duplicate chunk_id"):
                store.upsert("strategy-a", chunks, [[1.0, 0.0], [0.0, 1.0]])

        connect.assert_not_called()

    def test_search_builds_parameterized_metadata_filter_and_maps_rows(self) -> None:
        store = PgVectorStore("postgresql://example", 2)
        connection = MagicMock()
        cursor = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchall.return_value = [
            (
                "chunk-1",
                "doc-1",
                "law.pdf",
                4,
                "Some text.",
                {"chapter": "chapter_IV"},
                0.95,
            )
        ]

        with patch("src.store.psycopg.connect", return_value=connection):
            results = store.search(
                "strategy-a",
                [1.0, 0.0],
                top_k=2,
                filters={"chapter": "chapter_IV"},
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk-1")
        self.assertEqual(results[0]["metadata"], {"chapter": "chapter_IV"})
        self.assertEqual(results[0]["score"], 0.95)
        sql, parameters = cursor.execute.call_args.args
        self.assertIn("metadata @> %s", sql)
        self.assertNotIn("chapter_IV", sql)
        self.assertEqual(parameters[1], "strategy-a")
        self.assertEqual(parameters[-2:], ["[1.0,0.0]", 2])

    def test_rejects_invalid_vectors_and_search_limits(self) -> None:
        store = PgVectorStore("postgresql://example", 2)
        with self.assertRaises(ValueError):
            store.search("strategy-a", [1.0, 0.0], top_k=0)
        with self.assertRaises(ValueError):
            store.search("strategy-a", [float("nan"), 0.0])
        with self.assertRaises(ValueError):
            store.search("strategy-a", [1.0])


if __name__ == "__main__":
    unittest.main()
