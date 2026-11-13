import unittest
from unittest.mock import Mock, patch

from src.embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder


class SentenceTransformerEmbedderTests(unittest.TestCase):
    @patch("src.embeddings.SentenceTransformer")
    def test_documents_and_query_use_same_model_and_normalized_vectors(
        self, model_factory: Mock
    ) -> None:
        model = Mock(
            spec=["get_embedding_dimension", "get_sentence_embedding_dimension", "encode"]
        )
        model.get_embedding_dimension.return_value = 2
        model.encode.side_effect = [
            [[0.6, 0.8], [0.0, 1.0]],
            [[1.0, 0.0]],
        ]
        model_factory.return_value = model

        embedder = SentenceTransformerEmbedder()

        vectors = embedder.embed_documents(["văn bản", "quy định"])
        query_vector = embedder.embed_query("quy định nào?")

        self.assertEqual(embedder.model_name, DEFAULT_MODEL)
        self.assertEqual(embedder.dimensions, 2)
        self.assertEqual(vectors, [[0.6, 0.8], [0.0, 1.0]])
        self.assertEqual(query_vector, [1.0, 0.0])
        self.assertEqual(model.encode.call_args_list[0].args[0], ["văn bản", "quy định"])
        self.assertEqual(model.encode.call_args_list[1].args[0], ["quy định nào?"])
        self.assertTrue(model.encode.call_args_list[0].kwargs["normalize_embeddings"])
        self.assertTrue(model.encode.call_args_list[1].kwargs["normalize_embeddings"])

    @patch("src.embeddings.SentenceTransformer")
    def test_empty_document_list_returns_no_vectors(self, model_factory: Mock) -> None:
        model = Mock(
            spec=["get_embedding_dimension", "get_sentence_embedding_dimension", "encode"]
        )
        model.get_embedding_dimension.return_value = 384
        model_factory.return_value = model

        embedder = SentenceTransformerEmbedder()

        self.assertEqual(embedder.embed_documents([]), [])
        model.encode.assert_not_called()

    @patch("src.embeddings.SentenceTransformer")
    def test_rejects_vectors_with_wrong_dimension(self, model_factory: Mock) -> None:
        model = Mock(
            spec=["get_embedding_dimension", "get_sentence_embedding_dimension", "encode"]
        )
        model.get_embedding_dimension.return_value = 3
        model.encode.return_value = [[0.1, 0.2]]
        model_factory.return_value = model
        embedder = SentenceTransformerEmbedder()

        with self.assertRaises(RuntimeError):
            embedder.embed_documents(["test"])

    @patch("src.embeddings.SentenceTransformer")
    def test_supports_older_dimension_method(self, model_factory: Mock) -> None:
        model = Mock(spec=["get_sentence_embedding_dimension", "encode"])
        model.get_sentence_embedding_dimension.return_value = 3
        model_factory.return_value = model

        embedder = SentenceTransformerEmbedder()

        self.assertEqual(embedder.dimensions, 3)

    @patch("src.embeddings.SentenceTransformer")
    def test_rejects_empty_model_name(self, model_factory: Mock) -> None:
        with self.assertRaises(ValueError):
            SentenceTransformerEmbedder(" ")
        model_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
