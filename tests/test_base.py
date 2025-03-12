import numpy as np
import pytest

from mxbai_rerank.base import BaseReranker, RankResult


class MockReranker(BaseReranker):
    """Mock reranker for testing."""

    def predict(self, queries, documents):
        # Simple mock implementation that returns random scores
        return np.random.random(len(queries))


def test_rank_result():
    # Test RankResult dataclass
    result = RankResult(index=1, score=0.5, document="test document")
    assert result.index == 1
    assert result.score == 0.5
    assert result.document == "test document"


class TestBaseReranker:
    @pytest.fixture
    def reranker(self):
        return MockReranker()

    def test_rank_single_query(self, reranker):
        query = "test query"
        documents = ["doc1", "doc2", "doc3"]
        results = reranker.rank(
            query=query,
            documents=documents,
            top_k=2,
            batch_size=2,
            return_documents=True,
        )
        assert len(results) == 2
        for result in results:
            assert isinstance(result, RankResult)
            assert isinstance(result.score, float)
            assert isinstance(result.index, int)
            assert result.document in documents

    def test_rank_multiple_queries(self, reranker):
        queries = ["query1", "query2", "query3"]
        documents = ["doc1", "doc2", "doc3"]
        
        # Test with mismatched lengths
        with pytest.raises(ValueError):
            reranker._rank(queries=queries[:2], documents=documents)

        # Test with matching lengths
        results = reranker._rank(
            queries=queries,
            documents=documents,
            top_k=2,
            batch_size=2,
            return_documents=True,
        )
        assert len(results) == 2
        for result in results:
            assert isinstance(result, RankResult)
            assert isinstance(result.score, float)
            assert isinstance(result.index, int)
            assert result.document in documents

    def test_compute_scores(self, reranker):
        queries = ["query1", "query2", "query3"]
        documents = ["doc1", "doc2", "doc3"]
        
        # Test with small batch
        scores = reranker._compute_scores(
            queries=queries,
            documents=documents,
            batch_size=2,
            show_progress=False,
        )
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(queries)
        
        # Test with large batch
        scores = reranker._compute_scores(
            queries=queries,
            documents=documents,
            batch_size=10,
            show_progress=False,
        )
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(queries) 