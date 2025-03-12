from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
from batched.utils import bucket_batch_iter
from tqdm import tqdm

from mxbai_rerank.utils import top_k_numpy


@dataclass
class RankResult:
    index: int
    score: float
    document: str | None


class BaseReranker:
    """Base class for reranker models."""

    @abc.abstractmethod
    def predict(self, queries: list[str], documents: list[str], **kwargs) -> list[RankResult]:
        """Abstract method to be implemented by subclasses.

        Args:
            queries: List of query strings.
            documents: List of document strings.
            **kwargs: Additional keyword arguments.

        Returns:
            list[RankResult]: The output of the reranker.

        """

    def rank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int = 100,
        batch_size: int = 32,
        sort: bool = True,
        return_documents: bool = True,
        show_progress: bool = False,
        **kwargs,
    ) -> list[RankResult]:
        """Rerank documents based on a single query.

        Args:
            query: The query string.
            documents: List of document strings to rerank.
            top_k: Number of top results to return. Defaults to 100.
            batch_size: Batch size for processing. Defaults to 32.
            sort: Whether to sort the results. Defaults to True.
            return_documents: Whether to include input documents in the result. Defaults to True.
            show_progress: Whether to show progress bar. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            list[RankResult]: Reranked results.
        """
        return self._rank(
            queries=[query] * len(documents),
            documents=documents,
            top_k=top_k,
            batch_size=batch_size,
            sort=sort,
            return_documents=return_documents,
            show_progress=show_progress,
            **kwargs,
        )

    def _rank(
        self,
        queries: list[str],
        documents: list[str],
        *,
        top_k: int = 100,
        batch_size: int = 32,
        sort: bool = True,
        return_documents: bool = True,
        show_progress: bool = False,
        **kwargs,
    ) -> list[RankResult]:
        """Rank documents based on queries.

        Args:
            queries: List of query strings.
            documents: List of document strings to rank.
            top_k: Number of top results to return. Defaults to 100.
            batch_size: Batch size for processing. Defaults to 32.
            sort: Whether to sort the results. Defaults to True.
            return_documents: Whether to include input documents in the result. Defaults to True.
            show_progress: Whether to show progress bar. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            list[RankResult]: Ranked results including rerank results, optionally input documents.

        Raises:
            ValueError: If the number of queries and documents are not equal.

        """
        if len(queries) != len(documents):
            msg = "The number of queries and documents must be the same."
            raise ValueError(msg)

        scores = self._compute_scores(
            queries=queries, documents=documents, batch_size=batch_size, show_progress=show_progress, **kwargs
        )
        top_k_scores, top_k_indices = top_k_numpy(scores=scores, k=top_k, sort=sort)

        return [
            RankResult(index=int(i), score=float(score), document=documents[i] if return_documents else None)
            for i, score in zip(top_k_indices, top_k_scores)
        ]

    def _compute_scores(
        self,
        *,
        queries: list[str],
        documents: list[str],
        batch_size: int,
        show_progress: bool,
        **kwargs,
    ) -> np.ndarray:
        """Compute scores for query-document pairs.

        Args:
            queries: List of query strings.
            documents: List of document strings.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Computed scores.

        """
        if len(queries) < batch_size:
            return np.array(self.predict(queries, documents, **kwargs))

        scores = np.zeros(len(queries), dtype=np.float32)
        for docs, org_indices in tqdm(
            bucket_batch_iter(documents, batch_size),
            desc="Ranking",
            disable=not show_progress,
            total=(len(queries) + batch_size - 1) // batch_size,
        ):
            scores[org_indices] = np.array(
                self.predict(queries=[queries[i] for i in org_indices], documents=docs, **kwargs)
            )

        return scores
