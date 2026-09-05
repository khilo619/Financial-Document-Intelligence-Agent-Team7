from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranks retrieved documents using a Cross-Encoder model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
    ):
        # Load the Cross-Encoder reranking model.
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """
        Rerank retrieved documents using the Cross-Encoder.

        Args:
            query:
                The original user search query.

            results:
                Candidate documents retrieved by the hybrid search.

            top_n:
                Number of final results to return.

        Returns:
            The top_n documents ranked by Cross-Encoder score.
        """

        # Create query-document pairs for the Cross-Encoder.
        pairs = [
            [query, result.get("content", "")]
            for result in results
        ]

        # Calculate relevance scores for all query-document pairs.
        scores = self.model.predict(pairs)

        # Attach the Cross-Encoder score to each result.
        reranked_results = []

        for result, score in zip(results, scores):
            updated_result = result.copy()

            updated_result["rerank_score"] = float(score)

            reranked_results.append(updated_result)

        # Sort by Cross-Encoder relevance score.
        reranked_results.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        # Return only the requested number of results.
        return reranked_results[:top_n]