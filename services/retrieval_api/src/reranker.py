from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class Reranker:
    """
    Reranks retrieved documents using LangChain's
    Hugging Face Cross-Encoder integration.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
    ):
        # Load the Cross-Encoder reranking model
        # through LangChain's Hugging Face integration.
        self.model = HuggingFaceCrossEncoder(
            model_name=model_name,
            model_kwargs={
                "device": "cpu",
            },
        )

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
            (query, result.get("content", ""))
            for result in results
        ]

        # Calculate relevance scores.
        scores = self.model.score(pairs)

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