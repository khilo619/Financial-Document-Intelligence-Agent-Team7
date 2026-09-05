class RRFFusion:
    """
    Combines multiple ranked search result lists
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, k: int = 60):
        # RRF constant used to reduce the effect of very high rankings.
        self.k = k

    def fuse(self, result_lists: list[list[dict]]) -> list[dict]:
        """
        Combine multiple ranked result lists using RRF.

        Args:
            result_lists:
                A list containing ranked result lists.
                Each result must contain a unique "chunk_id".

        Returns:
            A single list ranked by the final RRF score.
        """

        # Store the accumulated RRF score for each chunk.
        rrf_scores = {}

        # Store the original result for each chunk.
        documents = {}

        # Process every retrieval system.
        for result_list in result_lists:

            # Rank starts from 1, not 0.
            for rank, result in enumerate(result_list, start=1):

                chunk_id = result["chunk_id"]

                # RRF formula:
                # score = 1 / (k + rank)
                score = 1 / (self.k + rank)

                # Add this rank contribution to the chunk's total score.
                rrf_scores[chunk_id] = (
                    rrf_scores.get(chunk_id, 0.0) + score
                )

                # Keep the original result.
                documents[chunk_id] = result

        # Sort chunks by their final RRF score.
        ranked_chunks = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # Build the final result list.
        results = []

        for chunk_id, score in ranked_chunks:
            result = documents[chunk_id].copy()

            # Add the final RRF score.
            result["rrf_score"] = score

            results.append(result)

        return results