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
        Combine multiple ranked search result lists using RRF.

        Args:
            result_lists:
                A list containing ranked result lists.
                The first list is expected to be dense results
                and the second list is expected to be sparse results.

        Returns:
            A single list ranked by the final RRF score.
        """

        # Store the accumulated RRF score for each chunk.
        rrf_scores = {}

        # Store one copy of the document information.
        documents = {}

        # Store the original dense score for each chunk.
        dense_scores = {}

        # Store the original sparse score for each chunk.
        sparse_scores = {}

        # Process every retrieval system.
        for source_index, result_list in enumerate(result_lists):

            # Rank starts from 1, not 0.
            for rank, result in enumerate(result_list, start=1):

                chunk_id = result["chunk_id"]

                # RRF formula:
                # score = 1 / (k + rank)
                score = 1 / (self.k + rank)

                # Add this rank contribution to the chunk's total RRF score.
                rrf_scores[chunk_id] = (
                    rrf_scores.get(chunk_id, 0.0) + score
                )

                # Keep the first copy of the document.
                # This prevents BM25 from overwriting the Qdrant result.
                if chunk_id not in documents:
                    documents[chunk_id] = result.copy()

                # The first result list is the dense search.
                if source_index == 0:
                    dense_scores[chunk_id] = result.get("score")

                # The second result list is the sparse/BM25 search.
                elif source_index == 1:
                    sparse_scores[chunk_id] = result.get("score")

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

            # Preserve the original dense retrieval score.
            result["dense_score"] = dense_scores.get(chunk_id)

            # Preserve the original sparse/BM25 retrieval score.
            result["sparse_score"] = sparse_scores.get(chunk_id)

            # Add the final RRF score.
            result["rrf_score"] = score

            results.append(result)

        return results