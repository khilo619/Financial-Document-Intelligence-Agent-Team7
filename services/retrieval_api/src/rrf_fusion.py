class RRFFusion:
    """
    Combines multiple ranked search result lists
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, result_lists: list[list[dict]]) -> list[dict]:
        """
        Combine multiple ranked search result lists using RRF.

        A chunk is uniquely identified by both:
        - document_id
        - chunk_id

        because chunk_id alone is only unique inside one document.
        """

        rrf_scores = {}
        documents = {}
        dense_scores = {}
        sparse_scores = {}

        for source_index, result_list in enumerate(result_lists):
            for rank, result in enumerate(result_list, start=1):
                chunk_key = (
                    result["document_id"],
                    result["chunk_id"],
                )

                score = 1 / (self.k + rank)

                rrf_scores[chunk_key] = (
                    rrf_scores.get(chunk_key, 0.0)
                    + score
                )

                if chunk_key not in documents:
                    documents[chunk_key] = result.copy()

                if source_index == 0:
                    dense_scores[chunk_key] = result.get("score")

                elif source_index == 1:
                    sparse_scores[chunk_key] = result.get("score")

        ranked_chunks = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for chunk_key, score in ranked_chunks:
            result = documents[chunk_key].copy()

            result["dense_score"] = dense_scores.get(chunk_key)
            result["sparse_score"] = sparse_scores.get(chunk_key)
            result["rrf_score"] = score

            results.append(result)

        return results