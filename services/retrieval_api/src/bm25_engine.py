from rank_bm25 import BM25Okapi


class BM25Engine:
    """
    Handles sparse lexical search using BM25.
    """

    def __init__(self):
        # Store the original document blocks.
        self.documents = []

        # BM25 index will be created after documents are added.
        self.bm25 = None

    def _build_search_text(self, document: dict) -> str:
        """
        Combine document content and important metadata
        into one text string for BM25 indexing.
        """

        parts = [
            document.get("content", ""),
        ]

        metadata = document.get("metadata", {})

        parts.extend(
            [
                document.get("document_id", ""),
                metadata.get("company", ""),
                str(metadata.get("year", "")),
                metadata.get("section", ""),
            ]
        )

        return " ".join(parts)

    def index_documents(self, documents: list[dict]):
        """
        Build a BM25 index from document blocks.

        Args:
            documents: List of dictionaries containing document information.
        """

        self.documents = documents

        texts = [self._build_search_text(document) for document in documents]

        tokenized_documents = [text.lower().split() for text in texts]

        self.bm25 = BM25Okapi(tokenized_documents)

    def _matches_filters(
        self,
        document: dict,
        filters: dict | None,
    ) -> bool:
        """
        Check whether a document matches the requested filters.

        Supported fields:
            - document_id
            - page
            - content_type
            - company
            - year
            - section

        Filters use exact matching.
        """

        # No filters means every document is allowed.
        if not filters:
            return True

        metadata = document.get("metadata", {})

        for key, expected_value in filters.items():
            # Get the actual value from either the document itself
            # or its metadata.
            if key in document:
                actual_value = document.get(key)
            else:
                actual_value = metadata.get(key)

            # Normalize values to strings for flexible comparison.
            if str(actual_value).lower() != str(expected_value).lower():
                return False

        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ):
        """
        Search the BM25 index using a text query.

        Args:
            query: User search query.
            top_k: Number of results to return.
            filters: Optional metadata/document filters.

        Returns:
            Ranked BM25 search results.
        """

        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been initialized.")

        tokenized_query = query.lower().split()

        # Calculate BM25 relevance scores for all documents.
        scores = self.bm25.get_scores(tokenized_query)

        # Keep only documents matching the requested filters.
        filtered_indices = [
            index for index, document in enumerate(self.documents) if self._matches_filters(document, filters)
        ]

        # Sort only the allowed documents by BM25 score.
        ranked_indices = sorted(
            filtered_indices,
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:
            document = self.documents[index]

            results.append(
                {
                    "chunk_id": document["chunk_id"],
                    "document_id": document.get("document_id"),
                    "page": document.get("page"),
                    "content": document.get("content", ""),
                    "content_type": document.get("content_type", "text"),
                    "bbox": document.get("bbox"),
                    "metadata": document.get("metadata", {}),
                    "score": float(scores[index]),
                }
            )

        return results
