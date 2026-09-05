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

        # Start with the main document content.
        parts = [
            document.get("content", ""),
        ]

        # Add document-level metadata that may be important
        # for exact lexical matching.
        metadata = document.get("metadata", {})

        parts.extend(
            [
                document.get("document_id", ""),
                metadata.get("company", ""),
                str(metadata.get("year", "")),
                metadata.get("section", ""),
            ]
        )

        # Combine all searchable fields into one text string.
        return " ".join(parts)

    def index_documents(self, documents: list[dict]):
        """
        Build a BM25 index from document blocks.

        Args:
            documents: List of dictionaries containing document information.
        """

        # Store the original document blocks.
        self.documents = documents

        # Build searchable text using both content and metadata.
        texts = [
            self._build_search_text(document)
            for document in documents
        ]

        # Tokenize each document using simple whitespace tokenization.
        tokenized_documents = [
            text.lower().split()
            for text in texts
        ]

        # Build the BM25 index.
        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query: str, top_k: int = 5):
        """
        Search the BM25 index using a text query.

        Args:
            query: User search query.
            top_k: Number of results to return.

        Returns:
            Ranked BM25 search results.
        """

        # Make sure the index has been created.
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been initialized.")

        # Tokenize the query using the same method used for documents.
        tokenized_query = query.lower().split()

        # Calculate BM25 relevance scores.
        scores = self.bm25.get_scores(tokenized_query)

        # Sort document indices by descending BM25 score.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        # Return the top-k documents in a standardized format.
        # This format matches the Qdrant search output
        # and can later be passed to RRF.
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