import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from shared.models import DocumentBlock

from .embedder import Embedder


class QdrantStore:
    """
    Handles storing document embeddings in Qdrant.
    """

    COLLECTION_NAME = "ledger_documents"
    VECTOR_SIZE = 1024

    def __init__(self):
        # Read Qdrant connection settings from environment variables.
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))

        # Create a client for communicating with the Qdrant server.
        self.client = QdrantClient(
            host=host,
            port=port,
        )

        # Load the embedding model used to convert text into vectors.
        model_name = os.getenv(
            "EMBEDDING_MODEL_NAME",
            "BAAI/bge-large-en-v1.5",
        )

        self.embedder = Embedder(model_name)

        # Create the collection if it does not already exist.
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Create the Qdrant collection if it does not already exist.
        """

        # Retrieve the list of existing Qdrant collections.
        collections = self.client.get_collections().collections

        # Check whether our collection already exists.
        exists = any(
            collection.name == self.COLLECTION_NAME
            for collection in collections
        )

        # Create the collection only if it does not exist.
        if not exists:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def index_block(self, block: DocumentBlock):
        """
        Convert one DocumentBlock into an embedding
        and store it in Qdrant.
        """

        # Use the block's Markdown content as the text to embed.
        text = block.markdown_content

        # Generate a dense embedding using the BGE model.
        vector = self.embedder.encode(text)

        # Qdrant point IDs must be UUIDs or integers.
        # Create a deterministic UUID from the block ID.
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                block.block_id,
            )
        )

        # Store the original document information as Qdrant payload.
        # This metadata will be returned when a matching vector is retrieved.
        payload = {
            "chunk_id": block.block_id,
            "document_id": block.document_id,
            "page": block.page,
            "content_type": block.content_type,
            "content": block.markdown_content,
            "bbox": block.bbox,
            "metadata": block.metadata,
        }

        # Create a Qdrant point containing the vector and its payload.
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )

        # Insert or update the point in the Qdrant collection.
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )

        return point_id

    def search(self, query: str, top_k: int = 5):
        """
        Search Qdrant for the most semantically similar document blocks.
        """

        # Convert the search query into a dense embedding.
        query_vector = self.embedder.encode(query)

        # Search the collection using cosine similarity.
        search_result = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        # Convert Qdrant results into a standardized dictionary format.
        # This format can be shared with BM25 and later passed to RRF.
        results = []

        for point in search_result.points:
            payload = point.payload or {}

            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "page": payload.get("page"),
                    "content": payload.get("content", ""),
                    "content_type": payload.get("content_type", "text"),
                    "bbox": payload.get("bbox"),
                    "metadata": payload.get("metadata", {}),
                    "score": float(point.score),
                }
            )

        return results