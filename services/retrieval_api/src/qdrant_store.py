import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from shared.models import DocumentBlock

from .embedder import Embedder


class QdrantStore:
    """
    Handles storing document embeddings in Qdrant.
    """

    COLLECTION_NAME = "ledger_documents"
    VECTOR_SIZE = 1024

    def __init__(self):
        # -----------------------------------------------------
        # Qdrant connection settings
        # -----------------------------------------------------

        host = os.getenv(
            "QDRANT_HOST",
            "localhost",
        )

        port = int(
            os.getenv(
                "QDRANT_PORT",
                "6333",
            )
        )

        self.client = QdrantClient(
            host=host,
            port=port,
        )

        # -----------------------------------------------------
        # Embedding model
        # -----------------------------------------------------

        model_name = os.getenv(
            "EMBEDDING_MODEL_NAME",
            "BAAI/bge-large-en-v1.5",
        )

        self.embedder = Embedder(model_name)

        # -----------------------------------------------------
        # Ensure Qdrant collection exists
        # -----------------------------------------------------

        self._ensure_collection()

    # =========================================================
    # Collection initialization
    # =========================================================

    def _ensure_collection(self):
        """
        Create the Qdrant collection if it does not already exist.
        """

        collections = self.client.get_collections().collections

        exists = any(
            collection.name == self.COLLECTION_NAME
            for collection in collections
        )

        if not exists:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    # =========================================================
    # Single-block indexing
    # =========================================================

    def index_block(
        self,
        block: DocumentBlock,
    ):
        """
        Convert one DocumentBlock into an embedding
        and store it in Qdrant.
        """

        # -----------------------------------------------------
        # Generate embedding
        # -----------------------------------------------------

        text = block.markdown_content

        vector = self.embedder.encode(text)

        # -----------------------------------------------------
        # Deterministic Qdrant point ID
        # -----------------------------------------------------

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                block.block_id,
            )
        )

        # -----------------------------------------------------
        # Qdrant payload
        # -----------------------------------------------------

        payload = {
            "chunk_id": block.block_id,
            "document_id": block.document_id,
            "page": block.page,
            "content_type": block.content_type,
            "content": block.markdown_content,
            "bbox": block.bbox,
            "metadata": block.metadata,
        }

        # -----------------------------------------------------
        # Create point
        # -----------------------------------------------------

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )

        # -----------------------------------------------------
        # Upsert
        # -----------------------------------------------------

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )

        return point_id

    # =========================================================
    # Batch indexing
    # =========================================================

    def index_blocks(
        self,
        blocks: list[DocumentBlock],
        batch_size: int = 32,
    ):
        """
        Batch-index multiple DocumentBlocks into Qdrant.

        Embeddings are generated in batches to reduce
        the overhead of processing one block at a time.
        """

        if not blocks:
            return 0

        total_indexed = 0

        # -----------------------------------------------------
        # Process blocks in batches
        # -----------------------------------------------------

        for start in range(
            0,
            len(blocks),
            batch_size,
        ):
            batch = blocks[
                start:start + batch_size
            ]

            # -------------------------------------------------
            # Extract texts
            # -------------------------------------------------

            texts = [
                block.markdown_content
                for block in batch
            ]

            # -------------------------------------------------
            # Generate embeddings for the batch
            # -------------------------------------------------

            vectors = (
                self.embedder.embeddings.embed_documents(
                    texts
                )
            )

            # -------------------------------------------------
            # Build Qdrant points
            # -------------------------------------------------

            points = []

            for block, vector in zip(
                batch,
                vectors,
            ):
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS,
                        block.block_id,
                    )
                )

                payload = {
                    "chunk_id": block.block_id,
                    "document_id": block.document_id,
                    "page": block.page,
                    "content_type": block.content_type,
                    "content": block.markdown_content,
                    "bbox": block.bbox,
                    "metadata": block.metadata,
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            # -------------------------------------------------
            # Batch upsert into Qdrant
            # -------------------------------------------------

            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points,
            )

            total_indexed += len(points)

            print(
                f"Indexed "
                f"{total_indexed}/"
                f"{len(blocks)} blocks"
            )

        return total_indexed

    # =========================================================
    # Dense search
    # =========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ):
        """
        Search Qdrant for the most semantically
        similar document blocks.
        """

        # -----------------------------------------------------
        # 1. Encode query
        # -----------------------------------------------------

        query_vector = self.embedder.encode(
            query
        )

        # -----------------------------------------------------
        # 2. Build filters
        # -----------------------------------------------------

        query_filter = None

        if filters:
            conditions = []

            for key, value in filters.items():

                # ---------------------------------------------
                # Direct payload fields
                # ---------------------------------------------

                if key in {
                    "document_id",
                    "page",
                    "content_type",
                }:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(
                                value=value
                            ),
                        )
                    )

                # ---------------------------------------------
                # Metadata fields
                # ---------------------------------------------

                elif key in {
                    "company",
                    "year",
                    "section",
                }:
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(
                                value=value
                            ),
                        )
                    )

            # ---------------------------------------------
            # Combine all conditions
            # ---------------------------------------------

            if conditions:
                query_filter = Filter(
                    must=conditions
                )

        # -----------------------------------------------------
        # 3. Dense semantic search
        # -----------------------------------------------------

        search_result = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        # -----------------------------------------------------
        # 4. Convert results
        # -----------------------------------------------------

        results = []

        for point in search_result.points:

            payload = point.payload or {}

            results.append(
                {
                    "chunk_id": payload.get(
                        "chunk_id"
                    ),
                    "document_id": payload.get(
                        "document_id"
                    ),
                    "page": payload.get(
                        "page"
                    ),
                    "content": payload.get(
                        "content",
                        "",
                    ),
                    "content_type": payload.get(
                        "content_type",
                        "text",
                    ),
                    "bbox": payload.get(
                        "bbox"
                    ),
                    "metadata": payload.get(
                        "metadata",
                        {},
                    ),
                    "score": float(
                        point.score
                    ),
                }
            )

        return results

    # =========================================================
    # Retrieve ALL indexed documents
    # =========================================================

    def get_all_documents(self):
        """
        Retrieve ALL indexed document blocks from Qdrant.

        Uses pagination to ensure that collections larger than
        the Qdrant scroll batch size are fully retrieved.

        This is used to build the BM25 sparse index.
        """

        documents = []

        # -----------------------------------------------------
        # Qdrant pagination
        # -----------------------------------------------------

        offset = None

        while True:

            points, next_offset = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            # -------------------------------------------------
            # Convert Qdrant points to document dictionaries
            # -------------------------------------------------

            for point in points:

                payload = point.payload or {}

                documents.append(
                    {
                        "chunk_id": payload.get(
                            "chunk_id"
                        ),
                        "document_id": payload.get(
                            "document_id"
                        ),
                        "page": payload.get(
                            "page"
                        ),
                        "content": payload.get(
                            "content",
                            "",
                        ),
                        "content_type": payload.get(
                            "content_type",
                            "text",
                        ),
                        "bbox": payload.get(
                            "bbox"
                        ),
                        "metadata": payload.get(
                            "metadata",
                            {},
                        ),
                    }
                )

            # -------------------------------------------------
            # Stop when Qdrant has no more pages
            # -------------------------------------------------

            if next_offset is None:
                break

            # -------------------------------------------------
            # Continue from the next page
            # -------------------------------------------------

            offset = next_offset

        return documents