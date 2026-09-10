import json
import os
import uuid
from pathlib import Path

from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from shared.models import DocumentBlock

from .embedder import Embedder

# ============================================================
# Configuration
# ============================================================

PROCESSED_JSON = Path("/data/tat_dqa/processed_json/f8867689504c406cebeb5d25694d0327_processed.json")

COLLECTION_NAME = "ledger_documents_processed_test"
VECTOR_SIZE = 1024

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "/models/huggingface/bge-large-en-v1.5",
)


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 70)
    print("TEST: Processed JSON -> DocumentBlock -> BGE -> Qdrant")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check processed JSON
    # --------------------------------------------------------

    print(f"\nProcessed JSON: {PROCESSED_JSON}")

    if not PROCESSED_JSON.exists():
        raise FileNotFoundError(f"Processed JSON not found: {PROCESSED_JSON}")

    # --------------------------------------------------------
    # 2. Load JSON
    # --------------------------------------------------------

    with open(PROCESSED_JSON, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    raw_blocks = data["blocks"]

    print(f"Document ID: {data['document_id']}")
    print(f"Total blocks: {len(raw_blocks)}")

    # --------------------------------------------------------
    # 3. Convert to canonical DocumentBlock objects
    # --------------------------------------------------------

    blocks = [DocumentBlock.model_validate(block) for block in raw_blocks]

    print("\nDocumentBlock validation: OK")

    for block in blocks:
        print(f"  - {block.block_id} | {block.content_type} | page={block.page} | {len(block.markdown_content)} chars")

    # --------------------------------------------------------
    # 4. Initialize embedding model
    # --------------------------------------------------------

    print("\nInitializing Embedder...")

    embedder = Embedder(MODEL_PATH)

    # --------------------------------------------------------
    # 5. Generate embeddings
    # --------------------------------------------------------

    print("\nGenerating embeddings...")

    texts = [block.markdown_content for block in blocks]

    vectors = embedder.embeddings.embed_documents(texts)

    print(f"Generated vectors: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")

    # --------------------------------------------------------
    # 6. Connect to Qdrant
    # --------------------------------------------------------

    from qdrant_client import QdrantClient

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    print(f"\nConnected to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")

    # --------------------------------------------------------
    # 7. Create isolated test collection
    # --------------------------------------------------------

    collections = client.get_collections().collections

    exists = any(c.name == COLLECTION_NAME for c in collections)

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(f"Created collection: {COLLECTION_NAME}")

    else:
        print(f"Collection already exists: {COLLECTION_NAME}")

    # --------------------------------------------------------
    # 8. Build Qdrant points
    # --------------------------------------------------------

    points = []

    for block, vector in zip(blocks, vectors):
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"processed:{block.document_id}:{block.block_id}",
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
            "source": "zeina_processed",
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    # --------------------------------------------------------
    # 9. Upsert
    # --------------------------------------------------------

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"\nIndexed {len(points)} blocks successfully.")

    # --------------------------------------------------------
    # 10. Verify collection
    # --------------------------------------------------------

    info = client.get_collection(collection_name=COLLECTION_NAME)

    print(f"Collection points: {info.points_count}")

    print("\n" + "=" * 70)
    print("TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
