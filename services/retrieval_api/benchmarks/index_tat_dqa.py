"""
Project LEDGER - TAT-DQA Qdrant Indexer

Indexes TAT-DQA document blocks into Qdrant using BGE-Large embeddings.

Dataset structure:

data/
└── tat_dqa/
    ├── train/
    │   └── train/
    │       ├── *.json
    │       ├── *.pdf
    │       └── *_1.png
    ├── dev/
    │   └── dev/
    │       ├── *.json
    │       ├── *.pdf
    │       └── *_1.png
    └── test/
        └── test/
            ├── *.json
            ├── *.pdf
            └── *_1.png

Each JSON document contains:
{
    "pages": [
        {
            "bbox": [...],
            "blocks": [
                {
                    "bbox": [...],
                    "uuid": "...",
                    "text": "...",
                    "words": [...],
                    "order": 0
                }
            ]
        }
    ]
}
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "data" / "tat_dqa"

TRAIN_ROOT = DATASET_ROOT / "train" / "train"
DEV_ROOT = DATASET_ROOT / "dev" / "dev"
TEST_ROOT = DATASET_ROOT / "test" / "test"

COLLECTION_NAME = "ledger_documents"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

BATCH_SIZE = 32

VECTOR_SIZE = 1024


# =============================================================================
# IMPORT PROJECT EMBEDDER
# =============================================================================

RETRIEVAL_SRC = PROJECT_ROOT / "services" / "retrieval_api" / "src"

if str(RETRIEVAL_SRC) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_SRC))

from embedder import Embedder

# =============================================================================
# DATASET DISCOVERY
# =============================================================================


def find_json_files() -> dict[str, list[Path]]:
    """
    Discover TAT-DQA JSON files for train/dev/test.
    """

    split_roots = {
        "train": TRAIN_ROOT,
        "dev": DEV_ROOT,
        "test": TEST_ROOT,
    }

    result: dict[str, list[Path]] = {}

    for split_name, root in split_roots.items():
        if not root.exists():
            print(f"[WARNING] Split directory does not exist: {root}")

            result[split_name] = []

            continue

        files = sorted(root.glob("*.json"))

        result[split_name] = files

        print(f"[{split_name}] Found {len(files)} JSON files")

    return result


# =============================================================================
# JSON DOCUMENT LOADING
# =============================================================================


def load_document_blocks(
    split_name: str,
    file_path: Path,
) -> list[dict[str, Any]]:
    """
    Load one TAT-DQA JSON document and convert its page blocks
    into the canonical indexing format.
    """

    import json

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        document = json.load(f)

    # -------------------------------------------------------------------------
    # Document UID
    # -------------------------------------------------------------------------

    document_id = file_path.stem

    blocks: list[dict[str, Any]] = []

    pages = document.get("pages", [])

    for page_index, page in enumerate(pages):
        # TAT-DQA page numbering is 1-based.
        page_number = page_index + 1

        page_blocks = page.get("blocks", [])

        for block_index, block in enumerate(page_blocks):
            block_uuid = block.get("uuid")

            if not block_uuid:
                block_uuid = f"{document_id}:page-{page_number}:block-{block_index}"

            text = block.get("text", "")

            if text is None:
                text = ""

            text = str(text).strip()

            # Skip completely empty blocks.
            if not text:
                continue

            bbox = block.get("bbox")

            order = block.get(
                "order",
                block_index,
            )

            # -----------------------------------------------------------------
            # Canonical block representation
            # -----------------------------------------------------------------

            block_data = {
                "block_id": str(block_uuid),
                "document_id": document_id,
                "page": page_number,
                "content_type": "text",
                "markdown_content": text,
                "bbox": bbox,
                "metadata": {
                    "source_doc_uid": document_id,
                    "source_document": (f"{document_id}.pdf"),
                    "source_split": split_name,
                    "original_block_uuid": str(block_uuid),
                    "order": order,
                },
            }

            blocks.append(block_data)

    return blocks


# =============================================================================
# QDRANT CONNECTION
# =============================================================================


def create_qdrant_client() -> QdrantClient:
    """
    Create Qdrant client.
    """

    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    # Test connection.
    collections = client.get_collections()

    print("Qdrant connection successful.")

    print(f"Existing collections: {[c.name for c in collections.collections]}")

    return client


# =============================================================================
# RESET COLLECTION
# =============================================================================


def reset_collection(
    client: QdrantClient,
) -> None:
    """
    Delete and recreate the Qdrant collection.
    """

    print(f"[RESET] Deleting collection '{COLLECTION_NAME}'...")

    try:
        client.delete_collection(collection_name=COLLECTION_NAME)

        print("[RESET] Collection deleted.")

    except Exception as exc:
        print(f"[RESET] Collection did not exist or could not be deleted: {exc}")

    print(f"[RESET] Creating collection '{COLLECTION_NAME}'...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print("[RESET] Collection recreated.")


# =============================================================================
# ENSURE COLLECTION EXISTS
# =============================================================================


def ensure_collection(
    client: QdrantClient,
) -> None:
    """
    Make sure the Qdrant collection exists.
    """

    collections = client.get_collections()

    names = {collection.name for collection in collections.collections}

    if COLLECTION_NAME in names:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

        return

    print(f"Collection '{COLLECTION_NAME}' does not exist.")

    print("Creating collection...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print("Collection created.")


# =============================================================================
# DETERMINISTIC POINT ID
# =============================================================================


def make_point_id(
    document_id: str,
    block_id: str,
) -> str:
    """
    Create deterministic UUID for a document block.

    The same document_id + block_id always produces
    the same Qdrant point ID.

    This makes re-running the indexer safe because
    existing points will be overwritten instead of duplicated.
    """

    point_key = f"{document_id}:{block_id}"

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            point_key,
        )
    )


# =============================================================================
# INDEX ONE BATCH
# =============================================================================


def index_batch(
    client: QdrantClient,
    embedder: Embedder,
    blocks: list[dict[str, Any]],
) -> int:
    """
    Embed and index one batch of blocks.
    """

    if not blocks:
        return 0

    texts = [block["markdown_content"] for block in blocks]

    # -------------------------------------------------------------------------
    # Generate embeddings
    # -------------------------------------------------------------------------

    embeddings = embedder.embeddings.embed_documents(texts)

    # -------------------------------------------------------------------------
    # Build Qdrant points
    # -------------------------------------------------------------------------

    points: list[PointStruct] = []

    for block, embedding in zip(
        blocks,
        embeddings,
    ):
        point_id = make_point_id(
            block["document_id"],
            block["block_id"],
        )

        payload = {
            "chunk_id": block["block_id"],
            "document_id": block["document_id"],
            "page": block["page"],
            "content_type": block["content_type"],
            "content": block["markdown_content"],
            "bbox": block["bbox"],
            "metadata": block["metadata"],
        }

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload,
        )

        points.append(point)

    # -------------------------------------------------------------------------
    # Upload to Qdrant
    # -------------------------------------------------------------------------

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return len(points)


# =============================================================================
# INDEX ONE DOCUMENT
# =============================================================================


def index_document(
    client: QdrantClient,
    embedder: Embedder,
    split_name: str,
    file_path: Path,
    batch_size: int,
) -> int:
    """
    Load one document and index its blocks in batches.
    """

    blocks = load_document_blocks(
        split_name=split_name,
        file_path=file_path,
    )

    if not blocks:
        return 0

    indexed = 0

    for start in range(
        0,
        len(blocks),
        batch_size,
    ):
        batch = blocks[start : start + batch_size]

        indexed += index_batch(
            client=client,
            embedder=embedder,
            blocks=batch,
        )

    return indexed


# =============================================================================
# MAIN INDEXING FUNCTION
# =============================================================================


def run_indexing(
    client: QdrantClient,
    embedder: Embedder,
    files_by_split: dict[str, list[Path]],
    batch_size: int,
) -> int:
    """
    Index the entire TAT-DQA dataset.
    """

    total_documents = sum(len(files) for files in files_by_split.values())

    processed_documents = 0

    total_blocks = 0

    print()
    print("Starting TAT-DQA indexing...")

    print(f"Total documents: {total_documents}")

    print(f"Batch size: {batch_size}")

    print()

    # -------------------------------------------------------------------------
    # Process each split
    # -------------------------------------------------------------------------

    for split_name in (
        "train",
        "dev",
        "test",
    ):
        files = files_by_split.get(
            split_name,
            [],
        )

        if not files:
            continue

        print("=" * 70)

        print(f"INDEXING SPLIT: {split_name.upper()}")

        print("=" * 70)

        split_blocks = 0

        for file_path in files:
            try:
                indexed = index_document(
                    client=client,
                    embedder=embedder,
                    split_name=split_name,
                    file_path=file_path,
                    batch_size=batch_size,
                )

                split_blocks += indexed

                total_blocks += indexed

                processed_documents += 1

                # Print progress every document.
                print(
                    f"[{split_name}] "
                    f"{processed_documents}/{total_documents} "
                    f"| "
                    f"{file_path.name} "
                    f"| "
                    f"{indexed} blocks "
                    f"| "
                    f"total={total_blocks}"
                )

            except Exception as exc:
                print()
                print("[ERROR] Failed to index:")

                print(f"  Split: {split_name}")

                print(f"  File: {file_path}")

                print(f"  Error: {exc}")

                print()

                # Re-raise so we don't silently produce
                # an incomplete index.
                raise

        print()

        print(f"[{split_name}] Completed.")

        print(f"[{split_name}] Indexed blocks: {split_blocks}")

        print()

    return total_blocks


# =============================================================================
# VERIFY QDRANT
# =============================================================================


def verify_collection(
    client: QdrantClient,
) -> None:
    """
    Verify final Qdrant collection statistics.

    NOTE:
    qdrant-client version currently used by Project LEDGER
    does not expose `vectors_count` on CollectionInfo.
    Therefore we only use points_count here.
    """

    print()
    print("=" * 70)

    print("QDRANT VERIFICATION")

    print("=" * 70)

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)

    print(f"Collection: {COLLECTION_NAME}")

    print(f"Points count: {collection_info.points_count}")

    print(f"Vector size: {VECTOR_SIZE}")

    print("Distance: COSINE")

    print()

    print("Qdrant verification completed.")


# =============================================================================
# GPU INFORMATION
# =============================================================================


def print_gpu_info() -> None:
    """
    Print CUDA/GPU information.
    """

    print()
    print("GPU / CUDA INFORMATION")

    print("-" * 40)

    print(f"PyTorch version: {torch.__version__}")

    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

        print(f"GPU: {torch.cuda.get_device_name(0)}")

        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(f"GPU memory: {total_memory:.2f} GB")

    else:
        print("Running on CPU.")

    print()


# =============================================================================
# ARGUMENT PARSER
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=("Index TAT-DQA documents into Qdrant."))

    parser.add_argument(
        "--reset",
        action="store_true",
        help=("Delete and recreate the Qdrant collection before indexing."),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(f"Embedding batch size (default: {BATCH_SIZE})."),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    args = parse_args()

    print()
    print("=" * 70)

    print("PROJECT LEDGER - TAT-DQA INDEXING")

    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")

    print(f"Dataset root: {DATASET_ROOT}")

    print(f"Batch size: {args.batch_size}")

    print(f"Reset collection: {args.reset}")

    print()

    # =========================================================================
    # GPU
    # =========================================================================

    print_gpu_info()

    # =========================================================================
    # STEP 1 - Discover documents
    # =========================================================================

    print("[1/4] Discovering TAT-DQA documents...")

    files_by_split = find_json_files()

    total_documents = sum(len(files) for files in files_by_split.values())

    print(f"Total documents: {total_documents}")

    # =========================================================================
    # STEP 2 - Qdrant
    # =========================================================================

    print()
    print("[2/4] Connecting to Qdrant...")

    client = create_qdrant_client()

    if args.reset:
        reset_collection(client)

    else:
        ensure_collection(client)

    # =========================================================================
    # STEP 3 - Embedding model
    # =========================================================================

    print()
    print("[3/4] Loading embedding model...")

    embedder = Embedder(model_name=EMBEDDING_MODEL)

    print(f"Model: {EMBEDDING_MODEL}")

    print(f"Device: {embedder.device}")

    if embedder.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Embedding model loaded successfully.")

    # =========================================================================
    # STEP 4 - Index
    # =========================================================================

    print()
    print("[4/4] Indexing documents...")

    total_blocks = run_indexing(
        client=client,
        embedder=embedder,
        files_by_split=files_by_split,
        batch_size=args.batch_size,
    )

    # =========================================================================
    # Verification
    # =========================================================================

    verify_collection(client)

    print()
    print("=" * 70)

    print("INDEXING COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print(f"Documents discovered: {total_documents}")

    print(f"Blocks indexed: {total_blocks}")

    print(f"Qdrant collection: {COLLECTION_NAME}")

    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
