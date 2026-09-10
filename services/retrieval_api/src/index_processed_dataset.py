import json
import os
import time
from pathlib import Path

from shared.models import DocumentBlock
from src.qdrant_store import QdrantStore

# ============================================================
# Configuration
# ============================================================

PROCESSED_DIR = Path("/data/processed_json")

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "ledger_documents_processed",
)

BATCH_SIZE = 32

PROGRESS_FILE = Path("/data/processed_json/.index_progress.json")


# ============================================================
# Helpers
# ============================================================


def get_processed_files():
    """
    Return all real processed JSON files.

    Checkpoint files are excluded.
    """
    files = sorted(p for p in PROCESSED_DIR.glob("*_blocks.json") if "-checkpoint_blocks.json" not in p.name)

    return files


def load_blocks(json_path: Path):
    """
    Load and validate all DocumentBlocks from one JSON file.
    """

    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_blocks = data
    elif isinstance(data, dict):
        raw_blocks = data.get(
            "blocks",
            data.get("document_blocks", []),
        )
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data)}")

    blocks = [DocumentBlock.model_validate(block) for block in raw_blocks]

    return blocks


def load_progress():
    """
    Load completed files from the progress file.
    """

    if not PROGRESS_FILE.exists():
        return set()

    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return set(data.get("completed_files", []))

    except Exception:
        print("Warning: progress file could not be read.")
        print("Starting without progress information.")
        return set()


def save_progress(completed_files):
    """
    Save completed files atomically.
    """

    temp_file = PROGRESS_FILE.with_suffix(".tmp")

    data = {
        "completed_files": sorted(completed_files),
        "completed_count": len(completed_files),
    }

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    temp_file.replace(PROGRESS_FILE)


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 70)
    print("TAT-DQA PROCESSED DATASET INDEXING")
    print("=" * 70)

    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Qdrant collection:   {COLLECTION_NAME}")
    print(f"Batch size:          {BATCH_SIZE}")

    files = get_processed_files()

    print(f"Files found:         {len(files)}")

    if not files:
        raise RuntimeError("No processed JSON files found.")

    completed_files = load_progress()

    remaining_files = [path for path in files if path.name not in completed_files]

    print(f"Already completed:   {len(completed_files)}")
    print(f"Remaining files:     {len(remaining_files)}")

    if not remaining_files:
        print("\nAll files are already indexed.")
        return

    # --------------------------------------------------------
    # Initialize Qdrant + local embedding model
    # --------------------------------------------------------

    print("\nInitializing QdrantStore...")
    store = QdrantStore()

    print("QdrantStore initialized.")
    print(f"Collection: {store.COLLECTION_NAME}")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_blocks = 0
    total_indexed = 0
    start_time = time.time()

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    for file_index, json_path in enumerate(
        remaining_files,
        start=1,
    ):
        file_start = time.time()

        try:
            blocks = load_blocks(json_path)

            if not blocks:
                print(f"[{file_index}/{len(remaining_files)}] {json_path.name}: 0 blocks")

                completed_files.add(json_path.name)
                save_progress(completed_files)
                continue

            # ------------------------------------------------
            # Index this document in batches
            # ------------------------------------------------

            indexed = store.index_blocks(
                blocks,
                batch_size=BATCH_SIZE,
            )

            total_blocks += len(blocks)
            total_indexed += indexed

            # ------------------------------------------------
            # Mark file complete ONLY after successful indexing
            # ------------------------------------------------

            completed_files.add(json_path.name)
            save_progress(completed_files)

            elapsed_file = time.time() - file_start
            elapsed_total = time.time() - start_time

            completed_now = len(completed_files)

            avg_file = elapsed_total / file_index

            remaining = len(files) - completed_now

            eta_seconds = avg_file * remaining

            print(f"\n[{file_index}/{len(remaining_files)}] COMPLETED: {json_path.name}")

            print(f"  Blocks: {len(blocks)}")

            print(f"  File time: {elapsed_file:.2f}s")

            print(f"  Progress: {completed_now}/{len(files)} files")

            print(f"  ETA: {eta_seconds / 3600:.2f} hours")

        except KeyboardInterrupt:
            print("\n\nIndexing interrupted by user.")
            print("Progress has been saved for completed files.")
            raise

        except Exception as exc:
            print(f"\nERROR processing {json_path.name}:")
            print(exc)

            print("\nStopping so the failed file can be investigated.")

            raise

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    total_time = time.time() - start_time

    print("\n")
    print("=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)

    print(f"Files completed: {len(completed_files)}/{len(files)}")
    print(f"Blocks indexed this run: {total_indexed}")
    print(f"Blocks loaded this run:  {total_blocks}")
    print(f"Time: {total_time / 3600:.2f} hours")

    print("\nCollection:")
    print(COLLECTION_NAME)


if __name__ == "__main__":
    main()
