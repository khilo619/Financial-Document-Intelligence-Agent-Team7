import json
import time
from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "tat_dqa"
OUTPUT_DIR = DATA_DIR / "processed_json"

DOC_PROCESSOR_URL = "http://localhost:8002/process_pdf"

# None = process all PDFs
# Set to a number like 3 for a small test.
MAX_FILES = None


# ============================================================
# Helpers
# ============================================================

def get_pdf_files() -> list[Path]:
    """Return dataset PDFs only, excluding checkpoints."""

    pdf_files = []

    for split in ["train", "dev", "test"]:
        split_dir = DATA_DIR / split

        if not split_dir.exists():
            continue

        for pdf_path in split_dir.rglob("*.pdf"):
            if ".ipynb_checkpoints" not in pdf_path.parts:
                pdf_files.append(pdf_path)

    return sorted(pdf_files)


def get_document_id(pdf_path: Path) -> str:
    """Use the PDF filename without extension as document ID."""
    return pdf_path.stem


def get_output_path(document_id: str) -> Path:
    """Return the processed JSON path for a document."""
    return OUTPUT_DIR / f"{document_id}_processed.json"


def process_pdf(pdf_path: Path) -> dict:
    """Send one PDF to the Doc Processor API."""

    relative_path = pdf_path.relative_to(DATA_DIR)

    # Host:
    # data/tat_dqa/train/train/file.pdf
    #
    # Container:
    # /data/tat_dqa/train/train/file.pdf

    container_pdf_path = (
        "/data/tat_dqa/"
        + str(relative_path).replace("\\", "/")
    )

    document_id = get_document_id(pdf_path)

    body = {
        "pdf_path": container_pdf_path,
        "document_id": document_id,
    }

    response = requests.post(
        DOC_PROCESSOR_URL,
        json=body,
        timeout=3600,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = get_pdf_files()

    print("=" * 70)
    print("LEDGER - TAT-DQA Batch Processing")
    print("=" * 70)

    print(f"Total PDFs found: {len(pdf_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"API: {DOC_PROCESSOR_URL}")
    print()

    # --------------------------------------------------------
    # Select files
    # --------------------------------------------------------

    if MAX_FILES is None:
        files_to_process = pdf_files
    else:
        files_to_process = pdf_files[:MAX_FILES]

    print(f"Files selected for this run: {len(files_to_process)}")
    print()

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    success = 0
    skipped = 0
    failed = 0

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    total_files = len(files_to_process)

    for index, pdf_path in enumerate(files_to_process, start=1):

        document_id = get_document_id(pdf_path)
        output_path = get_output_path(document_id)

        print("-" * 70)
        print(f"[{index}/{total_files}] {pdf_path.name}")
        print(f"Document ID: {document_id}")

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if output_path.exists():
            print("SKIPPED - processed JSON already exists.")
            skipped += 1
            continue

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        start = time.perf_counter()

        try:

            result = process_pdf(pdf_path)

            elapsed = time.perf_counter() - start

            # ------------------------------------------------
            # Save JSON
            # ------------------------------------------------

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    result,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            success += 1

            print("SUCCESS")
            print(f"Pages: {result.get('total_pages')}")
            print(f"Blocks: {result.get('total_blocks')}")
            print(
                f"Processing time: "
                f"{result.get('processing_time_s')} sec"
            )
            print(f"Client time: {elapsed:.2f} sec")
            print(f"Saved: {output_path}")

            # ------------------------------------------------
            # Progress / ETA
            # ------------------------------------------------

            processed_count = success + skipped

            elapsed_total = time.perf_counter() - total_start

            if processed_count > 0:
                avg_time = elapsed_total / processed_count
                remaining = total_files - processed_count
                eta_seconds = avg_time * remaining

                eta_minutes = eta_seconds / 60
                eta_hours = eta_minutes / 60

                print()
                print(
                    f"Progress: {processed_count}/{total_files}"
                )
                print(
                    f"Average time/file: {avg_time:.2f} sec"
                )

                if eta_hours >= 1:
                    print(
                        f"Estimated remaining time: "
                        f"{eta_hours:.2f} hours"
                    )
                else:
                    print(
                        f"Estimated remaining time: "
                        f"{eta_minutes:.2f} minutes"
                    )

        except Exception as exc:

            elapsed = time.perf_counter() - start

            failed += 1

            print("FAILED")
            print(f"Error: {exc}")
            print(f"Elapsed: {elapsed:.2f} sec")

    # ========================================================
    # Final Summary
    # ========================================================

    total_elapsed = time.perf_counter() - total_start

    print()
    print("=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)

    print(f"Selected: {total_files}")
    print(f"Success:  {success}")
    print(f"Skipped:  {skipped}")
    print(f"Failed:   {failed}")

    print(
        f"Elapsed:  "
        f"{total_elapsed:.2f} sec"
    )

    if success > 0:
        avg_success_time = total_elapsed / success
        print(
            f"Average successful file time: "
            f"{avg_success_time:.2f} sec"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()