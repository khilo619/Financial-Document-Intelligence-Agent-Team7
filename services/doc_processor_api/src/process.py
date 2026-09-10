"""
process.py: Standalone and batch runner for processing TAT-DQA PDFs.
Owned by Zeina (Member 1).
"""

import argparse
import json
import os
import statistics
import sys
import zipfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

# Setup import paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
SHARED_DIR = PROJECT_ROOT / "shared"

for p in (SRC_DIR, PROJECT_ROOT, SHARED_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from services.doc_processor_api.src.ocr import (
        get_doc_converter,
        process_single_pdf_with_metrics,
    )
except ImportError:
    from src.ocr import (
        get_doc_converter,
        process_single_pdf_with_metrics,
    )

# Configurable paths with environment variable overrides
TAT_DQA_DIR = Path(os.getenv("TAT_DQA_DIR", PROJECT_ROOT / "TAT-DQA"))
ZIP_FILENAME = "tatdqa_docs_train.zip"
EXTRACT_DIR = TAT_DQA_DIR / "extracted_one_pdf"
API_URL = os.getenv("DOC_PROCESSOR_URL", "http://127.0.0.1:8002/process_pdf")


def ensure_zip_downloaded(target_dir: Path, filename: str) -> Path:
    """Ensures the TAT-DQA zip archive is downloaded from Hugging Face."""
    zip_path = target_dir / filename
    if zip_path.exists():
        print(f"Zip already present: {zip_path}")
        return zip_path

    print(f"Zip not found locally. Downloading {filename} from Hugging Face...")
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id="next-tat/TAT-DQA",
        repo_type="dataset",
        filename=filename,
        local_dir=str(target_dir),
    )
    print(f"Downloaded to: {downloaded_path}")
    return Path(downloaded_path)


def extract_one_pdf(
    zip_path: Path,
    extract_dir: Path,
    target_name: str | None = None,
    pdf_index: int = 0,
) -> Path:
    """Extracts a specific PDF by name or deterministic sorted index."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Enforce deterministic alphabetical sort across all operating systems
        pdf_names = sorted([n for n in zf.namelist() if n.lower().endswith(".pdf")])
        if not pdf_names:
            raise FileNotFoundError("No PDF files found inside the zip.")

        if target_name:
            matches = [n for n in pdf_names if Path(n).name == target_name]
            if not matches:
                raise FileNotFoundError(f"'{target_name}' not found in zip.")
            chosen = matches[0]
        else:
            if pdf_index >= len(pdf_names):
                raise IndexError(f"pdf_index={pdf_index} out of range — zip has {len(pdf_names)} PDFs.")
            chosen = pdf_names[pdf_index]

        print(f"Extracting: {chosen}")
        extracted_path = zf.extract(chosen, path=extract_dir)

    return Path(extracted_path)


def save_blocks_as_json(blocks, doc_id: str, out_dir: Path | None = None) -> Path:
    """Serializes DocumentBlock list to JSON."""
    target_dir = out_dir if out_dir is not None else TAT_DQA_DIR / "processed_json"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{doc_id}_blocks.json"

    def to_dict(b):
        if hasattr(b, "model_dump"):
            return b.model_dump()
        if hasattr(b, "dict"):
            return b.dict()
        return b

    data = [to_dict(b) for b in blocks]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    content_type_counts: dict[str, int] = {}
    for b in data:
        ct = b.get("content_type", "unknown")
        content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

    print(f"Saved {len(data)} blocks to: {out_path}")
    print(f"Content type breakdown: {content_type_counts}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Process TAT-DQA PDFs into DocumentBlocks")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting PDF index")
    parser.add_argument("--count", type=int, default=1, help="Number of PDFs to process")
    parser.add_argument("--target-name", type=str, default=None, help="Target PDF filename")
    parser.add_argument("--use-api", action="store_true", help="Send request to FastAPI service")
    args = parser.parse_args()

    zip_path = ensure_zip_downloaded(TAT_DQA_DIR, ZIP_FILENAME)

    with zipfile.ZipFile(zip_path, "r") as zf:
        pdf_names = sorted([n for n in zf.namelist() if n.lower().endswith(".pdf")])
    total_pdfs = len(pdf_names)
    print(f"Total PDFs in archive: {total_pdfs}")

    doc_converter = None
    if not args.use_api:
        print("Initializing Docling converter on local accelerator...")
        doc_converter = get_doc_converter()
        print("Docling converter ready.")

    processing_times = []
    end_idx = min(args.start_idx + args.count, total_pdfs)

    for idx in range(args.start_idx, end_idx):
        print(f"\n--- Processing PDF #{idx + 1} / {end_idx} ---")
        pdf_path = extract_one_pdf(zip_path, EXTRACT_DIR, target_name=args.target_name, pdf_index=idx)
        doc_id = pdf_path.stem

        if args.use_api:
            payload = {"pdf_path": str(pdf_path), "document_id": doc_id}
            response = requests.post(API_URL, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            pure_time = data.get("processing_time_s", 0.0)
        else:
            blocks, pure_time = process_single_pdf_with_metrics(
                pdf_path=str(pdf_path),
                doc_id=doc_id,
                doc_converter=doc_converter,
            )
            save_blocks_as_json(blocks, doc_id)

        print(f"Finished {pdf_path.name} in {pure_time:.3f}s")
        processing_times.append(pure_time)

    if processing_times:
        avg_time = statistics.mean(processing_times)
        print("\n" + "=" * 40)
        print(f"Total Processed        : {len(processing_times)} files")
        print(f"Average Processing Time: {avg_time:.3f} seconds / PDF")
        print("=" * 40)


if __name__ == "__main__":
    main()
