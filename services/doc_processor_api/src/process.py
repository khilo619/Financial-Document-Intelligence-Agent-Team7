# import sys
# import json
# import time
# import zipfile
# import statistics
# import importlib.util
# from pathlib import Path

# import requests
# from huggingface_hub import hf_hub_download

# # Setup import paths
# SRC_DIR = Path(__file__).parent
# PROJECT_ROOT = SRC_DIR.parent.parent.parent
# SHARED_DIR = PROJECT_ROOT / "shared"

# for p in (SRC_DIR, SHARED_DIR):
#     if str(p) not in sys.path:
#         sys.path.insert(0, str(p))

# TAT_DQA_DIR = Path(r"D:\member1\Financial-Document-Intelligence-Agent-Team7\TAT-DQA")
# ZIP_FILENAME = "tatdqa_docs_train.zip"
# EXTRACT_DIR = TAT_DQA_DIR / "extracted_one_pdf"

# TARGET_PDF_NAME = None 
# PDF_INDEX = 160

# USE_API = False              
# API_URL = "http://127.0.0.1:8002/process_pdf"


# def ensure_zip_downloaded(target_dir: Path, filename: str) -> Path:
#     zip_path = target_dir / filename
#     if zip_path.exists():
#         print(f"Zip already present: {zip_path}")
#         return zip_path

#     print(f"Zip not found locally. Downloading {filename} from Hugging Face...")
#     target_dir.mkdir(parents=True, exist_ok=True)

#     downloaded_path = hf_hub_download(
#         repo_id="next-tat/TAT-DQA",
#         repo_type="dataset",
#         filename=filename,
#         local_dir=str(target_dir),
#     )
#     print(f"Downloaded to: {downloaded_path}")
#     return Path(downloaded_path)


# def extract_one_pdf(
#     zip_path: Path,
#     extract_dir: Path,
#     target_name: str | None = None,
#     pdf_index: int = 0,
# ) -> Path:
#     extract_dir.mkdir(parents=True, exist_ok=True)

#     with zipfile.ZipFile(zip_path, "r") as zf:
#         pdf_names =[n for n in zf.namelist() if n.lower().endswith(".pdf")]
#         if not pdf_names:
#             raise FileNotFoundError("No PDF files found inside the zip.")

#         if target_name:
#             matches = [n for n in pdf_names if Path(n).name == target_name]
#             if not matches:
#                 raise FileNotFoundError(f"'{target_name}' not found in zip.")
#             chosen = matches[0]
#         else:
#             if pdf_index >= len(pdf_names):
#                 raise IndexError(
#                     f"pdf_index={pdf_index} out of range — zip only has {len(pdf_names)} pdfs."
#                 )
#             chosen = pdf_names[pdf_index]

#         print(f"Extracting: {chosen}")
#         extracted_path = zf.extract(chosen, path=extract_dir)

#     return Path(extracted_path)


# def save_blocks_as_json(blocks, doc_id: str) -> Path:
#     out_dir = TAT_DQA_DIR / "processed_json"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_path = out_dir / f"{doc_id}_blocks.json"

#     def to_dict(b):
#         if hasattr(b, "model_dump"):
#             return b.model_dump()
#         return b.dict()

#     data = [to_dict(b) for b in blocks]

#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)

#     content_type_counts = {}
#     for b in data:
#         ct = b.get("content_type", "unknown")
#         content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

#     print(f"Saved {len(data)} blocks to: {out_path}")
#     print(f"Content type breakdown: {content_type_counts}")

#     return out_path


# if __name__ == "__main__":
#     zip_path = ensure_zip_downloaded(TAT_DQA_DIR, ZIP_FILENAME)
#     processing_times = []
#     ocr_copy = None
#     if not USE_API:
#         print("Loading OCR module and initializing models into memory...")
#         spec = importlib.util.spec_from_file_location("ocr_copy", SRC_DIR / "ocr copy.py")
#         ocr_copy = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(ocr_copy)
#         print("Models successfully loaded!")
#     with zipfile.ZipFile(zip_path, "r") as zf:
#      pdf_names =[n for n in zf.namelist() if n.lower().endswith(".pdf")]
#     total=len(pdf_names)
#     for idx in range(105,total):
#         print(f"\n--- Processing PDF #{idx + 1} / 4 ---")
#         pdf_path = extract_one_pdf(zip_path, EXTRACT_DIR, pdf_index=idx)
#         doc_id = pdf_path.stem

#         if USE_API:
#             payload = {"pdf_path": str(pdf_path), "document_id": doc_id}
#             response = requests.post(API_URL, json=payload, timeout=600)
#             response.raise_for_status()
#             data = response.json()
#             pure_time = data.get("processing_time_s", 0.0)
#         else:
#             start = time.perf_counter()
#             blocks, _ = ocr_copy.process_single_pdf_with_metrics(str(pdf_path), doc_id)
#             pure_time = time.perf_counter() - start

#             save_blocks_as_json(blocks, doc_id)

#         print(f"Finished {pdf_path.name} in {pure_time:.3f}s")
#         processing_times.append(pure_time)

#     avg_time = statistics.mean(processing_times)
#     print("\n" + "=" * 40)
#     print(f"Total Processed        : {len(processing_times)} files")
#     print(f"Average Processing Time: {avg_time:.3f} seconds / PDF")
#     print("=" * 40)


import sys
import json
import time
import zipfile
import statistics
import importlib.util
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download


# ==============================================================================
# Setup import paths
# ==============================================================================

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
SHARED_DIR = PROJECT_ROOT / "shared"

for p in (SRC_DIR, SHARED_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ==============================================================================
# TAT-DQA paths
# ==============================================================================

TAT_DQA_DIR = PROJECT_ROOT / "data" / "tat_dqa"

ZIP_FILENAME = "tatdqa_docs_train.zip"

ZIP_PATH = TAT_DQA_DIR / ZIP_FILENAME

EXTRACT_DIR = TAT_DQA_DIR / "extracted_one_pdf"

PROCESSED_DIR = TAT_DQA_DIR / "processed_json"


# ==============================================================================
# Processing configuration
# ==============================================================================

# If you want to process a specific PDF by filename, put its filename here.
# Example:
# TARGET_PDF_NAME = "f8867689504c406cebeb5d25694d0327.pdf"
TARGET_PDF_NAME = None

# For the first test, process only PDF #1.
# Index is zero-based:
# 0 = first PDF
# 1 = second PDF
# 160 = PDF #161
PDF_INDEX = 0

# False = use OCR/Docling directly in this Python process.
# True  = send the PDF to doc-processor-api through FastAPI.
USE_API = False

API_URL = "http://127.0.0.1:8002/process_pdf"


# ==============================================================================
# Download ZIP if it does not already exist
# ==============================================================================

def ensure_zip_downloaded(target_dir: Path, filename: str) -> Path:
    """
    Make sure the requested TAT-DQA ZIP exists locally.

    If it is already present, use the local copy.
    Otherwise, download it from Hugging Face.
    """

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


# ==============================================================================
# Find PDFs inside ZIP
# ==============================================================================

def get_pdf_names(zip_path: Path) -> list[str]:
    """
    Return all PDF paths stored inside the ZIP.
    """

    with zipfile.ZipFile(zip_path, "r") as zf:
        pdf_names = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".pdf")
        ]

    if not pdf_names:
        raise FileNotFoundError(
            f"No PDF files found inside: {zip_path}"
        )

    return pdf_names


# ==============================================================================
# Extract one PDF
# ==============================================================================

def extract_one_pdf(
    zip_path: Path,
    extract_dir: Path,
    target_name: str | None = None,
    pdf_index: int = 0,
) -> Path:
    """
    Extract exactly one PDF from the ZIP.
    """

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:

        pdf_names = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".pdf")
        ]

        if not pdf_names:
            raise FileNotFoundError(
                "No PDF files found inside the zip."
            )

        # --------------------------------------------------------------
        # Select by explicit filename
        # --------------------------------------------------------------

        if target_name:
            matches = [
                name
                for name in pdf_names
                if Path(name).name == target_name
            ]

            if not matches:
                raise FileNotFoundError(
                    f"'{target_name}' not found in zip."
                )

            chosen = matches[0]

        # --------------------------------------------------------------
        # Otherwise select by index
        # --------------------------------------------------------------

        else:
            if pdf_index < 0 or pdf_index >= len(pdf_names):
                raise IndexError(
                    f"pdf_index={pdf_index} out of range. "
                    f"ZIP contains {len(pdf_names)} PDFs."
                )

            chosen = pdf_names[pdf_index]

        print(f"Extracting: {chosen}")

        extracted_path = zf.extract(
            chosen,
            path=extract_dir,
        )

    return Path(extracted_path)


# ==============================================================================
# Save DocumentBlocks as JSON
# ==============================================================================

def save_blocks_as_json(blocks, doc_id: str) -> Path:
    """
    Save the processed DocumentBlock objects as JSON.

    Output:
        data/tat_dqa/processed_json/<doc_id>_blocks.json
    """

    out_dir = PROCESSED_DIR

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = out_dir / f"{doc_id}_blocks.json"

    def to_dict(block):
        if hasattr(block, "model_dump"):
            return block.model_dump()

        return block.dict()

    data = [
        to_dict(block)
        for block in blocks
    ]

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------------
    # Print content type statistics
    # --------------------------------------------------------------

    content_type_counts = {}

    for block in data:
        content_type = block.get(
            "content_type",
            "unknown",
        )

        content_type_counts[content_type] = (
            content_type_counts.get(content_type, 0) + 1
        )

    print()
    print(f"Saved {len(data)} blocks to:")
    print(f"  {out_path}")

    print(
        f"Content type breakdown: {content_type_counts}"
    )

    return out_path


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LEDGER - TAT-DQA Document Preprocessing Test")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Make sure ZIP exists
    # --------------------------------------------------------------------------

    zip_path = ensure_zip_downloaded(
        TAT_DQA_DIR,
        ZIP_FILENAME,
    )

    print(f"ZIP: {zip_path}")

    # --------------------------------------------------------------------------
    # Get PDF list
    # --------------------------------------------------------------------------

    pdf_names = get_pdf_names(zip_path)

    total = len(pdf_names)

    print(f"Total PDFs in ZIP: {total}")

    # --------------------------------------------------------------------------
    # Select PDF
    # --------------------------------------------------------------------------

    if TARGET_PDF_NAME:

        selected_index = next(
            (
                idx
                for idx, name in enumerate(pdf_names)
                if Path(name).name == TARGET_PDF_NAME
            ),
            None,
        )

        if selected_index is None:
            raise FileNotFoundError(
                f"Target PDF '{TARGET_PDF_NAME}' was not found."
            )

        selected_pdf_index = selected_index

    else:
        selected_pdf_index = PDF_INDEX

    if selected_pdf_index < 0 or selected_pdf_index >= total:
        raise IndexError(
            f"PDF_INDEX={selected_pdf_index} is invalid. "
            f"Valid range: 0-{total - 1}"
        )

    print(
        f"Selected PDF #{selected_pdf_index + 1}: "
        f"{pdf_names[selected_pdf_index]}"
    )

    # --------------------------------------------------------------------------
    # Load OCR / Docling once
    # --------------------------------------------------------------------------

    processing_times = []

    ocr_copy = None

    if not USE_API:

        print()
        print("Loading OCR module and initializing models into memory...")

        ocr_file = SRC_DIR / "ocr copy.py"

        if not ocr_file.exists():
            raise FileNotFoundError(
                f"OCR module not found:\n{ocr_file}"
            )

        spec = importlib.util.spec_from_file_location(
            "ocr_copy",
            ocr_file,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load OCR module from: {ocr_file}"
            )

        ocr_copy = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(ocr_copy)

        print("Models successfully loaded!")

    # --------------------------------------------------------------------------
    # Extract selected PDF
    # --------------------------------------------------------------------------

    print()
    print("-" * 70)
    print(
        f"Processing PDF #{selected_pdf_index + 1} / {total}"
    )
    print("-" * 70)

    pdf_path = extract_one_pdf(
        zip_path=zip_path,
        extract_dir=EXTRACT_DIR,
        target_name=TARGET_PDF_NAME,
        pdf_index=selected_pdf_index,
    )

    doc_id = pdf_path.stem

    print(f"PDF path: {pdf_path}")
    print(f"Document ID: {doc_id}")

    # --------------------------------------------------------------------------
    # Process PDF
    # --------------------------------------------------------------------------

    if USE_API:

        print()
        print("Sending PDF to doc-processor-api...")

        start = time.perf_counter()

        payload = {
            "pdf_path": str(pdf_path),
            "document_id": doc_id,
        }

        response = requests.post(
            API_URL,
            json=payload,
            timeout=600,
        )

        response.raise_for_status()

        data = response.json()

        pure_time = data.get(
            "processing_time_s",
            0.0,
        )

        print(
            f"API request completed in "
            f"{time.perf_counter() - start:.3f}s"
        )

    else:

        print()
        print("Processing PDF with Docling/OCR...")

        start = time.perf_counter()

        blocks, _ = (
            ocr_copy.process_single_pdf_with_metrics(
                str(pdf_path),
                doc_id,
            )
        )

        pure_time = time.perf_counter() - start

        # --------------------------------------------------------------
        # Save processed DocumentBlocks
        # --------------------------------------------------------------

        output_path = save_blocks_as_json(
            blocks,
            doc_id,
        )

        print()
        print(f"Processed JSON: {output_path}")

    # --------------------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------------------

    print()
    print(f"Finished {pdf_path.name}")
    print(f"Processing time: {pure_time:.3f} seconds")

    processing_times.append(pure_time)

    # --------------------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------------------

    avg_time = statistics.mean(
        processing_times
    )

    print()
    print("=" * 70)
    print("PREPROCESSING TEST SUMMARY")
    print("=" * 70)

    print(f"Total PDFs in ZIP     : {total}")
    print(f"PDFs processed        : {len(processing_times)}")
    print(f"PDF index             : {selected_pdf_index}")
    print(f"Average processing    : {avg_time:.3f} seconds / PDF")

    print()
    print(f"Processed output dir  : {PROCESSED_DIR}")
    print("=" * 70)