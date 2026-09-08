import sys
import json
import time
import zipfile
import statistics
import importlib.util
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

# Setup import paths
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
SHARED_DIR = PROJECT_ROOT / "shared"

for p in (SRC_DIR, SHARED_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TAT_DQA_DIR = Path(r"D:\member1\Financial-Document-Intelligence-Agent-Team7\TAT-DQA")
ZIP_FILENAME = "tatdqa_docs_train.zip"
EXTRACT_DIR = TAT_DQA_DIR / "extracted_one_pdf"

TARGET_PDF_NAME = None 
PDF_INDEX = 160

USE_API = False              
API_URL = "http://127.0.0.1:8002/process_pdf"


def ensure_zip_downloaded(target_dir: Path, filename: str) -> Path:
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
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        pdf_names =[n for n in zf.namelist() if n.lower().endswith(".pdf")]
        if not pdf_names:
            raise FileNotFoundError("No PDF files found inside the zip.")

        if target_name:
            matches = [n for n in pdf_names if Path(n).name == target_name]
            if not matches:
                raise FileNotFoundError(f"'{target_name}' not found in zip.")
            chosen = matches[0]
        else:
            if pdf_index >= len(pdf_names):
                raise IndexError(
                    f"pdf_index={pdf_index} out of range — zip only has {len(pdf_names)} pdfs."
                )
            chosen = pdf_names[pdf_index]

        print(f"Extracting: {chosen}")
        extracted_path = zf.extract(chosen, path=extract_dir)

    return Path(extracted_path)


def save_blocks_as_json(blocks, doc_id: str) -> Path:
    out_dir = TAT_DQA_DIR / "processed_json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}_blocks.json"

    def to_dict(b):
        if hasattr(b, "model_dump"):
            return b.model_dump()
        return b.dict()

    data = [to_dict(b) for b in blocks]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    content_type_counts = {}
    for b in data:
        ct = b.get("content_type", "unknown")
        content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

    print(f"Saved {len(data)} blocks to: {out_path}")
    print(f"Content type breakdown: {content_type_counts}")

    return out_path


if __name__ == "__main__":
    zip_path = ensure_zip_downloaded(TAT_DQA_DIR, ZIP_FILENAME)
    processing_times = []
    ocr_copy = None
    if not USE_API:
        print("Loading OCR module and initializing models into memory...")
        spec = importlib.util.spec_from_file_location("ocr_copy", SRC_DIR / "ocr copy.py")
        ocr_copy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ocr_copy)
        print("Models successfully loaded!")
    with zipfile.ZipFile(zip_path, "r") as zf:
     pdf_names =[n for n in zf.namelist() if n.lower().endswith(".pdf")]
    total=len(pdf_names)
    for idx in range(105,total):
        print(f"\n--- Processing PDF #{idx + 1} / 4 ---")
        pdf_path = extract_one_pdf(zip_path, EXTRACT_DIR, pdf_index=idx)
        doc_id = pdf_path.stem

        if USE_API:
            payload = {"pdf_path": str(pdf_path), "document_id": doc_id}
            response = requests.post(API_URL, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            pure_time = data.get("processing_time_s", 0.0)
        else:
            start = time.perf_counter()
            blocks, _ = ocr_copy.process_single_pdf_with_metrics(str(pdf_path), doc_id)
            pure_time = time.perf_counter() - start

            save_blocks_as_json(blocks, doc_id)

        print(f"Finished {pdf_path.name} in {pure_time:.3f}s")
        processing_times.append(pure_time)

    avg_time = statistics.mean(processing_times)
    print("\n" + "=" * 40)
    print(f"Total Processed        : {len(processing_times)} files")
    print(f"Average Processing Time: {avg_time:.3f} seconds / PDF")
    print("=" * 40)