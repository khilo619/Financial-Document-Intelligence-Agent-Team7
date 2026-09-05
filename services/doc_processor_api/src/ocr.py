import os
import zipfile
from pathlib import Path
import warnings
import json
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions,TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import SectionHeaderItem, TextItem, TableItem
from docling_core.types.doc import PictureItem
from pypdf import PdfReader
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
zippath=Path("./TAT-DQA")/"tatdqa_docs_train.zip"
file_dir=Path("./TAT-DQA")/"samples"
file_dir.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(zippath, 'r') as r:
  pdf_files = [f for f in r.namelist() if f.lower().endswith('.pdf')]
  selected_pdfs = pdf_files[62:68]
  for name in selected_pdfs:
     target_path = os.path.join(file_dir, name)
     if os.path.exists(target_path):
        print(f"Skipping extraction: {name} already exists in {file_dir}")
        continue  
     r.extract(name, path=file_dir)
     print(f"Extracted: {name}")
output_dir = Path("./TAT-DQA/processed_json")
images_dir = output_dir / "extracted_images"
output_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)
pipeline_options = PdfPipelineOptions()
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
pdf_dir=Path("./TAT-DQA")/"samples"/"train"
pdf_paths=list(pdf_dir.glob("*.pdf"))
def process_pdfs_to_custom_schema():
    for pdf_path in pdf_paths:
        doc_filename = pdf_path.name
        print(f"Parsing: {doc_filename}...")
        result = doc_converter.convert(pdf_path)
        doc = result.document
        total_pages = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 1
        blocks = []
        block_counter = 1
        current_section = "General"
        processed_picture_ids = set()
        for item, level in doc.iterate_items():
            page_no = item.prov[0].page_no if (hasattr(item, "prov") and item.prov) else 1
            bbox = None
            if hasattr(item, "prov") and item.prov and hasattr(item.prov[0], "bbox"):
                bbox = [round(c, 2) for c in item.prov[0].bbox.as_tuple()]
            if isinstance(item,SectionHeaderItem):
                current_section = item.text.strip()
                block_id = f"blk_{block_counter:03d}"
                blocks.append({
                "block_id": block_id,
                "page": page_no,
                "content_type": "header",
                "section_header": current_section,
                "markdown_content": current_section,
                "bbox": bbox
                    })
                block_counter += 1
                
            elif isinstance(item, TextItem) and hasattr(item, "text") and item.text.strip():
                block_id = f"blk_{block_counter:03d}"
                blocks.append({
                    "block_id": block_id,
                    "page": page_no,
                    "content_type": "text",
                    "section_header": current_section,
                    "markdown_content": item.text.strip(),
                    "bbox": bbox
                })
                block_counter += 1
            elif  isinstance(item, PictureItem):
               item_id = id(item)
               if item_id in processed_picture_ids:
                 continue
               processed_picture_ids.add(item_id)
               block_id = f"blk_{block_counter:03d}"
               clean_doc_id = os.path.splitext(doc_filename)[0]
               image_filename = f"{clean_doc_id}_{block_id}.png"
               image_path = os.path.join(images_dir, image_filename)
               markdown_image_path = image_path.replace("\\", "/")
               if os.path.exists(image_path):
                pass
               else:
                try:
                 image = item.get_image(doc)
                 if image:
                  image.save(image_path, format="PNG")
                 else:
                  markdown_image_path = ""
                except Exception as e:
                 print(f"Failed to save image for {block_id}: {e}")
                 markdown_image_path = ""
               caption_text = item.caption.text.strip() if (hasattr(item, "caption") and item.caption) else "Figure/Chart"
               blocks.append({
                 "block_id": block_id,
                 "page": page_no, 
                 "content_type": "figure",
                 "section_header": current_section,
                 "markdown_content": f"![{caption_text}]({markdown_image_path})" if markdown_image_path else f"![{caption_text}]",
                 "bbox": bbox  
                   })
               block_counter += 1  
            elif isinstance(item,TableItem) :
               block_id = f"blk_{block_counter:03d}"
               try:
                if hasattr(item, "export_to_markdown"):
                 table_md = item.export_to_markdown(doc)
                else:
                 df = item.export_to_dataframe(doc)
                 table_md = df.to_markdown(index=False)
               except Exception as e:
                 print(f"Failed to export table for {block_id}: {e}")
                 table_md = ""
               blocks.append({
                "block_id": block_id,
                "page": page_no,
                "content_type": "table",
                "section_header": current_section,
                "markdown_content": table_md,
                "bbox": bbox
                   })
               block_counter += 1     
        blocks.sort(key=lambda b: (b["page"],-b["bbox"][1] if b["bbox"] else 0,b["bbox"][0] if b["bbox"] else 0 ))
        merged_blocks = []
        for b in blocks:
          if not merged_blocks:
            merged_blocks.append(b)
            continue
          prev = merged_blocks[-1]
          if (
            b["content_type"] == "text" 
            and prev["content_type"] == "text"
            and b["page"] == prev["page"]
            and b["section_header"] == prev["section_header"]
            and not prev["markdown_content"].endswith(('.', ':', '?', '!'))
              ):

             prev["markdown_content"] += " " + b["markdown_content"]
             if prev["bbox"] and b["bbox"]:
               prev["bbox"][0] = min(prev["bbox"][0], b["bbox"][0])  
               prev["bbox"][1] = min(prev["bbox"][1], b["bbox"][1])  
               prev["bbox"][2] = max(prev["bbox"][2], b["bbox"][2])  
               prev["bbox"][3] = max(prev["bbox"][3], b["bbox"][3])  
             else:
               merged_blocks.append(b)
        final_blocks = []
        for idx, b in enumerate(merged_blocks, start=1):
          final_blocks.append({
        "block_id": f"blk_{idx:03d}",
        "page": b["page"],
        "content_type": b["content_type"],
        "section_header": b["section_header"],
        "markdown_content": b["markdown_content"],
        "bbox": b["bbox"]
            })
        final_doc = {
            "document_id": doc_filename,
            "total_pages": total_pages,
            "blocks": blocks
         }

        out_json_path = Path("./TAT-DQA")/"processed_json"/f"{pdf_path.stem}.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(final_doc, f, indent=2, ensure_ascii=False)
        print(f" Saved matching JSON: {out_json_path.name}")
import json
import csv
from pathlib import Path

INPUT_DIR =  Path("./TAT-DQA")/"processed_json" 
MIN_CHAR_COUNT = 50 

def validate_custom_json(file_path: Path):
    errors = []
    stats = {
        "file_name": file_path.name,
        "document_id": "",
        "total_pages": 0,
        "total_blocks": 0,
        "empty_blocks": 0,
        "total_chars": 0,
        "table_blocks": 0,
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return False, ["Corrupted JSON file"], stats
    except Exception as e:
        return False, [f"Read Error: {str(e)}"], stats
    stats["document_id"] = data.get("document_id", "UNKNOWN")
    stats["total_pages"] = data.get("total_pages", 0)
    blocks = data.get("blocks", [])
    stats["total_blocks"] = len(blocks)
    if not blocks:
        errors.append("No blocks found in file")
        return False, errors, stats

    for idx, block in enumerate(blocks):
        content = block.get("markdown_content", "").strip()
        stats["total_chars"] += len(content)
        if not content:
            stats["empty_blocks"] += 1
        if block.get("content_type") == "table":
            stats["table_blocks"] += 1
        missing_keys = [key for key in ["block_id", "page", "content_type", "bbox"] if key not in block]
        if missing_keys:
            errors.append(1)
    if stats["empty_blocks"] > 0:
        errors.append(1)
    if stats["total_chars"] < MIN_CHAR_COUNT:
        errors.append(1)
    is_valid = len(errors) == 0
    return is_valid


def run_batch_validation():
    json_files = list(INPUT_DIR.glob("*.json"))
    passed_count = 0
    failed_count = 0
    for file_path in json_files:
            is_valid = validate_custom_json(file_path)
            if is_valid:
                passed_count += 1
            else:
                failed_count += 1


    print("VALIDATION RESULTS")
    print(f"Total Scanned : {len(json_files)}")
    print(f"Valid Files  : {passed_count}")
    print(f"Flagged Files: {failed_count}")



if __name__ == "__main__":
    process_pdfs_to_custom_schema()
    run_batch_validation()
          
   
          