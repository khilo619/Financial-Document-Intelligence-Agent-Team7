import os
import zipfile
from pathlib import Path
import warnings
import json
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions,TableFormerMode,AcceleratorOptions,AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import SectionHeaderItem, TextItem, TableItem
from docling_core.types.doc import PictureItem
from pypdf import PdfReader
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch

def get_doc_converter() -> DocumentConverter:
    """Helper function to construct DocumentConverter once."""

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(device=device)
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

import os
from pathlib import Path
from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem
import time

from pathlib import Path
from typing import List
import uuid
from models import DocumentBlock

def process_pdfs_to_custom_schema(
    pdf_path: str, 
    doc_id: str, 
    doc_converter: DocumentConverter 
) -> List[DocumentBlock]:
    result = doc_converter.convert(pdf_path)
    doc = result.document
    # images_dir = Path("./TAT-DQA/processed_json/extracted_images")
    images_dir = Path("./data/tat_dqa/processed_json/extracted_images")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    raw_blocks = []
    block_counter = 1
    processed_picture_ids = set()

    for item, level in doc.iterate_items():
        page_no = (
            item.prov[0].page_no if (hasattr(item, "prov") and item.prov) else 1
        )
        bbox = None
        if hasattr(item, "prov") and item.prov and hasattr(item.prov[0], "bbox"):
            bbox = [round(c, 2) for c in item.prov[0].bbox.as_tuple()]

      
        if isinstance(item, SectionHeaderItem):
            section_text = item.text.strip()
            raw_blocks.append(
                {
                    "page": page_no,
                    "content_type": "header",
                    "markdown_content": section_text,
                    "table_rows": None,
                    "bbox": bbox,
                }
            )

        elif (
            isinstance(item, TextItem)
            and hasattr(item, "text")
            and item.text.strip()
        ):
            raw_blocks.append(
                {
                    "page": page_no,
                    "content_type": "text",
                    "markdown_content": item.text.strip(),
                    "table_rows": None,
                    "bbox": bbox,
                }
            )

        elif isinstance(item, PictureItem):
            item_id = id(item)
            if item_id in processed_picture_ids:
                continue
            processed_picture_ids.add(item_id)

            block_id = f"blk_{block_counter:03d}"
            clean_doc_id = Path(doc_id).stem
            image_filename = f"{clean_doc_id}_{block_id}.png"
            image_path = images_dir / image_filename
            markdown_image_path = str(image_path).replace("\\", "/")

            if not image_path.exists():
                try:
                    image = item.get_image(doc)
                    if image:
                        image.save(image_path, format="PNG")
                    else:
                        markdown_image_path = ""
                except Exception as e:
                    print(f"Failed to save image for {block_id}: {e}")
                    markdown_image_path = ""

            caption_text = (
                item.caption.text.strip()
                if (hasattr(item, "caption") and item.caption)
                else "Figure/Chart"
            )
            
            md_content = (
                f"![{caption_text}]({markdown_image_path})"
                if markdown_image_path
                else f"![{caption_text}]"
            )

            raw_blocks.append(
                {
                    "page": page_no,
                    "content_type": "text", 
                    "markdown_content": md_content,
                    "table_rows": None,
                    "bbox": bbox,
                }
            )
            block_counter += 1

        
        elif isinstance(item, TableItem):
            table_md = ""
            table_rows_grid = None

            try:
                if hasattr(item, "export_to_markdown"):
                    table_md = item.export_to_markdown(doc)
                else:
                    df = item.export_to_dataframe(doc)
                    table_md = df.to_markdown(index=False)
            except Exception as e:
                print(f"Failed to export table markdown: {e}")

            try:
                df = item.export_to_dataframe(doc)
                headers = [str(c) for c in df.columns]
                data = df.astype(str).values.tolist()
                table_rows_grid = [headers] + data
            except Exception as e:
                print(f"Failed to extract 2D grid for table: {e}")

            raw_blocks.append(
                {
                    "page": page_no,
                    "content_type": "table",
                    "markdown_content": table_md,
                    "table_rows": table_rows_grid,
                    "bbox": bbox,
                }
            )


    merged_blocks = []
    for b in raw_blocks:
        if not merged_blocks:
            merged_blocks.append(b)
            continue
        prev = merged_blocks[-1]
        same_column = True
        if prev["bbox"] and b["bbox"]:
         prev_x_center = (prev["bbox"][0] + prev["bbox"][2]) / 2
         curr_x_center = (b["bbox"][0] + b["bbox"][2]) / 2
         same_column = abs(prev_x_center - curr_x_center) < 150  
        is_mergeable = (
            b["content_type"] == "text"
            and prev["content_type"] == "text"
            and b["page"] == prev["page"]
            and not prev["markdown_content"].endswith((".", ":", "?", "!"))
        )

        if is_mergeable:
            prev["markdown_content"] += " " + b["markdown_content"]
            if prev["bbox"] and b["bbox"]:
                prev["bbox"][0] = min(prev["bbox"][0], b["bbox"][0])
                prev["bbox"][1] = min(prev["bbox"][1], b["bbox"][1])
                prev["bbox"][2] = max(prev["bbox"][2], b["bbox"][2])
                prev["bbox"][3] = max(prev["bbox"][3], b["bbox"][3])
        else:
            merged_blocks.append(b)


    final_document_blocks: List[DocumentBlock] = []
    for idx, b in enumerate(merged_blocks, start=1):
        block_obj = DocumentBlock(
            block_id=f"blk_{idx:03d}",
            document_id=doc_id,
            page=b["page"],
            content_type=b["content_type"], 
            markdown_content=b["markdown_content"],
            table_rows=b["table_rows"],
            bbox=b["bbox"],
            metadata={},
        )
        final_document_blocks.append(block_obj)

    return final_document_blocks
doc_converter = get_doc_converter()
def process_single_pdf_with_metrics(pdf_path: str, doc_id: str):

    start_time = time.perf_counter()

    blocks = process_pdfs_to_custom_schema(
    pdf_path=pdf_path, 
    doc_id=doc_id, 
    doc_converter=doc_converter  
      )

    elapsed_seconds = round(time.perf_counter() - start_time, 3)

    total_pages = max((b.page for b in blocks), default=1) if blocks else 1
    seconds_per_page = round(elapsed_seconds / total_pages, 3)

    print(f"File: {Path(pdf_path).name}")
    print(f"Total Pages: {total_pages}")
    print(f"Total Time: {elapsed_seconds} seconds")
    print(f"Speed: {seconds_per_page} seconds/page")

    return blocks, elapsed_seconds