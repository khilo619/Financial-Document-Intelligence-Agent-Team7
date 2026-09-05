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

import os
from pathlib import Path
from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem


def process_pdfs_to_custom_schema(pdf_path: str, doc_id: str):
    result = doc_converter.convert(pdf_path)
    doc = result.document
    images_dir = Path("./TAT-DQA/processed_json/extracted_images")
    images_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
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
            blocks.append(
                {
                    "block_id": f"blk_{block_counter:03d}",
                    "document_id": doc_id,
                    "page": page_no,
                    "content_type": "header",
                    "markdown_content": section_text,
                    "bbox": bbox,
                }
            )
            block_counter += 1

        elif (
            isinstance(item, TextItem)
            and hasattr(item, "text")
            and item.text.strip()
        ):
            blocks.append(
                {
                    "block_id": f"blk_{block_counter:03d}",
                    "document_id": doc_id,
                    "page": page_no,
                    "content_type": "text",
                    "markdown_content": item.text.strip(),
                    "bbox": bbox,
                }
            )
            block_counter += 1

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
            blocks.append(
                {
                    "block_id": block_id,
                    "document_id": doc_id,
                    "page": page_no,
                    "content_type": "figure",
                    "markdown_content": f"![{caption_text}]({markdown_image_path})"
                    if markdown_image_path
                    else f"![{caption_text}]",
                    "bbox": bbox,
                }
            )
            block_counter += 1
        elif isinstance(item, TableItem):
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

            blocks.append(
                {
                    "block_id": block_id,
                    "document_id": doc_id,
                    "page": page_no,
                    "content_type": "table",
                    "markdown_content": table_md,
                    "bbox": bbox,
                }
            )
            block_counter += 1
    blocks.sort(
        key=lambda b: (
            b["page"],
            -b["bbox"][1] if b["bbox"] else 0,
            b["bbox"][0] if b["bbox"] else 0,
        )
    )
    merged_blocks = []
    for b in blocks:
        if not merged_blocks:
            merged_blocks.append(b)
            continue
        prev = merged_blocks[-1]
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
    final_blocks = []
    for idx, b in enumerate(merged_blocks, start=1):
        final_blocks.append(
            {
                "block_id": f"blk_{idx:03d}",
                "document_id": doc_id,
                "page": b["page"],
                "content_type": b["content_type"],
                "markdown_content": b["markdown_content"],
                "bbox": b["bbox"],
            }
        )

    return final_blocks
