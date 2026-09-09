"""
ocr.py: Layout OCR and Table Parsing Engine powered by IBM Docling.
Owned by Zeina (Member 1).
"""

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch = None

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
        TableFormerMode,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import (
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
    )
except ImportError:
    InputFormat = None
    AcceleratorDevice = None
    AcceleratorOptions = None
    PdfPipelineOptions = None
    TableFormerMode = None
    DocumentConverter = Any
    PdfFormatOption = None
    PictureItem = None
    SectionHeaderItem = None
    TableItem = None
    TextItem = None

from shared.models import DocumentBlock

logger = logging.getLogger("DocProcessorAPI.OCR")


def get_doc_converter(mode: str | None = None) -> DocumentConverter:
    """
    Constructs and returns a Docling DocumentConverter configured for high-fidelity
    financial document understanding with dynamic hardware profiling and configurable
    TableFormer modes (ACCURATE vs FAST).
    """
    if DocumentConverter is None or PdfPipelineOptions is None:
        raise RuntimeError("IBM Docling is not installed in the environment.")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    use_cuda = bool(torch is not None and torch.cuda.is_available())
    device = AcceleratorDevice.CUDA if use_cuda else AcceleratorDevice.CPU

    if use_cuda:
        # Prevent VRAM fragmentation on 6GB-8GB consumer GPUs
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(
                "CUDA detected: %s (%.1f GB VRAM). Allocator configured with expandable segments.",
                device_name,
                vram_gb,
            )
        except (RuntimeError, ValueError, AttributeError, OSError) as e:
            logger.debug("Could not inspect GPU device properties: %s", e)
    else:
        logger.info("Initializing DocumentConverter on CPU accelerator device.")

    target_mode = (
        mode.lower() if mode else os.getenv("DOCLING_TABLE_MODE", "accurate").lower()
    )
    selected_table_mode = (
        TableFormerMode.FAST if target_mode == "fast" else TableFormerMode.ACCURATE
    )
    logger.info("Configuring TableFormer engine mode: %s", selected_table_mode.value)

    images_scale = float(os.getenv("DOCLING_IMAGES_SCALE", "1.5"))

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(device=device)
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = images_scale
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = selected_table_mode

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def process_pdfs_to_custom_schema(
    pdf_path: str | Path,
    doc_id: str,
    doc_converter: DocumentConverter,
    images_dir: str | Path | None = None,
) -> list[DocumentBlock]:
    """
    Converts a single PDF into a list of schema-compliant DocumentBlock objects,
    extracting headers, prose text, 2D table grids, and image bounding boxes.
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path_obj}")

    target_images_dir = (
        Path(images_dir)
        if images_dir is not None
        else Path("./TAT-DQA/processed_json/extracted_images")
    )
    target_images_dir.mkdir(parents=True, exist_ok=True)

    result = doc_converter.convert(str(pdf_path_obj))
    doc = result.document

    raw_blocks: list[dict[str, Any]] = []
    block_counter = 1
    processed_picture_ids: set[int] = set()

    for item, _level in doc.iterate_items():
        page_no = item.prov[0].page_no if (hasattr(item, "prov") and item.prov) else 1
        bbox = None
        if hasattr(item, "prov") and item.prov and hasattr(item.prov[0], "bbox"):
            bbox = [round(float(c), 2) for c in item.prov[0].bbox.as_tuple()]

        if isinstance(item, TableItem):
            table_md = ""
            table_rows_grid = None

            try:
                if hasattr(item, "export_to_markdown"):
                    table_md = item.export_to_markdown(doc)
                else:
                    df = item.export_to_dataframe(doc)
                    table_md = df.to_markdown(index=False)
            except (ValueError, RuntimeError, AttributeError, KeyError) as e:
                logger.warning("Failed to export table markdown: %s", e)

            try:
                df = item.export_to_dataframe(doc)
                df = df.fillna("")
                headers = [str(c) for c in df.columns]
                data = df.astype(str).values.tolist()
                table_rows_grid = [headers] + data
            except (ValueError, RuntimeError, AttributeError, KeyError) as e:
                logger.warning("Failed to extract 2D grid for table: %s", e)

            raw_blocks.append(
                {
                    "page": page_no,
                    "content_type": "table",
                    "markdown_content": table_md,
                    "table_rows": table_rows_grid,
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
            image_path = target_images_dir / image_filename
            markdown_image_path = str(image_path).replace("\\", "/")

            if not image_path.exists():
                try:
                    image = item.get_image(doc)
                    if image:
                        image.save(image_path, format="PNG")
                    else:
                        markdown_image_path = ""
                except (OSError, RuntimeError, AttributeError, ValueError) as e:
                    logger.warning("Failed to save image for %s: %s", block_id, e)
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

        elif isinstance(item, SectionHeaderItem):
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

        elif isinstance(item, TextItem) and hasattr(item, "text") and item.text.strip():
            raw_blocks.append(
                {
                    "page": page_no,
                    "content_type": "text",
                    "markdown_content": item.text.strip(),
                    "table_rows": None,
                    "bbox": bbox,
                }
            )

    # Paragraph merging (preserves headers and tables, merges continuous narrative lines)
    merged_blocks: list[dict[str, Any]] = []
    for b in raw_blocks:
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

    final_document_blocks: list[DocumentBlock] = []
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


def process_single_pdf_with_metrics(
    pdf_path: str | Path,
    doc_id: str,
    doc_converter: DocumentConverter | None = None,
    images_dir: str | Path | None = None,
) -> tuple[list[DocumentBlock], float]:
    """
    Convenience wrapper that runs conversion, benchmarks wall-clock execution time,
    and purges GPU memory allocations to prevent cache fragmentation.
    """
    converter = doc_converter if doc_converter is not None else get_doc_converter()
    start_time = time.perf_counter()

    blocks = process_pdfs_to_custom_schema(
        pdf_path=pdf_path,
        doc_id=doc_id,
        doc_converter=converter,
        images_dir=images_dir,
    )

    elapsed_seconds = round(time.perf_counter() - start_time, 3)

    # Free transient GPU allocations
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return blocks, elapsed_seconds
