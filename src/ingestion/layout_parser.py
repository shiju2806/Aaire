"""
Layout-aware document parser using Docling.

Parses PDF, DOCX, PPTX, and HTML documents into typed DocumentElement
objects, preserving tables, formulas, callouts, images, and section
structure. Falls back to PyMuPDF for basic extraction if Docling is
not installed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ElementType(enum.Enum):
    """Classification of document regions."""

    TEXT = "text"
    TABLE = "table"
    FORMULA = "formula"
    CALLOUT = "callout"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    CODE = "code"
    PAGE_BREAK = "page_break"


@dataclass
class BBox:
    """Bounding box in page coordinates (points from top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class DocumentElement:
    """A single typed region extracted from a document.

    Attributes:
        element_type: Classification of this region.
        content: Raw text content (or caption for images).
        structured_content: For tables: markdown/JSON representation.
                           For formulas: LaTeX string.
                           For images: base64 or file path.
        page_number: 1-indexed page number.
        bounding_box: Location on page (if available).
        parent_section: Section heading this element belongs to.
        metadata: Arbitrary metadata (confidence, language, etc.).
    """

    element_type: ElementType
    content: str
    structured_content: Optional[Dict[str, Any]] = None
    page_number: int = 1
    bounding_box: Optional[BBox] = None
    parent_section: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Docling integration
# ---------------------------------------------------------------------------

_DOCLING_AVAILABLE = False

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption

    _DOCLING_AVAILABLE = True
except ImportError:
    logger.info("Docling not installed — layout parser will use PyMuPDF fallback")

# Map Docling label strings to our ElementType enum.
_DOCLING_LABEL_MAP: Dict[str, ElementType] = {
    "text": ElementType.TEXT,
    "paragraph": ElementType.TEXT,
    "title": ElementType.HEADER,
    "section_header": ElementType.HEADER,
    "section-header": ElementType.HEADER,
    "page_header": ElementType.HEADER,
    "page-header": ElementType.HEADER,
    "page_footer": ElementType.FOOTER,
    "page-footer": ElementType.FOOTER,
    "table": ElementType.TABLE,
    "figure": ElementType.IMAGE,
    "picture": ElementType.IMAGE,
    "formula": ElementType.FORMULA,
    "equation": ElementType.FORMULA,
    "caption": ElementType.TEXT,
    "footnote": ElementType.TEXT,
    "list_item": ElementType.LIST,
    "list-item": ElementType.LIST,
    "code": ElementType.CODE,
    "page_break": ElementType.PAGE_BREAK,
}

SUPPORTED_FORMATS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md"}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class LayoutParser:
    """Parse documents into structured DocumentElement lists.

    Uses Docling when available, falls back to PyMuPDF for PDFs and
    python-docx/python-pptx for Office formats.
    """

    def __init__(self) -> None:
        self._converter: Optional[Any] = None

    # -- public API ---------------------------------------------------------

    def parse(self, file_path: str | Path) -> List[DocumentElement]:
        """Parse a document into typed elements.

        Args:
            file_path: Path to PDF, DOCX, PPTX, or HTML file.

        Returns:
            Ordered list of DocumentElement objects.

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If format is not supported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        if _DOCLING_AVAILABLE:
            return self._parse_with_docling(path)

        # Fallback paths
        if suffix == ".pdf":
            return self._parse_pdf_fallback(path)
        if suffix == ".docx":
            return self._parse_docx_fallback(path)
        if suffix == ".pptx":
            return self._parse_pptx_fallback(path)

        # HTML / Markdown — minimal text extraction
        return self._parse_text_fallback(path)

    @staticmethod
    def supported_formats() -> List[str]:
        """Return list of supported file extensions."""
        return sorted(SUPPORTED_FORMATS)

    # -- Docling backend ----------------------------------------------------

    def _get_converter(self) -> "DocumentConverter":
        """Lazy-init the Docling converter."""
        if self._converter is None:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),
                }
            )
        return self._converter

    def _parse_with_docling(self, path: Path) -> List[DocumentElement]:
        """Parse document using Docling layout analysis."""
        logger.info("Parsing with Docling", path=str(path))
        converter = self._get_converter()
        result = converter.convert(str(path))
        doc = result.document

        elements: List[DocumentElement] = []
        current_section = ""

        for item in doc.iterate_items():
            # Docling yields (item, level) tuples or just items depending on version.
            if isinstance(item, tuple):
                item = item[0]

            label = getattr(item, "label", "text")
            if isinstance(label, enum.Enum):
                label = label.value
            label = str(label).lower().replace(" ", "_")

            element_type = _DOCLING_LABEL_MAP.get(label, ElementType.TEXT)

            # Track section headings.
            if element_type == ElementType.HEADER:
                text = self._extract_text(item)
                if text:
                    current_section = text

            # Build element.
            content = self._extract_text(item)
            if not content and element_type not in (ElementType.IMAGE, ElementType.PAGE_BREAK):
                continue

            page_num = self._extract_page_number(item)
            bbox = self._extract_bbox(item, page_num)
            structured = self._extract_structured_content(item, element_type)

            elements.append(
                DocumentElement(
                    element_type=element_type,
                    content=content,
                    structured_content=structured,
                    page_number=page_num,
                    bounding_box=bbox,
                    parent_section=current_section,
                    metadata={"source_label": label, "parser": "docling"},
                )
            )

        logger.info(
            "Docling parsing complete",
            path=str(path),
            element_count=len(elements),
            types={t.value: sum(1 for e in elements if e.element_type == t) for t in ElementType if any(e.element_type == t for e in elements)},
        )
        return elements

    # -- Docling helpers ----------------------------------------------------

    @staticmethod
    def _extract_text(item: Any) -> str:
        """Extract text content from a Docling item."""
        # Docling items expose .text or .export_to_markdown()
        if hasattr(item, "text") and item.text:
            return str(item.text).strip()
        if hasattr(item, "export_to_markdown"):
            try:
                return str(item.export_to_markdown()).strip()
            except Exception:
                pass
        return ""

    @staticmethod
    def _extract_page_number(item: Any) -> int:
        """Extract 1-indexed page number from a Docling item."""
        # Docling items have .prov (provenance) with page info.
        prov = getattr(item, "prov", None)
        if prov and isinstance(prov, list) and len(prov) > 0:
            page = getattr(prov[0], "page_no", None) or getattr(prov[0], "page", None)
            if page is not None:
                return int(page)
        return 1

    @staticmethod
    def _extract_bbox(item: Any, page_num: int) -> Optional[BBox]:
        """Extract bounding box from a Docling item."""
        prov = getattr(item, "prov", None)
        if prov and isinstance(prov, list) and len(prov) > 0:
            bbox_data = getattr(prov[0], "bbox", None)
            if bbox_data is not None:
                try:
                    # Docling BBox has l, t, r, b or x0, y0, x1, y1
                    if hasattr(bbox_data, "l"):
                        return BBox(
                            x0=float(bbox_data.l),
                            y0=float(bbox_data.t),
                            x1=float(bbox_data.r),
                            y1=float(bbox_data.b),
                            page=page_num,
                        )
                    if hasattr(bbox_data, "x0"):
                        return BBox(
                            x0=float(bbox_data.x0),
                            y0=float(bbox_data.y0),
                            x1=float(bbox_data.x1),
                            y1=float(bbox_data.y1),
                            page=page_num,
                        )
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _extract_structured_content(
        item: Any, element_type: ElementType
    ) -> Optional[Dict[str, Any]]:
        """Extract structured content for tables and formulas."""
        if element_type == ElementType.TABLE:
            structured: Dict[str, Any] = {}
            # Docling tables expose .export_to_markdown() and .export_to_dataframe()
            if hasattr(item, "export_to_markdown"):
                try:
                    structured["markdown"] = item.export_to_markdown()
                except Exception:
                    pass
            if hasattr(item, "export_to_dataframe"):
                try:
                    df = item.export_to_dataframe()
                    structured["columns"] = list(df.columns)
                    structured["row_count"] = len(df)
                    # Store as list of dicts for JSON serialization
                    structured["data"] = df.to_dict(orient="records")
                except Exception:
                    pass
            return structured if structured else None

        if element_type == ElementType.FORMULA:
            # Docling may expose LaTeX via text or specific attributes.
            latex = None
            if hasattr(item, "text"):
                latex = item.text
            return {"latex": latex} if latex else None

        return None

    # -- Fallback: PyMuPDF for PDFs -----------------------------------------

    @staticmethod
    def _parse_pdf_fallback(path: Path) -> List[DocumentElement]:
        """Basic PDF parsing using PyMuPDF when Docling is unavailable."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("Neither Docling nor PyMuPDF available for PDF parsing")
            return []

        logger.info("Parsing PDF with PyMuPDF fallback", path=str(path))
        elements: List[DocumentElement] = []
        doc = fitz.open(str(path))

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1

            # Extract text blocks with positions.
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                if block["type"] == 0:  # text block
                    text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text += span.get("text", "")
                        text += "\n"
                    text = text.strip()
                    if not text:
                        continue

                    bbox = BBox(
                        x0=block["bbox"][0],
                        y0=block["bbox"][1],
                        x1=block["bbox"][2],
                        y1=block["bbox"][3],
                        page=page_num,
                    )

                    # Simple heuristic: large bold font → header.
                    spans = []
                    for line in block.get("lines", []):
                        spans.extend(line.get("spans", []))
                    is_header = (
                        len(spans) > 0
                        and spans[0].get("size", 12) > 14
                        and len(text) < 200
                    )

                    elements.append(
                        DocumentElement(
                            element_type=ElementType.HEADER if is_header else ElementType.TEXT,
                            content=text,
                            page_number=page_num,
                            bounding_box=bbox,
                            metadata={"parser": "pymupdf"},
                        )
                    )

                elif block["type"] == 1:  # image block
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.IMAGE,
                            content=f"[Image on page {page_num}]",
                            page_number=page_num,
                            bounding_box=BBox(
                                x0=block["bbox"][0],
                                y0=block["bbox"][1],
                                x1=block["bbox"][2],
                                y1=block["bbox"][3],
                                page=page_num,
                            ),
                            metadata={
                                "parser": "pymupdf",
                                "image_index": block.get("number", 0),
                            },
                        )
                    )

            # Extract tables using PyMuPDF's table finder.
            try:
                tables = page.find_tables()
                for table in tables:
                    df = table.to_pandas()
                    markdown = df.to_markdown(index=False)
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.TABLE,
                            content=markdown,
                            structured_content={
                                "markdown": markdown,
                                "columns": list(df.columns),
                                "row_count": len(df),
                                "data": df.to_dict(orient="records"),
                            },
                            page_number=page_num,
                            metadata={"parser": "pymupdf"},
                        )
                    )
            except Exception:
                # Table detection not available in all PyMuPDF versions.
                pass

        doc.close()

        # Assign parent sections.
        current_section = ""
        for elem in elements:
            if elem.element_type == ElementType.HEADER:
                current_section = elem.content
            elem.parent_section = current_section

        logger.info(
            "PyMuPDF parsing complete",
            path=str(path),
            element_count=len(elements),
        )
        return elements

    # -- Fallback: python-docx for DOCX -------------------------------------

    @staticmethod
    def _parse_docx_fallback(path: Path) -> List[DocumentElement]:
        """Basic DOCX parsing using python-docx."""
        try:
            from docx import Document
        except ImportError:
            logger.error("python-docx not available for DOCX parsing")
            return []

        logger.info("Parsing DOCX with python-docx fallback", path=str(path))
        doc = Document(str(path))
        elements: List[DocumentElement] = []
        current_section = ""

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            style_name = (para.style.name or "").lower()
            is_heading = "heading" in style_name

            element_type = ElementType.HEADER if is_heading else ElementType.TEXT
            if is_heading:
                current_section = text

            elements.append(
                DocumentElement(
                    element_type=element_type,
                    content=text,
                    parent_section=current_section,
                    metadata={"parser": "python-docx", "style": para.style.name},
                )
            )

        # Extract tables.
        for table in doc.tables:
            rows = []
            for row in table.rows:
                rows.append([cell.text.strip() for cell in row.cells])
            if not rows:
                continue

            # Build markdown.
            headers = rows[0]
            md_lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
            for row in rows[1:]:
                md_lines.append(" | ".join(row))
            markdown = "\n".join(md_lines)

            elements.append(
                DocumentElement(
                    element_type=ElementType.TABLE,
                    content=markdown,
                    structured_content={
                        "markdown": markdown,
                        "columns": headers,
                        "row_count": len(rows) - 1,
                    },
                    parent_section=current_section,
                    metadata={"parser": "python-docx"},
                )
            )

        logger.info(
            "DOCX parsing complete",
            path=str(path),
            element_count=len(elements),
        )
        return elements

    # -- Fallback: python-pptx for PPTX -------------------------------------

    @staticmethod
    def _parse_pptx_fallback(path: Path) -> List[DocumentElement]:
        """Basic PPTX parsing using python-pptx."""
        try:
            from pptx import Presentation
        except ImportError:
            logger.error("python-pptx not available for PPTX parsing")
            return []

        logger.info("Parsing PPTX with python-pptx fallback", path=str(path))
        prs = Presentation(str(path))
        elements: List[DocumentElement] = []

        for slide_idx, slide in enumerate(prs.slides):
            page_num = slide_idx + 1
            slide_title = ""

            # Get slide title.
            if slide.shapes.title and slide.shapes.title.text:
                slide_title = slide.shapes.title.text.strip()
                elements.append(
                    DocumentElement(
                        element_type=ElementType.HEADER,
                        content=slide_title,
                        page_number=page_num,
                        parent_section=slide_title,
                        metadata={"parser": "python-pptx", "shape_type": "title"},
                    )
                )

            for shape in slide.shapes:
                if shape.has_text_frame:
                    text = shape.text_frame.text.strip()
                    if not text or text == slide_title:
                        continue
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.TEXT,
                            content=text,
                            page_number=page_num,
                            parent_section=slide_title,
                            metadata={"parser": "python-pptx"},
                        )
                    )
                elif shape.has_table:
                    table = shape.table
                    rows = []
                    for row in table.rows:
                        rows.append([cell.text.strip() for cell in row.cells])
                    if rows:
                        headers = rows[0]
                        md_lines = [
                            " | ".join(headers),
                            " | ".join(["---"] * len(headers)),
                        ]
                        for row in rows[1:]:
                            md_lines.append(" | ".join(row))
                        markdown = "\n".join(md_lines)

                        elements.append(
                            DocumentElement(
                                element_type=ElementType.TABLE,
                                content=markdown,
                                structured_content={
                                    "markdown": markdown,
                                    "columns": headers,
                                    "row_count": len(rows) - 1,
                                },
                                page_number=page_num,
                                parent_section=slide_title,
                                metadata={"parser": "python-pptx"},
                            )
                        )
                elif shape.shape_type == 13:  # Picture
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.IMAGE,
                            content=f"[Image on slide {page_num}]",
                            page_number=page_num,
                            parent_section=slide_title,
                            metadata={"parser": "python-pptx"},
                        )
                    )

        logger.info(
            "PPTX parsing complete",
            path=str(path),
            element_count=len(elements),
        )
        return elements

    # -- Fallback: plain text/HTML/Markdown ---------------------------------

    @staticmethod
    def _parse_text_fallback(path: Path) -> List[DocumentElement]:
        """Minimal text-based parsing for HTML and Markdown."""
        logger.info("Parsing with text fallback", path=str(path))
        text = path.read_text(encoding="utf-8", errors="replace")
        return [
            DocumentElement(
                element_type=ElementType.TEXT,
                content=text,
                metadata={"parser": "text_fallback", "format": path.suffix},
            )
        ]
