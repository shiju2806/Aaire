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

from ..providers.config_loader import get_config, get_nested

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

# API VLM engine (Ollama/OpenAI) for sparse-page fallback
_API_VLM_AVAILABLE = False
try:
    from docling.datamodel.vlm_engine_options import (
        ApiVlmEngineOptions,
        VlmEngineType,
    )

    _API_VLM_AVAILABLE = True
except ImportError:
    pass

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

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._converter: Optional[Any] = None
        self._config = config or get_config("ingestion")
        self._sparse_fallback_enabled = get_nested(
            self._config, "layout_parser", "sparse_page_fallback", "enabled", default=True
        )
        self._sparse_fallback_config = get_nested(
            self._config, "layout_parser", "sparse_page_fallback", default={}
        ) or {}

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

        # Fallback paths (Docling not installed)
        if suffix == ".docx":
            return self._parse_docx_fallback(path)
        if suffix == ".pptx":
            return self._parse_pptx_fallback(path)

        # HTML / Markdown — minimal text extraction
        if suffix in (".html", ".htm", ".md"):
            return self._parse_text_fallback(path)

        raise RuntimeError(
            f"Docling is not installed and no fallback parser for '{suffix}'. "
            "Install docling: pip install docling"
        )

    @staticmethod
    def supported_formats() -> List[str]:
        """Return list of supported file extensions."""
        return sorted(SUPPORTED_FORMATS)

    # -- Docling backend ----------------------------------------------------

    def _get_converter(self) -> "DocumentConverter":
        """Lazy-init the Docling converter.

        Pre-renders page images for the sparse-page VLM fallback (Ollama).
        SmolVLM/MLX inline image description is disabled — all vision
        processing goes through the sparse-page fallback via Ollama API.
        """
        if self._converter is None:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True

            # Pre-render page images for sparse-page VLM fallback.
            # This caches page images at scale=1.0 during parsing so the
            # fallback can access them after Docling releases page backends.
            if self._sparse_fallback_enabled:
                pipeline_options.generate_page_images = True

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),
                }
            )
        return self._converter

    def _parse_with_docling(self, path: Path) -> List[DocumentElement]:
        """Parse document using Docling layout analysis.

        After Docling parsing, detects pages where Docling classified most
        content as empty images (e.g. org charts, flow diagrams rendered as
        vector graphics). For those pages, falls back to PyPDF2 text
        extraction which reads the embedded text layer that Docling misses.
        """
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

        # Recover text from image-heavy pages (org charts, flow diagrams).
        elements = self._recover_figure_text(elements, path)

        # VLM fallback for sparse pages (visual content classified as text).
        if self._sparse_fallback_enabled:
            elements = self._vlm_page_fallback(elements, result)

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

    # -- Figure text recovery -----------------------------------------------

    @staticmethod
    def _recover_figure_text(
        elements: List[DocumentElement], path: Path
    ) -> List[DocumentElement]:
        """Recover text from pages where Docling found mostly empty images.

        Org charts, flow diagrams, and similar visual content are often
        rendered as vector graphics in PDFs. Docling classifies them as
        "picture" regions and cannot extract the embedded text. PyPDF2 reads
        the PDF text layer directly and captures all text regardless of layout.

        For each page that is image-heavy (>50% image elements by count,
        with <100 chars of text extracted), replace the empty image elements
        with a single TEXT element containing the full page text from PyPDF2.
        """
        if not path.suffix.lower() == ".pdf":
            return elements

        # Group elements by page.
        pages: Dict[int, List[DocumentElement]] = {}
        for elem in elements:
            pages.setdefault(elem.page_number, []).append(elem)

        # Identify image-heavy pages.
        # A page qualifies if it has empty image elements AND Docling extracted
        # very little text (fragmented box labels from org charts/diagrams).
        image_heavy_pages: Dict[int, str] = {}  # page_num → section header
        for page_num, page_elements in pages.items():
            empty_images = sum(
                1 for e in page_elements
                if e.element_type == ElementType.IMAGE and not e.content.strip()
            )
            text_chars = sum(
                len(e.content) for e in page_elements
                if e.element_type in (ElementType.TEXT, ElementType.HEADER)
            )

            # Trigger recovery when: page has any empty image AND the text
            # Docling found is short (likely just box labels, not real prose).
            if empty_images >= 1 and text_chars < 200:
                # Find section context from this page or preceding pages.
                section = ""
                for e in page_elements:
                    if e.parent_section:
                        section = e.parent_section
                        break
                image_heavy_pages[page_num] = section

        if not image_heavy_pages:
            return elements

        # Extract text from image-heavy pages using PyPDF2.
        try:
            import PyPDF2
        except ImportError:
            logger.warning(
                "PyPDF2 not available for figure text recovery — "
                "image-heavy pages will have empty content"
            )
            return elements

        recovered_pages: Dict[int, str] = {}
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in image_heavy_pages:
                    if page_num <= len(reader.pages):
                        text = reader.pages[page_num - 1].extract_text() or ""
                        if text.strip():
                            recovered_pages[page_num] = text.strip()
        except Exception as e:
            logger.warning("PyPDF2 figure text recovery failed", error=str(e))
            return elements

        if not recovered_pages:
            return elements

        # Rebuild elements: replace empty images on recovered pages with text.
        new_elements: List[DocumentElement] = []
        pages_replaced: set = set()

        for elem in elements:
            if elem.page_number in recovered_pages and elem.page_number not in pages_replaced:
                # Insert recovered text element (once per page).
                pages_replaced.add(elem.page_number)
                section = image_heavy_pages.get(elem.page_number, "")

                # If no section header from Docling, try to infer from text.
                page_text = recovered_pages[elem.page_number]
                if not section:
                    # Org chart pages often end with a chart title like
                    # "Financial Planning & Analysis – Structure Chart".
                    for keyword in ("structure chart", "org chart", "organization"):
                        idx = page_text.lower().rfind(keyword)
                        if idx != -1:
                            # Walk backwards to find the start of this title.
                            start = page_text.rfind("\n", 0, idx)
                            section = page_text[start + 1:].strip() if start != -1 else page_text[idx:].strip()
                            break

                new_elements.append(
                    DocumentElement(
                        element_type=ElementType.TEXT,
                        content=page_text,
                        page_number=elem.page_number,
                        parent_section=section,
                        metadata={"parser": "docling+pypdf2_recovery", "recovery": "figure_text"},
                    )
                )
            elif elem.page_number in recovered_pages:
                # Skip remaining elements from this page (already replaced).
                continue
            else:
                # Keep non-recovered pages as-is.
                new_elements.append(elem)

        logger.info(
            "Figure text recovery complete",
            recovered_pages=list(recovered_pages.keys()),
            text_chars={p: len(t) for p, t in recovered_pages.items()},
        )
        return new_elements

    # -- Sparse-page VLM fallback -------------------------------------------

    def _create_api_vlm_engine(self, engine_type: str) -> Optional[Any]:
        """Create a standalone API VLM engine for sparse page fallback.

        Supports OpenAI (gpt-4o-mini) and Ollama (llama3.2-vision) backends
        via Docling's OpenAI-compatible ApiVlmEngine.
        """
        try:
            import os
            from docling.models.inference_engines.vlm.api_openai_compatible_engine import (
                ApiVlmEngine,
            )

            model_name = self._sparse_fallback_config.get("model", "gpt-4o-mini")

            if engine_type == "api_openai":
                api_key = os.environ.get("OPENAI_API_KEY", "")
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}"}
                timeout = 60.0
                vlm_engine_type = VlmEngineType.API
            else:  # api_ollama
                ollama_cfg = get_nested(
                    self._config, "layout_parser", "api_ollama", default={}
                ) or {}
                base_url = ollama_cfg.get("url", "http://localhost:11434")
                url = f"{base_url}/v1/chat/completions"
                headers = {}
                timeout = float(ollama_cfg.get("timeout", 120))
                vlm_engine_type = VlmEngineType.API_OLLAMA

            options = ApiVlmEngineOptions(
                engine_type=vlm_engine_type,
                url=url,
                timeout=timeout,
                headers=headers,
                params={"model": model_name},
            )
            engine = ApiVlmEngine(
                enable_remote_services=True,
                options=options,
            )
            engine.initialize()
            logger.info(
                "API VLM engine created for sparse page fallback",
                engine=engine_type,
                model=model_name,
            )
            return engine
        except Exception as e:
            logger.warning(
                "Could not create API VLM engine",
                engine=engine_type,
                error=str(e),
            )
            return None

    def _vlm_page_fallback(
        self,
        elements: List[DocumentElement],
        conv_result: Any,
    ) -> List[DocumentElement]:
        """Render sparse pages as images and describe them with the VLM.

        When Docling produces very few text elements for a page (e.g., org
        charts or flow diagrams classified as text rather than images), this
        method renders the page as an image and sends it through the
        already-loaded VLM for a proper visual description.

        Complements ``_recover_figure_text()`` which handles pages with empty
        IMAGE elements.  This method handles pages with only sparse TEXT
        elements that are actually visual content.
        """
        # Config thresholds.
        max_elements = self._sparse_fallback_config.get("max_elements_per_page", 3)
        min_avg = self._sparse_fallback_config.get("min_avg_elements_per_page", 5)
        image_scale = self._sparse_fallback_config.get("vlm_image_scale", 2.0)
        max_tokens = self._sparse_fallback_config.get("vlm_max_tokens", 500)
        prompt = self._sparse_fallback_config.get(
            "vlm_prompt",
            "Describe this document page in detail. Include all text, names, "
            "titles, labels, data values, visual structure, layout, and "
            "hierarchical relationships. Use indentation or bullet points to "
            "show hierarchy.",
        )

        # Need access to Docling page images.
        pages = getattr(conv_result, "pages", None)
        if not pages:
            return elements

        # Group elements by page.
        page_elements: Dict[int, List[DocumentElement]] = {}
        for elem in elements:
            page_elements.setdefault(elem.page_number, []).append(elem)

        if not page_elements:
            return elements

        # Document-level guard: skip dense documents.
        # If any page was already recovered by _recover_figure_text, the
        # document is known to be visual — skip the density check.
        has_recovered = any(
            e.metadata.get("recovery") == "figure_text"
            for e in elements
        )
        if not has_recovered:
            total_pages = len(page_elements)
            avg_elements = len(elements) / max(total_pages, 1)
            if avg_elements >= min_avg:
                return elements

        # Identify sparse pages.
        sparse_pages: Dict[int, str] = {}  # page_num → section name
        for page_num, page_elems in page_elements.items():
            non_break = [
                e for e in page_elems if e.element_type != ElementType.PAGE_BREAK
            ]

            # Pages recovered by _recover_figure_text still have garbled
            # concatenated text — VLM can do better, so treat them as
            # candidates too.  Only skip if the page has unrecovered IMAGE
            # elements (those need different handling).
            was_recovered = any(
                e.metadata.get("recovery") == "figure_text" for e in page_elems
            )
            if not was_recovered:
                # For non-recovered pages, apply the normal sparse check.
                if any(e.element_type == ElementType.IMAGE for e in page_elems):
                    continue
                if len(non_break) > max_elements:
                    continue

            total_text = sum(len(e.content) for e in non_break)
            if total_text == 0:
                continue

            # Skip title-only pages (single header is expected).
            if len(non_break) == 1 and non_break[0].element_type == ElementType.HEADER:
                continue

            section = next(
                (e.parent_section for e in page_elems if e.parent_section), ""
            )
            sparse_pages[page_num] = section

        if not sparse_pages:
            return elements

        # Choose VLM engine based on config: Ollama API (high quality) or
        # Docling's built-in engine (fast, lower quality).
        fallback_engine = self._sparse_fallback_config.get("engine", "api_ollama")
        vlm_engine = self._create_api_vlm_engine(fallback_engine)
        if vlm_engine is None:
            logger.warning(
                "VLM engine not available for sparse page fallback — "
                "sparse pages will retain original Docling text",
                engine=fallback_engine,
            )
            return elements

        from docling.models.inference_engines.vlm.base import VlmEngineInput

        # Build a quick lookup: page_no → Page object.
        page_lookup: Dict[int, Any] = {}
        for p in pages:
            page_lookup[p.page_no] = p

        # Process each sparse page through VLM.
        vlm_descriptions: Dict[int, str] = {}
        for page_num in sorted(sparse_pages):
            docling_page = page_lookup.get(page_num)
            if docling_page is None:
                logger.warning("No Docling page object", page_num=page_num)
                continue
            try:
                # Backends are released after Docling assembly, so prefer
                # the pre-rendered cache.  Fall back to get_image() only if
                # the cache is empty (backend still alive).
                page_image = None
                cache = getattr(docling_page, "_image_cache", {})
                if cache:
                    # Prefer the requested scale; else take whatever is cached.
                    page_image = cache.get(image_scale) or next(
                        iter(cache.values())
                    )
                if page_image is None:
                    page_image = docling_page.get_image(scale=image_scale)
                if page_image is None:
                    logger.warning("Could not render page image", page_num=page_num)
                    continue

                engine_input = VlmEngineInput(
                    image=page_image,
                    prompt=prompt,
                    temperature=0.0,
                    max_new_tokens=max_tokens,
                )
                output = vlm_engine.predict(engine_input)
                description = output.text.strip()

                logger.info(
                    "VLM page fallback result",
                    page_num=page_num,
                    output_len=len(description),
                    preview=description[:200] if description else "(empty)",
                )

                # Only use VLM output if it's substantive (short summaries
                # like "This is an org chart" are less useful than the raw
                # recovered text which at least contains the names).
                min_chars = 150
                if description and len(description) >= min_chars:
                    vlm_descriptions[page_num] = description
                else:
                    logger.info(
                        "VLM output too short, keeping original text",
                        page_num=page_num,
                        output_len=len(description),
                    )
            except Exception as e:
                logger.warning(
                    "VLM page fallback failed",
                    page_num=page_num,
                    error=str(e),
                )

        if not vlm_descriptions:
            return elements

        # Rebuild element list: replace elements on VLM-described pages.
        new_elements: List[DocumentElement] = []
        pages_replaced: set = set()

        for elem in elements:
            if elem.page_number in vlm_descriptions:
                if elem.page_number not in pages_replaced:
                    pages_replaced.add(elem.page_number)
                    section = sparse_pages.get(elem.page_number, "")
                    original_count = len(page_elements.get(elem.page_number, []))
                    new_elements.append(
                        DocumentElement(
                            element_type=ElementType.TEXT,
                            content=vlm_descriptions[elem.page_number],
                            page_number=elem.page_number,
                            parent_section=section,
                            metadata={
                                "parser": "docling+vlm_page_fallback",
                                "recovery": "sparse_page_vlm",
                                "original_element_count": original_count,
                            },
                        )
                    )
                # Skip remaining elements from this page (already replaced).
            else:
                new_elements.append(elem)

        logger.info(
            "VLM sparse page fallback complete",
            sparse_pages=list(vlm_descriptions.keys()),
            description_lengths={p: len(d) for p, d in vlm_descriptions.items()},
        )
        return new_elements

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
