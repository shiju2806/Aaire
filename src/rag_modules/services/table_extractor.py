"""
Table Extraction Service for AAIRE
Uses pdfplumber + GPT-4o-mini for intelligent table extraction and structuring
"""

import json
import structlog
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = structlog.get_logger()

# Try importing pdfplumber - graceful degradation if not available
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available - table extraction disabled. Install with: pip install pdfplumber")


class TableExtractor:
    """
    Extract and structure tables using PDF parsers + GPT-4o-mini

    Design principles:
    - No hard-coded table patterns
    - LLM interprets everything
    - Works for any regulatory framework
    """

    def __init__(self, llm_client=None, async_client=None, config=None):
        """
        Initialize table extractor

        Args:
            llm_client: Synchronous LLM client (llama-index)
            async_client: Async LLM client (OpenAI AsyncOpenAI)
            config: Configuration dictionary (from mvp_config.yaml)
        """
        self.llm = llm_client
        self.async_client = async_client
        self.config = config or {}

        # Extract table extraction config
        table_config = self.config.get('table_extraction', {})

        # LLM settings resolved via provider; config overrides still supported
        from ...providers import get_llm_provider
        self._llm = get_llm_provider()
        self.model = self._llm.get_model_name("extraction")
        self.temperature = table_config.get('temperature', 0)
        self.max_tokens = table_config.get('max_tokens', 2000)

        # Table types from config
        self.table_types = table_config.get('table_types', [
            'requirement', 'example', 'reference', 'comparison', 'calculation'
        ])

        # Min table size
        self.min_table_rows = table_config.get('min_table_rows', 2)

        if not PDFPLUMBER_AVAILABLE:
            logger.warning("TableExtractor initialized without pdfplumber - table extraction will be disabled")
        else:
            logger.info(f"TableExtractor initialized: model={self.model}, types={self.table_types}")

    async def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract all tables from a PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of structured table dictionaries with metadata
        """
        if not PDFPLUMBER_AVAILABLE:
            logger.info("Skipping table extraction - pdfplumber not available")
            return []

        tables_data = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"📊 Extracting tables from {Path(pdf_path).name} ({len(pdf.pages)} pages)")

                for page_num, page in enumerate(pdf.pages):
                    # Extract all tables from this page
                    tables = page.extract_tables()

                    if tables:
                        logger.info(f"Found {len(tables)} table(s) on page {page_num + 1}")

                    for table_idx, raw_table in enumerate(tables):
                        # Skip empty or invalid tables (use config min_table_rows)
                        if not raw_table or len(raw_table) < self.min_table_rows:
                            continue

                        # Use GPT-4o-mini to interpret and structure the table
                        structured_table = await self._structure_table_with_llm(
                            raw_table,
                            page_num + 1,
                            table_idx + 1,
                            Path(pdf_path).name
                        )

                        if structured_table:
                            tables_data.append(structured_table)

            logger.info(f"✅ Extracted {len(tables_data)} tables from {Path(pdf_path).name}")
            return tables_data

        except Exception as e:
            logger.error(f"Failed to extract tables from {pdf_path}: {str(e)}")
            return []

    async def _structure_table_with_llm(
        self,
        raw_table: List[List[str]],
        page_num: int,
        table_idx: int,
        filename: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use GPT-4o-mini to understand and structure a table

        This is pure prompt engineering - no hard-coded rules!
        """
        if not self.async_client:
            logger.warning("No async LLM client available - skipping table structuring")
            return None

        try:
            # Convert raw table to text representation
            table_text = self._format_table_as_text(raw_table)

            # Generate table type options from config (dynamic!)
            table_type_options = "|".join(self.table_types)

            # Prompt GPT-4o-mini to interpret the table
            prompt = f"""Analyze this table extracted from a regulatory/insurance document.

TABLE DATA:
{table_text}

SOURCE: {filename}, Page {page_num}, Table {table_idx}

Your task:
1. Determine the table's purpose (what does it represent?)
2. Identify column headers and their meanings
3. Extract key numeric values, thresholds, or requirements
4. Identify any formulas or calculations shown
5. Classify the table type based on available categories

Return a JSON object with this structure:
{{
    "title": "Inferred table title or purpose",
    "type": "{table_type_options}",
    "headers": ["Column 1 name", "Column 2 name", ...],
    "key_values": {{
        "threshold_1": "value and unit",
        "threshold_2": "value and unit"
    }},
    "formulas": ["Any formulas found in the table"],
    "summary": "One sentence describing what this table shows",
    "is_example": true/false  // true if this is an illustrative example, false if actual requirements
}}

IMPORTANT:
- Infer the table's purpose from context
- Extract ALL numeric values with their context
- Identify if values are examples/illustrations vs. actual regulatory thresholds
- Be specific about what each column represents
- Choose the most appropriate type from: {table_type_options}
"""

            # Call LLM with config settings (fully configurable!)
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Parse the response
            llm_analysis = response.choices[0].message.content.strip()

            # Try to extract JSON from the response
            structured_data = self._extract_json_from_response(llm_analysis)

            if structured_data:
                # Add metadata
                structured_data["page"] = page_num
                structured_data["table_index"] = table_idx
                structured_data["filename"] = filename
                structured_data["raw_data"] = raw_table  # Keep original for reference

                logger.info(f"✅ Structured table: {structured_data.get('title', 'Unknown')} (type: {structured_data.get('type', 'unknown')})")
                return structured_data
            else:
                logger.warning(f"Failed to parse LLM response for table on page {page_num}")
                return None

        except Exception as e:
            logger.error(f"Error structuring table: {str(e)}")
            return None

    def _format_table_as_text(self, raw_table: List[List[str]]) -> str:
        """Convert raw table data to formatted text"""
        lines = []
        for row in raw_table:
            # Filter out None values and convert to strings
            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
            lines.append(" | ".join(cleaned_row))
        return "\n".join(lines)

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response (handles markdown code blocks)"""
        try:
            # Try direct JSON parse first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract from markdown code block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            # Try to find any JSON object in the response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            logger.warning("Could not extract JSON from LLM response")
            return None

    def create_table_index_entry(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a searchable index entry for a table

        This allows tables to be retrieved independently from their parent document
        """
        return {
            "type": "table",
            "title": table.get("title", "Unknown Table"),
            "table_type": table.get("type", "unknown"),
            "summary": table.get("summary", ""),
            "source": f"{table.get('filename')} - Page {table.get('page')}",
            "page": table.get("page"),
            "filename": table.get("filename"),
            "headers": table.get("headers", []),
            "key_values": table.get("key_values", {}),
            "formulas": table.get("formulas", []),
            "is_example": table.get("is_example", False),
            # Create searchable text content
            "content": self._create_searchable_content(table),
            "metadata": {
                "content_type": "table",
                "table_type": table.get("type"),
                "is_example": table.get("is_example", False)
            }
        }

    def _create_searchable_content(self, table: Dict[str, Any]) -> str:
        """Create searchable text representation of table"""
        parts = [
            f"Table: {table.get('title', 'Unknown')}",
            f"Summary: {table.get('summary', '')}",
            f"Columns: {', '.join(table.get('headers', []))}",
        ]

        # Add key values
        if table.get('key_values'):
            parts.append("Key Values:")
            for key, value in table.get('key_values', {}).items():
                parts.append(f"  - {key}: {value}")

        # Add formulas
        if table.get('formulas'):
            parts.append("Formulas:")
            for formula in table.get('formulas', []):
                parts.append(f"  - {formula}")

        # Add raw table data as text
        if table.get('raw_data'):
            parts.append("\nTable Data:")
            for row in table.get('raw_data', []):
                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                parts.append(row_text)

        return "\n".join(parts)


def create_table_extractor(llm_client=None, async_client=None, config=None):
    """Factory function to create a TableExtractor instance"""
    return TableExtractor(llm_client=llm_client, async_client=async_client, config=config)
