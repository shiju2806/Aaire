"""
Type-specific element processors.

Each processor takes DocumentElement objects of a specific type and
produces ProcessedChunk objects ready for embedding and storage.
"""

from .text_processor import TextProcessor
from .table_processor import TableProcessor
from .formula_processor import FormulaProcessor
from .callout_processor import CalloutProcessor
from .image_processor import ImageProcessor

__all__ = [
    "TextProcessor",
    "TableProcessor",
    "FormulaProcessor",
    "CalloutProcessor",
    "ImageProcessor",
]
