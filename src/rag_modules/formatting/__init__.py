"""
Formatting Module for RAG Pipeline

This module contains all formatting-related functionality extracted from the main
RAG pipeline to provide clean, professional text formatting and presentation.

Main Components:
- FormattingManager: Core class handling all formatting operations
- create_formatting_manager: Factory function for creating manager instances
"""

from .manager import FormattingManager, create_formatting_manager

__all__ = [
    'FormattingManager',
    'create_formatting_manager'
]