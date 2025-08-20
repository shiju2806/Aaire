"""
PDF Spatial Extractor - Shape-aware document processing for organizational charts
Extracts text while preserving spatial relationships and box structures
"""

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    # Create mock fitz for testing/development
    class MockFitz:
        def open(self, *args, **kwargs):
            raise ImportError("PyMuPDF not available - install with: pip install PyMuPDF==1.24.7")
    fitz = MockFitz()
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import structlog
import re
from pathlib import Path
from advanced_org_parser import AdvancedOrgParser

logger = structlog.get_logger()

@dataclass
class TextElement:
    """Individual text element with spatial information"""
    text: str
    x0: float  # Left coordinate
    y0: float  # Bottom coordinate  
    x1: float  # Right coordinate
    y1: float  # Top coordinate
    page: int
    font_size: float
    font_name: str
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property 
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0

@dataclass
class TextCluster:
    """Group of related text elements forming a logical unit (e.g., org chart box)"""
    elements: List[TextElement]
    cluster_id: str
    confidence: float = 0.0
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box of entire cluster"""
        if not self.elements:
            return (0, 0, 0, 0)
        
        x0 = min(elem.x0 for elem in self.elements)
        y0 = min(elem.y0 for elem in self.elements)
        x1 = max(elem.x1 for elem in self.elements)
        y1 = max(elem.y1 for elem in self.elements)
        return (x0, y0, x1, y1)
    
    @property
    def text_lines(self) -> List[str]:
        """Text lines sorted by vertical position (top to bottom)"""
        # Sort by Y coordinate (descending - top to bottom in PDF coordinates)
        sorted_elements = sorted(self.elements, key=lambda e: -e.center_y)
        return [elem.text.strip() for elem in sorted_elements if elem.text.strip()]

@dataclass
class OrganizationalUnit:
    """Structured organizational information extracted from a cluster"""
    name: str
    title: str
    department: str
    cluster_id: str
    confidence: float
    source_box: Tuple[float, float, float, float]  # Bounding box
    warnings: List[str]

class PDFSpatialExtractor:
    """
    Advanced PDF extractor that preserves spatial relationships
    Specifically designed for organizational charts and structured documents
    """
    
    def __init__(self):
        self.proximity_threshold = 25  # Reduced for tighter clustering
        self.min_cluster_size = 2  # Minimum elements per cluster
        self.max_cluster_size = 6   # Reduced max - org boxes typically have 2-4 lines
        self.advanced_parser = AdvancedOrgParser()  # Use advanced parsing logic
        
    def extract_with_coordinates(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF while preserving spatial relationships
        Returns structured data with organizational information
        """
        try:
            logger.info(f"Starting spatial extraction for: {pdf_path}")
            
            # Step 1: Extract all text elements with coordinates
            text_elements = self._extract_text_elements(pdf_path)
            
            if not text_elements:
                return self._create_empty_result("No text elements found")
            
            logger.info(f"Extracted {len(text_elements)} text elements")
            
            # Step 2: Group elements into spatial clusters (potential boxes)
            clusters = self._create_spatial_clusters(text_elements)
            
            logger.info(f"Created {len(clusters)} spatial clusters")
            
            # Step 3: Parse each cluster for organizational information
            org_units = self._parse_organizational_clusters(clusters)
            
            logger.info(f"Identified {len(org_units)} organizational units")
            
            # Step 4: Create structured output
            result = self._create_structured_result(org_units, clusters, text_elements)
            
            logger.info("Spatial extraction completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Spatial extraction failed: {e}")
            return self._create_error_result(str(e))
    
    def _extract_text_elements(self, pdf_path: str) -> List[TextElement]:
        """Extract individual text elements with spatial coordinates"""
        elements = []
        
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available - spatial extraction disabled")
            return elements
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text with detailed information
                text_dict = page.get_text("dict")
                
                for block in text_dict["blocks"]:
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text and len(text) > 1:  # Skip single characters and empty
                                    elements.append(TextElement(
                                        text=text,
                                        x0=span["bbox"][0],
                                        y0=span["bbox"][1], 
                                        x1=span["bbox"][2],
                                        y1=span["bbox"][3],
                                        page=page_num,
                                        font_size=span["size"],
                                        font_name=span["font"]
                                    ))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Failed to extract text elements: {e}")
            
        return elements
    
    def _create_spatial_clusters(self, elements: List[TextElement]) -> List[TextCluster]:
        """Group text elements into spatial clusters based on proximity"""
        if not elements:
            return []
        
        clusters = []
        used_elements = set()
        
        for i, element in enumerate(elements):
            if i in used_elements:
                continue
                
            # Start new cluster
            cluster_elements = [element]
            used_elements.add(i)
            
            # Find nearby elements
            for j, other_element in enumerate(elements):
                if j in used_elements or i == j:
                    continue
                
                # Check if elements are on same page
                if element.page != other_element.page:
                    continue
                
                # Calculate distance between elements
                distance = self._calculate_distance(element, other_element)
                
                if distance <= self.proximity_threshold:
                    cluster_elements.append(other_element)
                    used_elements.add(j)
            
            # Only create cluster if it has enough elements and forms a coherent box
            if (len(cluster_elements) >= self.min_cluster_size and 
                len(cluster_elements) <= self.max_cluster_size and
                self._is_valid_org_cluster(cluster_elements)):
                
                cluster_id = f"cluster_{len(clusters)}_page_{element.page}"
                confidence = self._calculate_cluster_confidence(cluster_elements)
                
                clusters.append(TextCluster(
                    elements=cluster_elements,
                    cluster_id=cluster_id,
                    confidence=confidence
                ))
        
        return clusters
    
    def _calculate_distance(self, elem1: TextElement, elem2: TextElement) -> float:
        """Calculate spatial distance between two text elements with org chart awareness"""
        
        # For organizational charts, prioritize vertical alignment over pure distance
        x_distance = abs(elem1.center_x - elem2.center_x)
        y_distance = abs(elem1.center_y - elem2.center_y)
        
        # If elements are vertically aligned (same column), weight y-distance more
        if x_distance < 30:  # Elements are roughly in same column
            return y_distance * 0.7  # Prefer vertical grouping
        
        # If elements are horizontally aligned (same row), they probably don't belong together
        if y_distance < 10:  # Same horizontal line
            return x_distance * 2.0  # Discourage horizontal grouping
        
        # Standard euclidean distance for other cases
        return ((x_distance ** 2) + (y_distance ** 2)) ** 0.5
    
    def _is_valid_org_cluster(self, elements: List[TextElement]) -> bool:
        """Validate that elements form a coherent organizational chart box"""
        if len(elements) < 2:
            return False
        
        # Sort elements by vertical position (top to bottom)
        sorted_elements = sorted(elements, key=lambda e: e.y0, reverse=True)
        
        # Check if elements are roughly vertically aligned (same column)
        x_positions = [e.center_x for e in elements]
        x_variance = max(x_positions) - min(x_positions)
        
        # Elements should be in roughly the same column (within 50 pixels)
        if x_variance > 50:
            return False
        
        # Check vertical spacing - elements should be close vertically
        for i in range(len(sorted_elements) - 1):
            current = sorted_elements[i]
            next_elem = sorted_elements[i + 1]
            vertical_gap = current.y0 - next_elem.y1  # Gap between elements
            
            # Gap should be reasonable (not too large)
            if vertical_gap > 30:  # More than 30 pixels gap
                return False
        
        # Check if cluster contains typical org chart content patterns
        combined_text = ' '.join([e.text.lower() for e in elements])
        
        # Should contain typical org chart indicators
        has_name_pattern = any(
            len(word) > 2 and word[0].isupper() and word[1:].islower()
            for word in ' '.join([e.text for e in elements]).split()
        )
        
        has_title_indicators = any(
            indicator in combined_text 
            for indicator in ['manager', 'director', 'vp', 'avp', 'analyst', 'accountant', 'mvp']
        )
        
        return has_name_pattern or has_title_indicators
    
    def _calculate_cluster_confidence(self, elements: List[TextElement]) -> float:
        """Calculate confidence score for a cluster based on spatial coherence"""
        if len(elements) < 2:
            return 0.0
        
        # Check alignment and spacing consistency
        confidence = 0.5  # Base confidence
        
        # Bonus for consistent font sizes
        font_sizes = [elem.font_size for elem in elements]
        if len(set(font_sizes)) <= 2:  # Max 2 different font sizes
            confidence += 0.2
        
        # Bonus for good vertical alignment
        x_positions = [elem.center_x for elem in elements]
        x_range = max(x_positions) - min(x_positions)
        if x_range < 100:  # Well aligned horizontally
            confidence += 0.2
        
        # Bonus for reasonable number of elements (org chart box typically has 2-4 lines)
        if 2 <= len(elements) <= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _parse_organizational_clusters(self, clusters: List[TextCluster]) -> List[OrganizationalUnit]:
        """Parse clusters to extract organizational information"""
        org_units = []
        
        for cluster in clusters:
            if cluster.confidence < 0.6:  # Skip low-confidence clusters
                continue
            
            try:
                org_unit = self._parse_single_cluster(cluster)
                if org_unit:
                    org_units.append(org_unit)
            except Exception as e:
                logger.warning(f"Failed to parse cluster {cluster.cluster_id}: {e}")
        
        return org_units
    
    def _parse_single_cluster(self, cluster: TextCluster) -> Optional[OrganizationalUnit]:
        """Parse a single cluster to extract name, title, department using advanced parser"""
        
        # Combine cluster text lines into single text for advanced parser
        cluster_text = '\n'.join(cluster.text_lines)
        
        # Use advanced parser
        person_record = self.advanced_parser.parse_spatial_cluster(
            cluster_text, 
            cluster.bbox
        )
        
        if not person_record:
            logger.debug(f"Advanced parser failed for cluster {cluster.cluster_id}")
            return None
        
        # Map PersonRecord to OrganizationalUnit
        warnings = []
        if person_record.source_info.get('method') == 'fallback':
            warnings.append("Used fallback parsing pattern")
        
        return OrganizationalUnit(
            name=person_record.name,
            title=person_record.title,
            department=person_record.department,
            cluster_id=cluster.cluster_id,
            confidence=person_record.confidence,
            source_box=cluster.bbox,
            warnings=warnings
        )
    
    
    def _create_structured_result(self, org_units: List[OrganizationalUnit], 
                                clusters: List[TextCluster], 
                                elements: List[TextElement]) -> Dict[str, Any]:
        """Create final structured result"""
        
        return {
            "extraction_method": "pdf_spatial",
            "success": True,
            "organizational_units": [
                {
                    "name": unit.name,
                    "title": unit.title,
                    "department": unit.department,
                    "confidence": unit.confidence,
                    "source_box": unit.source_box,
                    "cluster_id": unit.cluster_id,
                    "warnings": unit.warnings
                }
                for unit in org_units
            ],
            "metadata": {
                "total_clusters": len(clusters),
                "successful_extractions": len(org_units),
                "total_text_elements": len(elements),
                "average_confidence": sum(unit.confidence for unit in org_units) / len(org_units) if org_units else 0.0
            },
            "spatial_data": {
                "clusters": [
                    {
                        "id": cluster.cluster_id,
                        "bbox": cluster.bbox,
                        "text_lines": cluster.text_lines,
                        "confidence": cluster.confidence
                    }
                    for cluster in clusters
                ]
            }
        }
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result structure"""
        return {
            "extraction_method": "pdf_spatial",
            "success": False,
            "organizational_units": [],
            "metadata": {
                "total_clusters": 0,
                "successful_extractions": 0,
                "total_text_elements": 0,
                "average_confidence": 0.0,
                "error": reason
            },
            "spatial_data": {"clusters": []}
        }
    
    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Create error result structure"""
        return {
            "extraction_method": "pdf_spatial",
            "success": False,
            "organizational_units": [],
            "metadata": {
                "total_clusters": 0,
                "successful_extractions": 0,
                "total_text_elements": 0,
                "average_confidence": 0.0,
                "error": error
            },
            "spatial_data": {"clusters": []}
        }

# Convenience function for easy integration
async def extract_pdf_spatial(pdf_path: str) -> Dict[str, Any]:
    """
    Async wrapper for PDF spatial extraction
    Returns structured organizational data from PDF
    """
    extractor = PDFSpatialExtractor()
    return extractor.extract_with_coordinates(pdf_path)