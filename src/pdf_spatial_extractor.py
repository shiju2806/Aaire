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
            logger.info(f"Starting shape-aware spatial extraction for: {pdf_path}")
            
            # Step 1: Extract PDF shapes/rectangles (org chart boxes)
            shapes_by_page = self._extract_pdf_shapes(pdf_path)
            total_shapes = sum(len(shapes) for shapes in shapes_by_page.values())
            logger.info(f"Detected {total_shapes} shapes across {len(shapes_by_page)} pages")
            
            # Step 2: Extract all text elements with coordinates
            text_elements = self._extract_text_elements(pdf_path)
            
            if not text_elements:
                return self._create_empty_result("No text elements found")
            
            logger.info(f"Extracted {len(text_elements)} text elements")
            
            # Step 3: Group elements into spatial clusters using shape awareness
            clusters = self._create_shape_aware_clusters(text_elements, shapes_by_page)
            
            logger.info(f"Created {len(clusters)} shape-aware spatial clusters")
            
            # Step 4: Parse each cluster for organizational information
            org_units = self._parse_organizational_clusters(clusters)
            
            logger.info(f"Identified {len(org_units)} organizational units")
            
            # Step 5: Create structured output
            result = self._create_structured_result(org_units, clusters, text_elements)
            
            # Add shape detection metadata
            result["metadata"]["shapes_detected"] = total_shapes
            result["metadata"]["shape_based_clusters"] = sum(1 for c in clusters if "shape_cluster" in c.cluster_id)
            
            logger.info("Shape-aware spatial extraction completed successfully")
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
    
    def _extract_pdf_shapes(self, pdf_path: str) -> Dict[int, List[Tuple[float, float, float, float]]]:
        """Extract actual shapes/rectangles from PDF for each page"""
        shapes_by_page = {}
        
        if not PYMUPDF_AVAILABLE:
            return shapes_by_page
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                shapes = []
                
                # Get drawing objects (rectangles, lines, etc.)
                drawings = page.get_drawings()
                
                for drawing in drawings:
                    # Look for rectangular shapes
                    if 'rect' in drawing:
                        rect = drawing['rect']
                        # Filter shapes for org chart boxes
                        width = rect.width
                        height = rect.height
                        
                        # Org chart boxes: reasonable size, not too large (not page-wide)
                        if (50 < width < 400 and 30 < height < 200 and
                            rect.x0 > 10 and rect.y0 > 10):  # Not at page edges
                            shapes.append((rect.x0, rect.y0, rect.x1, rect.y1))
                    
                    # Also check for closed paths that might be rectangles
                    elif 'items' in drawing:
                        # Analyze path items to detect rectangular shapes
                        path_rect = self._analyze_path_for_rectangle(drawing['items'])
                        if path_rect:
                            x0, y0, x1, y1 = path_rect
                            width = x1 - x0
                            height = y1 - y0
                            
                            # Apply same filtering as above
                            if (50 < width < 400 and 30 < height < 200 and
                                x0 > 10 and y0 > 10):
                                shapes.append(path_rect)
                
                # Also detect implicit boxes from text clustering patterns
                implicit_boxes = self._detect_implicit_text_boxes(page)
                
                # Filter implicit boxes as well
                filtered_implicit = []
                for box in implicit_boxes:
                    x0, y0, x1, y1 = box
                    width = x1 - x0
                    height = y1 - y0
                    if (50 < width < 400 and 30 < height < 200):
                        filtered_implicit.append(box)
                
                shapes.extend(filtered_implicit)
                
                shapes_by_page[page_num] = shapes
                logger.debug(f"Found {len(shapes)} shapes on page {page_num}")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Failed to extract PDF shapes: {e}")
        
        return shapes_by_page
    
    def _analyze_path_for_rectangle(self, path_items: List) -> Optional[Tuple[float, float, float, float]]:
        """Analyze path items to detect if they form a rectangle"""
        if not path_items or len(path_items) < 4:
            return None
        
        # Look for 4-5 items that form a closed rectangle (moveto + 4 lineto + closepath)
        points = []
        for item in path_items:
            if item[0] in ['m', 'l']:  # moveto or lineto
                points.append((item[1].x, item[1].y))
        
        if len(points) >= 4:
            # Check if points form a rectangle
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            
            # Verify it's roughly rectangular (not too skewed)
            if (x1 - x0) > 20 and (y1 - y0) > 20:
                return (x0, y0, x1, y1)
        
        return None
    
    def _detect_implicit_text_boxes(self, page) -> List[Tuple[float, float, float, float]]:
        """Detect implicit boxes from text layout patterns when no explicit shapes exist"""
        implicit_boxes = []
        
        # Get text blocks
        text_dict = page.get_text("dict")
        text_blocks = []
        
        for block in text_dict["blocks"]:
            if "lines" in block and block["lines"]:
                # Calculate block bounding box
                all_spans = []
                for line in block["lines"]:
                    all_spans.extend(line["spans"])
                
                if all_spans:
                    x0 = min(span["bbox"][0] for span in all_spans)
                    y0 = min(span["bbox"][1] for span in all_spans)
                    x1 = max(span["bbox"][2] for span in all_spans)
                    y1 = max(span["bbox"][3] for span in all_spans)
                    
                    # Only consider blocks that could be org chart boxes
                    width = x1 - x0
                    height = y1 - y0
                    
                    if 30 < width < 300 and 20 < height < 150:
                        text_blocks.append((x0, y0, x1, y1))
        
        # Filter overlapping blocks to avoid duplicates
        for i, box1 in enumerate(text_blocks):
            is_unique = True
            for j, box2 in enumerate(text_blocks):
                if i != j and self._boxes_overlap_significantly(box1, box2):
                    # Keep the larger box
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    if area1 < area2:
                        is_unique = False
                        break
            
            if is_unique:
                implicit_boxes.append(box1)
        
        return implicit_boxes
    
    def _boxes_overlap_significantly(self, box1: Tuple[float, float, float, float], 
                                   box2: Tuple[float, float, float, float]) -> bool:
        """Check if two boxes overlap significantly (>50% area)"""
        x0_1, y0_1, x1_1, y1_1 = box1
        x0_2, y0_2, x1_2, y1_2 = box2
        
        # Calculate intersection
        x0_i = max(x0_1, x0_2)
        y0_i = max(y0_1, y0_2)
        x1_i = min(x1_1, x1_2)
        y1_i = min(y1_1, y1_2)
        
        if x1_i <= x0_i or y1_i <= y0_i:
            return False  # No overlap
        
        # Calculate areas
        intersection_area = (x1_i - x0_i) * (y1_i - y0_i)
        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        
        # Check if intersection is >50% of smaller box
        smaller_area = min(area1, area2)
        return intersection_area / smaller_area > 0.5
    
    def _create_shape_aware_clusters(self, elements: List[TextElement], 
                                   shapes_by_page: Dict[int, List[Tuple[float, float, float, float]]]) -> List[TextCluster]:
        """Group text elements into spatial clusters using shape awareness"""
        if not elements:
            return []
        
        clusters = []
        
        # Group elements by page first
        elements_by_page = {}
        for element in elements:
            if element.page not in elements_by_page:
                elements_by_page[element.page] = []
            elements_by_page[element.page].append(element)
        
        # Process each page
        for page_num, page_elements in elements_by_page.items():
            page_shapes = shapes_by_page.get(page_num, [])
            
            if page_shapes:
                # Shape-based clustering: assign text to detected shapes
                logger.debug(f"Using shape-based clustering for page {page_num} ({len(page_shapes)} shapes)")
                page_clusters = self._create_shape_based_clusters(page_elements, page_shapes, page_num)
            else:
                # Fallback: improved proximity-based clustering
                logger.debug(f"Using improved proximity clustering for page {page_num} (no shapes detected)")
                page_clusters = self._create_improved_proximity_clusters(page_elements, page_num)
            
            clusters.extend(page_clusters)
        
        return clusters
    
    def _create_shape_based_clusters(self, elements: List[TextElement], 
                                   shapes: List[Tuple[float, float, float, float]], 
                                   page_num: int) -> List[TextCluster]:
        """Create clusters by assigning text elements to detected shapes"""
        clusters = []
        used_elements = set()
        
        for i, shape in enumerate(shapes):
            shape_x0, shape_y0, shape_x1, shape_y1 = shape
            shape_elements = []
            
            # Find all text elements within this shape
            for j, element in enumerate(elements):
                if j in used_elements:
                    continue
                
                # Check if element is within shape boundaries (with small margin)
                margin = 5
                if (shape_x0 - margin <= element.center_x <= shape_x1 + margin and
                    shape_y0 - margin <= element.center_y <= shape_y1 + margin):
                    shape_elements.append(element)
                    used_elements.add(j)
            
            # Create cluster if shape has text and looks like org chart box
            if (len(shape_elements) >= self.min_cluster_size and 
                len(shape_elements) <= self.max_cluster_size and
                self._is_valid_org_cluster(shape_elements)):
                
                cluster_id = f"shape_cluster_{i}_page_{page_num}"
                confidence = self._calculate_cluster_confidence(shape_elements)
                confidence += 0.2  # Bonus for being shape-based
                
                clusters.append(TextCluster(
                    elements=shape_elements,
                    cluster_id=cluster_id,
                    confidence=min(confidence, 1.0)
                ))
        
        # Handle remaining elements with proximity clustering
        remaining_elements = [elem for j, elem in enumerate(elements) if j not in used_elements]
        if remaining_elements:
            remaining_clusters = self._create_improved_proximity_clusters(remaining_elements, page_num)
            clusters.extend(remaining_clusters)
        
        return clusters
    
    def _create_improved_proximity_clusters(self, elements: List[TextElement], page_num: int) -> List[TextCluster]:
        """Improved proximity-based clustering with better org chart awareness"""
        clusters = []
        used_elements = set()
        
        # Sort elements by position (top-to-bottom, left-to-right)
        sorted_elements = sorted(elements, key=lambda e: (-e.center_y, e.center_x))
        
        for i, element in enumerate(sorted_elements):
            element_idx = elements.index(element)
            if element_idx in used_elements:
                continue
            
            # Start new cluster with improved logic
            cluster_elements = [element]
            used_elements.add(element_idx)
            
            # Look for elements that should be grouped together
            search_radius = 60  # Larger search radius
            candidates = []
            
            for j, other_element in enumerate(sorted_elements):
                other_idx = elements.index(other_element)
                if other_idx in used_elements or i == j:
                    continue
                
                # Calculate distance with org chart bias
                distance = self._calculate_org_aware_distance(element, other_element)
                
                if distance <= search_radius:
                    candidates.append((other_element, other_idx, distance))
            
            # Sort candidates by distance and add the best ones
            candidates.sort(key=lambda x: x[2])
            
            for other_element, other_idx, distance in candidates:
                # Check if adding this element makes sense
                potential_cluster = cluster_elements + [other_element]
                
                if (len(potential_cluster) <= self.max_cluster_size and
                    self._would_improve_cluster(cluster_elements, other_element)):
                    cluster_elements.append(other_element)
                    used_elements.add(other_idx)
            
            # Only create cluster if it meets org chart criteria
            if (len(cluster_elements) >= self.min_cluster_size and
                self._is_valid_org_cluster(cluster_elements)):
                
                cluster_id = f"proximity_cluster_{len(clusters)}_page_{page_num}"
                confidence = self._calculate_cluster_confidence(cluster_elements)
                
                clusters.append(TextCluster(
                    elements=cluster_elements,
                    cluster_id=cluster_id,
                    confidence=confidence
                ))
        
        return clusters
    
    def _calculate_org_aware_distance(self, elem1: TextElement, elem2: TextElement) -> float:
        """Calculate distance with org chart awareness - improved version"""
        x_distance = abs(elem1.center_x - elem2.center_x)
        y_distance = abs(elem1.center_y - elem2.center_y)
        
        # Strong preference for vertical alignment (same column)
        if x_distance < 40:  # Elements are in same column
            # Very close vertically -> likely same box
            if y_distance < 25:
                return y_distance * 0.3
            # Moderately close vertically -> might be same box
            elif y_distance < 60:
                return y_distance * 0.6
            else:
                return y_distance * 1.2
        
        # If horizontally aligned but different columns -> discourage
        if y_distance < 15:  # Same row
            return x_distance * 3.0  # Heavily penalize horizontal grouping
        
        # Diagonal - use standard euclidean with slight penalty
        euclidean = ((x_distance ** 2) + (y_distance ** 2)) ** 0.5
        return euclidean * 1.1  # Slight penalty for diagonal relationships
    
    def _would_improve_cluster(self, current_cluster: List[TextElement], new_element: TextElement) -> bool:
        """Check if adding an element would improve the cluster quality"""
        if not current_cluster:
            return True
        
        # Check alignment consistency
        current_x_positions = [elem.center_x for elem in current_cluster]
        x_variance_current = max(current_x_positions) - min(current_x_positions)
        
        new_x_positions = current_x_positions + [new_element.center_x]
        x_variance_new = max(new_x_positions) - min(new_x_positions)
        
        # Don't add if it significantly increases horizontal spread
        if x_variance_new > x_variance_current * 1.5 and x_variance_new > 80:
            return False
        
        # Check vertical consistency (elements should be reasonably close vertically)
        current_y_positions = [elem.center_y for elem in current_cluster]
        y_min, y_max = min(current_y_positions), max(current_y_positions)
        
        # New element should be within reasonable vertical range
        if not (y_min - 40 <= new_element.center_y <= y_max + 40):
            return False
        
        return True
    
    def _calculate_distance(self, elem1: TextElement, elem2: TextElement) -> float:
        """Legacy distance calculation - kept for compatibility"""
        return self._calculate_org_aware_distance(elem1, elem2)
    
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