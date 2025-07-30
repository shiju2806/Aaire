"""
Advanced OCR Processor for AAIRE
Combines PaddleOCR + OpenCV + Document Structure Analysis
"""

import cv2
import numpy as np
from PIL import Image
# Handle Pillow version compatibility
try:
    ANTIALIAS = Image.ANTIALIAS
except AttributeError:
    ANTIALIAS = Image.LANCZOS
import io
import base64
from typing import Dict, List, Tuple, Optional, Any
import structlog
from pathlib import Path

logger = structlog.get_logger()

class AdvancedOCRProcessor:
    def __init__(self):
        """Initialize OCR processor with PaddleOCR and OpenCV"""
        self.ocr_engine = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize EasyOCR engine with optimal settings"""
        try:
            import easyocr
            
            # Initialize EasyOCR with English language support
            self.ocr_engine = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR processor initialized successfully")
            
        except ImportError as e:
            logger.warning(f"EasyOCR not available, falling back to basic mode: {e}")
            self.ocr_engine = None
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {e}")
            self.ocr_engine = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Advanced image preprocessing using OpenCV"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Enhance contrast
            enhanced = cv2.convertScaleAbs(denoised, alpha=1.3, beta=10)
            
            # Adaptive thresholding for better text detection
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            logger.debug("Image preprocessing completed successfully")
            return cleaned
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {e}")
            return image
    
    def detect_document_structure(self, ocr_results: List) -> Dict[str, Any]:
        """Analyze document structure from OCR results"""
        try:
            elements = {
                'titles': [],
                'tables': [],
                'charts': [],
                'text_blocks': [],
                'numbers': [],
                'x_axis': [],  # For chart x-axis labels (FY19, FY20, etc.)
                'legends': []  # For chart legends (Revenue, Operating Expenses, etc.)
            }
            
            for result in ocr_results:
                bbox, (text, confidence) = result
                
                # Skip low confidence results
                if confidence < 0.6:
                    continue
                
                # Analyze text properties
                text_clean = text.strip()
                if not text_clean:
                    continue
                
                # Get bounding box coordinates for spatial analysis
                if isinstance(bbox, list) and len(bbox) >= 4:
                    # bbox is typically [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    min_x = min(point[0] for point in bbox)
                    max_x = max(point[0] for point in bbox)
                    min_y = min(point[1] for point in bbox)
                    max_y = max(point[1] for point in bbox)
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                else:
                    center_x = center_y = min_x = max_x = min_y = max_y = 0
                
                # Detect fiscal year patterns (FY19, FY20, etc.)
                import re
                if re.search(r'FY\d{2,4}|fy\d{2,4}|20\d{2}', text_clean):
                    elements['x_axis'].append({
                        'text': text_clean,
                        'bbox': bbox,
                        'confidence': confidence,
                        'x_pos': center_x,
                        'y_pos': center_y
                    })
                
                # Detect chart legends (Revenue, Operating Expenses, Gross Margin, etc.)
                legend_keywords = ['revenue', 'expense', 'margin', 'profit', 'loss', 'income', 'cost', 'gross', 'operating', 'net']
                if any(keyword in text_clean.lower() for keyword in legend_keywords) and len(text_clean) < 50:
                    elements['legends'].append({
                        'text': text_clean,
                        'bbox': bbox,
                        'confidence': confidence,
                        'x_pos': center_x,
                        'y_pos': center_y
                    })
                
                # Detect financial values with units - expanded patterns
                # Matches: $47.4B, 47.4B, 47B, $47 billion, 47 billion, 47.4, 10B, 20B, etc.
                if re.search(r'\$?[\d,]+\.?\d*\s*[BMK](?![a-zA-Z])|\$[\d,]+\.?\d*|\d+\.?\d*\s*(?:billion|million|thousand)|\d+\.?\d*%|^\d+B$|^\d+\.?\d*$', text_clean):
                    # Check if it's likely an axis label (typically on far left)
                    is_axis_label = min_x < 100 and re.search(r'^\d+B?$|^\d+\.?\d*$', text_clean)
                    
                    elements['numbers'].append({
                        'text': text_clean,
                        'bbox': bbox,
                        'confidence': confidence,
                        'x_pos': center_x,
                        'y_pos': center_y,
                        'is_percentage': '%' in text_clean,
                        'is_currency': '$' in text_clean,
                        'is_billions': 'B' in text_clean.upper() or 'billion' in text_clean.lower(),
                        'is_axis_label': is_axis_label
                    })
                
                # General text blocks
                elements['text_blocks'].append({
                    'text': text_clean,
                    'bbox': bbox,
                    'confidence': confidence
                })
            
            logger.debug(f"Document structure detected: {len(elements['titles'])} titles, {len(elements['numbers'])} numbers")
            logger.debug(f"Chart elements: {len(elements['x_axis'])} x-axis labels, {len(elements['legends'])} legends")
            return elements
            
        except Exception as e:
            logger.error(f"Document structure analysis failed: {e}")
            return {'titles': [], 'tables': [], 'charts': [], 'text_blocks': [], 'numbers': [], 'x_axis': [], 'legends': []}
    
    def extract_table_structure(self, image: np.ndarray) -> List[List[str]]:
        """Detect and extract table structure from image"""
        try:
            # Find horizontal and vertical lines
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Detect vertical lines  
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines to detect table structure  
            table_mask = cv2.add(horizontal_lines, vertical_lines)
            
            # Find contours (table cells)
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by position (top to bottom, left to right)
            contours = sorted(contours, key=lambda x: (cv2.boundingRect(x)[1], cv2.boundingRect(x)[0]))
            
            table_data = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20:  # Filter small contours
                    cell_image = gray[y:y+h, x:x+w]
                    cell_text = self.extract_text_from_image(cell_image)
                    if cell_text.strip():
                        table_data.append(cell_text.strip())
            
            logger.debug(f"Table structure extracted: {len(table_data)} cells")
            return [table_data[i:i+4] for i in range(0, len(table_data), 4)]  # Assume 4 columns
            
        except Exception as e:
            logger.warning(f"Table structure extraction failed: {e}")
            return []
    
    def reconstruct_chart_data(self, structure: Dict[str, Any]) -> str:
        """Reconstruct chart data by matching x-axis labels with values"""
        try:
            chart_text = []
            
            # Sort x-axis labels by position (left to right)
            x_labels = sorted(structure.get('x_axis', []), key=lambda x: x.get('x_pos', 0))
            
            # Sort legends and values by position
            legends = structure.get('legends', [])
            values = structure.get('numbers', [])
            
            # Group values by their vertical position (same row = same metric)
            value_rows = {}
            for value in values:
                y_pos = value.get('y_pos', 0)
                # Round y position to group nearby values
                y_bucket = round(y_pos / 20) * 20
                if y_bucket not in value_rows:
                    value_rows[y_bucket] = []
                value_rows[y_bucket].append(value)
            
            # Build structured output
            if x_labels:
                chart_text.append("[CHART DATA STRUCTURE]")
                chart_text.append(f"X-Axis (Fiscal Years): {', '.join([x['text'] for x in x_labels])}")
                chart_text.append("")
            
            if legends:
                chart_text.append("Chart Metrics:")
                for legend in legends:
                    chart_text.append(f"- {legend['text']}")
                chart_text.append("")
            
            # Try to match values with x-axis labels
            if x_labels and value_rows:
                chart_text.append("[EXTRACTED DATA BY YEAR]")
                for x_label in x_labels:
                    x_pos = x_label.get('x_pos', 0)
                    year_text = x_label['text']
                    chart_text.append(f"\n{year_text}:")
                    
                    # Find values near this x position
                    for y_bucket, row_values in value_rows.items():
                        nearby_values = [
                            v for v in row_values 
                            if abs(v.get('x_pos', 0) - x_pos) < 50  # Values within 50 pixels
                        ]
                        for value in nearby_values:
                            if value.get('is_currency') or value.get('is_billions'):
                                # Try to identify if it's revenue or expense based on legend proximity
                                value_type = "Revenue/Expense"
                                if legends:
                                    # Find closest legend by y-position
                                    closest_legend = min(legends, key=lambda l: abs(l.get('y_pos', 0) - value.get('y_pos', 0)), default=None)
                                    if closest_legend:
                                        value_type = closest_legend['text']
                                chart_text.append(f"  {value_type}: {value['text']}")
                            elif value.get('is_percentage'):
                                chart_text.append(f"  Gross Margin: {value['text']}")
                            else:
                                chart_text.append(f"  Value: {value['text']}")
            
            # Also show axis scale if detected
            axis_values = [v for v in values if v.get('is_axis_label')]
            if axis_values:
                chart_text.append("\n[Y-AXIS SCALE]")
                for av in sorted(axis_values, key=lambda x: x.get('y_pos', 0)):
                    chart_text.append(f"- {av['text']}")
            
            return "\n".join(chart_text) if chart_text else ""
            
        except Exception as e:
            logger.error(f"Chart reconstruction failed: {e}")
            return ""
    
    def extract_text_from_image(self, image: np.ndarray, preprocess: bool = True) -> str:
        """Extract text from image using advanced OCR"""
        try:
            if self.ocr_engine is None:
                logger.warning("OCR engine not available")
                return ""
            
            # Preprocess image for better OCR
            if preprocess:
                processed_image = self.preprocess_image(image)
            else:
                processed_image = image
            
            # Run EasyOCR
            results = self.ocr_engine.readtext(processed_image)
            
            if not results:
                return ""
            
            # Convert EasyOCR format to our format
            ocr_results = []
            for (bbox, text, confidence) in results:
                ocr_results.append([bbox, (text, confidence)])
            structure = self.detect_document_structure(ocr_results)
            
            # Build structured text output
            extracted_text = []
            
            # Add reconstructed chart data first if available
            if structure.get('x_axis') or structure.get('legends'):
                chart_reconstruction = self.reconstruct_chart_data(structure)
                if chart_reconstruction:
                    extracted_text.append(chart_reconstruction)
                    extracted_text.append("")
            
            # Add titles
            if structure['titles']:
                extracted_text.append("[ADDITIONAL TITLES]")
                for title in structure['titles']:
                    extracted_text.append(f"TITLE: {title['text']}")
                extracted_text.append("")
            
            # Add any remaining unstructured data
            if structure['numbers'] and not structure.get('x_axis'):
                # Only show raw numbers if we couldn't structure them
                extracted_text.append("[UNSTRUCTURED FINANCIAL DATA]")
                for num in structure['numbers']:
                    extracted_text.append(f"VALUE: {num['text']}")
                extracted_text.append("")
            
            result = "\n".join(extracted_text)
            logger.info(f"OCR extraction completed: {len(result)} characters extracted")
            
            # Log detailed extraction for debugging
            if result:
                logger.info(f"OCR extracted text preview: {result[:500]}...")
                # Log financial data specifically
                if structure['numbers']:
                    logger.info(f"Financial values found: {[num['text'] for num in structure['numbers'][:10]]}")
            
            return result
            
        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}")
            return ""
    
    def process_chart_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process chart image and extract structured data"""
        try:
            # Convert bytes to OpenCV image
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image data")
            
            # Extract text using advanced OCR
            extracted_text = self.extract_text_from_image(image)
            
            # Attempt table extraction
            table_data = self.extract_table_structure(image)
            
            # Build comprehensive result
            result = {
                'extracted_text': extracted_text,
                'table_data': table_data,
                'image_dimensions': image.shape[:2],
                'processing_status': 'success'
            }
            
            logger.info("Chart image processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Chart image processing failed: {e}")
            return {
                'extracted_text': '',
                'table_data': [],
                'image_dimensions': (0, 0),
                'processing_status': f'error: {str(e)}'
            }
    
    def is_available(self) -> bool:
        """Check if OCR processor is available"""
        return self.ocr_engine is not None