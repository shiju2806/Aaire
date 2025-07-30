"""
Advanced OCR Processor using docTR for AAIRE
Optimized for financial charts and structured documents
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Dict, List, Tuple, Optional, Any
import structlog

logger = structlog.get_logger()

class DocTROCRProcessor:
    def __init__(self):
        """Initialize docTR OCR processor"""
        self.ocr_engine = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize docTR engine"""
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
            
            # Initialize docTR with pretrained models
            self.ocr_engine = ocr_predictor(
                det_arch='db_resnet50',  # Detection architecture
                reco_arch='crnn_vgg16_bn',  # Recognition architecture
                pretrained=True,
                assume_straight_pages=True,  # Charts are usually straight
                export_as_straight_boxes=True  # Better for structured data
            )
            
            logger.info("docTR OCR processor initialized successfully")
            
        except ImportError as e:
            logger.warning(f"docTR not available, falling back to basic mode: {e}")
            self.ocr_engine = None
        except Exception as e:
            logger.error(f"Failed to initialize docTR engine: {e}")
            self.ocr_engine = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Light preprocessing for docTR"""
        try:
            # docTR handles most preprocessing internally
            # Just ensure proper format
            if len(image.shape) == 2:
                # Convert grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def extract_text_from_image(self, image: np.ndarray, preprocess: bool = True) -> str:
        """Extract text from image using docTR"""
        try:
            if self.ocr_engine is None:
                logger.warning("docTR engine not available")
                return ""
            
            # Preprocess if needed
            if preprocess:
                image = self.preprocess_image(image)
            
            # Convert numpy array to PIL Image for docTR
            pil_image = Image.fromarray(image)
            
            # Save to bytes buffer
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Create document from buffer
            from doctr.io import DocumentFile
            doc = DocumentFile.from_images(img_buffer.read())
            
            # Run OCR
            result = self.ocr_engine(doc)
            
            # Extract structured text
            extracted_text = self._parse_doctr_results(result)
            
            logger.info(f"docTR extraction completed: {len(extracted_text)} characters extracted")
            print(f"[docTR] Extracted {len(extracted_text)} characters")  # Direct print
            
            # Log what was found for debugging
            if extracted_text:
                logger.info(f"docTR extracted preview: {extracted_text[:300]}...")
                print(f"[docTR] Preview: {extracted_text[:200]}...")  # Direct print
                # Log if fiscal years were found
                if "FY" in extracted_text or "fy" in extracted_text.lower():
                    logger.info("docTR found fiscal year data in chart")
                    print("[docTR] Found fiscal year data!")  # Direct print
                # Log if dollar amounts were found
                import re
                dollar_amounts = re.findall(r'\$?[\d,]+\.?\d*[BMK]|\d+B|\d+\.\d+B', extracted_text)
                if dollar_amounts:
                    logger.info(f"docTR found dollar amounts: {dollar_amounts[:5]}")
                    print(f"[docTR] Found amounts: {dollar_amounts}")  # Direct print
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"docTR text extraction failed: {e}")
            return ""
    
    def _parse_doctr_results(self, result) -> str:
        """Parse docTR results into structured text"""
        try:
            text_elements = []
            
            # docTR provides hierarchical structure: document -> page -> block -> line -> word
            for page in result.pages:
                blocks_by_position = []
                
                # Collect all text blocks with positions
                for block in page.blocks:
                    for line in block.lines:
                        line_text = " ".join([word.value for word in line.words])
                        if line_text.strip():
                            # Get bounding box center for spatial analysis
                            bbox = line.geometry
                            center_y = (bbox[0][1] + bbox[1][1]) / 2
                            center_x = (bbox[0][0] + bbox[1][0]) / 2
                            
                            blocks_by_position.append({
                                'text': line_text,
                                'x': center_x,
                                'y': center_y,
                                'confidence': min([word.confidence for word in line.words])
                            })
                
                # Sort by position (top to bottom, left to right)
                blocks_by_position.sort(key=lambda b: (b['y'], b['x']))
                
                # Group by vertical position to identify rows
                rows = self._group_by_rows(blocks_by_position)
                
                # Identify chart components
                chart_data = self._identify_chart_components(rows)
                
                # Build structured output
                if chart_data['has_chart']:
                    text_elements.append("[CHART DETECTED]")
                    
                    if chart_data['x_axis']:
                        text_elements.append(f"X-Axis: {' | '.join(chart_data['x_axis'])}")
                    
                    if chart_data['y_values']:
                        text_elements.append(f"Y-Axis Scale: {' | '.join(chart_data['y_values'])}")
                    
                    if chart_data['data_rows']:
                        text_elements.append("\n[DATA BY FISCAL YEAR]")
                        for row in chart_data['data_rows']:
                            text_elements.append(row)
                    
                    text_elements.append("")
                
                # Add any remaining text
                for row in rows:
                    row_text = " | ".join([b['text'] for b in row])
                    if row_text not in text_elements:
                        text_elements.append(row_text)
            
            return "\n".join(text_elements)
            
        except Exception as e:
            logger.error(f"Failed to parse docTR results: {e}")
            return ""
    
    def _group_by_rows(self, blocks: List[Dict]) -> List[List[Dict]]:
        """Group text blocks into rows based on Y position"""
        if not blocks:
            return []
        
        rows = []
        current_row = [blocks[0]]
        current_y = blocks[0]['y']
        
        for block in blocks[1:]:
            # If Y position is close (within 2% of image height), same row
            if abs(block['y'] - current_y) < 0.02:
                current_row.append(block)
            else:
                rows.append(sorted(current_row, key=lambda b: b['x']))
                current_row = [block]
                current_y = block['y']
        
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b['x']))
        
        return rows
    
    def _identify_chart_components(self, rows: List[List[Dict]]) -> Dict:
        """Identify chart components from grouped text"""
        import re
        
        chart_data = {
            'has_chart': False,
            'x_axis': [],
            'y_values': [],
            'data_rows': []
        }
        
        for row in rows:
            row_text = [b['text'] for b in row]
            
            # Check for fiscal year patterns (likely X-axis)
            fy_pattern = re.compile(r'FY\d{2,4}|20\d{2}|19\d{2}')
            fy_matches = [t for t in row_text if fy_pattern.search(t)]
            if len(fy_matches) > 2:  # Multiple fiscal years = X-axis
                chart_data['x_axis'] = fy_matches
                chart_data['has_chart'] = True
            
            # Check for axis scale values (10, 20, 30, etc. or 10B, 20B)
            scale_pattern = re.compile(r'^\d+[BMK]?$|^\d+\.?\d*$')
            scale_matches = [t for t in row_text if scale_pattern.match(t) and len(t) < 5]
            if len(scale_matches) > 2 and row[0]['x'] < 0.15:  # Left side values
                chart_data['y_values'] = scale_matches
            
            # Check for data values with fiscal years
            if chart_data['x_axis'] and any('$' in t or 'B' in t or '%' in t for t in row_text):
                # This row likely contains data values
                formatted_row = self._format_data_row(row_text, chart_data['x_axis'])
                if formatted_row:
                    chart_data['data_rows'].append(formatted_row)
        
        return chart_data
    
    def _format_data_row(self, row_text: List[str], x_axis: List[str]) -> Optional[str]:
        """Format a data row matching values to fiscal years"""
        import re
        
        # Extract metric name (Revenue, Operating Expenses, etc.)
        metric = None
        for text in row_text:
            if any(word in text.lower() for word in ['revenue', 'expense', 'margin', 'profit', 'income']):
                metric = text
                break
        
        if not metric:
            metric = "Data"
        
        # Extract values
        value_pattern = re.compile(r'\$?[\d,]+\.?\d*[BMK]?|\d+\.?\d*%')
        values = [t for t in row_text if value_pattern.search(t)]
        
        if values:
            return f"{metric}: {' | '.join(values)}"
        
        return None
    
    def process_chart_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process chart image and extract structured data"""
        try:
            # Convert bytes to numpy array
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image data")
            
            # Extract text using docTR
            extracted_text = self.extract_text_from_image(image)
            
            result = {
                'extracted_text': extracted_text,
                'processing_status': 'success'
            }
            
            logger.info("Chart image processing completed with docTR")
            return result
            
        except Exception as e:
            logger.error(f"Chart image processing failed: {e}")
            return {
                'extracted_text': '',
                'processing_status': f'error: {str(e)}'
            }
    
    def is_available(self) -> bool:
        """Check if OCR processor is available"""
        return self.ocr_engine is not None

# Create alias for backward compatibility
AdvancedOCRProcessor = DocTROCRProcessor