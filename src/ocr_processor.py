"""
Advanced OCR Processor for AAIRE
Combines PaddleOCR + OpenCV + Document Structure Analysis
"""

import cv2
import numpy as np
from PIL import Image
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
        """Initialize PaddleOCR engine with optimal settings"""
        try:
            from paddleocr import PaddleOCR
            
            # Initialize with optimal settings for financial documents
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,  # Detect text angle
                lang='en',  # English language
                use_gpu=False,  # CPU mode (compatible everywhere)
                show_log=False,  # Reduce logging noise
                det_model_dir=None,  # Use default detection model
                rec_model_dir=None,  # Use default recognition model
                cls_model_dir=None,  # Use default classification model
            )
            logger.info("Advanced OCR processor initialized successfully")
            
        except ImportError as e:
            logger.warning(f"PaddleOCR not available, falling back to basic mode: {e}")
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
                'numbers': []
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
                
                # Detect titles (usually larger, centered, short)
                if len(text_clean) < 50 and confidence > 0.8:
                    if any(word in text_clean.upper() for word in ['REVENUE', 'PROFIT', 'QUARTER', 'YEAR', 'TOTAL']):
                        elements['titles'].append({
                            'text': text_clean,
                            'bbox': bbox,
                            'confidence': confidence
                        })
                
                # Detect numbers (financial data)
                if any(char in text_clean for char in ['$', '%', ',']) or text_clean.replace('.', '').replace(',', '').isdigit():
                    elements['numbers'].append({
                        'text': text_clean,
                        'bbox': bbox,
                        'confidence': confidence
                    })
                
                # General text blocks
                elements['text_blocks'].append({
                    'text': text_clean,
                    'bbox': bbox,
                    'confidence': confidence
                })
            
            logger.debug(f"Document structure detected: {len(elements['titles'])} titles, {len(elements['numbers'])} numbers")
            return elements
            
        except Exception as e:
            logger.error(f"Document structure analysis failed: {e}")
            return {'titles': [], 'tables': [], 'charts': [], 'text_blocks': [], 'numbers': []}
    
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
            
            # Run OCR
            results = self.ocr_engine.ocr(processed_image, cls=True)
            
            if not results or not results[0]:
                return ""
            
            # Extract text and analyze structure
            ocr_results = results[0]
            structure = self.detect_document_structure(ocr_results)
            
            # Build structured text output
            extracted_text = []
            
            # Add titles first
            if structure['titles']:
                extracted_text.append("[CHART TITLES]")
                for title in structure['titles']:
                    extracted_text.append(f"TITLE: {title['text']}")
                extracted_text.append("")
            
            # Add financial numbers
            if structure['numbers']:
                extracted_text.append("[FINANCIAL DATA]")
                for num in structure['numbers']:
                    extracted_text.append(f"VALUE: {num['text']}")
                extracted_text.append("")
            
            # Add other text
            remaining_text = []
            for block in structure['text_blocks']:
                if block not in structure['titles'] and block not in structure['numbers']:
                    remaining_text.append(block['text'])
            
            if remaining_text:
                extracted_text.append("[CHART TEXT]")
                extracted_text.extend(remaining_text)
            
            result = "\n".join(extracted_text)
            logger.info(f"OCR extraction completed: {len(result)} characters extracted")
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