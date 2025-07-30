"""
Simple Tesseract OCR Processor for AAIRE
Fallback option when advanced OCR models fail
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Dict, List, Optional, Any
import structlog
import re
from .chart_analyzer import ChartAnalyzer

logger = structlog.get_logger()

class TesseractOCRProcessor:
    def __init__(self):
        """Initialize Tesseract OCR processor"""
        self.ocr_available = self._check_tesseract()
        self.chart_analyzer = ChartAnalyzer()
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available"""
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available")
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply threshold to get better text
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Denoise
            denoised = cv2.medianBlur(thresh, 1)
            
            return denoised
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return image
    
    def extract_text_from_image(self, image: np.ndarray, preprocess: bool = True) -> str:
        """Extract text using Tesseract"""
        try:
            if not self.ocr_available:
                return ""
            
            # Preprocess if needed
            if preprocess:
                processed = self.preprocess_image(image)
            else:
                processed = image
            
            # Configure Tesseract specifically for financial charts
            # PSM 6: Uniform block of text (good for charts)
            # OEM 3: Default engine mode
            # Whitelist digits, currency symbols, and common chart text
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$%.,FYfy '
            
            # Try multiple PSM modes for better results
            configs_to_try = [
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$%.,FYfy ',  # Uniform text
                r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$%.,FYfy ',  # Sparse text
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$%.,FYfy '   # Single word
            ]
            
            best_text = ""
            max_length = 0
            
            for config in configs_to_try:
                try:
                    text = pytesseract.image_to_string(processed, config=config)
                    if len(text) > max_length:
                        max_length = len(text)
                        best_text = text
                        custom_config = config
                except:
                    continue
            
            text = best_text if best_text else pytesseract.image_to_string(processed, config=custom_config)
            
            print(f"[Tesseract] Raw extraction: {len(text)} chars")
            print(f"[Tesseract] Raw text:\n{text}")
            
            # Parse and structure the text
            structured = self._structure_chart_text(text)
            
            # If no explicit values found, try to estimate from chart analysis
            if not re.search(r'\$?[\d,]+\.?\d*[BMK]', structured):
                print("[Tesseract] No explicit values found, analyzing chart...")
                analysis = self.chart_analyzer.analyze_chart(processed, text)
                estimates = self.chart_analyzer.format_estimates_as_text(analysis)
                if estimates:
                    structured += "\n\n" + estimates
            
            return structured
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return ""
    
    def _structure_chart_text(self, raw_text: str) -> str:
        """Structure the raw OCR text for charts"""
        lines = raw_text.strip().split('\n')
        
        structured = []
        fiscal_years = []
        values = []
        labels = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for fiscal years
            if re.search(r'FY\s*\d{2,4}', line, re.IGNORECASE):
                fiscal_years.extend(re.findall(r'FY\s*\d{2,4}', line, re.IGNORECASE))
            
            # Look for dollar amounts or billions
            if re.search(r'\$?[\d,]+\.?\d*[BMK]?|\d+\.?\d*B(?:\s|$)', line):
                amounts = re.findall(r'\$?[\d,]+\.?\d*[BMK]?|\d+\.?\d*B(?:\s|$)', line)
                values.extend(amounts)
            
            # Look for percentages
            if '%' in line:
                percentages = re.findall(r'\d+\.?\d*%', line)
                values.extend(percentages)
            
            # Look for labels
            if any(word in line.lower() for word in ['revenue', 'expense', 'margin', 'profit']):
                labels.append(line)
        
        # Build structured output
        if fiscal_years or values:
            structured.append("[TESSERACT OCR EXTRACTION]")
            
            if fiscal_years:
                structured.append(f"Fiscal Years Found: {', '.join(fiscal_years)}")
            
            if values:
                structured.append(f"Values Found: {', '.join(values)}")
            
            if labels:
                structured.append(f"Labels Found: {', '.join(labels)}")
            
            structured.append("\n[RAW TEXT]")
            structured.append(raw_text)
        else:
            structured.append(raw_text)
        
        return '\n'.join(structured)
    
    def process_chart_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process chart image"""
        try:
            # Convert bytes to numpy array
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
            
            # Extract text
            extracted_text = self.extract_text_from_image(image)
            
            return {
                'extracted_text': extracted_text,
                'processing_status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Chart processing failed: {e}")
            return {
                'extracted_text': '',
                'processing_status': f'error: {str(e)}'
            }
    
    def is_available(self) -> bool:
        """Check if processor is available"""
        return self.ocr_available

# Alias for compatibility
AdvancedOCRProcessor = TesseractOCRProcessor