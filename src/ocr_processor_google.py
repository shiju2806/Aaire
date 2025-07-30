"""
Google Cloud Vision OCR Processor - Premium Feature
High accuracy OCR for financial charts and documents
"""

import base64
import json
from typing import Dict, List, Optional, Any
import structlog
import os

logger = structlog.get_logger()

class GoogleVisionOCRProcessor:
    def __init__(self):
        """Initialize Google Cloud Vision OCR processor"""
        self.client = None
        self.available = self._initialize_client()
    
    def _initialize_client(self) -> bool:
        """Initialize Google Cloud Vision client"""
        try:
            from google.cloud import vision
            
            # Check for credentials
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if not credentials_path:
                logger.warning("Google Cloud Vision: GOOGLE_APPLICATION_CREDENTIALS not set")
                return False
            
            if not os.path.exists(credentials_path):
                logger.warning(f"Google Cloud Vision: Credentials file not found: {credentials_path}")
                return False
            
            self.client = vision.ImageAnnotatorClient()
            logger.info("Google Cloud Vision OCR processor initialized")
            return True
            
        except ImportError:
            logger.info("Google Cloud Vision not available - install with: pip install google-cloud-vision")
            return False
        except Exception as e:
            logger.warning(f"Google Cloud Vision initialization failed: {e}")
            return False
    
    def extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text using Google Cloud Vision API"""
        try:
            if not self.available:
                return ""
            
            from google.cloud import vision
            
            # Create image object
            image = vision.Image(content=image_data)
            
            # Perform text detection
            response = self.client.text_detection(image=image)
            texts = response.text_annotations
            
            if texts:
                # Return the full text (first annotation contains all text)
                full_text = texts[0].description
                
                print(f"[GoogleVision] Extracted {len(full_text)} characters")
                print(f"[GoogleVision] Text preview: {full_text[:200]}...")
                
                return full_text
            else:
                print("[GoogleVision] No text detected")
                return ""
                
        except Exception as e:
            logger.error(f"Google Vision text extraction failed: {e}")
            return ""
    
    def process_chart_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process chart image with Google Cloud Vision"""
        try:
            extracted_text = self.extract_text_from_image(image_data)
            
            return {
                'extracted_text': extracted_text,
                'processing_status': 'success',
                'processor': 'google_cloud_vision'
            }
            
        except Exception as e:
            logger.error(f"Google Vision chart processing failed: {e}")
            return {
                'extracted_text': '',
                'processing_status': f'error: {str(e)}',
                'processor': 'google_cloud_vision'
            }
    
    def is_available(self) -> bool:
        """Check if processor is available"""
        return self.available

# Optional: Add to requirements-premium.txt
# google-cloud-vision==3.4.4