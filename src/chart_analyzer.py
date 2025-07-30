"""
Chart Analysis for Bar Height Estimation
Extracts approximate values when numbers aren't explicitly shown on bars
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import structlog
import re

logger = structlog.get_logger()

class ChartAnalyzer:
    def __init__(self):
        """Initialize chart analyzer"""
        pass
    
    def analyze_bar_chart(self, image: np.ndarray, ocr_text: str) -> Dict:
        """Analyze bar chart to estimate values from heights"""
        try:
            print("[ChartAnalyzer] Starting bar chart analysis")
            
            # Extract chart metadata from OCR
            metadata = self._extract_chart_metadata(ocr_text)
            print(f"[ChartAnalyzer] Metadata: {metadata}")
            
            if not metadata['has_y_axis_scale']:
                print("[ChartAnalyzer] No Y-axis scale found, cannot estimate values")
                return {'estimates': [], 'metadata': metadata}
            
            # Detect chart region and bars
            chart_analysis = self._detect_chart_elements(image, metadata)
            
            # Estimate values based on bar heights
            estimates = self._estimate_bar_values(chart_analysis, metadata)
            
            print(f"[ChartAnalyzer] Generated {len(estimates)} value estimates")
            return {
                'estimates': estimates,
                'metadata': metadata,
                'chart_analysis': chart_analysis
            }
            
        except Exception as e:
            logger.error(f"Chart analysis failed: {e}")
            return {'estimates': [], 'metadata': {}}
    
    def _extract_chart_metadata(self, ocr_text: str) -> Dict:
        """Extract metadata from OCR text"""
        metadata = {
            'fiscal_years': [],
            'y_axis_scale': [],
            'labels': [],
            'has_y_axis_scale': False,
            'scale_max': 0,
            'scale_min': 0
        }
        
        lines = ocr_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract fiscal years
            fy_matches = re.findall(r'FY\s*(\d{2,4})', line, re.IGNORECASE)
            if fy_matches:
                metadata['fiscal_years'].extend([f"FY{fy}" for fy in fy_matches])
            
            # Extract year patterns like "FY23"
            year_matches = re.findall(r'FY\d{2,4}', line, re.IGNORECASE)
            if year_matches:
                metadata['fiscal_years'].extend(year_matches)
            
            # Extract scale numbers (likely Y-axis)
            scale_numbers = re.findall(r'^\d+$', line)
            if scale_numbers:
                metadata['y_axis_scale'].extend([int(n) for n in scale_numbers])
            
            # Extract labels
            if any(keyword in line.lower() for keyword in ['revenue', 'expense', 'margin', 'profit']):
                metadata['labels'].append(line)
        
        # Process scale
        if metadata['y_axis_scale']:
            metadata['y_axis_scale'] = sorted(set(metadata['y_axis_scale']))
            metadata['has_y_axis_scale'] = True
            metadata['scale_max'] = max(metadata['y_axis_scale'])
            metadata['scale_min'] = min(metadata['y_axis_scale'])
        
        # Remove duplicates and sort fiscal years
        metadata['fiscal_years'] = sorted(set(metadata['fiscal_years']))
        
        return metadata
    
    def _detect_chart_elements(self, image: np.ndarray, metadata: Dict) -> Dict:
        """Detect chart elements like bars and axes"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # Detect potential chart area (usually central region)
            chart_region = {
                'left': int(width * 0.15),    # Skip Y-axis labels
                'right': int(width * 0.85),   # Skip legend area
                'top': int(height * 0.15),    # Skip title
                'bottom': int(height * 0.85)  # Skip X-axis labels
            }
            
            # Extract chart area
            chart_area = gray[chart_region['top']:chart_region['bottom'], 
                            chart_region['left']:chart_region['right']]
            
            # Detect vertical bars using edge detection
            bars = self._detect_bars(chart_area, chart_region)
            
            return {
                'chart_region': chart_region,
                'bars': bars,
                'image_dimensions': (width, height)
            }
            
        except Exception as e:
            logger.error(f"Chart element detection failed: {e}")
            return {'bars': [], 'chart_region': {}}
    
    def _detect_bars(self, chart_area: np.ndarray, chart_region: Dict) -> List[Dict]:
        """Detect individual bars in the chart"""
        try:
            # Apply threshold to highlight bars
            _, thresh = cv2.threshold(chart_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bars = []
            chart_height = chart_area.shape[0]
            chart_width = chart_area.shape[1]
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter for bar-like shapes (taller than wide, reasonable size)
                if h > w and h > chart_height * 0.1 and w > chart_width * 0.02:
                    # Calculate relative position and height
                    bar_info = {
                        'x_center': x + w/2,
                        'x_relative': (x + w/2) / chart_width,  # 0-1 position
                        'height_pixels': h,
                        'height_relative': h / chart_height,    # 0-1 height
                        'bottom_y': y + h,
                        'bottom_relative': (y + h) / chart_height
                    }
                    bars.append(bar_info)
            
            # Sort bars by x position (left to right)
            bars = sorted(bars, key=lambda b: b['x_center'])
            
            print(f"[ChartAnalyzer] Detected {len(bars)} potential bars")
            return bars
            
        except Exception as e:
            logger.error(f"Bar detection failed: {e}")
            return []
    
    def _estimate_bar_values(self, chart_analysis: Dict, metadata: Dict) -> List[Dict]:
        """Estimate values based on bar heights and scale"""
        estimates = []
        
        try:
            bars = chart_analysis.get('bars', [])
            fiscal_years = metadata.get('fiscal_years', [])
            scale_max = metadata.get('scale_max', 0)
            scale_min = metadata.get('scale_min', 0)
            
            print(f"[ChartAnalyzer] Processing {len(bars)} bars with scale {scale_min}-{scale_max}")
            
            # Estimate scale unit (billions if scale goes to 90)
            scale_unit = "B" if scale_max >= 50 else "M" if scale_max >= 5 else ""
            scale_unit_name = "billion" if scale_unit == "B" else "million" if scale_unit == "M" else ""
            
            for i, bar in enumerate(bars):
                # Estimate value based on relative height
                estimated_value = scale_min + (bar['height_relative'] * (scale_max - scale_min))
                
                # Match with fiscal year if available
                fiscal_year = fiscal_years[i] if i < len(fiscal_years) else f"Period_{i+1}"
                
                estimate = {
                    'fiscal_year': fiscal_year,
                    'estimated_value': round(estimated_value, 1),
                    'display_value': f"${estimated_value:.1f}{scale_unit}",
                    'confidence': 'estimated_from_bar_height',
                    'method': f'bar_height_relative_to_scale_0_{scale_max}'
                }
                estimates.append(estimate)
                
                print(f"[ChartAnalyzer] {fiscal_year}: ~${estimated_value:.1f}{scale_unit}")
            
            return estimates
            
        except Exception as e:
            logger.error(f"Value estimation failed: {e}")
            return []
    
    def format_estimates_as_text(self, analysis_result: Dict) -> str:
        """Format analysis results as readable text"""
        try:
            estimates = analysis_result.get('estimates', [])
            metadata = analysis_result.get('metadata', {})
            
            if not estimates:
                return ""
            
            output = ["[ESTIMATED VALUES FROM BAR HEIGHTS]"]
            output.append("Note: These are approximate values based on visual bar heights relative to the Y-axis scale")
            output.append("")
            
            # Add scale info
            if metadata.get('has_y_axis_scale'):
                scale_min = metadata.get('scale_min', 0)
                scale_max = metadata.get('scale_max', 0)
                output.append(f"Y-axis scale: {scale_min} to {scale_max} (likely billions)")
                output.append("")
            
            # Add estimates
            for estimate in estimates:
                output.append(f"{estimate['fiscal_year']}: {estimate['display_value']} (estimated)")
            
            output.append("")
            output.append("Methodology: Bar heights measured relative to Y-axis scale")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Failed to format estimates: {e}")
            return ""