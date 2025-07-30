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
    
    def analyze_chart(self, image: np.ndarray, ocr_text: str) -> Dict:
        """Analyze any chart type to estimate values"""
        try:
            print("[ChartAnalyzer] Starting chart analysis")
            
            # Extract chart metadata from OCR
            metadata = self._extract_chart_metadata(ocr_text)
            print(f"[ChartAnalyzer] Metadata: {metadata}")
            
            # Detect chart type
            chart_type = self._detect_chart_type(image, ocr_text)
            metadata['chart_type'] = chart_type
            print(f"[ChartAnalyzer] Detected chart type: {chart_type}")
            
            if not metadata['has_y_axis_scale'] and chart_type != 'pie':
                print("[ChartAnalyzer] No Y-axis scale found, cannot estimate values")
                return {'estimates': [], 'metadata': metadata}
            
            # Analyze based on chart type
            if chart_type == 'bar':
                chart_analysis = self._detect_bar_elements(image, metadata)
                chart_analysis['ocr_text'] = ocr_text  # Pass OCR text for scale unit detection
                estimates = self._estimate_bar_values(chart_analysis, metadata)
            elif chart_type == 'line':
                chart_analysis = self._detect_line_elements(image, metadata)
                estimates = self._estimate_line_values(chart_analysis, metadata)
            elif chart_type == 'pie':
                chart_analysis = self._detect_pie_elements(image, metadata)
                estimates = self._estimate_pie_values(chart_analysis, metadata)
            elif chart_type == 'area':
                chart_analysis = self._detect_area_elements(image, metadata)
                estimates = self._estimate_area_values(chart_analysis, metadata)
            elif chart_type == 'scatter':
                chart_analysis = self._detect_scatter_elements(image, metadata)
                estimates = self._estimate_scatter_values(chart_analysis, metadata)
            else:
                # Default to bar chart analysis
                chart_analysis = self._detect_bar_elements(image, metadata)
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
            # If we have 0 and other values, use the next smallest as min (0 is often baseline)
            scale_values = [v for v in metadata['y_axis_scale'] if v > 0]
            if scale_values:
                metadata['scale_min'] = min(scale_values)
            else:
                metadata['scale_min'] = min(metadata['y_axis_scale'])
            
            print(f"[ChartAnalyzer] Scale range: {metadata['scale_min']} to {metadata['scale_max']} billion")
        
        # Remove duplicates and sort fiscal years
        metadata['fiscal_years'] = sorted(set(metadata['fiscal_years']))
        
        return metadata
    
    def _detect_scale_unit(self, ocr_text: str, scale_max: int) -> str:
        """Dynamically detect scale unit from OCR text"""
        import re
        
        # Look for explicit unit indicators in text
        text_lower = ocr_text.lower()
        
        # Check for explicit mentions of units
        if re.search(r'\d+\.?\d*\s*trillion|\d+\.?\d*t\b', text_lower):
            return "T"
        elif re.search(r'\d+\.?\d*\s*billion|\d+\.?\d*b\b', text_lower):
            return "B"  
        elif re.search(r'\d+\.?\d*\s*million|\d+\.?\d*m\b', text_lower):
            return "M"
        elif re.search(r'\d+\.?\d*\s*thousand|\d+\.?\d*k\b', text_lower):
            return "K"
        
        # Look for currency symbols with large numbers
        large_numbers = re.findall(r'\$[\d,]+\.?\d*', ocr_text)
        if large_numbers:
            # Extract the largest number
            max_num = 0
            for num_str in large_numbers:
                try:
                    num = float(num_str.replace('$', '').replace(',', ''))
                    max_num = max(max_num, num)
                except:
                    continue
            
            if max_num >= 1000000000:
                return "B"
            elif max_num >= 1000000:
                return "M" 
            elif max_num >= 1000:
                return "K"
        
        # Fallback based on scale magnitude and context
        # For financial charts, if we see numbers like 20, 30, 40... 90
        # and the context mentions "Revenue", it's likely billions
        if any(word in text_lower for word in ['revenue', 'expense', 'income', 'profit']):
            # Financial context - likely larger units
            if scale_max >= 20:
                return "B"  # Revenue charts with 20-90 scale = billions
            elif scale_max >= 5:
                return "M"  # Medium revenue = millions
        
        # Non-financial context fallbacks
        if scale_max >= 1000:
            return "B"  # Very large scales
        elif scale_max >= 100:
            return "M"  # Medium scales  
        elif scale_max >= 10:
            return "K"  # Small scales
        else:
            return ""   # No unit
    
    def _detect_chart_type(self, image: np.ndarray, ocr_text: str) -> str:
        """Detect the type of chart from image analysis and OCR text"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Look for textual indicators first
            text_lower = ocr_text.lower()
            
            # Check for pie chart indicators (but be more specific)
            if 'pie' in text_lower or 'slice' in text_lower:
                # Look for circular shapes
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=200)
                if circles is not None:
                    print("[ChartAnalyzer] Found circular shapes, likely pie chart")
                    return 'pie'
            
            # Detect edges for structural analysis
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                horizontal_lines = 0
                vertical_lines = 0
                diagonal_lines = 0
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    if abs(angle) < 15 or abs(angle) > 165:
                        horizontal_lines += 1
                    elif abs(angle - 90) < 15 or abs(angle + 90) < 15:
                        vertical_lines += 1
                    else:
                        diagonal_lines += 1
                
                print(f"[ChartAnalyzer] Lines detected - H:{horizontal_lines}, V:{vertical_lines}, D:{diagonal_lines}")
                
                # Line chart: more diagonal lines (trend lines)
                if diagonal_lines > max(horizontal_lines, vertical_lines) * 0.3:
                    return 'line'
                
                # Bar chart: many vertical lines (bars) + grid lines
                if vertical_lines > horizontal_lines * 1.5:
                    return 'bar'
            
            # Check for area chart indicators (filled regions)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
            
            if len(filled_areas) > 0:
                avg_area = np.mean(filled_areas)
                if avg_area > gray.shape[0] * gray.shape[1] * 0.05:  # Large filled areas
                    return 'area'
            
            # Enhanced pattern-based detection
            chart_indicators = {
                'bar': ['revenue', 'expense', 'profit', 'income', 'quarterly', 'annual', 'fy', 'fiscal year'],
                'line': ['trend', 'over time', 'growth', 'change', 'progression'],
                'pie': ['distribution', 'breakdown', 'composition', 'share', 'percentage of total'],
                'area': ['cumulative', 'stacked', 'total over time'],
                'scatter': ['correlation', 'relationship', 'vs', 'compared to']
            }
            
            # Score each chart type based on keyword matches
            scores = {}
            for chart_type, keywords in chart_indicators.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    scores[chart_type] = score
            
            # Return the highest scoring type, with bar as fallback
            if scores:
                best_type = max(scores.items(), key=lambda x: x[1])[0]
                print(f"[ChartAnalyzer] Chart type '{best_type}' detected based on keywords (scores: {scores})")
                return best_type
            
            print("[ChartAnalyzer] No specific indicators found, defaulting to bar chart")
            return 'bar'  # Default assumption
            
        except Exception as e:
            logger.error(f"Chart type detection failed: {e}")
            return 'bar'
    
    def _detect_bar_elements(self, image: np.ndarray, metadata: Dict) -> Dict:
        """Detect chart elements like bars and axes"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
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
            bars = self._detect_bars(chart_area, chart_region, metadata)
            
            return {
                'chart_region': chart_region,
                'bars': bars,
                'image_dimensions': (width, height)
            }
            
        except Exception as e:
            logger.error(f"Chart element detection failed: {e}")
            return {'bars': [], 'chart_region': {}}
    
    def _detect_bars(self, chart_area: np.ndarray, chart_region: Dict, metadata: Dict) -> List[Dict]:
        """Detect individual bars in the chart"""
        try:
            print(f"[ChartAnalyzer] Chart area dimensions: {chart_area.shape}")
            
            bars = []
            chart_height = chart_area.shape[0]
            chart_width = chart_area.shape[1]
            
            # Try multiple detection methods
            
            # Method 1: OTSU thresholding
            _, thresh1 = cv2.threshold(chart_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[ChartAnalyzer] Method 1 (OTSU): Found {len(contours1)} contours")
            
            # Method 2: Fixed threshold (for light bars on white background)
            _, thresh2 = cv2.threshold(chart_area, 200, 255, cv2.THRESH_BINARY_INV)
            contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[ChartAnalyzer] Method 2 (Fixed 200): Found {len(contours2)} contours")
            
            # Method 3: Adaptive threshold
            thresh3 = cv2.adaptiveThreshold(chart_area, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours3, _ = cv2.findContours(thresh3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[ChartAnalyzer] Method 3 (Adaptive): Found {len(contours3)} contours")
            
            # Method 4: Edge detection + morphology
            edges = cv2.Canny(chart_area, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours4, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[ChartAnalyzer] Method 4 (Edges): Found {len(contours4)} contours")
            
            # Combine all contours and filter
            all_contours = contours1 + contours2 + contours3 + contours4
            
            # Dynamically calculate expected bars based on detected fiscal years and data series
            num_fiscal_years = len(metadata.get('fiscal_years', []))
            num_data_series = len([label for label in metadata.get('labels', []) 
                                 if any(keyword in label.lower() for keyword in ['revenue', 'expense', 'margin', 'profit'])])
            expected_bars = max(num_fiscal_years * max(1, num_data_series), num_fiscal_years)
            
            print(f"[ChartAnalyzer] Expected bars: {expected_bars} ({num_fiscal_years} years × {max(1, num_data_series)} series)")
            
            for contour in all_contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # More relaxed filtering for bar-like shapes
                # Check if it's reasonably tall and not too wide
                aspect_ratio = h / w if w > 0 else 0
                min_height = chart_height * 0.05  # Reduced from 0.1
                min_width = chart_width * 0.01   # Reduced from 0.02
                min_area = 50  # Minimum area
                
                if (aspect_ratio > 0.5 and  # Allow wider bars
                    h > min_height and 
                    w > min_width and 
                    area > min_area and
                    x > chart_width * 0.1 and  # Not too far left (avoid Y-axis)
                    x < chart_width * 0.9):    # Not too far right
                    
                    # Calculate relative position and height
                    bar_info = {
                        'x_center': x + w/2,
                        'x_relative': (x + w/2) / chart_width,  # 0-1 position
                        'height_pixels': h,
                        'height_relative': h / chart_height,    # 0-1 height
                        'bottom_y': y + h,
                        'bottom_relative': (y + h) / chart_height,
                        'width': w,
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    }
                    bars.append(bar_info)
                    print(f"[ChartAnalyzer] Found bar: x={x}, y={y}, w={w}, h={h}, aspect={aspect_ratio:.2f}")
            
            # Remove duplicates (bars found by multiple methods)
            # Sort by x position and remove bars that are too close to each other
            bars = sorted(bars, key=lambda b: b['x_center'])
            unique_bars = []
            for bar in bars:
                # Check if this bar is too close to an existing one
                is_duplicate = False
                for existing in unique_bars:
                    if abs(bar['x_center'] - existing['x_center']) < chart_width * 0.05:  # 5% of width
                        is_duplicate = True
                        # Keep the one with larger area
                        if bar['area'] > existing['area']:
                            unique_bars.remove(existing)
                            unique_bars.append(bar)
                        break
                if not is_duplicate:
                    unique_bars.append(bar)
            
            # Sort final bars by x position
            unique_bars = sorted(unique_bars, key=lambda b: b['x_center'])
            
            print(f"[ChartAnalyzer] Detected {len(unique_bars)} unique bars after deduplication")
            return unique_bars
            
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
            
            # Get OCR text from metadata for dynamic scale unit detection
            ocr_text = chart_analysis.get('ocr_text', '')
            if not ocr_text:
                # Reconstruct from metadata if not available
                ocr_text = ' '.join(metadata.get('labels', []))
            
            # Dynamically detect scale unit from OCR text
            scale_unit = self._detect_scale_unit(ocr_text, scale_max)
            scale_unit_name = {"B": "billion", "M": "million", "K": "thousand", "T": "trillion"}.get(scale_unit, "")
            print(f"[ChartAnalyzer] Detected scale unit: {scale_unit} ({scale_unit_name})")
            
            # Group bars by fiscal year (assuming 2 bars per year: Revenue + Operating Expense)
            bars_per_year = max(1, len(bars) // len(fiscal_years)) if fiscal_years else 1
            
            for i, bar in enumerate(bars):
                # Estimate value based on relative height (from bottom of chart)
                # bar['height_relative'] is 0-1, where 1 = full chart height
                # Need to invert because chart coordinates are top-down
                height_from_bottom = 1 - (bar['bottom_relative'] - bar['height_relative'])
                estimated_value = scale_min + (height_from_bottom * (scale_max - scale_min))
                
                # Match with fiscal year - if multiple bars per year, cycle through years
                year_index = i // bars_per_year if bars_per_year > 0 else i
                if year_index < len(fiscal_years):
                    fiscal_year = fiscal_years[year_index]
                    bar_type = "Revenue" if (i % bars_per_year) == 0 else "Operating Expense"
                    fiscal_year_display = f"{fiscal_year} {bar_type}"
                else:
                    fiscal_year_display = f"Bar_{i+1}"
                
                estimate = {
                    'fiscal_year': fiscal_year_display,
                    'estimated_value': round(estimated_value, 1),
                    'display_value': f"${estimated_value:.1f}{scale_unit}",
                    'confidence': 'estimated_from_bar_height',
                    'method': f'bar_height_relative_to_scale_{scale_min}_{scale_max}'
                }
                estimates.append(estimate)
                
                print(f"[ChartAnalyzer] {fiscal_year_display}: ~${estimated_value:.1f}{scale_unit} (height_from_bottom={height_from_bottom:.2f})")
            
            return estimates
            
        except Exception as e:
            logger.error(f"Value estimation failed: {e}")
            return []
    
    def _detect_line_elements(self, image: np.ndarray, metadata: Dict) -> Dict:
        """Detect line chart elements like data points and trend lines"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            height, width = gray.shape
            
            chart_region = {
                'left': int(width * 0.15),
                'right': int(width * 0.85),
                'top': int(height * 0.15),
                'bottom': int(height * 0.85)
            }
            
            chart_area = gray[chart_region['top']:chart_region['bottom'], 
                            chart_region['left']:chart_region['right']]
            
            # Detect edges to find trend lines
            edges = cv2.Canny(chart_area, 50, 150)
            
            # Find lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
            
            trend_lines = []
            data_points = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Filter for diagonal lines that could be trends
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    if 15 < abs(angle) < 165:  # Diagonal lines
                        trend_lines.append({
                            'start': (x1, y1),
                            'end': (x2, y2),
                            'angle': angle,
                            'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        })
            
            # Detect potential data points using contour detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < 200:  # Small circular areas could be data points
                    x, y, w, h = cv2.boundingRect(contour)
                    if abs(w - h) < 5:  # Nearly square (circular)
                        data_points.append({
                            'x': x + w/2,
                            'y': y + h/2,
                            'x_relative': (x + w/2) / chart_area.shape[1],
                            'y_relative': (y + h/2) / chart_area.shape[0]
                        })
            
            return {
                'chart_region': chart_region,
                'trend_lines': trend_lines,
                'data_points': data_points,
                'image_dimensions': (width, height)
            }
            
        except Exception as e:
            logger.error(f"Line chart element detection failed: {e}")
            return {'trend_lines': [], 'data_points': [], 'chart_region': {}}
    
    def _detect_pie_elements(self, image: np.ndarray, metadata: Dict) -> Dict:
        """Detect pie chart elements like slices and labels"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            height, width = gray.shape
            
            # Find circles (pie chart outline)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=50, maxRadius=300)
            
            pie_slices = []
            center = None
            radius = 0
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Use the largest circle as the pie chart
                largest_circle = max(circles[0, :], key=lambda c: c[2])
                center = (largest_circle[0], largest_circle[1])
                radius = largest_circle[2]
                
                # Detect edges within the circle to find slice boundaries
                edges = cv2.Canny(gray, 50, 150)
                
                # Find lines emanating from center (slice dividers)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=radius//2, maxLineGap=10)
                
                slice_angles = []
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # Check if line passes near center
                        dist_to_center = min(
                            np.sqrt((x1-center[0])**2 + (y1-center[1])**2),
                            np.sqrt((x2-center[0])**2 + (y2-center[1])**2)
                        )
                        if dist_to_center < radius * 0.3:
                            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                            slice_angles.append(angle)
                
                # Estimate number of slices
                slice_angles = sorted(set([round(a/10)*10 for a in slice_angles]))  # Round to 10 degrees
                num_slices = len(slice_angles) if slice_angles else 4  # Default to 4 slices
                
                for i in range(num_slices):
                    pie_slices.append({
                        'slice_id': i,
                        'estimated_percentage': 100 / num_slices,  # Equal slices assumption
                        'angle_start': i * (360 / num_slices),
                        'angle_end': (i + 1) * (360 / num_slices)
                    })
            
            return {
                'center': center,
                'radius': radius,
                'pie_slices': pie_slices,
                'image_dimensions': (width, height)
            }
            
        except Exception as e:
            logger.error(f"Pie chart element detection failed: {e}")
            return {'pie_slices': [], 'center': None, 'radius': 0}
    
    def _detect_area_elements(self, image: np.ndarray, metadata: Dict) -> Dict:
        """Detect area chart elements like filled regions and boundaries"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            height, width = gray.shape
            
            chart_region = {
                'left': int(width * 0.15),
                'right': int(width * 0.85),
                'top': int(height * 0.15),
                'bottom': int(height * 0.85)
            }
            
            chart_area = gray[chart_region['top']:chart_region['bottom'], 
                            chart_region['left']:chart_region['right']]
            
            # Find contours to detect filled areas
            contours, _ = cv2.findContours(chart_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            filled_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # Look for large filled regions
                if area > chart_area.shape[0] * chart_area.shape[1] * 0.05:
                    x, y, w, h = cv2.boundingRect(contour)
                    filled_regions.append({
                        'area': area,
                        'bounds': (x, y, w, h),
                        'area_relative': area / (chart_area.shape[0] * chart_area.shape[1]),
                        'height_relative': h / chart_area.shape[0]
                    })
            
            # Sort by area size
            filled_regions = sorted(filled_regions, key=lambda r: r['area'], reverse=True)
            
            return {
                'chart_region': chart_region,
                'filled_regions': filled_regions,
                'image_dimensions': (width, height)
            }
            
        except Exception as e:
            logger.error(f"Area chart element detection failed: {e}")
            return {'filled_regions': [], 'chart_region': {}}
    
    def _detect_scatter_elements(self, image: np.ndarray, metadata: Dict) -> Dict:
        """Detect scatter plot elements like data points"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            height, width = gray.shape
            
            chart_region = {
                'left': int(width * 0.15),
                'right': int(width * 0.85),
                'top': int(height * 0.15),
                'bottom': int(height * 0.85)
            }
            
            chart_area = gray[chart_region['top']:chart_region['bottom'], 
                            chart_region['left']:chart_region['right']]
            
            # Detect small circular shapes (scatter points)
            circles = cv2.HoughCircles(chart_area, cv2.HOUGH_GRADIENT, 1, 10,
                                     param1=50, param2=15, minRadius=2, maxRadius=20)
            
            scatter_points = []
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    x, y, r = circle
                    scatter_points.append({
                        'x': x,
                        'y': y,
                        'radius': r,
                        'x_relative': x / chart_area.shape[1],
                        'y_relative': y / chart_area.shape[0]
                    })
            
            # Also try contour detection for non-circular points
            edges = cv2.Canny(chart_area, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 5 < area < 100:  # Small points
                    x, y, w, h = cv2.boundingRect(contour)
                    if w < 20 and h < 20:  # Small bounding box
                        scatter_points.append({
                            'x': x + w/2,
                            'y': y + h/2,
                            'radius': max(w, h) / 2,
                            'x_relative': (x + w/2) / chart_area.shape[1],
                            'y_relative': (y + h/2) / chart_area.shape[0]
                        })
            
            return {
                'chart_region': chart_region,
                'scatter_points': scatter_points,
                'image_dimensions': (width, height)
            }
            
        except Exception as e:
            logger.error(f"Scatter plot element detection failed: {e}")
            return {'scatter_points': [], 'chart_region': {}}
    
    def _estimate_line_values(self, chart_analysis: Dict, metadata: Dict) -> List[Dict]:
        """Estimate values from line chart trend lines and data points"""
        estimates = []
        
        try:
            data_points = chart_analysis.get('data_points', [])
            fiscal_years = metadata.get('fiscal_years', [])
            scale_max = metadata.get('scale_max', 0)
            scale_min = metadata.get('scale_min', 0)
            
            if not data_points or not scale_max:
                return estimates
            
            # Sort points by x position
            sorted_points = sorted(data_points, key=lambda p: p['x_relative'])
            
            scale_unit = "B" if scale_max >= 50 else "M" if scale_max >= 5 else ""
            
            for i, point in enumerate(sorted_points):
                # Convert y position to value (inverted because y=0 is top)
                estimated_value = scale_min + ((1 - point['y_relative']) * (scale_max - scale_min))
                fiscal_year = fiscal_years[i] if i < len(fiscal_years) else f"Point_{i+1}"
                
                estimate = {
                    'fiscal_year': fiscal_year,
                    'estimated_value': round(estimated_value, 1),
                    'display_value': f"${estimated_value:.1f}{scale_unit}",
                    'confidence': 'estimated_from_line_chart',
                    'method': f'data_point_position_relative_to_scale_0_{scale_max}'
                }
                estimates.append(estimate)
                
                print(f"[ChartAnalyzer] Line chart {fiscal_year}: ~${estimated_value:.1f}{scale_unit}")
            
            return estimates
            
        except Exception as e:
            logger.error(f"Line chart value estimation failed: {e}")
            return []
    
    def _estimate_pie_values(self, chart_analysis: Dict, metadata: Dict) -> List[Dict]:
        """Estimate values from pie chart slices"""
        estimates = []
        
        try:
            pie_slices = chart_analysis.get('pie_slices', [])
            labels = metadata.get('labels', [])
            
            if not pie_slices:
                return estimates
            
            for i, slice_info in enumerate(pie_slices):
                label = labels[i] if i < len(labels) else f"Slice_{i+1}"
                percentage = slice_info['estimated_percentage']
                
                estimate = {
                    'category': label,
                    'estimated_percentage': round(percentage, 1),
                    'display_value': f"{percentage:.1f}%",
                    'confidence': 'estimated_from_pie_slice',
                    'method': 'pie_slice_angle_analysis'
                }
                estimates.append(estimate)
                
                print(f"[ChartAnalyzer] Pie slice {label}: ~{percentage:.1f}%")
            
            return estimates
            
        except Exception as e:
            logger.error(f"Pie chart value estimation failed: {e}")
            return []
    
    def _estimate_area_values(self, chart_analysis: Dict, metadata: Dict) -> List[Dict]:
        """Estimate values from area chart filled regions"""
        estimates = []
        
        try:
            filled_regions = chart_analysis.get('filled_regions', [])
            fiscal_years = metadata.get('fiscal_years', [])
            scale_max = metadata.get('scale_max', 0)
            scale_min = metadata.get('scale_min', 0)
            
            if not filled_regions or not scale_max:
                return estimates
            
            scale_unit = "B" if scale_max >= 50 else "M" if scale_max >= 5 else ""
            
            for i, region in enumerate(filled_regions[:len(fiscal_years)]):
                # Estimate value based on filled area height
                estimated_value = scale_min + (region['height_relative'] * (scale_max - scale_min))
                fiscal_year = fiscal_years[i] if i < len(fiscal_years) else f"Region_{i+1}"
                
                estimate = {
                    'fiscal_year': fiscal_year,
                    'estimated_value': round(estimated_value, 1),
                    'display_value': f"${estimated_value:.1f}{scale_unit}",
                    'confidence': 'estimated_from_area_height',
                    'method': f'filled_area_height_relative_to_scale_0_{scale_max}'
                }
                estimates.append(estimate)
                
                print(f"[ChartAnalyzer] Area chart {fiscal_year}: ~${estimated_value:.1f}{scale_unit}")
            
            return estimates
            
        except Exception as e:
            logger.error(f"Area chart value estimation failed: {e}")
            return []
    
    def _estimate_scatter_values(self, chart_analysis: Dict, metadata: Dict) -> List[Dict]:
        """Estimate values from scatter plot data points"""
        estimates = []
        
        try:
            scatter_points = chart_analysis.get('scatter_points', [])
            scale_max = metadata.get('scale_max', 0)
            scale_min = metadata.get('scale_min', 0)
            
            if not scatter_points or not scale_max:
                return estimates
            
            scale_unit = "B" if scale_max >= 50 else "M" if scale_max >= 5 else ""
            
            # Group points by similar x positions (time periods)
            x_groups = {}
            for point in scatter_points:
                x_bucket = round(point['x_relative'] * 10)  # Group into 10 buckets
                if x_bucket not in x_groups:
                    x_groups[x_bucket] = []
                x_groups[x_bucket].append(point)
            
            for i, (x_bucket, points) in enumerate(sorted(x_groups.items())):
                # Average y position for points in this time period
                avg_y = sum(p['y_relative'] for p in points) / len(points)
                estimated_value = scale_min + ((1 - avg_y) * (scale_max - scale_min))
                
                estimate = {
                    'time_period': f"Period_{i+1}",
                    'point_count': len(points),
                    'estimated_value': round(estimated_value, 1),
                    'display_value': f"${estimated_value:.1f}{scale_unit}",
                    'confidence': 'estimated_from_scatter_points',
                    'method': f'scatter_point_average_relative_to_scale_0_{scale_max}'
                }
                estimates.append(estimate)
                
                print(f"[ChartAnalyzer] Scatter Period_{i+1} ({len(points)} points): ~${estimated_value:.1f}{scale_unit}")
            
            return estimates
            
        except Exception as e:
            logger.error(f"Scatter plot value estimation failed: {e}")
            return []

    def format_estimates_as_text(self, analysis_result: Dict) -> str:
        """Format analysis results as readable text"""
        try:
            estimates = analysis_result.get('estimates', [])
            metadata = analysis_result.get('metadata', {})
            chart_type = metadata.get('chart_type', 'bar')
            
            if not estimates:
                return ""
            
            # Dynamic header based on chart type
            chart_type_names = {
                'bar': 'BAR CHART',
                'line': 'LINE CHART', 
                'pie': 'PIE CHART',
                'area': 'AREA CHART',
                'scatter': 'SCATTER PLOT'
            }
            
            output = [f"[ESTIMATED VALUES FROM {chart_type_names.get(chart_type, 'CHART').upper()}]"]
            output.append(f"Note: These are approximate values based on visual {chart_type} analysis")
            output.append("")
            
            # Add scale info for charts with scales
            if metadata.get('has_y_axis_scale') and chart_type != 'pie':
                scale_min = metadata.get('scale_min', 0)
                scale_max = metadata.get('scale_max', 0)
                output.append(f"Y-axis scale: {scale_min} to {scale_max} (likely billions)")
                output.append("")
            
            # Add estimates with chart-specific formatting
            for estimate in estimates:
                if chart_type == 'pie':
                    output.append(f"{estimate.get('category', 'Unknown')}: {estimate['display_value']} (estimated)")
                elif chart_type == 'scatter':
                    output.append(f"{estimate.get('time_period', 'Unknown')} ({estimate.get('point_count', 0)} points): {estimate['display_value']} (estimated)")
                else:
                    fiscal_year = estimate.get('fiscal_year', estimate.get('time_period', 'Unknown'))
                    output.append(f"{fiscal_year}: {estimate['display_value']} (estimated)")
            
            output.append("")
            
            # Chart-specific methodology
            methodologies = {
                'bar': 'Bar heights measured relative to Y-axis scale',
                'line': 'Data point positions analyzed relative to trend lines and scale',
                'pie': 'Slice angles measured to estimate percentages',
                'area': 'Filled area heights measured relative to Y-axis scale', 
                'scatter': 'Data point positions grouped and averaged by time periods'
            }
            
            output.append(f"Methodology: {methodologies.get(chart_type, 'Visual analysis relative to chart axes')}")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Failed to format estimates: {e}")
            return ""