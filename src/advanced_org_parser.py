#!/usr/bin/env python3
"""
Advanced Organizational Chart Parser - Future-proof solution
Handles any organizational chart format by understanding common patterns
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class PersonRecord:
    """Clean person record with properly separated fields"""
    name: str
    title: str
    department: str
    hierarchy_level: str
    confidence: float
    source_info: Dict[str, Any]
    
class AdvancedOrgParser:
    """
    Future-proof organizational chart parser that can handle:
    - Any text layout within boxes
    - Multiple name/title formats
    - Various hierarchy patterns
    - Different department structures
    """
    
    def __init__(self):
        # Hierarchy levels (order matters - most senior first)
        self.hierarchy_levels = [
            'MVP', 'CEO', 'President', 'Chief',
            'VP', 'Vice President', 
            'AVP', 'Assistant Vice President',
            'Director', 'Senior Director',
            'Manager', 'Senior Manager',
            'Lead', 'Team Lead', 'Senior Lead',
            'Senior', 'Principal',
            'Analyst', 'Senior Analyst',
            'Accountant', 'Senior Accountant', 'Financial Accountant',
            'Specialist', 'Coordinator', 'Associate',
            'Intern', 'Apprentice', 'Trainee'
        ]
        
        # Common department patterns
        self.department_patterns = [
            r'Financial\s+Reporting\s*(?:&|and)?\s*Tax',
            r'Financial\s+Planning\s*(?:&|and)?\s*Analysis',
            r'Financial\s+Reporting',
            r'Financial\s+Management',
            r'Financial\s+Operations',
            r'Tax(?:\s+Management)?',
            r'Treasury(?:\s+Management)?',
            r'Accounting\s+Policy',
            r'Internal\s+Control(?:\s+and)?\s+Financial\s+Reporting|ICFR',
            r'Payroll(?:\s*(?:&|and)?\s*Benefits)?',
            r'Finance(?:\s+Team)?',
            r'Audit(?:\s+Team)?',
            r'Risk\s+Management',
            r'Compliance'
        ]
        
        # Name patterns (proper nouns)
        self.name_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z\-]+(?:\s+[A-Z][a-z\-]+)*)',  # First Middle? Last (multiple parts)
            r'([A-Z][a-z]+\s+[A-Z][a-z]+-[A-Z][a-z]+)',  # Hyphenated names
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z\']+){1,4})'     # Multiple names with apostrophes
        ]
        
        # Title indicators
        self.title_indicators = [
            'Manager', 'Director', 'Analyst', 'Accountant', 'Specialist',
            'Lead', 'VP', 'AVP', 'MVP', 'Senior', 'Principal', 'Coordinator',
            'Intern', 'Trainee', 'Associate', 'Chief', 'President'
        ]
    
    def parse_spatial_cluster(self, cluster_text: str, coordinates: Tuple[float, float, float, float]) -> Optional[PersonRecord]:
        """
        Parse a single spatial cluster (box) to extract person information
        
        Args:
            cluster_text: Raw text from the spatial cluster
            coordinates: (x0, y0, x1, y1) bounding box
            
        Returns:
            PersonRecord with properly separated name, title, department
        """
        
        if not cluster_text or len(cluster_text.strip()) < 3:
            return None
        
        # Clean and normalize text
        text = self._clean_text(cluster_text)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        logger.debug(f"Parsing cluster with {len(lines)} lines", text=text[:100])
        
        # Strategy 1: Try structured parsing (most common)
        result = self._parse_structured_format(lines)
        if result and result.confidence > 0.7:
            result.source_info = {'coordinates': coordinates, 'method': 'structured'}
            return result
        
        # Strategy 2: Try pattern-based parsing
        result = self._parse_pattern_based(text)
        if result and result.confidence > 0.6:
            result.source_info = {'coordinates': coordinates, 'method': 'pattern'}
            return result
        
        # Strategy 3: Fallback with line analysis
        result = self._parse_line_analysis(lines)
        if result:
            result.source_info = {'coordinates': coordinates, 'method': 'fallback'}
            return result
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for parsing - preserve line structure"""
        # Fix common OCR issues but preserve newlines
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space before capitals
        text = re.sub(r'([,&])([A-Z])', r'\1 \2', text)   # Space after punctuation
        # Clean up each line but preserve structure
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove extra whitespace within each line
            line = re.sub(r'\s+', ' ', line.strip())
            if line:  # Only keep non-empty lines
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    def _parse_structured_format(self, lines: List[str]) -> Optional[PersonRecord]:
        """
        Parse common structured formats:
        Line 1: Title, Department
        Line 2: Name
        
        Or:
        Line 1: Name
        Line 2: Title, Department
        """
        
        if len(lines) < 2:
            return None
        
        # Try format: Title/Department first, then Name
        for name_line_idx in range(1, len(lines)):
            name_candidate = lines[name_line_idx]
            title_dept_lines = lines[:name_line_idx] + lines[name_line_idx+1:]
            
            if self._looks_like_name(name_candidate):
                name = self._extract_name(name_candidate)
                title, department = self._extract_title_department(' '.join(title_dept_lines))
                
                if name and (title or department):
                    hierarchy = self._determine_hierarchy_level(title)
                    confidence = self._calculate_confidence(name, title, department)
                    
                    return PersonRecord(
                        name=name,
                        title=title or 'Not specified',
                        department=department or 'Not specified',
                        hierarchy_level=hierarchy,
                        confidence=confidence,
                        source_info={}
                    )
        
        return None
    
    def _parse_pattern_based(self, text: str) -> Optional[PersonRecord]:
        """Parse using regex patterns for common formats"""
        
        # Pattern 1: "Title, Department Name"
        pattern1 = r'([A-Z][^,]+),\s*([^A-Z]*[A-Z][a-z]+(?:\s+[A-Z][a-z\']+)*)'
        match1 = re.search(pattern1, text)
        if match1:
            title_dept = match1.group(1).strip()
            name = match1.group(2).strip()
            
            title, department = self._extract_title_department(title_dept)
            if self._looks_like_name(name):
                hierarchy = self._determine_hierarchy_level(title)
                confidence = 0.8
                
                return PersonRecord(
                    name=name,
                    title=title or 'Not specified',
                    department=department or 'Not specified', 
                    hierarchy_level=hierarchy,
                    confidence=confidence,
                    source_info={}
                )
        
        # Pattern 2: "Name Title, Department"
        pattern2 = r'([A-Z][a-z]+(?:\s+[A-Z][a-z\']+)*)\s+([A-Z][^,]+?)(?:,\s*(.+))?'
        match2 = re.search(pattern2, text)
        if match2 and self._looks_like_name(match2.group(1)):
            name = match2.group(1).strip()
            title_part = match2.group(2).strip()
            dept_part = match2.group(3).strip() if match2.group(3) else ''
            
            title, department = self._extract_title_department(f"{title_part} {dept_part}")
            hierarchy = self._determine_hierarchy_level(title)
            confidence = 0.7
            
            return PersonRecord(
                name=name,
                title=title or title_part,
                department=department or 'Not specified',
                hierarchy_level=hierarchy,
                confidence=confidence,
                source_info={}
            )
        
        return None
    
    def _parse_line_analysis(self, lines: List[str]) -> Optional[PersonRecord]:
        """Fallback: analyze each line to determine most likely name/title"""
        
        if not lines:
            return None
        
        # Score each line for likelihood of being name vs title vs department
        line_scores = []
        for line in lines:
            scores = {
                'name_score': self._score_as_name(line),
                'title_score': self._score_as_title(line), 
                'dept_score': self._score_as_department(line),
                'text': line
            }
            line_scores.append(scores)
        
        # Find best name candidate
        name_line = max(line_scores, key=lambda x: x['name_score'])
        if name_line['name_score'] < 0.3:
            return None
        
        name = self._extract_name(name_line['text'])
        
        # Find best title/department from remaining lines
        remaining_lines = [line for line in lines if line != name_line['text']]
        title_dept_text = ' '.join(remaining_lines)
        title, department = self._extract_title_department(title_dept_text)
        
        hierarchy = self._determine_hierarchy_level(title)
        confidence = 0.5  # Lower confidence for fallback
        
        return PersonRecord(
            name=name or 'Unknown',
            title=title or 'Not specified',
            department=department or 'Not specified',
            hierarchy_level=hierarchy,
            confidence=confidence,
            source_info={}
        )
    
    def _looks_like_name(self, text: str) -> bool:
        """Check if text looks like a person's name"""
        if not text or len(text) < 2:
            return False
        
        # Must start with capital letter
        if not text[0].isupper():
            return False
        
        # Check for name patterns
        for pattern in self.name_patterns:
            if re.match(pattern, text):
                return True
        
        # Additional heuristics
        words = text.split()
        if len(words) == 2 and all(word[0].isupper() and word[1:].islower() for word in words):
            return True
        
        return False
    
    def _score_as_name(self, text: str) -> float:
        """Score how likely text is to be a person's name (0.0 to 1.0)"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Proper name pattern
        if self._looks_like_name(text):
            score += 0.6
        
        # Length indicators
        words = text.split()
        if 2 <= len(words) <= 4:
            score += 0.2
        
        # No title indicators
        if not any(indicator.lower() in text.lower() for indicator in self.title_indicators):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_as_title(self, text: str) -> float:
        """Score how likely text is to be a job title"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Contains title indicators
        for indicator in self.title_indicators:
            if indicator.lower() in text.lower():
                score += 0.4
                break
        
        # Contains hierarchy levels
        for level in self.hierarchy_levels:
            if level.lower() in text.lower():
                score += 0.3
                break
        
        # Patterns like "Manager, Department"
        if ',' in text:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_as_department(self, text: str) -> float:
        """Score how likely text is to be a department"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Contains department patterns
        for pattern in self.department_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.6
                break
        
        # Common department words
        dept_words = ['financial', 'reporting', 'tax', 'analysis', 'management', 'operations']
        if any(word in text.lower() for word in dept_words):
            score += 0.3
        
        return min(score, 1.0)
    
    def _extract_name(self, text: str) -> str:
        """Extract clean name from text"""
        # Try name patterns
        for pattern in self.name_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Fallback: take first reasonable name-like part
        words = text.split()
        name_words = []
        for word in words:
            if word[0].isupper() and len(word) > 1:
                name_words.append(word)
            else:
                break
        
        return ' '.join(name_words) if name_words else text.strip()
    
    def _extract_title_department(self, text: str) -> Tuple[str, str]:
        """Extract title and department from combined text"""
        
        # Find department first (more specific patterns)
        department = ''
        for pattern in self.department_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                department = match.group(0)
                # Remove department from text to get title
                text = text.replace(department, '').strip()
                break
        
        # Clean remaining text as title
        title = text.strip(' ,&')
        
        # If title is empty, try to extract from original
        if not title and department:
            # Look for title indicators before department
            for indicator in self.title_indicators:
                if indicator.lower() in text.lower():
                    title = indicator
                    break
        
        return title, department
    
    def _determine_hierarchy_level(self, title: str) -> str:
        """Determine hierarchy level from title"""
        if not title:
            return 'Unknown'
        
        title_lower = title.lower()
        
        # Direct matches first
        for level in self.hierarchy_levels:
            if level.lower() in title_lower:
                return level
        
        # Pattern matching
        if any(word in title_lower for word in ['chief', 'president']):
            return 'CEO'
        elif 'vice' in title_lower:
            return 'VP'
        elif 'assistant' in title_lower and 'vice' in title_lower:
            return 'AVP'
        elif 'senior' in title_lower and 'manager' in title_lower:
            return 'Senior Manager'
        elif 'manager' in title_lower:
            return 'Manager'
        elif 'senior' in title_lower:
            return 'Senior'
        elif 'director' in title_lower:
            return 'Director'
        elif 'analyst' in title_lower:
            return 'Analyst'
        elif 'accountant' in title_lower:
            return 'Accountant'
        
        return 'Unknown'
    
    def _calculate_confidence(self, name: str, title: str, department: str) -> float:
        """Calculate parsing confidence"""
        confidence = 0.0
        
        # Name quality
        if name and self._looks_like_name(name):
            confidence += 0.4
        
        # Title quality  
        if title and any(indicator.lower() in title.lower() for indicator in self.title_indicators):
            confidence += 0.3
        
        # Department quality
        if department and any(re.search(pattern, department, re.IGNORECASE) for pattern in self.department_patterns):
            confidence += 0.3
        
        return min(confidence, 1.0)

# Test function
def test_advanced_parser():
    """Test the advanced parser with problematic examples"""
    
    parser = AdvancedOrgParser()
    
    test_cases = [
        "MVP, Financial Reporting & Tax\nOkobea Antwi-Boasiako",
        "VP, Financial Reporting\nYoke Yin Lee", 
        "Manager, Tax\nSandip Pabari",
        "Financial Reporting\nAccountant,\nChloe Teo"
    ]
    
    print("🧪 **TESTING ADVANCED PARSER:**")
    
    for i, test_text in enumerate(test_cases):
        print(f"\n--- Test {i+1} ---")
        print(f"Input: {repr(test_text)}")
        
        result = parser.parse_spatial_cluster(test_text, (0, 0, 100, 100))
        
        if result:
            print(f"✅ Parsed successfully:")
            print(f"   Name: '{result.name}'")
            print(f"   Title: '{result.title}'")  
            print(f"   Department: '{result.department}'")
            print(f"   Level: '{result.hierarchy_level}'")
            print(f"   Confidence: {result.confidence:.2f}")
        else:
            print(f"❌ Parsing failed")

if __name__ == "__main__":
    test_advanced_parser()