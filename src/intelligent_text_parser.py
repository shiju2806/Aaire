#!/usr/bin/env python3
"""
Intelligent Text Parser for Organizational Charts
Parses concatenated job title/name text without spatial coordinates
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class Person:
    """Individual person in organizational chart"""
    name: str
    title: str
    department: str = ""
    level: str = ""  # VP, AVP, Manager, etc.
    confidence: float = 0.0

@dataclass
class ParseResult:
    """Result of parsing organizational text"""
    people: List[Person]
    departments: List[str]
    hierarchy: Dict[str, List[str]]
    confidence: float

class IntelligentTextParser:
    """Parse concatenated organizational text using pattern recognition"""
    
    def __init__(self):
        # Job title patterns (in order of hierarchy)
        self.title_patterns = [
            r'MVP,?\s+(.+?)(?=[A-Z][a-z])',  # MVP, Financial Reporting & Tax
            r'VP,?\s+(.+?)(?=[A-Z][a-z])',   # VP, Financial Reporting
            r'AVP,?\s+(.+?)(?=[A-Z][a-z])',  # AVP, Financial Reporting
            r'Manager,?\s+(.+?)(?=[A-Z][a-z])', # Manager, Financial Reporting
            r'Senior\s+(.+?)(?=[A-Z][a-z])',    # Senior Financial Accountant
            r'(.+?)\s+Analyst(?=[A-Z][a-z])',   # Finance Analyst
            r'(.+?)\s+Accountant(?=[A-Z][a-z])', # Financial Accountant
            r'Intern,?\s+(.+?)(?=[A-Z][a-z])',  # Intern, Finance
        ]
        
        # Name patterns (capital letters followed by mixed case)
        self.name_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z\']+)*)'
        
        # Department extraction patterns
        self.dept_patterns = [
            r'Financial\s+Reporting(?:\s+&\s+Tax)?',
            r'Financial\s+Planning\s+&\s+Analysis',
            r'Financial\s+Management',
            r'Tax',
            r'Payroll(?:\s+&\s+Benefits)?',
            r'ICFR',
            r'Accounting\s+Policy',
            r'Finance'
        ]
        
        # Hierarchy levels
        self.hierarchy_levels = {
            'MVP': 1,
            'VP': 2, 
            'AVP': 3,
            'Manager': 4,
            'Senior': 5,
            'Analyst': 6,
            'Accountant': 7,
            'Intern': 8
        }
    
    def parse_organizational_text(self, text: str) -> ParseResult:
        """Parse concatenated organizational chart text"""
        
        logger.info("Starting intelligent text parsing", text_length=len(text))
        
        # Step 1: Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        # Step 2: Extract people using pattern matching
        people = self._extract_people(cleaned_text)
        
        # Step 3: Extract departments
        departments = self._extract_departments(text)
        
        # Step 4: Build hierarchy
        hierarchy = self._build_hierarchy(people)
        
        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(people, text)
        
        result = ParseResult(
            people=people,
            departments=departments,
            hierarchy=hierarchy,
            confidence=confidence
        )
        
        logger.info("Parsing complete", 
                   people_count=len(people),
                   departments_count=len(departments),
                   confidence=confidence)
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for parsing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Add spaces before capital letters that follow lowercase (word boundaries)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Clean up common patterns
        text = re.sub(r'([,&])([A-Z])', r'\1 \2', text)
        
        return text
    
    def _extract_people(self, text: str) -> List[Person]:
        """Extract individual people from text using advanced pattern matching"""
        people = []
        
        # Strategy 1: Look for Title + Name patterns
        people.extend(self._extract_title_name_pairs(text))
        
        # Strategy 2: Look for Name + Title patterns  
        people.extend(self._extract_name_title_pairs(text))
        
        # Strategy 3: Handle complex concatenated strings
        people.extend(self._extract_concatenated_entries(text))
        
        # Remove duplicates and validate
        people = self._deduplicate_people(people)
        
        return people
    
    def _extract_title_name_pairs(self, text: str) -> List[Person]:
        """Extract Title + Name patterns like 'VP, Financial Reporting John Smith'"""
        people = []
        
        for title_pattern in self.title_patterns:
            matches = re.finditer(title_pattern, text)
            for match in matches:
                title_dept = match.group(1).strip()
                
                # Look for name after the title
                remaining_text = text[match.end():match.end()+100]
                name_match = re.search(self.name_pattern, remaining_text)
                
                if name_match:
                    name = name_match.group(1).strip()
                    title, dept = self._parse_title_department(title_dept)
                    level = self._extract_level(match.group(0))
                    
                    person = Person(
                        name=name,
                        title=title,
                        department=dept,
                        level=level,
                        confidence=0.8
                    )
                    people.append(person)
        
        return people
    
    def _extract_name_title_pairs(self, text: str) -> List[Person]:
        """Extract Name + Title patterns like 'John Smith VP, Financial Reporting'"""
        people = []
        
        # Look for Name followed by title keywords
        pattern = rf'({self.name_pattern})\s+((?:MVP|VP|AVP|Manager|Senior|Analyst|Accountant|Intern).+?)(?=[A-Z][a-z]|$)'
        
        matches = re.finditer(pattern, text)
        for match in matches:
            name = match.group(1).strip()
            title_text = match.group(2).strip()
            
            title, dept = self._parse_title_department(title_text)
            level = self._extract_level(title_text)
            
            person = Person(
                name=name,
                title=title,
                department=dept,
                level=level,
                confidence=0.7
            )
            people.append(person)
        
        return people
    
    def _extract_concatenated_entries(self, text: str) -> List[Person]:
        """Handle complex concatenated strings like the Finance Structures document"""
        people = []
        
        # Split on common patterns that indicate person boundaries
        boundary_pattern = r'(?=[A-Z][a-z]+(?:\s+[A-Z][a-z\']+)*(?:MVP|VP|AVP|Manager|Senior|Analyst|Accountant|Intern))'
        segments = re.split(boundary_pattern, text)
        
        for segment in segments:
            if len(segment.strip()) < 10:  # Skip very short segments
                continue
                
            person = self._parse_single_entry(segment.strip())
            if person:
                people.append(person)
        
        return people
    
    def _parse_single_entry(self, entry: str) -> Optional[Person]:
        """Parse a single concatenated entry like 'MVP, Financial Reporting & TaxOkobea Antwi-Boasiako'"""
        
        # Strategy: Use more precise pattern to find title + name combinations
        
        # Pattern 1: Title, Department + Name
        pattern1 = r'(MVP|VP|AVP|Manager|Senior\s+[\w\s]+|[\w\s]*Analyst|[\w\s]*Accountant|Intern),?\s*([\w\s&,]+?)([A-Z][a-z]+(?:\s+[A-Z][a-z\']+)*)'
        match1 = re.search(pattern1, entry)
        
        if match1:
            level_title = match1.group(1).strip()
            dept_part = match1.group(2).strip()
            name = match1.group(3).strip()
            
            # Clean up department part
            dept = dept_part.replace(',', '').strip()
            
            # Extract level
            level = self._extract_level(level_title)
            
            # Construct full title
            if level in ['MVP', 'VP', 'AVP']:
                title = level
            else:
                title = level_title
            
            return Person(
                name=name,
                title=title,
                department=dept,
                level=level,
                confidence=0.9
            )
        
        # Pattern 2: Handle complex cases with better name detection
        title_keywords = ['MVP', 'VP', 'AVP', 'Manager', 'Senior', 'Analyst', 'Accountant', 'Intern']
        
        for keyword in title_keywords:
            if keyword in entry:
                # Find the keyword position
                keyword_pos = entry.find(keyword)
                
                # Look for proper name pattern (first letter caps, rest lowercase)
                name_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z\']+){0,3})(?![a-z])'
                names_found = re.findall(name_pattern, entry[keyword_pos:])
                
                if names_found:
                    # Take the most likely name (longest match that looks like a person name)
                    name = max(names_found, key=len)
                    
                    # Extract department from before the name
                    before_name = entry[:entry.find(name)]
                    dept = ""
                    for dept_pattern in self.dept_patterns:
                        if re.search(dept_pattern, before_name):
                            dept = re.search(dept_pattern, before_name).group(0)
                            break
                    
                    level = self._extract_level(before_name)
                    
                    return Person(
                        name=name,
                        title=keyword if keyword in ['MVP', 'VP', 'AVP'] else keyword,
                        department=dept,
                        level=level,
                        confidence=0.8
                    )
        
        return None
    
    def _parse_title_department(self, title_text: str) -> Tuple[str, str]:
        """Separate job title from department"""
        
        # Extract department
        dept = ""
        for dept_pattern in self.dept_patterns:
            match = re.search(dept_pattern, title_text)
            if match:
                dept = match.group(0)
                break
        
        # Extract title (remove department part)
        title = re.sub(r',?\s*(?:' + '|'.join(self.dept_patterns) + r')', '', title_text).strip()
        title = re.sub(r'^,\s*', '', title).strip()  # Remove leading comma
        
        return title, dept
    
    def _extract_level(self, text: str) -> str:
        """Extract hierarchy level from text"""
        for level in self.hierarchy_levels.keys():
            if level in text:
                return level
        return "Unknown"
    
    def _extract_departments(self, text: str) -> List[str]:
        """Extract all departments mentioned in text"""
        departments = set()
        
        for dept_pattern in self.dept_patterns:
            matches = re.findall(dept_pattern, text)
            departments.update(matches)
        
        return sorted(list(departments))
    
    def _build_hierarchy(self, people: List[Person]) -> Dict[str, List[str]]:
        """Build organizational hierarchy"""
        hierarchy = {}
        
        # Group by level
        for person in people:
            level = person.level
            if level not in hierarchy:
                hierarchy[level] = []
            hierarchy[level].append(person.name)
        
        # Sort by hierarchy level
        sorted_hierarchy = {}
        for level in sorted(self.hierarchy_levels.keys(), key=lambda x: self.hierarchy_levels[x]):
            if level in hierarchy:
                sorted_hierarchy[level] = hierarchy[level]
        
        return sorted_hierarchy
    
    def _deduplicate_people(self, people: List[Person]) -> List[Person]:
        """Remove duplicate people entries"""
        seen_names = set()
        unique_people = []
        
        # Sort by confidence (highest first)
        people.sort(key=lambda x: x.confidence, reverse=True)
        
        for person in people:
            if person.name not in seen_names:
                seen_names.add(person.name)
                unique_people.append(person)
        
        return unique_people
    
    def _calculate_confidence(self, people: List[Person], original_text: str) -> float:
        """Calculate overall parsing confidence"""
        if not people:
            return 0.0
        
        # Base confidence on successful extractions
        avg_confidence = sum(p.confidence for p in people) / len(people)
        
        # Boost confidence if we found multiple hierarchy levels
        levels_found = len(set(p.level for p in people))
        level_bonus = min(0.2, levels_found * 0.05)
        
        # Boost confidence if we found departments
        dept_bonus = 0.1 if any(p.department for p in people) else 0.0
        
        total_confidence = min(1.0, avg_confidence + level_bonus + dept_bonus)
        
        return round(total_confidence, 2)

# Test function
def test_parser():
    """Test the parser with Finance Structures data"""
    
    test_text = """MVP, Financial Reporting & TaxOkobea Antwi-BoasiakoVP, Financial ReportingYoke Yin LeeAVP, Financial ReportingMing OwManager, Financial ReportingEunjeong KoFinancial Accountant, Financial ReportingJaehyeong JunManager, Financial ReportingYuen Kuan HoonFinancial Accountant, Financial ReportingHui Zhen SeahFinance Analyst, Financial ReportingChloe TeoAVP, Financial Reporting & TaxSimon PercivalManager, TaxSandip PabariManager, Financial ReportingAoife KellySenior Financial Accountant, Financial ReportingAisling HennessyFinancial Accountant, Financial ReportingGiao DinhAVP, Financial ReportingSarah ChowManager, Financial ReportingShubham AsijaSenior Financial Accountant, Financial ReportingNirav PatelSenior Financial Accountant, Financial ReportingJoel CrosbyFinance Analyst, Financial ReportingTom WhiteleyManager, Accounting PolicyBen GaoManager, ICFRBernard DamhuisAVP, Financial ReportingVivian CaiManager, Financial ReportingNiluka AbeygunaratneFinancial Accountant, Financial ReportingMiranda LeeManager, Financial ReportingBen CochangcoSenior Finance Analyst, Financial ReportingWinston LamIntern, FinanceAngel XuManager, Payroll & BenefitsMarina TolstounAVP, Financial ReportingLan Jiang"""
    
    parser = IntelligentTextParser()
    result = parser.parse_organizational_text(test_text)
    
    print(f"🎯 **PARSING RESULTS:**")
    print(f"   People found: {len(result.people)}")
    print(f"   Departments: {len(result.departments)}")
    print(f"   Confidence: {result.confidence}")
    
    print(f"\n👥 **PEOPLE BY HIERARCHY:**")
    for level, names in result.hierarchy.items():
        print(f"   {level}: {len(names)} people")
        for name in names[:3]:  # Show first 3
            print(f"     • {name}")
        if len(names) > 3:
            print(f"     ... and {len(names) - 3} more")
    
    print(f"\n🏢 **DEPARTMENTS:**")
    for dept in result.departments:
        print(f"   • {dept}")

if __name__ == "__main__":
    test_parser()