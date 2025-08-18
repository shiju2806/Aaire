#!/usr/bin/env python3
"""
Finance Structures Specific Parser
Handles the exact concatenated format from the Finance Structures PDF
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class Employee:
    """Individual employee in organizational chart"""
    name: str
    title: str
    department: str
    hierarchy_level: str
    raw_text: str = ""

class FinanceStructuresParser:
    """Parse the specific format from Finance Structures PDF"""
    
    def __init__(self):
        # Hierarchy levels in order
        self.hierarchy_order = ['MVP', 'VP', 'AVP', 'Manager', 'Senior', 'Analyst', 'Accountant', 'Intern']
        
        # Common department patterns
        self.departments = [
            'Financial Reporting & Tax',
            'Financial Reporting', 
            'Financial Planning & Analysis',
            'Financial Management',
            'Accounting Policy',
            'Tax',
            'ICFR',
            'Payroll & Benefits',
            'Finance'
        ]
    
    def parse_concatenated_text(self, text: str) -> List[Employee]:
        """Parse the specific concatenated format"""
        
        logger.info("Parsing Finance Structures format", text_length=len(text))
        
        employees = []
        
        # Step 1: Split text into potential employee entries
        # Look for patterns like "TitleDepartmentNameNextTitle"
        entries = self._split_into_entries(text)
        
        # Step 2: Parse each entry
        for entry in entries:
            employee = self._parse_single_employee(entry)
            if employee:
                employees.append(employee)
        
        # Step 3: Group and validate by hierarchy
        employees = self._group_by_hierarchy(employees)
        
        logger.info("Parsing complete", employees_found=len(employees))
        
        return employees
    
    def _split_into_entries(self, text: str) -> List[str]:
        """Split concatenated text into individual employee entries"""
        
        # The pattern appears to be: Title, Department + Name + Next Title
        # Example: "MVP, Financial Reporting & TaxOkobea Antwi-BoasiakoVP"
        
        # Strategy: Split on title keywords when they're preceded by a name
        title_keywords = '|'.join(self.hierarchy_order)
        
        # Split pattern: Name followed by Title
        # Looking for: [Capital letter name] + [Title keyword]
        split_pattern = rf'([A-Z][a-z]+(?:\s+[A-Z][a-z\']+)*)\s*({title_keywords})'
        
        entries = []
        last_end = 0
        
        for match in re.finditer(split_pattern, text):
            # Add the text from last split to current match
            if last_end < match.start():
                entry = text[last_end:match.start() + len(match.group(1))]
                if entry.strip():
                    entries.append(entry.strip())
            
            last_end = match.start() + len(match.group(1))
        
        # Add remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                entries.append(remaining)
        
        # Filter out very short entries
        entries = [e for e in entries if len(e) > 10]
        
        return entries
    
    def _parse_single_employee(self, entry: str) -> Optional[Employee]:
        """Parse a single employee entry"""
        
        # Clean the entry
        entry = entry.strip()
        if len(entry) < 5:
            return None
        
        # Strategy: Find title + department, then extract name
        
        # Find hierarchy level
        hierarchy_level = self._find_hierarchy_level(entry)
        if not hierarchy_level:
            return None
        
        # Find department
        department = self._find_department(entry)
        
        # Extract name (look for proper name pattern after title/department)
        name = self._extract_name(entry, hierarchy_level, department)
        if not name:
            return None
        
        # Build title
        title = self._build_title(entry, hierarchy_level, department)
        
        employee = Employee(
            name=name,
            title=title,
            department=department,
            hierarchy_level=hierarchy_level,
            raw_text=entry
        )
        
        return employee
    
    def _find_hierarchy_level(self, entry: str) -> str:
        """Find the hierarchy level in the entry"""
        for level in self.hierarchy_order:
            if level in entry:
                return level
        return ""
    
    def _find_department(self, entry: str) -> str:
        """Find the department in the entry"""
        for dept in self.departments:
            if dept in entry:
                return dept
        return ""
    
    def _extract_name(self, entry: str, hierarchy_level: str, department: str) -> str:
        """Extract the person's name from the entry"""
        
        # Remove title and department parts to isolate the name
        cleaned = entry
        
        # Remove hierarchy level
        if hierarchy_level:
            cleaned = cleaned.replace(hierarchy_level, ' ')
        
        # Remove department
        if department:
            cleaned = cleaned.replace(department, ' ')
        
        # Remove common words
        cleaned = re.sub(r'\b(and|&|,)\b', ' ', cleaned)
        
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Find proper name pattern (2-4 words, each starting with capital)
        name_patterns = [
            r'([A-Z][a-z]+\s+[A-Z][a-z\']+(?:\s+[A-Z][a-z\']+)?(?:\s+[A-Z][a-z\']+)?)',
            r'([A-Z][a-z]+\s+[A-Z][a-z\']+)',
            r'([A-Z][a-z]+)'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, cleaned)
            if matches:
                # Take the longest reasonable name
                best_match = max(matches, key=len)
                if len(best_match.split()) <= 4:  # Reasonable name length
                    return best_match.strip()
        
        return ""
    
    def _build_title(self, entry: str, hierarchy_level: str, department: str) -> str:
        """Build the full job title"""
        
        # For executive levels, title is just the level
        if hierarchy_level in ['MVP', 'VP', 'AVP']:
            if department:
                return f"{hierarchy_level}, {department}"
            else:
                return hierarchy_level
        
        # For other levels, look for more specific title
        if 'Senior' in entry:
            if 'Accountant' in entry:
                return "Senior Financial Accountant"
            elif 'Analyst' in entry:
                return "Senior Financial Analyst"
            else:
                return f"Senior {hierarchy_level}"
        
        # Default titles
        title_map = {
            'Manager': 'Manager',
            'Analyst': 'Finance Analyst', 
            'Accountant': 'Financial Accountant',
            'Intern': 'Intern'
        }
        
        title = title_map.get(hierarchy_level, hierarchy_level)
        
        if department:
            return f"{title}, {department}"
        else:
            return title
    
    def _group_by_hierarchy(self, employees: List[Employee]) -> List[Employee]:
        """Group employees by hierarchy and validate"""
        
        # Create hierarchy grouping
        hierarchy_groups = {}
        for emp in employees:
            level = emp.hierarchy_level
            if level not in hierarchy_groups:
                hierarchy_groups[level] = []
            hierarchy_groups[level].append(emp)
        
        # Log hierarchy summary
        logger.info("Hierarchy breakdown:")
        for level in self.hierarchy_order:
            if level in hierarchy_groups:
                count = len(hierarchy_groups[level])
                logger.info(f"  {level}: {count} people")
                
                # Show first few names
                for i, emp in enumerate(hierarchy_groups[level][:3]):
                    logger.info(f"    • {emp.name} - {emp.title}")
                if len(hierarchy_groups[level]) > 3:
                    logger.info(f"    ... and {len(hierarchy_groups[level]) - 3} more")
        
        return employees
    
    def create_summary(self, employees: List[Employee]) -> Dict[str, Any]:
        """Create a summary of the organizational structure"""
        
        summary = {
            'total_employees': len(employees),
            'by_hierarchy': {},
            'by_department': {},
            'extraction_quality': self._assess_quality(employees)
        }
        
        # Group by hierarchy
        for emp in employees:
            level = emp.hierarchy_level
            if level not in summary['by_hierarchy']:
                summary['by_hierarchy'][level] = []
            summary['by_hierarchy'][level].append({
                'name': emp.name,
                'title': emp.title,
                'department': emp.department
            })
        
        # Group by department
        for emp in employees:
            dept = emp.department or 'Unknown'
            if dept not in summary['by_department']:
                summary['by_department'][dept] = []
            summary['by_department'][dept].append({
                'name': emp.name,
                'title': emp.title,
                'hierarchy_level': emp.hierarchy_level
            })
        
        return summary
    
    def _assess_quality(self, employees: List[Employee]) -> Dict[str, Any]:
        """Assess the quality of extraction"""
        
        total = len(employees)
        if total == 0:
            return {'score': 0.0, 'issues': ['No employees extracted']}
        
        issues = []
        
        # Check for missing names
        no_name = sum(1 for emp in employees if not emp.name.strip())
        if no_name > 0:
            issues.append(f"{no_name} employees missing names")
        
        # Check for missing departments
        no_dept = sum(1 for emp in employees if not emp.department.strip())
        if no_dept > total * 0.5:
            issues.append(f"{no_dept} employees missing departments")
        
        # Check hierarchy distribution
        hierarchy_counts = {}
        for emp in employees:
            hierarchy_counts[emp.hierarchy_level] = hierarchy_counts.get(emp.hierarchy_level, 0) + 1
        
        if len(hierarchy_counts) < 3:
            issues.append("Limited hierarchy diversity")
        
        # Calculate score
        score = 1.0
        score -= (no_name / total) * 0.5  # Heavy penalty for missing names
        score -= (no_dept / total) * 0.2  # Light penalty for missing departments
        score = max(0.0, score)
        
        return {
            'score': round(score, 2),
            'issues': issues,
            'hierarchy_levels_found': len(hierarchy_counts),
            'total_extracted': total
        }

# Test function
def test_finance_parser():
    """Test the Finance Structures parser"""
    
    test_text = """MVP, Financial Reporting & TaxOkobea Antwi-BoasiakoVP, Financial ReportingYoke Yin LeeAVP, Financial ReportingMing OwManager, Financial ReportingEunjeong KoFinancial Accountant, Financial ReportingJaehyeong JunManager, Financial ReportingYuen Kuan HoonFinancial Accountant, Financial ReportingHui Zhen SeahFinance Analyst, Financial ReportingChloe TeoAVP, Financial Reporting & TaxSimon PercivalManager, TaxSandip PabariManager, Financial ReportingAoife KellySenior Financial Accountant, Financial ReportingAisling HennessyFinancial Accountant, Financial ReportingGiao DinhAVP, Financial ReportingSarah ChowManager, Financial ReportingShubham AsijaSenior Financial Accountant, Financial ReportingNirav PatelSenior Financial Accountant, Financial ReportingJoel CrosbyFinance Analyst, Financial ReportingTom WhiteleyManager, Accounting PolicyBen GaoManager, ICFRBernard DamhuisAVP, Financial ReportingVivian CaiManager, Financial ReportingNiluka AbeygunaratneFinancial Accountant, Financial ReportingMiranda LeeManager, Financial ReportingBen CochangcoSenior Finance Analyst, Financial ReportingWinston LamIntern, FinanceAngel XuManager, Payroll & BenefitsMarina TolstounAVP, Financial ReportingLan Jiang"""
    
    parser = FinanceStructuresParser()
    employees = parser.parse_concatenated_text(test_text)
    summary = parser.create_summary(employees)
    
    print(f"🎯 **FINANCE STRUCTURES PARSING RESULTS:**")
    print(f"   Total employees: {summary['total_employees']}")
    print(f"   Extraction quality: {summary['extraction_quality']['score']}")
    
    if summary['extraction_quality']['issues']:
        print(f"   Issues: {', '.join(summary['extraction_quality']['issues'])}")
    
    print(f"\n👥 **BY HIERARCHY:**")
    for level, people in summary['by_hierarchy'].items():
        print(f"   {level}: {len(people)} people")
        for person in people[:3]:
            print(f"     • {person['name']} - {person['title']}")
        if len(people) > 3:
            print(f"     ... and {len(people) - 3} more")
    
    print(f"\n🏢 **BY DEPARTMENT:**")
    for dept, people in summary['by_department'].items():
        print(f"   {dept}: {len(people)} people")

if __name__ == "__main__":
    test_finance_parser()