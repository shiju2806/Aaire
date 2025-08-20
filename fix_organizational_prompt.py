#!/usr/bin/env python3
"""
Fix the organizational structure prompt to properly use spatial extraction data
"""

import re
import sys
sys.path.append('/Users/shijuprakash/AAIRE/src')

def create_enhanced_org_prompt():
    """Create a better prompt for organizational structure queries"""
    
    return """You are AAIRE, an expert in insurance accounting and actuarial matters.
You provide accurate information based on US GAAP, IFRS, and company policies.

Current User Question: {query}

SPATIAL EXTRACTION DATA FROM ORGANIZATIONAL CHARTS:
{context}

CRITICAL INSTRUCTIONS FOR ORGANIZATIONAL STRUCTURE RESPONSES:

1. **PARSE SPATIAL EXTRACTION FORMAT:**
   - Look for "[SHAPE-AWARE ORGANIZATIONAL EXTRACTION]" sections
   - Each person is formatted as: "** Name - Title" with department info
   - Group people by their actual departments/levels as shown in the extraction

2. **RESPONSE FORMAT:**
   Use this exact structure:
   
   ## [Department Name]
   
   ### [Hierarchy Level] (e.g., VP, AVP, Manager)
   • **[Full Name]** - [Complete Job Title]
   
   ### [Next Hierarchy Level]
   • **[Full Name]** - [Complete Job Title]
   • **[Full Name]** - [Complete Job Title]

3. **HIERARCHY ORDERING:**
   - MVP (Most Senior)
   - VP (Vice President) 
   - AVP (Assistant Vice President)
   - Manager
   - Senior [Role]
   - [Role] (Analyst, Accountant, etc.)
   - Intern (Most Junior)

4. **KEY REQUIREMENTS:**
   - Use ONLY the names and titles from the spatial extraction
   - Group by actual departments (Financial Reporting & Tax, Financial Planning & Analysis, etc.)
   - Maintain hierarchy order within each department
   - Show clear organizational structure, not random lists
   - Don't make up job titles or invent positions
   - If extraction shows "warnings" or fallback patterns, still use the data but note any limitations

5. **CITATION FORMAT:**
   Use source names with page numbers: "Finance Structures.pdf, Page 2"

6. **AVOID:**
   - Generic text like "multiple positions listed"
   - Invented role descriptions
   - Mixing up names and titles
   - Losing department groupings

Provide a clear, well-structured organizational breakdown based on the spatial extraction data.

Response:"""

def extract_org_data_from_context(context_text):
    """Extract and structure organizational data from spatial extraction context"""
    
    # Look for the shape-aware extraction section
    extraction_match = re.search(r'\[SHAPE-AWARE ORGANIZATIONAL EXTRACTION\](.*?)(?=\[|$)', context_text, re.DOTALL)
    
    if not extraction_match:
        return None
    
    extraction_text = extraction_match.group(1)
    
    # Parse the departments and people
    departments = {}
    current_dept = None
    
    lines = extraction_text.split('\n')
    for line in lines:
        line = line.strip()
        
        # Department headers (end with colon)
        if line.endswith(':') and not line.startswith('**'):
            current_dept = line.rstrip(':')
            departments[current_dept] = []
        
        # Person entries (start with **)
        elif line.startswith('**') and current_dept:
            # Extract name and title from "** Name - Title"
            person_match = re.search(r'\*\*\s*(.+?)\s*-\s*(.+)', line)
            if person_match:
                name = person_match.group(1).strip()
                title = person_match.group(2).strip()
                departments[current_dept].append({
                    'name': name,
                    'title': title,
                    'raw_line': line
                })
    
    return departments

def test_extraction():
    """Test the extraction logic"""
    
    sample_context = """
[SHAPE-AWARE ORGANIZATIONAL EXTRACTION]
==================================================

Reporting & Tax:
---------------
** Okobea Antwi-Boasiako - MVP, Financial Reporting & Tax
   Source: Page 1, cluster_0_page_1
** Simon Percival - AVP, Financial Reporting & Tax
   Source: Page 1, cluster_10_page_1

Financial Reporting:
-------------------
** Yoke Yin Lee - VP, Financial Reporting
   Source: Page 1, cluster_1_page_1
** Ming Ow - AVP, Financial Reporting
   Source: Page 1, cluster_2_page_1
** Sarah Chow - Manager, Financial Reporting
   Source: Page 1, cluster_15_page_1
"""
    
    departments = extract_org_data_from_context(sample_context)
    
    print("🔍 **EXTRACTED ORGANIZATIONAL DATA:**")
    for dept, people in departments.items():
        print(f"\n📂 {dept}:")
        for person in people:
            print(f"   • {person['name']} - {person['title']}")

if __name__ == "__main__":
    test_extraction()