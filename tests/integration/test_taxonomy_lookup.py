#!/usr/bin/env python3
"""Test taxonomy fuzzy matching for reserves → NPR/SR/DR"""

import sys
sys.path.insert(0, '/Users/shijuprakash/AAIRE/src')

from rag_modules.query.insurance_taxonomy_extractor import InsuranceTaxonomyExtractor
import json

# Load existing taxonomy
tax = InsuranceTaxonomyExtractor()
tax.load_taxonomy('/Users/shijuprakash/AAIRE/data/taxonomy.json')

print("=" * 60)
print("Testing Taxonomy Fuzzy Matching")
print("=" * 60)

# Test terms that should now match
test_terms = [
    'reserves',        # Should match via singular variation + compound "reserve methods"
    'reserve',         # Should match via compound "reserve methods"
    'reserve methods', # Should match exactly
    'calculation',     # General test
    'NPR',            # Acronym test
]

for term in test_terms:
    related = tax.get_related_terms(term, max_depth=1)
    print(f'\n{term}:')
    if related:
        print(f'  ✅ Found {len(related)} related terms:')
        for r in related[:10]:  # Show first 10
            print(f'     - {r}')
        if len(related) > 10:
            print(f'     ... and {len(related) - 10} more')
    else:
        print('  ❌ No matches')

print("\n" + "=" * 60)
print("Checking taxonomy structure:")
print("=" * 60)

# Show what's actually in the taxonomy
if 'reserve methods' in tax.hierarchies:
    print(f"\n✅ 'reserve methods' hierarchy found:")
    print(f"   {tax.hierarchies['reserve methods']}")
else:
    print("\n❌ 'reserve methods' not in hierarchies")

# Show a few hierarchy keys
print(f"\nAll hierarchy keys containing 'reserve':")
for key in tax.hierarchies.keys():
    if 'reserve' in key:
        print(f"  - {key}: {tax.hierarchies[key][:3]}")