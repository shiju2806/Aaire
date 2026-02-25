#!/usr/bin/env python3
"""
Debug the spatial extraction to see exactly what's being extracted
"""

import sys
sys.path.append('/Users/shijuprakash/AAIRE/src')

from pdf_spatial_extractor import PDFSpatialExtractor
import json

def debug_extraction():
    """Debug the spatial extraction results"""
    
    pdf_path = "/Users/shijuprakash/AAIRE/data/uploads/finance_structures.pdf"
    
    print("🔍 **DEBUGGING SPATIAL EXTRACTION**\n")
    
    extractor = PDFSpatialExtractor()
    result = extractor.extract_with_coordinates(pdf_path)
    
    print(f"📊 **RESULT STRUCTURE:**")
    print(f"   Type: {type(result)}")
    print(f"   Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"\n🔑 **{key.upper()}:**")
            print(f"   Type: {type(value)}")
            if isinstance(value, list):
                print(f"   Length: {len(value)}")
                if len(value) > 0:
                    print(f"   Sample item: {value[0]}")
                    if isinstance(value[0], dict):
                        print(f"   Sample keys: {list(value[0].keys())}")
            elif isinstance(value, dict):
                print(f"   Keys: {list(value.keys())}")
            else:
                print(f"   Value: {str(value)[:100]}...")
    
    # Try to find the organizational data
    if 'organizational_units' in result:
        units = result['organizational_units']
        print(f"\n🎯 **FIRST 3 ORGANIZATIONAL UNITS:**")
        for i, unit in enumerate(units[:3]):
            print(f"\n  Unit {i+1}:")
            if isinstance(unit, dict):
                for key, value in unit.items():
                    print(f"    {key}: {value}")
            else:
                print(f"    {unit}")
    
    # Check spatial clusters
    if 'spatial_clusters' in result:
        clusters = result['spatial_clusters']
        print(f"\n🧩 **FIRST 3 SPATIAL CLUSTERS:**")
        for i, cluster in enumerate(clusters[:3]):
            print(f"\n  Cluster {i+1}:")
            if isinstance(cluster, dict):
                for key, value in cluster.items():
                    if key == 'elements' and isinstance(value, list):
                        print(f"    {key}: {len(value)} elements")
                        if len(value) > 0:
                            print(f"      Sample: {value[0]}")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"    {cluster}")

if __name__ == "__main__":
    debug_extraction()