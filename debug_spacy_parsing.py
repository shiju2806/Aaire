#!/usr/bin/env python3
"""
Debug spaCy Parsing
Analyze how spaCy parses "whole life policy" vs "universal life policy"
"""

import sys
sys.path.append('/Users/shijuprakash/AAIRE/src')

import spacy

def debug_spacy_parsing():
    print("🔍 Debug spaCy Parsing")
    print("=" * 40)

    nlp = spacy.load('en_core_web_sm')

    queries = [
        "how do I calculate the reserves for a whole life policy in usstat",
        "how do I calculate the reserves for a universal life policy in usstat"
    ]

    for query in queries:
        print(f"\n📝 Query: '{query}'")
        doc = nlp(query)

        print("🔍 Token Analysis:")
        for token in doc:
            print(f"  {token.text:12} | POS: {token.pos_:4} | DEP: {token.dep_:10} | HEAD: {token.head.text}")

        print("\n🧠 Noun Chunks:")
        for chunk in doc.noun_chunks:
            print(f"  '{chunk.text}' | ROOT: {chunk.root.text} | LABEL: {chunk.label_}")

        print("\n🎯 ADJ + NOUN + NOUN patterns:")
        for token in doc:
            if token.pos_ == 'ADJ' and token.head.pos_ == 'NOUN':
                head = token.head
                if head.head.pos_ == 'NOUN':
                    pattern = f"{token.text} {head.text} {head.head.text}"
                    print(f"  3-word: '{pattern}'")
                pattern2 = f"{token.text} {head.text}"
                print(f"  2-word: '{pattern2}'")

        print("-" * 50)

if __name__ == "__main__":
    debug_spacy_parsing()