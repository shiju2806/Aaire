#!/usr/bin/env python3
"""
Debug script to isolate OpenAI initialization issue
"""
import os
import sys

print("🔍 OpenAI Initialization Debug")
print("=" * 50)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print(f"✅ OpenAI API Key set: {'OPENAI_API_KEY' in os.environ}")
print(f"✅ Model setting: {os.getenv('OPENAI_MODEL', 'not set')}")

# Test imports step by step
print("\n📦 Testing imports...")

try:
    # Try newer llama-index import structure (0.10.x+)
    from llama_index.llms.openai import OpenAI
    print("✅ Imported OpenAI from llama_index.llms.openai")
    import_path = "newer"
except ImportError:
    try:
        # Fall back to older import structure (0.9.x)
        from llama_index.llms import OpenAI
        print("✅ Imported OpenAI from llama_index.llms")
        import_path = "older"
    except ImportError:
        try:
            # Try even older structure
            from llama_index.llms import OpenAI
            print("✅ Imported OpenAI from llama_index.llms (oldest)")
            import_path = "oldest"
        except ImportError as e:
            print(f"❌ Failed to import OpenAI: {e}")
            sys.exit(1)

print(f"📍 Using import path: {import_path}")

# Test basic initialization
print("\n🧪 Testing OpenAI initialization...")

test_cases = [
    {
        "name": "No parameters",
        "params": {}
    },
    {
        "name": "Only temperature",
        "params": {"temperature": 0.7}
    },
    {
        "name": "Temperature + max_tokens",
        "params": {"temperature": 0.7, "max_tokens": 500}
    },
    {
        "name": "With gpt-3.5-turbo model",
        "params": {"model": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 500}
    },
    {
        "name": "With gpt-4o-mini model",
        "params": {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 500}
    }
]

for test in test_cases:
    print(f"\n  🧪 Test: {test['name']}")
    try:
        llm = OpenAI(**test["params"])
        print(f"    ✅ Success")
        
        # Check available attributes
        if hasattr(llm, '_model'):
            print(f"    📋 Has _model field: {getattr(llm, '_model', 'None')}")
        else:
            print(f"    ⚠️  No _model field")
            
        if hasattr(llm, 'model'):
            print(f"    📋 Has model field: {getattr(llm, 'model', 'None')}")
        else:
            print(f"    ⚠️  No model field")
            
        # Try to set _model field
        if test["name"] == "With gpt-4o-mini model":
            try:
                llm._model = "gpt-4o-mini"
                print(f"    ✅ Successfully set _model field")
            except Exception as e:
                print(f"    ❌ Failed to set _model field: {e}")
                
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        print(f"    📋 Error type: {type(e).__name__}")
        import traceback
        print(f"    📋 Traceback: {traceback.format_exc()}")

print("\n🏁 Debug complete")