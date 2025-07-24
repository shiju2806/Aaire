#!/usr/bin/env python3
"""
AAIRE Startup Script
Automatically chooses the best startup mode based on available dependencies
"""

import sys
import os
import subprocess
import importlib.util

# Load environment variables from .env file before checking
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed - .env file not loaded")
    print("   Environment variables must be set in shell")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

def check_module(module_name):
    """Check if a module can be imported"""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ImportError:
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'llama_index',
        'pinecone',
        'openai',
        'redis',
        'sqlalchemy'
    ]
    
    missing = []
    for module in required_modules:
        if not check_module(module):
            missing.append(module)
    
    return missing

def check_environment():
    """Check if required environment variables are set"""
    required_env = ['OPENAI_API_KEY']
    optional_env = ['PINECONE_API_KEY', 'FRED_API_KEY']
    
    missing_required = []
    missing_optional = []
    
    # Check required variables
    for var in required_env:
        value = os.getenv(var)
        if not value or value == f"your_{var.lower()}_here":
            missing_required.append(var)
    
    # Check optional variables
    for var in optional_env:
        value = os.getenv(var)
        if not value or value == f"your_{var.lower()}_here":
            missing_optional.append(var)
    
    # Check if .env file exists
    if not os.path.exists('.env') and (missing_required or missing_optional):
        print("💡 No .env file found. Create one with: cp .env.example .env")
    
    return missing_required, missing_optional

def main():
    print("🚀 AAIRE Startup Diagnostics")
    print("=" * 50)
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("📦 To install: pip install -r requirements.txt")
    else:
        print("✅ All Python dependencies available")
    
    # Check environment
    missing_required, missing_optional = check_environment()
    
    if missing_required:
        print(f"❌ Missing required environment variables: {', '.join(missing_required)}")
    else:
        print("✅ Required environment variables set")
    
    if missing_optional:
        print(f"⚠️  Missing optional environment variables: {', '.join(missing_optional)}")
        print("   (These will limit functionality but won't prevent startup)")
    
    print()
    
    # Decide startup mode
    if missing_deps or missing_required:
        print("🔧 Starting in SIMPLE MODE (limited functionality)")
        print("   - Basic API endpoints available")
        print("   - Sample responses for testing")
        print("   - No AI/RAG functionality")
        print()
        print("🌐 Access web interface at: http://localhost:8000")
        print("📊 Access API docs at: http://localhost:8000/api/docs")
        print("🔄 Configure missing items above and restart for full functionality")
        print()
        
        try:
            import uvicorn
            from simple_main import app
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        except Exception as e:
            print(f"❌ Failed to start simple mode: {e}")
            print("💡 Try: python simple_main.py")
            sys.exit(1)
    
    else:
        print("🎉 Starting in FULL MODE")
        print("   - Complete RAG pipeline")
        print("   - AI-powered responses")
        print("   - Document processing")
        print("   - External API integration")
        print()
        print("🌐 Access web interface at: http://localhost:8000")
        print("📊 Access API docs at: http://localhost:8000/api/docs")
        print()
        
        try:
            import uvicorn
            from main import app
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        except Exception as e:
            print(f"❌ Failed to start full mode: {e}")
            print("🔄 Falling back to simple mode...")
            print()
            
            try:
                from simple_main import app
                uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
            except Exception as e2:
                print(f"❌ Simple mode also failed: {e2}")
                print("💡 Please check your installation and try again")
                sys.exit(1)

if __name__ == "__main__":
    main()