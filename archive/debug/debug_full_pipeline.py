#!/usr/bin/env python3
"""
Comprehensive debug script to identify document processing issues
"""

import os
import sys
from pathlib import Path

def debug_environment():
    """Check environment and configuration"""
    print("=== ENVIRONMENT DEBUG ===")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check for essential files
    essential_files = ['.env', 'main.py', 'src/rag_pipeline.py', 'src/document_processor.py']
    for file_path in essential_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
    
    # Check data directory structure
    data_dir = Path('data')
    if data_dir.exists():
        print(f"✅ data directory exists")
        subdirs = ['uploads', 'analytics', 'workflows']
        for subdir in subdirs:
            subdir_path = data_dir / subdir
            if subdir_path.exists():
                file_count = len(list(subdir_path.iterdir()))
                print(f"  ✅ {subdir}: {file_count} files")
            else:
                print(f"  ❌ {subdir}: missing")
    else:
        print(f"❌ data directory missing")

def debug_dependencies():
    """Check if required dependencies are available"""
    print("\n=== DEPENDENCIES DEBUG ===")
    
    required_modules = [
        'fastapi', 'uvicorn', 'openai', 'qdrant_client', 
        'llama_index', 'PyPDF2', 'redis', 'structlog'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")

def debug_config():
    """Check configuration issues"""
    print("\n=== CONFIGURATION DEBUG ===")
    
    # Check .env file format
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file exists")
        try:
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Check for common formatting issues
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' not in line:
                        print(f"⚠️  Line {i}: No '=' found: {line}")
                    elif line.count('=') > 1:
                        print(f"⚠️  Line {i}: Multiple '=' found: {line}")
                    elif '#' in line and not line.startswith('#'):
                        print(f"⚠️  Line {i}: Inline comment may cause issues: {line}")
            
            # Check for required keys
            required_keys = ['OPENAI_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
            for key in required_keys:
                if key in content:
                    print(f"✅ {key} present")
                else:
                    print(f"❌ {key} missing")
                    
        except Exception as e:
            print(f"❌ Error reading .env: {e}")
    else:
        print("❌ .env file not found")

def debug_permissions():
    """Check file permissions"""
    print("\n=== PERMISSIONS DEBUG ===")
    
    # Check write permissions for data directories
    dirs_to_check = ['data/uploads', 'data/analytics']
    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            if os.access(path, os.W_OK):
                print(f"✅ {dir_path}: writable")
            else:
                print(f"❌ {dir_path}: not writable")
        else:
            print(f"❌ {dir_path}: doesn't exist")

def debug_recent_activity():
    """Check for recent activity and logs"""
    print("\n=== RECENT ACTIVITY DEBUG ===")
    
    # Check server logs
    log_file = Path('server.log')
    if log_file.exists():
        print("✅ server.log exists")
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Show last 10 lines
            print("Last 10 log entries:")
            for line in lines[-10:]:
                print(f"  {line.strip()}")
                
        except Exception as e:
            print(f"❌ Error reading server.log: {e}")
    else:
        print("❌ server.log not found")
    
    # Check for recent uploads
    uploads_dir = Path('data/uploads')
    if uploads_dir.exists():
        files = list(uploads_dir.iterdir())
        if files:
            # Sort by modification time
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            print(f"\nMost recent uploads:")
            for f in files[:5]:
                import datetime
                mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  {f.name}: {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({size_mb:.2f} MB)")
        else:
            print("❌ No files in uploads directory")

if __name__ == "__main__":
    debug_environment()
    debug_dependencies()
    debug_config()
    debug_permissions()
    debug_recent_activity()
    print("\n=== DEBUG COMPLETE ===")
    print("Run this script on your EC2 instance to identify the specific issue.")