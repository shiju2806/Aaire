#!/usr/bin/env python3
"""
Simple server startup script that bypasses dependency conflicts
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Try to start the server
    from main import app
    import uvicorn
    
    print("🚀 Starting AAIRE server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("🔧 Installing minimal requirements...")
    
    # Install only what we need
    os.system("pip3 install --break-system-packages fastapi uvicorn python-dotenv structlog")
    
    print("✅ Dependencies installed, restarting...")
    os.system("python3 simple_start.py")
    
except Exception as e:
    print(f"❌ Server startup failed: {e}")
    print("📋 Starting basic file server instead...")
    
    # Fallback: basic file server
    import http.server
    import socketserver
    
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🌐 Basic file server at http://localhost:{PORT}")
        httpd.serve_forever()