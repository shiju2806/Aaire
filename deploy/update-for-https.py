#!/usr/bin/env python3
"""
Update AAIRE configuration for HTTPS domain hosting
Usage: python3 update-for-https.py your-domain.com
"""

import sys
import os
import re
from pathlib import Path

def update_main_py(domain):
    """Update main.py with HTTPS domain configuration"""
    main_files = ['main.py', 'start.py']
    
    for main_file in main_files:
        if os.path.exists(main_file):
            print(f"📝 Updating {main_file}...")
            
            with open(main_file, 'r') as f:
                content = f.read()
            
            # Update CORS origins
            cors_pattern = r'allow_origins=\["[^"]*"\]'
            new_cors = f'allow_origins=["https://{domain}", "https://www.{domain}"]'
            content = re.sub(cors_pattern, new_cors, content)
            
            # Add security middleware if not present
            security_middleware = f'''
# Security middleware for HTTPS
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY" 
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self' https://{domain}; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'"
    return response
'''
            
            if "add_security_headers" not in content:
                # Insert after imports but before app creation
                app_creation_pattern = r'(from fastapi\.responses import.*?\n)(.*?)(app = FastAPI)'
                if re.search(app_creation_pattern, content, re.DOTALL):
                    content = re.sub(app_creation_pattern, r'\1\2' + security_middleware + r'\3', content, flags=re.DOTALL)
            
            with open(main_file, 'w') as f:
                f.write(content)
            
            print(f"✅ Updated {main_file}")
            break

def update_frontend_config(domain):
    """Update frontend configuration for new domain"""
    frontend_files = [
        'static/js/app.js',
        'templates/index.html'
    ]
    
    for file_path in frontend_files:
        if os.path.exists(file_path):
            print(f"📝 Updating {file_path}...")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Update API endpoints
            # Replace IP-based URLs with domain-based URLs
            ip_pattern = r'http://\d+\.\d+\.\d+\.\d+:8000'
            new_url = f'https://{domain}'
            content = re.sub(ip_pattern, new_url, content)
            
            # Update WebSocket URLs
            ws_pattern = r'ws://\d+\.\d+\.\d+\.\d+:8000'
            new_ws_url = f'wss://{domain}'
            content = re.sub(ws_pattern, new_ws_url, content)
            
            # Update localhost references
            localhost_pattern = r'http://localhost:8000'
            content = re.sub(localhost_pattern, new_url, content)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"✅ Updated {file_path}")

def update_env_file(domain):
    """Update .env file with HTTPS configuration"""
    env_file = '.env'
    env_config = {
        'ALLOWED_ORIGINS': f'https://{domain},https://www.{domain}',
        'DOMAIN': domain,
        'SSL_ENABLED': 'true',
        'SECURE_COOKIES': 'true'
    }
    
    if os.path.exists(env_file):
        print("📝 Updating .env file...")
        
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Update existing keys or add new ones
        updated_keys = set()
        for i, line in enumerate(lines):
            for key, value in env_config.items():
                if line.startswith(f'{key}='):
                    lines[i] = f'{key}={value}\n'
                    updated_keys.add(key)
                    break
        
        # Add new keys that weren't found
        for key, value in env_config.items():
            if key not in updated_keys:
                lines.append(f'{key}={value}\n')
        
        with open(env_file, 'w') as f:
            f.writelines(lines)
        
        print("✅ Updated .env file")
    else:
        print("⚠️  .env file not found, creating with basic HTTPS config...")
        with open(env_file, 'w') as f:
            f.write("# HTTPS Configuration\n")
            for key, value in env_config.items():
                f.write(f'{key}={value}\n')
            f.write("\n# Add your API keys here:\n")
            f.write("# OPENAI_API_KEY=your_key_here\n")
            f.write("# QDRANT_URL=your_qdrant_url\n")
            f.write("# QDRANT_API_KEY=your_qdrant_key\n")

def create_systemd_service(domain):
    """Create systemd service file for AAIRE"""
    service_content = f"""[Unit]
Description=AAIRE - AI Insurance Accounting Assistant  
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/AAIRE
ExecStart=/usr/bin/python3 start.py
Restart=always
RestartSec=10
Environment=PATH=/usr/bin:/usr/local/bin
Environment=PYTHONPATH=/home/ubuntu/AAIRE
Environment=DOMAIN={domain}
Environment=SSL_ENABLED=true

[Install]
WantedBy=multi-user.target
"""
    
    service_file = 'deploy/aaire.service'
    with open(service_file, 'w') as f:
        f.write(service_content)
    
    print(f"📝 Created systemd service file: {service_file}")
    print("   To install: sudo cp deploy/aaire.service /etc/systemd/system/")
    print("   To enable: sudo systemctl enable aaire && sudo systemctl start aaire")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 update-for-https.py your-domain.com")
        print("Example: python3 update-for-https.py aaire.company.com")
        sys.exit(1)
    
    domain = sys.argv[1]
    
    print(f"🚀 Updating AAIRE configuration for HTTPS domain: {domain}")
    print()
    
    # Update configuration files
    update_main_py(domain)
    update_frontend_config(domain)
    update_env_file(domain)
    create_systemd_service(domain)
    
    print()
    print("🎉 Configuration update completed!")
    print()
    print("📋 Next Steps:")
    print("1. Review the updated files")
    print("2. Test the application locally")
    print("3. Deploy to your server")
    print("4. Install systemd service if using systemctl")
    print()
    print("⚠️  Don't forget to:")
    print("- Add your API keys to .env file")
    print("- Configure your domain's DNS")
    print("- Test all functionality after deployment")

if __name__ == "__main__":
    main()