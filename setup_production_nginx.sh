#!/bin/bash

# AAIRE Production Setup with Nginx + SSL
# Run this on your EC2 server as sudo

echo "🚀 Setting up AAIRE production environment..."

# Update system
echo "📦 Updating system packages..."
sudo dnf update -y

# Install nginx and certbot
echo "📦 Installing nginx and SSL tools..."
sudo dnf install -y nginx certbot python3-certbot-nginx

# Start and enable nginx
echo "🔧 Starting nginx..."
sudo systemctl start nginx
sudo systemctl enable nginx

# Create nginx configuration for AAIRE
echo "📝 Creating nginx configuration..."
sudo tee /etc/nginx/conf.d/aaire.conf > /dev/null <<EOF
server {
    listen 80;
    server_name aaire.xyz www.aaire.xyz;
    
    # Redirect all HTTP to HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name aaire.xyz www.aaire.xyz;
    
    # SSL certificates (will be added by certbot)
    # ssl_certificate /etc/letsencrypt/live/aaire.xyz/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/aaire.xyz/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # Client max body size for file uploads
    client_max_body_size 100M;
    
    # Proxy to AAIRE application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Host \$host;
        proxy_set_header X-Forwarded-Port \$server_port;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files (if any)
    location /static/ {
        alias /home/ec2-user/aaire/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
EOF

# Test nginx configuration
echo "🔍 Testing nginx configuration..."
sudo nginx -t

if [ \$? -eq 0 ]; then
    echo "✅ Nginx configuration is valid"
    sudo systemctl reload nginx
else
    echo "❌ Nginx configuration error"
    exit 1
fi

# Set up SSL certificate
echo "🔐 Setting up SSL certificate with Let's Encrypt..."
echo "Note: Make sure your domain DNS is pointing to this server before running this!"
echo "Run this command manually after DNS is confirmed:"
echo "sudo certbot --nginx -d aaire.xyz -d www.aaire.xyz"

# Create AAIRE service file for systemd
echo "📝 Creating systemd service for AAIRE..."
sudo tee /etc/systemd/system/aaire.service > /dev/null <<EOF
[Unit]
Description=AAIRE AI Assistant API
After=network.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=/home/ec2-user/aaire
Environment=PATH=/home/ec2-user/.local/bin:/usr/bin
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start AAIRE service
echo "🚀 Setting up AAIRE as a system service..."
sudo systemctl daemon-reload
sudo systemctl enable aaire.service
sudo systemctl start aaire.service

# Check status
echo "📊 Checking service status..."
sudo systemctl status nginx
sudo systemctl status aaire.service

echo "🎯 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Verify your DNS points to this server: dig aaire.xyz"
echo "2. Run SSL setup: sudo certbot --nginx -d aaire.xyz -d www.aaire.xyz"
echo "3. Test: curl https://aaire.xyz/health"
echo ""
echo "Your site should be available at: https://aaire.xyz"
echo "Logs: sudo journalctl -u aaire.service -f"
EOF