#!/bin/bash

# Fix nginx configuration for AAIRE
# This creates a working HTTP-first config, then adds SSL

echo "🔧 Fixing nginx configuration..."

# Create a working HTTP-only configuration first
echo "📝 Creating HTTP-only nginx configuration..."
sudo tee /etc/nginx/conf.d/aaire.conf > /dev/null <<EOF
server {
    listen 80;
    server_name aaire.xyz www.aaire.xyz;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
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

# Test the configuration
echo "🔍 Testing nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Nginx configuration is valid"
    sudo systemctl reload nginx
    echo "✅ Nginx reloaded successfully"
else
    echo "❌ Nginx configuration still has errors"
    exit 1
fi

# Check if AAIRE is running
echo "🔍 Checking AAIRE service..."
if systemctl is-active --quiet aaire.service; then
    echo "✅ AAIRE service is running"
else
    echo "🚀 Starting AAIRE service..."
    sudo systemctl start aaire.service
    sleep 3
    if systemctl is-active --quiet aaire.service; then
        echo "✅ AAIRE service started"
    else
        echo "❌ AAIRE service failed to start"
        sudo journalctl -u aaire.service --no-pager -n 20
    fi
fi

# Test the setup
echo "🧪 Testing the setup..."
echo "Testing localhost:8000..."
curl -s http://localhost:8000/health && echo "✅ AAIRE app working" || echo "❌ AAIRE app not responding"

echo "Testing nginx proxy..."
curl -s http://localhost/health && echo "✅ Nginx proxy working" || echo "❌ Nginx proxy not working"

echo "Testing external access..."
curl -s http://aaire.xyz/health && echo "✅ External access working" || echo "❌ External access not working"

echo ""
echo "🎯 HTTP setup complete!"
echo "Your site should now work at: http://aaire.xyz"
echo ""
echo "To add HTTPS later, run:"
echo "sudo certbot --nginx -d aaire.xyz -d www.aaire.xyz"
echo ""
echo "This will automatically:"
echo "1. Generate SSL certificates"
echo "2. Update nginx config for HTTPS"
echo "3. Set up automatic renewal"
EOF