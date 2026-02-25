#!/bin/bash

# Fix SSL compatibility issues with external clients
echo "🔧 Fixing SSL compatibility for external clients..."

# The issue: TLS handshake fails after Client Hello
# This suggests cipher suite or SSL configuration incompatibility

echo "📝 Creating SSL-compatible nginx configuration..."
sudo tee /etc/nginx/conf.d/aaire.conf > /dev/null <<'EOF'
server {
    listen 80;
    listen [::]:80;
    server_name aaire.xyz www.aaire.xyz;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name aaire.xyz www.aaire.xyz;
    
    # SSL Configuration (managed by certbot)
    ssl_certificate /etc/letsencrypt/live/aaire.xyz/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aaire.xyz/privkey.pem;
    
    # More compatible SSL settings (remove strict options-ssl-nginx.conf)
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;
    
    # Remove strict HSTS for now
    # add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # Basic security headers only
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # Client max body size for file uploads
    client_max_body_size 100M;
    
    # Proxy all requests to AAIRE application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

echo "🔍 Testing nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Nginx configuration is valid"
    echo "🔄 Restarting nginx to apply compatibility fixes..."
    sudo systemctl restart nginx
    echo "✅ Nginx restarted"
else
    echo "❌ Nginx configuration has errors"
    exit 1
fi

echo "📊 Checking nginx listening ports..."
sudo ss -tlnp | grep nginx

echo ""
echo "🧪 Testing the fix..."
echo "Local HTTPS test:"
curl -s -I https://localhost/health -k | head -1

echo ""
echo "🎯 SSL compatibility fix complete!"
echo "The fix removes the strict certbot SSL options that may be incompatible with some clients"
echo "Try accessing https://aaire.xyz again"