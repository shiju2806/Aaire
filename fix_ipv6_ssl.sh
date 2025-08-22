#!/bin/bash

# Fix IPv6 SSL configuration issue
echo "🔧 Fixing IPv6 SSL configuration..."

# The issue: nginx is listening on IPv4:443 but not IPv6:443
# This causes connection resets for external HTTPS connections

echo "📝 Creating fixed nginx configuration with proper IPv6 SSL support..."
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
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
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
    echo "🔄 Restarting nginx to apply IPv6 SSL fix..."
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
echo "External HTTPS test (this should now work):"
timeout 10 curl -s -I https://aaire.xyz/health | head -1

echo ""
echo "🎯 IPv6 SSL fix complete!"
echo "Your site should now work at: https://aaire.xyz"
echo ""
echo "The issue was that nginx was only listening on IPv4:443, not IPv6:443"
echo "This caused connection resets for external HTTPS connections"