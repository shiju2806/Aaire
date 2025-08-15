#!/bin/bash
# AAIRE HTTPS Setup for Amazon Linux 2023 with ec2-user
# Usage: ./setup-https-nginx-al2023.sh aaire.xyz

set -e

DOMAIN=$1
if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 <domain-name>"
    echo "Example: $0 aaire.xyz"
    exit 1
fi

echo "🚀 Setting up HTTPS for AAIRE on domain: $DOMAIN (Amazon Linux 2023)"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

USER_HOME="/home/ec2-user"
WEB_USER="ec2-user"

echo "🏠 User home: $USER_HOME"

# Update system packages
echo "📦 Updating system packages..."
dnf update -y

# Install Nginx
echo "🌐 Installing Nginx..."
dnf install nginx -y
systemctl enable nginx
systemctl start nginx

# Install Certbot for Let's Encrypt
echo "🔒 Installing Certbot..."
dnf install python3-certbot python3-certbot-nginx -y

# Create Nginx configuration
echo "⚙️  Creating Nginx configuration..."
cat > /etc/nginx/conf.d/aaire.conf << 'EOF'
server {
    listen 80;
    server_name aaire.xyz www.aaire.xyz;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    # Serve static files directly
    location /static/ {
        alias /home/ec2-user/aaire/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Main application
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
        
        # CORS headers
        add_header Access-Control-Allow-Origin "https://aaire.xyz" always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization" always;
        
        if ($request_method = 'OPTIONS') {
            return 204;
        }
    }
    
    # WebSocket support for real-time chat
    location /api/v1/chat/ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # API endpoints
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_send_timeout 300;
        
        # CORS for API
        add_header Access-Control-Allow-Origin "https://aaire.xyz" always;
        add_header Access-Control-Allow-Credentials true always;
    }
    
    # Health check
    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
EOF

# Remove default nginx config that might conflict
rm -f /etc/nginx/conf.d/default.conf 2>/dev/null || true

# Test Nginx configuration
nginx -t

# Reload Nginx
systemctl reload nginx

# Get SSL certificate
echo "🔐 Obtaining SSL certificate from Let's Encrypt..."
certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN --redirect

# Setup auto-renewal
echo "🔄 Setting up SSL certificate auto-renewal..."
(crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/certbot renew --quiet") | crontab -

# Create systemd service for AAIRE (if not exists)
if [ ! -f /etc/systemd/system/aaire.service ]; then
    echo "🎯 Creating AAIRE systemd service..."
    cat > /etc/systemd/system/aaire.service << EOF
[Unit]
Description=AAIRE - AI Insurance Accounting Assistant
After=network.target

[Service]
Type=simple
User=$WEB_USER
WorkingDirectory=$USER_HOME/aaire
ExecStart=/usr/bin/python3 start.py
Restart=always
RestartSec=10
Environment=PATH=/usr/bin:/usr/local/bin
Environment=PYTHONPATH=$USER_HOME/aaire

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable aaire
fi

# Update AAIRE configuration for HTTPS
echo "⚙️  Updating AAIRE configuration for HTTPS..."
cd $USER_HOME/aaire

# Create/update .env file for production
if [ -f .env ]; then
    # Update existing .env
    sed -i "s|ALLOWED_ORIGINS=.*|ALLOWED_ORIGINS=https://$DOMAIN,https://www.$DOMAIN|" .env
    sed -i "s|DOMAIN=.*|DOMAIN=$DOMAIN|" .env
    grep -q "SSL_ENABLED" .env || echo "SSL_ENABLED=true" >> .env
else
    echo "⚠️  .env file not found. Please create one with your API keys and configuration."
fi

# Change ownership to correct user
chown -R $WEB_USER:$WEB_USER $USER_HOME/aaire

# Restart AAIRE service
echo "🔄 Starting AAIRE service..."
systemctl restart aaire

# Test the setup
echo "🧪 Testing the setup..."
sleep 5

if curl -s -f https://$DOMAIN/health > /dev/null; then
    echo "✅ Success! AAIRE is now running on https://$DOMAIN"
    echo ""
    echo "🎉 Setup completed successfully!"
    echo "📋 Summary:"
    echo "   - Domain: https://$DOMAIN"  
    echo "   - SSL: Let's Encrypt (auto-renewing)"
    echo "   - Web Server: Nginx (reverse proxy)"
    echo "   - Service: systemd (auto-starting)"
    echo "   - User: $WEB_USER"
    echo ""
    echo "🔍 Useful commands:"
    echo "   - Check status: systemctl status aaire"
    echo "   - View logs: journalctl -u aaire -f"  
    echo "   - Nginx logs: tail -f /var/log/nginx/access.log"
    echo "   - SSL renewal: certbot renew"
else
    echo "❌ Setup completed but health check failed."
    echo "Please check:"
    echo "   - AAIRE service: systemctl status aaire"
    echo "   - Nginx status: systemctl status nginx"
    echo "   - Nginx logs: tail /var/log/nginx/error.log"
fi

echo ""
echo "⚠️  Don't forget to:"
echo "   1. Test all functionality on https://$DOMAIN"
echo "   2. Update frontend URLs if needed"