#!/bin/bash
# AAIRE HTTPS Setup with Nginx + Let's Encrypt for ec2-user
# Usage: ./setup-https-nginx-ec2user.sh aaire.xyz

set -e

DOMAIN=$1
if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 <domain-name>"
    echo "Example: $0 aaire.xyz"
    exit 1
fi

echo "🚀 Setting up HTTPS for AAIRE on domain: $DOMAIN (ec2-user version)"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Detect the OS and package manager
if [ -f /etc/amazon-linux-release ] || [ -f /etc/system-release ]; then
    # Amazon Linux
    PKG_MANAGER="yum"
    NGINX_SERVICE="nginx"
    USER_HOME="/home/ec2-user"
    WEB_USER="ec2-user"
elif [ -f /etc/redhat-release ]; then
    # RHEL/CentOS
    PKG_MANAGER="yum"
    NGINX_SERVICE="nginx"
    USER_HOME="/home/ec2-user"
    WEB_USER="ec2-user"
elif [ -f /etc/debian_version ]; then
    # Ubuntu/Debian
    PKG_MANAGER="apt"
    NGINX_SERVICE="nginx"
    USER_HOME="/home/ubuntu"
    WEB_USER="ubuntu"
else
    echo "⚠️  Unsupported OS. This script supports Amazon Linux, RHEL/CentOS, and Ubuntu/Debian."
    exit 1
fi

echo "📦 Detected OS: Using $PKG_MANAGER package manager"
echo "🏠 User home: $USER_HOME"

# Update system packages
echo "📦 Updating system packages..."
if [ "$PKG_MANAGER" = "yum" ]; then
    yum update -y
    # Enable EPEL repository for additional packages
    amazon-linux-extras install epel -y 2>/dev/null || yum install epel-release -y
elif [ "$PKG_MANAGER" = "apt" ]; then
    apt update && apt upgrade -y
fi

# Install Nginx
echo "🌐 Installing Nginx..."
if [ "$PKG_MANAGER" = "yum" ]; then
    yum install nginx -y
elif [ "$PKG_MANAGER" = "apt" ]; then
    apt install nginx -y
fi

systemctl enable nginx
systemctl start nginx

# Install Certbot for Let's Encrypt
echo "🔒 Installing Certbot..."
if [ "$PKG_MANAGER" = "yum" ]; then
    yum install certbot python3-certbot-nginx -y
elif [ "$PKG_MANAGER" = "apt" ]; then
    apt install certbot python3-certbot-nginx -y
fi

# Create Nginx configuration
echo "⚙️  Creating Nginx configuration..."
cat > /etc/nginx/conf.d/aaire.conf << EOF
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;
    
    # Serve static files directly
    location /static/ {
        alias $USER_HOME/AAIRE/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Main application
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
        
        # CORS headers
        add_header Access-Control-Allow-Origin "https://$DOMAIN" always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization" always;
        
        if (\$request_method = 'OPTIONS') {
            return 204;
        }
    }
    
    # WebSocket support for real-time chat
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # API endpoints
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300;
        proxy_send_timeout 300;
        
        # CORS for API
        add_header Access-Control-Allow-Origin "https://$DOMAIN" always;
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
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
rm -f /etc/nginx/conf.d/default.conf 2>/dev/null || true

# Test Nginx configuration
nginx -t

# Reload Nginx
systemctl reload nginx

# Get SSL certificate
echo "🔐 Obtaining SSL certificate from Let's Encrypt..."
echo "Make sure your domain $DOMAIN points to this server's IP address!"
read -p "Press Enter when DNS is configured..."

certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN --redirect

# Setup auto-renewal
echo "🔄 Setting up SSL certificate auto-renewal..."
(crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/certbot renew --quiet") | crontab -

# Update firewall (Amazon Linux uses different firewall)
if command -v firewall-cmd &> /dev/null; then
    echo "🛡️  Updating firewall rules (firewalld)..."
    firewall-cmd --permanent --add-service=http
    firewall-cmd --permanent --add-service=https
    firewall-cmd --reload
elif command -v ufw &> /dev/null; then
    echo "🛡️  Updating firewall rules (ufw)..."
    ufw allow 'Nginx Full'
    ufw delete allow 'Nginx HTTP' 2>/dev/null || true
fi

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
WorkingDirectory=$USER_HOME/AAIRE
ExecStart=/usr/bin/python3 start.py
Restart=always
RestartSec=10
Environment=PATH=/usr/bin:/usr/local/bin
Environment=PYTHONPATH=$USER_HOME/AAIRE

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable aaire
fi

# Update AAIRE configuration for HTTPS
echo "⚙️  Updating AAIRE configuration for HTTPS..."
cd $USER_HOME/AAIRE

# Create/update .env file for production
if [ -f .env ]; then
    # Update existing .env
    sed -i "s|ALLOWED_ORIGINS=.*|ALLOWED_ORIGINS=https://$DOMAIN,https://www.$DOMAIN|" .env
    sed -i "s|DOMAIN=.*|DOMAIN=$DOMAIN|" .env
    echo "SSL_ENABLED=true" >> .env
else
    echo "⚠️  .env file not found. Please create one with your API keys and configuration."
fi

# Change ownership to correct user
chown -R $WEB_USER:$WEB_USER $USER_HOME/AAIRE

# Restart AAIRE service
echo "🔄 Restarting AAIRE service..."
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
echo "   1. Update your domain's DNS to point to this server"
echo "   2. Configure your .env file with API keys"
echo "   3. Test all functionality on https://$DOMAIN"