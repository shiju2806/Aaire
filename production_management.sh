#!/bin/bash

# AAIRE Production Management Script
# Useful commands for managing your production deployment

case "$1" in
    "start")
        echo "🚀 Starting AAIRE services..."
        sudo systemctl start nginx
        sudo systemctl start aaire.service
        echo "✅ Services started"
        ;;
    "stop")
        echo "🛑 Stopping AAIRE services..."
        sudo systemctl stop aaire.service
        sudo systemctl stop nginx
        echo "✅ Services stopped"
        ;;
    "restart")
        echo "🔄 Restarting AAIRE services..."
        sudo systemctl restart aaire.service
        sudo systemctl restart nginx
        echo "✅ Services restarted"
        ;;
    "status")
        echo "📊 Service Status:"
        echo "Nginx:"
        sudo systemctl status nginx --no-pager
        echo ""
        echo "AAIRE:"
        sudo systemctl status aaire.service --no-pager
        ;;
    "logs")
        echo "📝 AAIRE Application Logs:"
        sudo journalctl -u aaire.service -f
        ;;
    "nginx-logs")
        echo "📝 Nginx Logs:"
        sudo tail -f /var/log/nginx/access.log /var/log/nginx/error.log
        ;;
    "deploy")
        echo "📦 Deploying latest code..."
        git pull origin main
        sudo systemctl restart aaire.service
        echo "✅ Deployment complete"
        ;;
    "ssl")
        echo "🔐 Setting up SSL certificate..."
        sudo certbot --nginx -d aaire.xyz -d www.aaire.xyz
        ;;
    "renew-ssl")
        echo "🔐 Renewing SSL certificate..."
        sudo certbot renew
        sudo systemctl reload nginx
        ;;
    "test")
        echo "🧪 Testing AAIRE deployment..."
        echo "Local health check:"
        curl -s http://localhost:8000/health || echo "❌ Local service not responding"
        echo ""
        echo "Nginx proxy test:"
        curl -s -I http://localhost/health || echo "❌ Nginx proxy not working"
        echo ""
        echo "External HTTPS test:"
        curl -s -I https://aaire.xyz/health || echo "❌ HTTPS not working"
        ;;
    "backup")
        echo "💾 Creating configuration backup..."
        sudo mkdir -p /home/ec2-user/backups
        sudo cp /etc/nginx/conf.d/aaire.conf /home/ec2-user/backups/aaire.conf.bak
        sudo cp /etc/systemd/system/aaire.service /home/ec2-user/backups/aaire.service.bak
        echo "✅ Backup created in /home/ec2-user/backups/"
        ;;
    *)
        echo "🛠️  AAIRE Production Management"
        echo ""
        echo "Usage: ./production_management.sh [command]"
        echo ""
        echo "Commands:"
        echo "  start        - Start all services"
        echo "  stop         - Stop all services"
        echo "  restart      - Restart all services"
        echo "  status       - Show service status"
        echo "  logs         - Show AAIRE application logs"
        echo "  nginx-logs   - Show Nginx logs"
        echo "  deploy       - Deploy latest code from git"
        echo "  ssl          - Set up SSL certificate"
        echo "  renew-ssl    - Renew SSL certificate"
        echo "  test         - Test deployment"
        echo "  backup       - Backup configurations"
        echo ""
        ;;
esac