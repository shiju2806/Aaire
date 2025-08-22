#!/bin/bash

# AAIRE Production Deployment Script
# Run this on your EC2 server

echo "🚀 Starting AAIRE production deployment..."

# Kill any existing processes
echo "Stopping existing processes..."
sudo pkill -f "python3 main.py" || echo "No existing processes found"
sleep 2

# Pull latest code
echo "Pulling latest code from GitHub..."
git pull origin main

# Install/update dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt --user

# Check if port 8000 is free
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Port 8000 is still in use, force killing..."
    sudo lsof -ti :8000 | xargs sudo kill -9
    sleep 2
fi

# Start server with proper logging
echo "Starting AAIRE server..."
nohup python3 main.py > aaire.log 2>&1 &

# Wait a moment for startup
sleep 5

# Check if server started successfully
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ AAIRE server started successfully!"
    echo "📝 Server logs: tail -f aaire.log"
    echo "🌐 Server URL: http://18.119.14.61:8000"
    echo "🔍 Health check: curl http://18.119.14.61:8000/health"
else
    echo "❌ Server failed to start. Check logs:"
    tail aaire.log
fi

echo "🎯 Deployment complete!"