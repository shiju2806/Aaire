#!/bin/bash

# EC2 Setup Verification Script
# Run this on your EC2 server to check configuration

echo "🔍 Checking EC2 setup for AAIRE..."

echo "1. Checking current processes on port 8000:"
sudo lsof -i :8000 || echo "Port 8000 is free"

echo -e "\n2. Checking if Python/AAIRE is running:"
ps aux | grep python | grep -v grep || echo "No Python processes found"

echo -e "\n3. Checking network connectivity:"
curl -I http://localhost:8000/health 2>/dev/null || echo "Server not responding on port 8000"

echo -e "\n4. Checking firewall status:"
sudo iptables -L | head -5

echo -e "\n5. Checking security group (if possible):"
curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null && echo "Instance ID found" || echo "Not on EC2 or metadata service unavailable"

echo -e "\n6. Checking system resources:"
free -h
df -h /

echo -e "\n7. Checking recent logs:"
if [ -f "aaire.log" ]; then
    echo "Recent AAIRE logs:"
    tail -10 aaire.log
else
    echo "No aaire.log found"
fi

echo -e "\n🎯 Setup check complete!"
echo "If server is not accessible from outside:"
echo "1. Check AWS Security Group allows port 8000 inbound"
echo "2. Check if EC2 firewall (iptables) allows port 8000"
echo "3. Verify server is binding to 0.0.0.0:8000 not 127.0.0.1:8000"