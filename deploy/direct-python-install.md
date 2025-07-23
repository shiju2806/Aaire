# 🐍 Direct Python Installation Guide (Docker Alternative)

If Docker is having issues, you can run AAIRE directly with Python on your EC2 instance.

## Quick Direct Installation

### 1. SSH into your instance
```bash
ssh -i your-key.pem ec2-user@your-instance-ip
```

### 2. Install system dependencies
```bash
# For Amazon Linux 2023
sudo yum update -y
sudo yum install -y python3 python3-pip git gcc python3-devel

# For Ubuntu (if using Ubuntu AMI)
# sudo apt update && sudo apt install -y python3 python3-pip git build-essential python3-dev
```

### 3. Clone and setup AAIRE
```bash
cd ~
git clone https://github.com/shiju2806/aaire.git
cd aaire

# Install Python dependencies
pip3 install --user -r requirements.txt
```

### 4. Configure environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

Add your actual API keys:
```bash
OPENAI_API_KEY=sk-your-actual-openai-key
PINECONE_API_KEY=your-actual-pinecone-key
PINECONE_ENVIRONMENT=us-east-1
FRED_API_KEY=your-fred-key  # optional
```

### 5. Test the application
```bash
# Run directly (for testing)
python3 start.py
```

Access at: `http://your-instance-ip:8000`

### 6. Set up as a service (Production)

Create systemd service:
```bash
sudo nano /etc/systemd/system/aaire.service
```

Add this content:
```ini
[Unit]
Description=AAIRE MVP Service
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/aaire
Environment=PATH=/home/ec2-user/.local/bin:/usr/local/bin:/bin:/usr/bin
ExecStart=/usr/bin/python3 start.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable aaire
sudo systemctl start aaire

# Check status
sudo systemctl status aaire

# View logs
sudo journalctl -u aaire -f
```

## 🔧 Troubleshooting

### Check Application Status
```bash
# Test health endpoint
curl http://localhost:8000/health

# Check what's running on port 8000
sudo netstat -tlnp | grep 8000

# View application logs
sudo journalctl -u aaire -f
```

### Common Issues

**1. Permission denied for pip install**
```bash
# Use --user flag
pip3 install --user -r requirements.txt

# Or create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Module not found errors**
```bash
# Check Python path
echo $PYTHONPATH

# Add current directory
export PYTHONPATH=/home/ec2-user/aaire:$PYTHONPATH
```

**3. Port 8000 already in use**
```bash
# Kill existing process
sudo kill $(sudo lsof -t -i:8000)

# Or use different port
export PORT=8080
python3 start.py
```

**4. Missing system dependencies**
```bash
# Install additional packages if needed
sudo yum install -y gcc python3-devel libffi-devel openssl-devel

# For compilation issues
sudo yum groupinstall -y "Development Tools"
```

## 🚀 Performance Tips

### 1. Use screen/tmux for long-running sessions
```bash
# Install screen
sudo yum install -y screen

# Start screen session
screen -S aaire

# Run application
python3 start.py

# Detach: Ctrl+A then D
# Reattach: screen -r aaire
```

### 2. Set up log rotation
```bash
# Create logrotate config
sudo nano /etc/logrotate.d/aaire
```

Add:
```
/var/log/aaire/*.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 0644 ec2-user ec2-user
}
```

### 3. Monitor resource usage
```bash
# Check CPU/Memory
htop

# Check disk space
df -h

# Check application performance
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/health"
```

## 📊 Monitoring

### Setup basic monitoring
```bash
# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "$(date): AAIRE is down, restarting..."
        sudo systemctl restart aaire
        sleep 30
    fi
    sleep 60
done
EOF

chmod +x monitor.sh

# Run in background
nohup ./monitor.sh &
```

This approach bypasses Docker entirely and should work reliably on any EC2 instance!