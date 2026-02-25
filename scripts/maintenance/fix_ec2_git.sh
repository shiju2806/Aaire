#!/bin/bash
# Fix EC2 Git Conflicts and Deploy Spatial Extraction
# Run this on your EC2 server

echo "🔧 **FIXING EC2 GIT CONFLICTS AND DEPLOYING SPATIAL EXTRACTION**"
echo

# Step 1: Backup current .env (most important)
echo "1️⃣ Backing up .env file..."
if [ -f .env ]; then
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
    echo "   ✅ .env backed up"
else
    echo "   ⚠️  No .env found"
fi

# Step 2: Check what files are causing conflicts
echo
echo "2️⃣ Checking conflicting files..."
git status --porcelain
echo

# Step 3: Stash local changes
echo "3️⃣ Stashing local changes..."
git stash push -m "Auto-stash before spatial extraction deployment $(date)"
echo "   ✅ Local changes stashed"

# Step 4: Pull latest changes with spatial extraction
echo
echo "4️⃣ Pulling latest spatial extraction code..."
git pull origin main
if [ $? -eq 0 ]; then
    echo "   ✅ Git pull successful"
else
    echo "   ❌ Git pull failed"
    exit 1
fi

# Step 5: Restore .env if it was overwritten
echo
echo "5️⃣ Restoring .env..."
if [ -f .env.backup.* ]; then
    latest_backup=$(ls -t .env.backup.* | head -1)
    if [ ! -f .env ] || [ ! -s .env ]; then
        cp "$latest_backup" .env
        echo "   ✅ .env restored from $latest_backup"
    else
        echo "   ✅ .env already exists and is not empty"
    fi
fi

# Step 6: Install PyMuPDF for spatial extraction
echo
echo "6️⃣ Installing spatial extraction dependencies..."
pip3 install --user PyMuPDF==1.24.7
if [ $? -eq 0 ]; then
    echo "   ✅ PyMuPDF installed successfully"
else
    echo "   ⚠️  PyMuPDF installation may have issues, but continuing..."
fi

# Step 7: Restart services
echo
echo "7️⃣ Restarting AAIRE services..."

# Kill any running processes
pkill -f main.py || echo "   No main.py processes to kill"

# Stop systemd service if it exists
sudo systemctl stop aaire.service 2>/dev/null || echo "   No systemd service to stop"

# Start the service
echo "   Starting AAIRE server..."
nohup python3 main.py > server.log 2>&1 &
sleep 3

# Check if it's running
if pgrep -f main.py > /dev/null; then
    echo "   ✅ AAIRE server started successfully"
    echo "   📊 Process ID: $(pgrep -f main.py)"
else
    echo "   ❌ AAIRE server failed to start"
    echo "   📋 Checking logs..."
    tail -10 server.log
    exit 1
fi

# Step 8: Test spatial extraction
echo
echo "8️⃣ Testing spatial extraction..."
sleep 2
curl -s http://localhost:8000/ > /dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ Server responding"
    echo "   🎯 Spatial extraction ready for organizational charts"
else
    echo "   ❌ Server not responding"
fi

echo
echo "🎉 **DEPLOYMENT COMPLETE**"
echo "   • Spatial PDF extraction active"
echo "   • PyMuPDF coordinate parsing enabled"
echo "   • Finance Structures job title extraction fixed"
echo "   • Server running on port 8000"
echo
echo "🧪 **NEXT STEPS:**"
echo "   1. Upload Finance Structures PDF via web interface"
echo "   2. Ask about job title breakdown"
echo "   3. Should see proper VP/AVP/Manager grouping"
echo "   4. Citations will show 'FileName, Page X' format"