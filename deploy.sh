#!/bin/bash
set -e

echo "🚀 **AAIRE PRODUCTION DEPLOYMENT SCRIPT**"
echo "   Handles environment, dependencies, and service management"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to backup and preserve .env
backup_env() {
    if [ -f .env ]; then
        cp .env .env.backup
        log_info "Backed up existing .env file"
    fi
}

# Function to restore .env if it got overwritten
restore_env() {
    if [ -f .env.backup ]; then
        if [ ! -f .env ] || [ ! -s .env ]; then
            cp .env.backup .env
            log_info "Restored .env from backup"
        fi
    fi
}

# Function to validate required environment variables
validate_env() {
    log_info "Validating environment variables..."
    
    if [ ! -f .env ]; then
        log_error ".env file not found!"
        echo "Please create .env with required variables:"
        echo "OPENAI_API_KEY=your_key_here"
        echo "OPENAI_MODEL=gpt-4o-mini"
        echo "REDIS_HOST=localhost"
        echo "REDIS_PORT=6379"
        exit 1
    fi
    
    # Source .env and check required variables
    source .env
    
    if [ -z "$OPENAI_API_KEY" ]; then
        log_error "OPENAI_API_KEY not set in .env"
        exit 1
    fi
    
    if [ -z "$OPENAI_MODEL" ]; then
        log_warn "OPENAI_MODEL not set, using default: gpt-4o-mini"
        echo "OPENAI_MODEL=gpt-4o-mini" >> .env
    fi
    
    log_info "Environment variables validated ✅"
}

# Function to update code
update_code() {
    log_info "Updating code from repository..."
    
    # Backup .env before git operations
    backup_env
    
    # Handle git conflicts gracefully
    if git status --porcelain | grep -q .; then
        log_warn "Local changes detected, stashing..."
        git stash push -m "Auto-stash before deployment $(date)"
    fi
    
    # Pull latest changes
    git pull origin main
    
    # Restore .env if needed
    restore_env
    
    log_info "Code updated ✅"
}

# Function to install/update dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Install/upgrade pip
    python3 -m pip install --upgrade pip --user
    
    # Install requirements
    pip3 install -r requirements.txt --user
    
    log_info "Dependencies installed ✅"
}

# Function to create systemd service
create_service() {
    log_info "Creating systemd service..."
    
    # Get current directory
    CURRENT_DIR=$(pwd)
    
    # Create service file
    sudo tee /etc/systemd/system/aaire.service > /dev/null << EOF
[Unit]
Description=AAIRE - AI Insurance Accounting Assistant
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$CURRENT_DIR
Environment=PATH=/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=$CURRENT_DIR/.env
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    sudo systemctl daemon-reload
    sudo systemctl enable aaire.service
    
    log_info "Systemd service created ✅"
}

# Function to restart service
restart_service() {
    log_info "Restarting AAIRE service..."
    
    # Stop any manual processes
    pkill -f main.py || true
    
    # Restart systemd service
    sudo systemctl stop aaire.service || true
    sudo systemctl start aaire.service
    
    # Wait for service to start
    sleep 5
    
    # Check service status
    if systemctl is-active --quiet aaire.service; then
        log_info "AAIRE service started successfully ✅"
    else
        log_error "AAIRE service failed to start"
        sudo systemctl status aaire.service
        exit 1
    fi
}

# Function to test deployment
test_deployment() {
    log_info "Testing deployment..."
    
    # Wait for service to fully start
    sleep 3
    
    # Test health endpoint
    for i in {1..5}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            log_info "Health check passed ✅"
            return 0
        fi
        log_warn "Health check attempt $i failed, retrying..."
        sleep 2
    done
    
    # Test root endpoint if health fails
    if curl -s http://localhost:8000/ > /dev/null; then
        log_info "Root endpoint responding ✅"
    else
        log_error "Service not responding"
        sudo systemctl status aaire.service
        exit 1
    fi
}

# Main deployment flow
main() {
    echo "Starting deployment..."
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        log_error "Don't run this script as root"
        exit 1
    fi
    
    validate_env
    update_code
    install_dependencies
    create_service
    restart_service
    test_deployment
    
    echo
    log_info "🎉 **DEPLOYMENT COMPLETE**"
    echo "   • Service: sudo systemctl status aaire.service"
    echo "   • Logs: sudo journalctl -u aaire.service -f"
    echo "   • Health: curl http://localhost:8000/health"
    echo "   • Enhanced citations and shape-aware extraction active"
}

# Run main function
main "$@"