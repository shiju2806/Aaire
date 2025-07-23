#!/bin/bash
# AAIRE AWS Docker Deployment Script - Fixed Version
# Handles Docker build issues and provides fallback options

set -e

echo "🚀 AAIRE AWS Deployment Starting (Fixed Version)..."

# Configuration
AWS_REGION="us-east-1"
KEY_PAIR_NAME="aaire-key"
INSTANCE_TYPE="t3.medium"
AMI_ID="ami-0c02fb55956c7d316"  # Amazon Linux 2023
SECURITY_GROUP_NAME="aaire-sg"

# Create key pair if it doesn't exist
echo "🔑 Setting up SSH key pair..."
if ! aws ec2 describe-key-pairs --key-names $KEY_PAIR_NAME --region $AWS_REGION 2>/dev/null; then
    aws ec2 create-key-pair \
        --key-name $KEY_PAIR_NAME \
        --region $AWS_REGION \
        --query 'KeyMaterial' \
        --output text > ${KEY_PAIR_NAME}.pem
    chmod 400 ${KEY_PAIR_NAME}.pem
    echo "✅ Key pair created: ${KEY_PAIR_NAME}.pem"
else
    echo "✅ Key pair already exists"
fi

# Create security group
echo "🛡️ Setting up security group..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text --region $AWS_REGION)

if ! aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME --region $AWS_REGION 2>/dev/null; then
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
        --group-name $SECURITY_GROUP_NAME \
        --description "AAIRE MVP Security Group" \
        --vpc-id $VPC_ID \
        --region $AWS_REGION \
        --query 'GroupId' \
        --output text)
    
    # Add rules
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $AWS_REGION
    
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0 \
        --region $AWS_REGION
    
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 \
        --region $AWS_REGION
    
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0 \
        --region $AWS_REGION
    
    echo "✅ Security group created: $SECURITY_GROUP_ID"
else
    SECURITY_GROUP_ID=$(aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME --query 'SecurityGroups[0].GroupId' --output text --region $AWS_REGION)
    echo "✅ Security group exists: $SECURITY_GROUP_ID"
fi

# Create improved user data script with fallback options
cat > user-data.sh << 'EOF'
#!/bin/bash
# AAIRE Installation Script with Docker Build Fixes

echo "Starting AAIRE installation..."

# Update system
yum update -y
yum install -y docker git python3 python3-pip

# Start Docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone repository
cd /home/ec2-user
git clone https://github.com/shiju2806/aaire.git
cd aaire

# Create environment file
cat > .env << 'ENVEOF'
ENVIRONMENT=production
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=us-east-1
REDIS_HOST=localhost
REDIS_PORT=6379
DATABASE_URL=postgresql://aaire:password@localhost:5432/aaire
JWT_SECRET_KEY=change-this-in-production-$(openssl rand -hex 32)
FRED_API_KEY=your_fred_key_here
SEC_EDGAR_USER_AGENT=AAIRE/1.0 (your-email@company.com)
ENVEOF

# Function to try Docker build with fallbacks
try_docker_build() {
    echo "Attempting Docker build (Method 1: Standard)..."
    if docker-compose build --no-cache aaire-api; then
        echo "✅ Standard Docker build successful"
        return 0
    fi
    
    echo "❌ Standard build failed, trying Alpine version..."
    if docker build -f Dockerfile.light -t aaire-api:latest .; then
        echo "✅ Alpine Docker build successful"
        # Update docker-compose to use the built image
        sed -i 's/build: ./image: aaire-api:latest/' docker-compose.yml
        return 0
    fi
    
    echo "❌ Docker builds failed, falling back to direct Python installation..."
    return 1
}

# Function for direct Python installation
direct_python_install() {
    echo "Installing AAIRE directly with Python..."
    
    # Install Python dependencies
    pip3 install --user -r requirements.txt
    
    # Create systemd service
    cat > /etc/systemd/system/aaire.service << 'SERVICEEOF'
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

[Install]
WantedBy=multi-user.target
SERVICEEOF
    
    # Set ownership
    chown -R ec2-user:ec2-user /home/ec2-user/aaire
    
    # Start service
    systemctl daemon-reload
    systemctl enable aaire
    systemctl start aaire
    
    echo "✅ Direct Python installation completed"
    return 0
}

# Try Docker first, fallback to direct Python
if try_docker_build; then
    echo "Starting with Docker..."
    docker-compose up -d
    
    # Wait and check if services started
    sleep 30
    if docker-compose ps | grep -q "Up"; then
        echo "✅ Docker services started successfully"
    else
        echo "❌ Docker services failed, trying direct installation..."
        direct_python_install
    fi
else
    echo "Docker build failed completely, using direct Python installation..."
    direct_python_install
fi

# Set final ownership
chown -R ec2-user:ec2-user /home/ec2-user/aaire

# Create status log
echo "AAIRE Installation completed at $(date)" > /var/log/aaire-install.log
echo "Access the application at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000" >> /var/log/aaire-install.log

echo "Installation script completed!"
EOF

# Launch EC2 instance
echo "🖥️ Launching EC2 instance with improved installation..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_PAIR_NAME \
    --security-group-ids $SECURITY_GROUP_ID \
    --user-data file://user-data.sh \
    --region $AWS_REGION \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=AAIRE-MVP-Fixed}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instance launched: $INSTANCE_ID"

# Wait for instance to be running
echo "⏳ Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $AWS_REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $AWS_REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "🎉 AAIRE MVP deployed successfully!"
echo ""
echo "📋 Deployment Details:"
echo "   Instance ID: $INSTANCE_ID"
echo "   Public IP: $PUBLIC_IP"
echo "   SSH: ssh -i ${KEY_PAIR_NAME}.pem ec2-user@$PUBLIC_IP"
echo ""
echo "🌐 Application URLs:"
echo "   API: http://$PUBLIC_IP:8000"
echo "   Docs: http://$PUBLIC_IP:8000/api/docs"
echo "   Health: http://$PUBLIC_IP:8000/health"
echo ""
echo "⏱️  Installation Progress:"
echo "   The application is installing in the background."
echo "   Wait 5-10 minutes, then check the URLs above."
echo "   If the app doesn't respond, it may still be installing."
echo ""
echo "🔧 Next Steps:"
echo "1. Wait for installation to complete (5-10 minutes)"
echo "2. Test the health endpoint: curl http://$PUBLIC_IP:8000/health"
echo "3. SSH in and update .env with your API keys:"
echo "   ssh -i ${KEY_PAIR_NAME}.pem ec2-user@$PUBLIC_IP"
echo "   cd aaire && nano .env"
echo "4. Restart the application after adding keys"
echo ""
echo "📜 Check Installation Logs:"
echo "   SSH in and run: sudo tail -f /var/log/cloud-init-output.log"
echo "   Or check: cat /var/log/aaire-install.log"

# Cleanup
rm -f user-data.sh

echo "✅ Deployment script completed!"