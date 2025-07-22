#!/bin/bash
# AAIRE AWS Docker Deployment Script
# Quick deployment using EC2 with Docker

set -e

echo "🚀 AAIRE AWS Deployment Starting..."

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

# Create user data script
cat > user-data.sh << 'EOF'
#!/bin/bash
yum update -y
yum install -y docker git

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
JWT_SECRET_KEY=change-this-in-production
FRED_API_KEY=your_fred_key_here
SEC_EDGAR_USER_AGENT=AAIRE/1.0 (your-email@company.com)
ENVEOF

# Start services
docker-compose up -d

# Set ownership
chown -R ec2-user:ec2-user /home/ec2-user/aaire
EOF

# Launch EC2 instance
echo "🖥️ Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_PAIR_NAME \
    --security-group-ids $SECURITY_GROUP_ID \
    --user-data file://user-data.sh \
    --region $AWS_REGION \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=AAIRE-MVP}]' \
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
echo "   API URL: http://$PUBLIC_IP:8000"
echo "   API Docs: http://$PUBLIC_IP:8000/api/docs"
echo ""
echo "🔧 Next Steps:"
echo "1. SSH into the instance and update .env with your API keys"
echo "2. Restart services: cd aaire && docker-compose restart"
echo "3. Configure your domain and SSL certificate"
echo "4. Set up monitoring and backups"

# Cleanup
rm -f user-data.sh

echo "✅ Deployment script completed!"