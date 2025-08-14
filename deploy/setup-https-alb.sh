#!/bin/bash
# AAIRE HTTPS Setup with AWS Application Load Balancer + ACM
# Usage: ./setup-https-alb.sh your-domain.com

set -e

DOMAIN=$1
if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 <domain-name>"
    echo "Example: $0 aaire.company.com"
    exit 1
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

# Get current region
REGION=$(aws configure get region)
if [ -z "$REGION" ]; then
    REGION="us-east-2"
    echo "🌍 No region set, using default: $REGION"
fi

echo "🚀 Setting up HTTPS for AAIRE using AWS ALB + ACM"
echo "   Domain: $DOMAIN"
echo "   Region: $REGION"

# Get VPC and subnet information
echo "🔍 Getting VPC and subnet information..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text)
SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[?MapPublicIpOnLaunch==`true`].SubnetId' --output text)
SUBNET_ARRAY=($SUBNET_IDS)

if [ ${#SUBNET_ARRAY[@]} -lt 2 ]; then
    echo "❌ Need at least 2 public subnets in different AZs for ALB"
    exit 1
fi

echo "   VPC ID: $VPC_ID"
echo "   Subnets: ${SUBNET_ARRAY[0]}, ${SUBNET_ARRAY[1]}"

# Get current EC2 instance ID
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
if [ -z "$INSTANCE_ID" ]; then
    echo "❌ Could not determine EC2 instance ID. Are you running this on EC2?"
    exit 1
fi

echo "   Instance ID: $INSTANCE_ID"

# Create security group for ALB
echo "🛡️  Creating security group for ALB..."
ALB_SG_ID=$(aws ec2 create-security-group \
    --group-name aaire-alb-sg \
    --description "Security group for AAIRE Application Load Balancer" \
    --vpc-id $VPC_ID \
    --query 'GroupId' --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
        --group-names aaire-alb-sg \
        --query 'SecurityGroups[0].GroupId' --output text)

# Add rules to ALB security group
aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 2>/dev/null || echo "HTTP rule already exists"

aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 2>/dev/null || echo "HTTPS rule already exists"

echo "   ALB Security Group: $ALB_SG_ID"

# Request SSL certificate
echo "🔐 Requesting SSL certificate from AWS Certificate Manager..."
CERT_ARN=$(aws acm request-certificate \
    --domain-name $DOMAIN \
    --subject-alternative-names "*.$DOMAIN" \
    --validation-method DNS \
    --query 'CertificateArn' --output text)

echo "   Certificate ARN: $CERT_ARN"
echo ""
echo "⚠️  IMPORTANT: You need to validate the SSL certificate!"
echo "   1. Go to AWS Console > Certificate Manager"
echo "   2. Find certificate for $DOMAIN"
echo "   3. Add the DNS validation records to your domain"
echo "   4. Wait for certificate status to become 'Issued'"
echo ""
read -p "Press Enter when certificate is validated and issued..."

# Wait for certificate to be issued
echo "🕐 Waiting for certificate to be validated..."
while true; do
    STATUS=$(aws acm describe-certificate --certificate-arn $CERT_ARN --query 'Certificate.Status' --output text)
    if [ "$STATUS" = "ISSUED" ]; then
        echo "✅ Certificate validated successfully!"
        break
    elif [ "$STATUS" = "FAILED" ]; then
        echo "❌ Certificate validation failed. Please check DNS records."
        exit 1
    else
        echo "   Status: $STATUS - waiting..."
        sleep 30
    fi
done

# Create Application Load Balancer
echo "🌐 Creating Application Load Balancer..."
ALB_ARN=$(aws elbv2 create-load-balancer \
    --name aaire-alb \
    --subnets ${SUBNET_ARRAY[0]} ${SUBNET_ARRAY[1]} \
    --security-groups $ALB_SG_ID \
    --scheme internet-facing \
    --type application \
    --ip-address-type ipv4 \
    --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null || \
    aws elbv2 describe-load-balancers \
        --names aaire-alb \
        --query 'LoadBalancers[0].LoadBalancerArn' --output text)

# Get ALB DNS name
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns $ALB_ARN \
    --query 'LoadBalancers[0].DNSName' --output text)

echo "   ALB ARN: $ALB_ARN"
echo "   ALB DNS: $ALB_DNS"

# Create target group
echo "🎯 Creating target group..."
TG_ARN=$(aws elbv2 create-target-group \
    --name aaire-targets \
    --protocol HTTP \
    --port 8000 \
    --vpc-id $VPC_ID \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 10 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 5 \
    --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null || \
    aws elbv2 describe-target-groups \
        --names aaire-targets \
        --query 'TargetGroups[0].TargetGroupArn' --output text)

# Register EC2 instance with target group
echo "📌 Registering EC2 instance with target group..."
aws elbv2 register-targets \
    --target-group-arn $TG_ARN \
    --targets Id=$INSTANCE_ID,Port=8000

echo "   Target Group ARN: $TG_ARN"

# Create HTTPS listener
echo "🔒 Creating HTTPS listener..."
aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTPS \
    --port 443 \
    --certificates CertificateArn=$CERT_ARN \
    --default-actions Type=forward,TargetGroupArn=$TG_ARN 2>/dev/null || echo "HTTPS listener already exists"

# Create HTTP listener (redirect to HTTPS)
echo "🔄 Creating HTTP redirect listener..."
aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=redirect,RedirectConfig='{Protocol=HTTPS,Port=443,StatusCode=HTTP_301}' 2>/dev/null || echo "HTTP listener already exists"

# Update EC2 security group to allow ALB access
echo "🔐 Updating EC2 security group..."
INSTANCE_SG=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)

aws ec2 authorize-security-group-ingress \
    --group-id $INSTANCE_SG \
    --protocol tcp \
    --port 8000 \
    --source-group $ALB_SG_ID 2>/dev/null || echo "ALB access rule already exists"

# Create Route 53 hosted zone (if using Route 53)
echo "🌍 DNS Configuration"
echo "You now need to configure DNS for $DOMAIN to point to the ALB:"
echo "   ALB DNS Name: $ALB_DNS"
echo ""
echo "Option 1 - Using Route 53:"
echo "   1. Create hosted zone: aws route53 create-hosted-zone --name $DOMAIN --caller-reference \$(date +%s)"
echo "   2. Create ALIAS record pointing to ALB"
echo ""
echo "Option 2 - Using external DNS provider:"
echo "   1. Create CNAME record: $DOMAIN -> $ALB_DNS"
echo "   2. Create CNAME record: www.$DOMAIN -> $ALB_DNS"
echo ""

read -p "Configure DNS now and press Enter when ready to test..."

# Test the setup
echo "🧪 Testing the setup..."
sleep 10

if curl -s -f https://$DOMAIN/health > /dev/null; then
    echo "✅ Success! AAIRE is now running on https://$DOMAIN"
else
    echo "⚠️  Health check failed. This might be normal if DNS hasn't propagated yet."
    echo "Please wait a few minutes and test manually: https://$DOMAIN"
fi

echo ""
echo "🎉 Setup completed!"
echo "📋 Summary:"
echo "   - Domain: https://$DOMAIN"
echo "   - SSL: AWS Certificate Manager"
echo "   - Load Balancer: Application Load Balancer"
echo "   - Target: EC2 instance $INSTANCE_ID on port 8000"
echo ""
echo "🔍 AWS Resources Created:"
echo "   - Certificate: $CERT_ARN"
echo "   - Load Balancer: $ALB_ARN"
echo "   - Target Group: $TG_ARN"
echo "   - Security Group: $ALB_SG_ID"
echo ""
echo "⚠️  Don't forget to:"
echo "   1. Configure DNS to point to: $ALB_DNS"
echo "   2. Update AAIRE CORS settings for https://$DOMAIN"
echo "   3. Test all functionality"