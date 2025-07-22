# 🚀 AAIRE AWS Deployment Guide

## Quick Start (Recommended for MVP Testing)

### Prerequisites
- AWS CLI installed and configured
- Docker installed locally
- Your API keys ready (OpenAI, Pinecone, FRED)

### 1. Quick Deploy with Docker on EC2

```bash
# Make script executable
chmod +x deploy/aws-docker-deploy.sh

# Run deployment
./deploy/aws-docker-deploy.sh
```

This will:
- Create EC2 instance (t3.medium)
- Install Docker and Docker Compose
- Deploy AAIRE automatically
- Provide you with access URLs

**Cost**: ~$50/month

### 2. Configure API Keys

```bash
# SSH into your instance
ssh -i aaire-key.pem ec2-user@YOUR_PUBLIC_IP

# Edit environment file
cd aaire
nano .env

# Update these values:
OPENAI_API_KEY=your_actual_key
PINECONE_API_KEY=your_actual_key
FRED_API_KEY=your_actual_key

# Restart services
docker-compose restart
```

### 3. Access Your Application

- **API**: http://YOUR_IP:8000
- **Documentation**: http://YOUR_IP:8000/api/docs
- **Health Check**: http://YOUR_IP:8000/health

## Production Deployment with Terraform

### Prerequisites
- Terraform installed
- AWS CLI configured with appropriate permissions
- Domain name (optional)

### 1. Deploy Infrastructure

```bash
cd deploy/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan

# Deploy (takes ~15 minutes)
terraform apply
```

This creates:
- VPC with public/private subnets
- Application Load Balancer
- Auto Scaling Group with EC2 instances
- RDS PostgreSQL database
- ElastiCache Redis cluster
- S3 bucket for document storage
- Security groups and networking

**Cost**: $550-850/month (as per SRS)

### 2. Deploy Application

```bash
# Get infrastructure outputs
ALB_DNS=$(terraform output -raw alb_dns_name)
RDS_ENDPOINT=$(terraform output -raw rds_endpoint)
REDIS_ENDPOINT=$(terraform output -raw redis_endpoint)

# Create deployment package
./deploy/package-app.sh

# Deploy to EC2 instances
./deploy/deploy-app.sh
```

## AWS Services Used

### Compute Layer
- **EC2 t3.medium**: 2 instances with auto-scaling (2-5 instances)
- **Application Load Balancer**: HTTPS termination and health checks

### Storage Layer
- **RDS PostgreSQL**: db.t3.micro with 100GB storage
- **ElastiCache Redis**: cache.t3.micro for query caching
- **S3**: Document storage with lifecycle policies

### Security Layer
- **VPC**: Isolated network with public/private subnets
- **Security Groups**: Restrictive firewall rules
- **IAM Roles**: Least privilege access

### Monitoring (Optional)
- **CloudWatch**: Logs and metrics
- **CloudTrail**: API audit logging

## Environment Variables

### Required API Keys
```bash
# OpenAI (Required)
OPENAI_API_KEY=sk-...

# Pinecone (Required)
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1

# FRED API (Optional but recommended)
FRED_API_KEY=...
```

### AWS Configuration
```bash
# Database
DATABASE_URL=postgresql://aaire:password@YOUR_RDS_ENDPOINT:5432/aaire

# Cache
REDIS_HOST=YOUR_REDIS_ENDPOINT
REDIS_PORT=6379

# Storage
S3_BUCKET_NAME=your-s3-bucket
AWS_REGION=us-east-1
```

## Scaling Configuration

### Auto Scaling Targets
- **Target CPU**: 70%
- **Min Instances**: 2
- **Max Instances**: 5
- **Target Response Time**: <3 seconds

### Database Scaling
- **Initial**: db.t3.micro
- **Scale to**: db.t3.small → db.t3.medium
- **Read Replicas**: Add when needed

### Cache Scaling
- **Initial**: cache.t3.micro
- **Scale to**: cache.t3.small with clustering

## Security Checklist

### ✅ Network Security
- VPC with isolated subnets
- Security groups with minimal access
- No direct internet access to databases

### ✅ Data Protection
- RDS encryption at rest
- S3 bucket encryption
- TLS 1.3 for all traffic

### ✅ Access Control
- IAM roles with least privilege
- No hardcoded credentials
- JWT token authentication

### ✅ Monitoring
- CloudWatch logging
- Failed login alerts
- Performance monitoring

## Monitoring & Alerts

### Key Metrics to Monitor
- **Response Time**: <3 seconds (p95)
- **Error Rate**: <1%
- **CPU Utilization**: <70%
- **Memory Usage**: <80%
- **Database Connections**: <80% of max

### Recommended Alerts
```bash
# High error rate
aws cloudwatch put-metric-alarm \
  --alarm-name "AAIRE-High-Error-Rate" \
  --alarm-description "Error rate > 5%" \
  --metric-name ErrorRate \
  --namespace AWS/ApplicationELB \
  --statistic Average \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold

# High response time
aws cloudwatch put-metric-alarm \
  --alarm-name "AAIRE-High-Response-Time" \
  --alarm-description "Response time > 3 seconds" \
  --metric-name TargetResponseTime \
  --namespace AWS/ApplicationELB \
  --statistic Average \
  --period 300 \
  --threshold 3 \
  --comparison-operator GreaterThanThreshold
```

## Cost Optimization

### Monthly Cost Breakdown
- **EC2 (2x t3.medium)**: $60
- **RDS (db.t3.micro)**: $25
- **ElastiCache**: $25
- **Load Balancer**: $25
- **Data Transfer**: $20
- **S3 Storage**: $10
- **CloudWatch**: $10
- **Total Infrastructure**: ~$175

### External Services
- **Pinecone Starter**: $70
- **OpenAI API**: $200-500 (usage dependent)
- **Total External**: $270-570

### **Grand Total**: $445-745/month

### Cost Optimization Tips
1. **Use Spot Instances**: Save 60-70% on EC2 costs
2. **Schedule Dev/Test**: Turn off non-prod environments
3. **Reserved Instances**: 1-year commitment saves 30%
4. **S3 Lifecycle**: Move old documents to cheaper storage

## Backup & Disaster Recovery

### Automated Backups
- **RDS**: 7-day retention with point-in-time recovery
- **S3**: Versioning enabled with lifecycle policies
- **Code**: GitHub repository

### Disaster Recovery Plan
1. **RTO (Recovery Time Objective)**: 4 hours
2. **RPO (Recovery Point Objective)**: 1 hour
3. **Multi-AZ Deployment**: Automatic failover
4. **Cross-Region Backup**: For critical data

## SSL/TLS Setup

### Option 1: AWS Certificate Manager (Free)
```bash
# Request SSL certificate
aws acm request-certificate \
  --domain-name yourdomain.com \
  --validation-method DNS

# Add HTTPS listener to load balancer
aws elbv2 create-listener \
  --load-balancer-arn YOUR_ALB_ARN \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=YOUR_CERT_ARN
```

### Option 2: Let's Encrypt (Free)
- Use Certbot in your EC2 instances
- Automatic renewal with cron jobs

## Troubleshooting

### Common Issues

**1. Application Won't Start**
```bash
# Check logs
docker-compose logs aaire-api

# Common fixes
- Verify API keys in .env
- Check database connection
- Ensure Pinecone index exists
```

**2. High Response Times**
```bash
# Check Redis connection
redis-cli -h YOUR_REDIS_ENDPOINT ping

# Scale up instances
aws autoscaling update-auto-scaling-group \
  --auto-scaling-group-name aaire-asg \
  --desired-capacity 3
```

**3. Database Connection Issues**
```bash
# Test connection
psql -h YOUR_RDS_ENDPOINT -U aaire -d aaire

# Check security groups
aws ec2 describe-security-groups --group-ids YOUR_RDS_SG_ID
```

## Next Steps

### Week 1: Basic Deployment
- [ ] Deploy with Docker script
- [ ] Configure API keys
- [ ] Test core functionality
- [ ] Set up basic monitoring

### Week 2: Production Ready
- [ ] Deploy with Terraform
- [ ] Configure custom domain
- [ ] Set up SSL certificates
- [ ] Implement backup procedures

### Week 3: Optimization
- [ ] Load testing
- [ ] Performance tuning
- [ ] Cost optimization
- [ ] Security hardening

### Week 4: Launch
- [ ] User acceptance testing
- [ ] Documentation complete
- [ ] Go-live checklist
- [ ] Support procedures

## Support

For deployment issues:
1. Check CloudWatch logs
2. Review security group settings
3. Verify API key configuration
4. Contact AWS Support if needed

The deployment is designed to be production-ready from day one while keeping costs manageable for an MVP.