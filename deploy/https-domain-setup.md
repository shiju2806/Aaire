# AAIRE HTTPS Domain Hosting Setup

## 🎯 Goal: Host AAIRE on https://yourdomain.com instead of http://18.119.14.61:8000

## 📋 Prerequisites

### **1. Domain Name**
- Purchase domain from registrar (Namecheap, GoDaddy, Route53, etc.)
- Example: `aaire.company.com` or `ai.company.com`

### **2. Current AWS Setup**
- ✅ EC2 instance running: `18.119.14.61`
- ✅ AAIRE running on port 8000
- ✅ Security group allows HTTP traffic

## 🚀 **Option 1: AWS Application Load Balancer + ACM (Recommended)**

### **Step 1: Configure Route 53 (AWS DNS)**
```bash
# 1. Create hosted zone in Route 53
aws route53 create-hosted-zone --name aaire.company.com --caller-reference $(date +%s)

# 2. Update domain registrar to use Route 53 name servers
# (Copy NS records from Route 53 to your domain registrar)
```

### **Step 2: Request SSL Certificate**
```bash
# Request certificate through AWS Certificate Manager
aws acm request-certificate \
    --domain-name aaire.company.com \
    --domain-name *.aaire.company.com \
    --validation-method DNS \
    --region us-east-2
```

### **Step 3: Create Application Load Balancer**
```bash
# 1. Create ALB
aws elbv2 create-load-balancer \
    --name aaire-alb \
    --subnets subnet-12345 subnet-67890 \
    --security-groups sg-web \
    --scheme internet-facing \
    --type application \
    --ip-address-type ipv4

# 2. Create target group
aws elbv2 create-target-group \
    --name aaire-targets \
    --protocol HTTP \
    --port 8000 \
    --vpc-id vpc-12345 \
    --health-check-path /health

# 3. Register EC2 instance
aws elbv2 register-targets \
    --target-group-arn arn:aws:elasticloadbalancing:... \
    --targets Id=i-1234567890abcdef0,Port=8000
```

### **Step 4: Configure HTTPS Listener**
```bash
# Create HTTPS listener with SSL certificate
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:... \
    --protocol HTTPS \
    --port 443 \
    --certificates CertificateArn=arn:aws:acm:... \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

### **Step 5: Update Route 53 DNS**
```bash
# Create A record pointing to ALB
aws route53 change-resource-record-sets \
    --hosted-zone-id Z123456789 \
    --change-batch '{
        "Changes": [{
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": "aaire.company.com",
                "Type": "A",
                "AliasTarget": {
                    "DNSName": "aaire-alb-123456789.us-east-2.elb.amazonaws.com",
                    "EvaluateTargetHealth": false,
                    "HostedZoneId": "Z3AADJGX6KTTL2"
                }
            }
        }]
    }'
```

## 🔧 **Option 2: Nginx + Let's Encrypt (Cost-effective)**

### **Step 1: Install Nginx on EC2**
```bash
# SSH to your EC2 instance
ssh -i your-key.pem ubuntu@18.119.14.61

# Install Nginx
sudo apt update
sudo apt install nginx -y
sudo systemctl enable nginx
sudo systemctl start nginx
```

### **Step 2: Configure Domain DNS**
```bash
# Point your domain to EC2 public IP
# In your domain registrar (Namecheap, GoDaddy, etc.):
# Create A record: aaire.company.com -> 18.119.14.61
```

### **Step 3: Configure Nginx Reverse Proxy**
```bash
# Create Nginx config
sudo nano /etc/nginx/sites-available/aaire

# Add this configuration:
server {
    listen 80;
    server_name aaire.company.com www.aaire.company.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
    
    # WebSocket support for real-time chat
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Enable the site
sudo ln -s /etc/nginx/sites-available/aaire /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### **Step 4: Install Let's Encrypt SSL**
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d aaire.company.com -d www.aaire.company.com

# Nginx config will be automatically updated for HTTPS
```

### **Step 5: Update Security Groups**
```bash
# Allow HTTPS traffic
aws ec2 authorize-security-group-ingress \
    --group-id sg-your-group-id \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

# Allow HTTP (for redirect to HTTPS)  
aws ec2 authorize-security-group-ingress \
    --group-id sg-your-group-id \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0
```

### **Step 6: Setup Auto-renewal**
```bash
# Test renewal
sudo certbot renew --dry-run

# Add to crontab for auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 🔒 **Security Enhancements**

### **1. Update AAIRE Configuration**
```python
# Update main.py or start.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aaire.company.com"],  # Update domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

### **2. Environment Variables**
```bash
# Update .env file
ALLOWED_ORIGINS=https://aaire.company.com,https://www.aaire.company.com
DOMAIN=aaire.company.com
SSL_ENABLED=true
```

## 📊 **Cost Comparison**

### **Option 1: ALB + ACM**
- Application Load Balancer: ~$20/month
- SSL Certificate: Free (ACM)
- Route 53: ~$0.50/month
- **Total: ~$20.50/month**

### **Option 2: Nginx + Let's Encrypt**
- SSL Certificate: Free (Let's Encrypt)
- Additional compute: Minimal
- **Total: ~$0/month (just existing EC2 costs)**

## 🎯 **Recommended Approach**

### **For Production: Option 1 (ALB)**
- Better scalability and availability
- AWS-managed SSL certificates
- Built-in DDoS protection
- Health checks and failover

### **For Development/Testing: Option 2 (Nginx)**
- Cost-effective
- Quick setup
- Full control over configuration

## 📝 **Next Steps**

1. **Choose your domain name**
2. **Select Option 1 or 2 based on requirements**
3. **Follow the step-by-step guide**
4. **Test the setup**: https://aaire.company.com
5. **Update any hardcoded URLs in frontend**

## ⚠️ **Important Notes**

- **DNS propagation** can take 24-48 hours
- **SSL certificate** validation may take a few minutes
- **Update all API endpoints** in frontend from IP to domain
- **Test WebSocket connections** work over HTTPS
- **Monitor logs** during transition

Would you like me to help you implement Option 1 or Option 2?