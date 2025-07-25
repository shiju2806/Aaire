# AAIRE Docker Deployment Guide

## 🚀 Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shiju2806/Aaire.git
   cd Aaire
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Start with Docker Compose:**
   
   **For Development (with external Qdrant Cloud):**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```
   
   **For Production (with all services):**
   ```bash
   docker-compose up -d
   ```

4. **Access AAIRE:**
   - Web Interface: http://localhost:8000
   - API Docs: http://localhost:8000/api/docs

## 🐳 Docker Images

### Building the Image

```bash
docker build -t aaire:latest .
```

### Running Standalone

```bash
docker run -d \
  --name aaire \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  -e QDRANT_URL=your_qdrant_url \
  -e QDRANT_API_KEY=your_qdrant_key \
  -v $(pwd)/data/uploads:/app/data/uploads \
  aaire:latest
```

## 📋 Environment Variables

Required:
- `OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_URL`: Qdrant cluster URL (or use local)
- `QDRANT_API_KEY`: Qdrant API key (if using cloud)

Optional:
- `OPENAI_MODEL`: Model to use (default: gpt-3.5-turbo, recommended: gpt-4o-mini)
- `REDIS_HOST`: Redis host (default: redis)
- `REDIS_PORT`: Redis port (default: 6379)

## 🏗️ Architecture

### Development Setup
- AAIRE App + Redis only
- Uses external Qdrant Cloud
- Source code mounted for hot reload

### Production Setup
- AAIRE App
- Redis (caching)
- PostgreSQL (user data)
- Qdrant (vector database)
- Nginx (reverse proxy, optional)

## 🛠️ Common Commands

### View logs:
```bash
docker-compose logs -f aaire
```

### Restart services:
```bash
docker-compose restart aaire
```

### Stop all services:
```bash
docker-compose down
```

### Remove all data (careful!):
```bash
docker-compose down -v
```

### Execute commands in container:
```bash
docker-compose exec aaire python cleanup_vector_db.py
```

## 🔧 Troubleshooting

### Dependencies Issues
The Docker image ensures consistent dependencies. No need to worry about llama-index versions!

### Port Conflicts
If port 8000 is already in use:
```bash
# Change the port mapping in docker-compose.yml
ports:
  - "8080:8000"  # Use 8080 instead
```

### Memory Issues
For large document processing, increase Docker memory:
- Docker Desktop: Preferences > Resources > Memory: 4GB+

## 🚀 Production Deployment

### Using Docker on EC2

1. **Install Docker:**
   ```bash
   sudo yum update -y
   sudo yum install -y docker
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   ```

2. **Install Docker Compose:**
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

3. **Deploy AAIRE:**
   ```bash
   git clone https://github.com/shiju2806/Aaire.git
   cd Aaire
   cp .env.example .env
   # Edit .env with your API keys
   docker-compose up -d
   ```

### Security Notes
- Never commit `.env` files with real API keys
- Use AWS Secrets Manager or similar for production
- Enable HTTPS with proper SSL certificates
- Restrict security group to necessary ports only

## 📊 Monitoring

Check container health:
```bash
docker-compose ps
docker stats
```

The AAIRE container includes a health check that verifies the API is responding.