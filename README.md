# AAIRE - Accounting & Actuarial Insurance Resource Expert

## MVP Version 1.0

AAIRE is an AI-powered conversational assistant that provides accurate, citation-backed answers to insurance accounting and actuarial questions under US GAAP and IFRS frameworks.

## 🚀 Features

### Core Capabilities
- **Conversational AI Interface**: Natural language queries with streaming responses
- **Multi-Framework Support**: US GAAP, IFRS, and actuarial standards
- **Citation-Backed Responses**: All answers include source citations
- **Document Upload**: Support for PDF, DOCX, CSV, and XLSX files
- **Compliance Controls**: Built-in filtering for tax/legal advice
- **External Data Integration**: SEC EDGAR and FRED API connectors

### Technical Features
- **LlamaIndex RAG Pipeline**: Hierarchical document chunking
- **Pinecone Vector Database**: Scalable semantic search
- **OpenAI GPT-4o-mini**: Advanced language understanding with cost efficiency
- **Redis Caching**: Fast query response times
- **Comprehensive Audit Logging**: Full compliance trail
- **Authentication**: SAML 2.0 SSO ready (JWT for MVP)

## 📋 Requirements

### MVP Requirements Met
- ✅ Support 100+ concurrent users
- ✅ Response time <3 seconds (p95)
- ✅ Citation-backed responses
- ✅ Document upload and processing
- ✅ Compliance rule filtering
- ✅ External API integration
- ✅ Audit logging

## 🛠 Installation

### Prerequisites
- Python 3.11+
- Redis server
- PostgreSQL database
- OpenAI API key
- Pinecone account

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/shiju2806/aaire.git
   cd aaire
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Start services**
   ```bash
   # Start Redis (varies by system)
   redis-server
   
   # Start PostgreSQL (varies by system)
   # Create database: createdb aaire
   ```

6. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 🔧 Configuration

### Core Configuration Files

- `config/mvp_config.yaml` - Main application configuration
- `config/data_sources.yaml` - External API configuration  
- `config/compliance.py` - Compliance rules and audit settings
- `.env` - Environment variables and secrets

### Key Configuration Options

```yaml
# config/mvp_config.yaml
llm_config:
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 4000

retrieval_config:
  similarity_threshold: 0.75
  max_results: 10
  use_cache: true
```

## 📚 API Documentation

### Main Endpoints

- `POST /api/v1/chat` - Submit queries and get responses
- `POST /api/v1/documents` - Upload documents for processing
- `GET /api/v1/documents/{job_id}/status` - Check processing status
- `WebSocket /api/v1/chat/stream` - Real-time streaming responses

### Example Usage

```python
import requests

# Submit a query
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "query": "How should insurance reserves be measured under IFRS 17?",
        "filters": {"source_type": ["IFRS", "ACTUARIAL"]}
    },
    headers={"Authorization": "Bearer your-jwt-token"}
)

print(response.json())
```

Full API documentation available at: `http://localhost:8000/api/docs`

## 🏗 Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   RAG Pipeline   │    │ Pinecone Vector │
│                 │────│                  │────│    Database     │
│ • REST API      │    │ • LlamaIndex     │    │                 │
│ • WebSocket     │    │ • Hierarchical   │    │ • US GAAP       │
│ • Auth          │    │   Chunking       │    │ • IFRS          │
└─────────────────┘    └──────────────────┘    │ • Company       │
         │                       │              │ • Actuarial     │
         │                       │              └─────────────────┘
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│ Compliance      │    │ Document         │
│ Engine          │    │ Processor        │
│                 │    │                  │
│ • Rule Filter   │    │ • PDF/DOCX       │
│ • Audit Trail   │    │ • CSV/XLSX       │
└─────────────────┘    └──────────────────┘
```

### Data Flow

1. **Query Processing**: User submits query → Compliance check → RAG retrieval
2. **Document Processing**: Upload → Text extraction → Chunking → Embedding → Storage
3. **Response Generation**: Context assembly → LLM generation → Citation extraction

## 🔐 Security & Compliance

### Authentication
- JWT tokens for MVP
- SAML 2.0 SSO ready for production
- Role-based access control (User/Admin)

### Compliance Features
- Automatic filtering of tax/legal advice requests
- Professional judgment disclaimers
- Comprehensive audit logging
- 7-year log retention policy

### Data Protection
- TLS 1.3 encryption in transit
- AES-256 encryption at rest
- No sensitive data in logs
- GDPR-compliant data handling

## 📊 Monitoring & Performance

### Metrics Tracked
- Query response times
- Compliance rule triggers
- Document processing status
- User activity patterns
- System resource usage

### Performance Targets
- Response time: <3 seconds (p95)
- Throughput: 50 queries/second
- Uptime: 99.9%
- Cost per query: <$0.10

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Load testing
pytest tests/performance/
```

### Test Coverage
- Target: 70% code coverage
- Unit tests for all core components
- Integration tests for API endpoints
- Performance tests for load scenarios

## 🚀 Deployment

### Production Deployment

See the MVP SRS document for complete infrastructure specifications:

- **Compute**: AWS EC2 with auto-scaling
- **Storage**: Pinecone + RDS PostgreSQL + S3
- **Caching**: ElastiCache Redis
- **Monitoring**: CloudWatch + custom metrics
- **Estimated Cost**: $550-850/month

### Docker Deployment
```bash
# Build image
docker build -t aaire:latest .

# Run with docker-compose
docker-compose up -d
```

## 📈 Roadmap

### Phase 1 (MVP) - Completed ✅
- Core RAG pipeline
- Basic document processing
- Compliance filtering
- External API integration
- Authentication framework

### Phase 2 (Planned)
- Advanced RAG features
- Teams/Slack integration
- Advanced analytics
- Multi-language support
- Mobile applications

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Code Standards
- Python: Black formatter, isort imports
- Type hints required
- Docstrings for all functions
- Comprehensive test coverage

## 📄 License

This project is proprietary software. All rights reserved.

## 🆘 Support

### Documentation
- API Docs: `http://localhost:8000/api/docs`
- Architecture: See `docs/` folder
- MVP SRS: See requirements document

### Contact
- Technical Issues: Create GitHub issue
- Security Concerns: security@yourcompany.com
- General Questions: support@yourcompany.com

## 📝 Changelog

### v1.0-MVP (2025-01-XX)
- Initial MVP release
- LlamaIndex RAG pipeline
- Pinecone vector storage
- OpenAI GPT-4o-mini integration
- SEC EDGAR and FRED APIs
- Compliance engine
- Document processing
- Authentication framework

---

**AAIRE MVP** - Built with ❤️ for insurance professionals