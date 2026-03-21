# AAIRE - Accounting & Actuarial Insurance Resource Expert

AI-powered conversational assistant for insurance and accounting professionals. Upload regulatory documents (VM-20, IFRS 17, ASC 944, LICAT, etc.), ask natural language questions, and receive accurate, citation-backed answers.

## Key Features

- **Layout-aware document ingestion** -- PDF, DOCX, CSV, XLSX with table/formula/image detection (Docling)
- **Hybrid retrieval** -- Vector similarity + BM25 keyword search in parallel
- **Cross-encoder reranking** -- ms-marco-MiniLM-L-6-v2 (~110ms for 20 docs)
- **Query decomposition** -- Heuristic-gated complex query splitting (zero LLM cost for simple queries)
- **Section expansion** -- Sibling chunks pulled for full section context
- **Priority context packing** -- Coverage-first, dedup, whole-element policy (30K token budget)
- **Verification pipeline** -- Generate + verify + correct (max 3 iterations)
- **Citation-backed responses** -- Every claim linked to source documents
- **Knowledge graph** -- Elasticsearch-backed entity resolution
- **SEC EDGAR integration** -- Company search, filing browse, direct ingestion
- **Streaming responses** -- SSE and WebSocket
- **Eval harness** -- 30 golden queries across 6 categories

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Qdrant (vector database)
- Elasticsearch (BM25 + knowledge graph)
- Redis (optional, for response caching)

### Setup

```bash
git clone https://github.com/shiju2806/aaire.git
cd aaire

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
python main.py
# API at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### Docker

```bash
docker-compose up -d
```

See `DOCKER_README.md` for details.

## Architecture

```
User --> FastAPI (30+ endpoints) --> RAG Pipeline --> Response

RAG Pipeline:
  Topic Gate --> Query Enhancement --> Decomposition --> Retrieval --> Reranking
  --> Section Expansion --> Context Packing --> Verification --> Compliance --> Citations

Storage:
  Qdrant (vectors) + Elasticsearch (BM25/KG) + Redis (cache)
```

The query pipeline runs 9 stages:

| Stage | What it does |
|-------|-------------|
| 1. Topic gate | Rejects off-domain queries (keyword check first, LLM fallback) |
| 2. Semantic enhancement | Expands query with domain terms (1 LLM call) |
| 3. Query decomposition | Splits complex queries into 2-4 sub-questions (gated) |
| 4. Element-aware retrieval | Hybrid + graph + targeted element search (parallel, RRF merge) |
| 5. Cross-encoder reranking | (query, doc) pair scoring with entity-overlap bonus |
| 6. Section expansion | Pulls sibling chunks from same section |
| 7. Priority context packing | Coverage pass + high-value pass + fill pass (30K tokens) |
| 8. Verification + compliance | Generate-verify-correct loop + tax/legal filtering |
| 9. Citation extraction | Maps [N] references back to source documents |

See `ARCHITECTURE.md` for full technical details.

## API

### Core

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/chat` | Buffered query + response |
| POST | `/api/v1/chat/stream` | SSE streaming response |
| WS | `/api/v1/chat/ws` | WebSocket streaming |
| POST | `/api/v1/upload` | Document upload + ingestion |

### Analytics & Feedback

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/feedback` | Submit response feedback |
| GET | `/api/v1/analytics/summary` | System analytics |
| GET | `/api/v1/analytics/knowledge-gaps` | Knowledge gap detection |

Full endpoint list in `ARCHITECTURE.md` or at `http://localhost:8000/docs`.

## Configuration

All settings in `config/` directory:

| File | Purpose |
|------|---------|
| `mvp_config.yaml` | LLM model, retrieval thresholds, compliance |
| `scoring.yaml` | Reranking, section expansion, query decomposition |
| `infrastructure.yaml` | Token limits, server settings |
| `knowledge_graph.yaml` | Entity extraction, graph traversal |

Key tuning knobs:

| Setting | Default | Purpose |
|---------|---------|---------|
| `reranking_enabled` | `true` | Cross-encoder on/off |
| `section_expansion.enabled` | `true` | Section sibling retrieval |
| `query_decomposition.enabled` | `true` | Complex query splitting |
| `target_tokens` | `30000` | Context window budget |
| `similarity_threshold` | `0.30` | Minimum retrieval score |

## Evaluation

30 golden queries across 6 categories (simple, comparison, aggregation, formula, section_span, edge_case).

```bash
python scripts/evaluate.py                      # All queries
python scripts/evaluate.py --category comparison # One category
python scripts/evaluate.py --save-baseline       # Save baseline
python scripts/evaluate.py --compare             # Check regressions
```

## Supported Document Types

| Format | Parsing | Element Detection |
|--------|---------|-------------------|
| PDF | Docling layout parser | Tables, formulas, images, callouts, text |
| DOCX | Docling | Tables, text, images |
| CSV | Direct pandas load | Tabular data |
| XLSX | Direct pandas load | Tabular data with sheets |

### Regulatory Frameworks

- **US GAAP**: ASC 944, ASC 842, ASC 326
- **IFRS**: IFRS 17, IFRS 9, IFRS 4
- **Actuarial**: VM-20, VM-21, VM-22
- **Canadian**: LICAT, CALM, OSFI guidelines
- **General**: SOX, Solvency II, SEC filings (10-K, 10-Q)

## Deployment

| Option | Guide |
|--------|-------|
| Docker Compose | `DOCKER_README.md` |
| AWS EC2 + RDS | `deploy/aws-setup-guide.md` |
| Direct Python on EC2 | `deploy/direct-python-install.md` |
| HTTPS setup | `deploy/https-domain-setup.md` |

## Documentation

| Document | Content |
|----------|---------|
| `ARCHITECTURE.md` | Full technical architecture with pipeline stages |
| `PRODUCT.md` | Product specification, feature matrix, roadmap |
| `CROSS_ENCODER_SOLUTION.md` | Cross-encoder rationale and benchmarks |
| `INTELLIGENCE_ARCHITECTURE.md` | Intelligence subsystems (historical) |
| `REBUILD_PLAN.md` | Phase 0-4 rebuild history |

## License

This project is proprietary software. All rights reserved.
