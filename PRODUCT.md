# AAIRE Product Specification

> Last updated: 2026-03-21

## What is AAIRE?

AAIRE (Accounting & Actuarial Insurance Resource Expert) is an AI-powered conversational assistant for insurance and accounting professionals. Users upload regulatory documents (VM-20, IFRS 17, ASC 944, LICAT, etc.), then ask natural language questions and receive accurate, citation-backed answers.

**Target users**: Actuaries, insurance accountants, regulatory analysts, audit teams.

**Domain scope**: Insurance accounting, actuarial science, reserve valuation, regulatory capital, financial reporting standards (US GAAP, IFRS, Canadian OSFI).

---

## Core Capabilities

### 1. Document Ingestion
- Upload PDF, DOCX, CSV, XLSX files
- Layout-aware parsing preserves tables, formulas, diagrams, and callouts
- Element-type-specific processing (tables stored as markdown, formulas as LaTeX)
- Automatic entity extraction and knowledge graph population
- Content deduplication via hash-based detection
- SEC EDGAR filing ingestion (search companies, browse filings, ingest directly)

### 2. Conversational Q&A
- Natural language queries with streaming responses (SSE + WebSocket)
- Multi-turn conversation with session memory
- Citation-backed answers — every claim linked to source documents
- Compliance controls prevent tax/legal advice with automatic disclaimers
- Follow-up question generation based on response context
- Topic gating rejects off-domain queries politely

### 3. Intelligent Retrieval
- Hybrid search: vector similarity + BM25 keyword matching (parallel)
- Cross-encoder reranking distinguishes semantically similar concepts
- Knowledge graph entity resolution for disambiguation
- Element-type-aware scoring (tables boosted for numerical queries)
- Section expansion pulls sibling chunks for full context
- Query decomposition splits complex queries into sub-questions automatically

### 4. Response Quality
- Multi-pass verification with self-correction (max 3 iterations)
- Post-generation compliance check with disclaimers
- Refusal detection suppresses citations when LLM can't answer
- Priority context packing ensures coverage across sections and document types
- Inline citations with source mapping to specific document sections

### 5. Analytics & Feedback
- Response feedback collection (helpful/not helpful + comments)
- Knowledge gap detection — identifies topics users ask about that lack coverage
- System analytics dashboard (query volume, latency, confidence distribution)
- Retrieval audit trail for debugging and optimization

---

## Feature Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| PDF upload + layout-aware parsing | Shipped | Docling-based, supports tables/formulas/images |
| Hybrid vector + keyword search | Shipped | Parallel execution, confidence-based early exit |
| Cross-encoder reranking | Shipped | ms-marco-MiniLM-L-6-v2, ~110ms latency |
| Section expansion | Shipped | Pulls sibling chunks from same section |
| Query decomposition | Shipped | Heuristic-gated, LLM decomposition for complex queries |
| Priority context packing | Shipped | Coverage-first, dedup, whole-element policy |
| Verification pipeline | Shipped | Generate + verify + correct (2-3 LLM calls) |
| Compliance check | Shipped | Tax/legal filtering with disclaimers |
| Inline citations | Shipped | Mapped to source documents via [N] references |
| Streaming responses | Shipped | SSE and WebSocket |
| Conversation memory | Shipped | Session-based with compression |
| Knowledge graph | Shipped | Elasticsearch-backed entity resolution |
| SEC EDGAR integration | Shipped | Company search, filing browse, direct ingestion |
| Eval harness | Shipped | 30 golden queries, 6 categories, baseline comparison |
| Redis caching | Shipped | TTL-based response cache |
| Feedback system | Shipped | Thumbs up/down + text feedback + analytics |
| JWT authentication | Shipped | Token-based auth |
| SAML 2.0 SSO | Partial | Skeleton implemented, 6 TODOs remaining |
| Workflow engine | Partial | Templates defined, step execution needs testing |
| FRED API integration | Partial | Connector exists, limited data coverage |
| Mobile/Slack integration | Not started | Roadmap item |
| Multi-language support | Not started | Roadmap item |

---

## Architecture Summary

```
User → FastAPI (30+ endpoints) → RAG Pipeline → Response

RAG Pipeline:
  Topic Gate → Query Enhancement → Decomposition → Retrieval → Reranking
  → Section Expansion → Context Packing → Verification → Compliance → Citations

Storage:
  Qdrant (vectors) + Elasticsearch (BM25/KG) + Redis (cache) + Structured Store (originals)
```

See `ARCHITECTURE.md` for full technical details.

---

## Supported Document Types

| Format | Parsing | Element Detection |
|--------|---------|-------------------|
| PDF | Docling layout parser | Tables, formulas, images, callouts, text |
| DOCX | Docling | Tables, text, images |
| CSV | Direct pandas load | Tabular data |
| XLSX | Direct pandas load | Tabular data with sheets |

### Supported Regulatory Frameworks
- **US GAAP**: ASC 944 (insurance contracts), ASC 842, ASC 326
- **IFRS**: IFRS 17 (insurance contracts), IFRS 9, IFRS 4
- **Actuarial**: VM-20, VM-21, VM-22 (valuation manual)
- **Canadian**: LICAT, CALM, OSFI guidelines
- **General**: SOX compliance, Solvency II, SEC filings (10-K, 10-Q)

---

## Performance Characteristics

| Metric | Target | Current |
|--------|--------|---------|
| Query latency (p50) | <3s | ~2-4s (depends on document count) |
| Query latency (p95) | <5s | ~4-6s |
| Cross-encoder reranking | <200ms | ~110ms for 20 docs |
| Document ingestion | <60s/doc | ~30-90s depending on size |
| Max context window | 30K tokens | Configurable |
| Cost per query | <$0.10 | ~$0.03-0.08 (GPT-4o-mini) |

---

## Evaluation

30 golden queries across 6 categories test the full pipeline:

| Category | Count | What it tests |
|----------|-------|---------------|
| Simple | 11 | Single-concept retrieval, no decomposition |
| Comparison | 6 | Query decomposition triggers correctly |
| Aggregation | 3 | Multi-pass retrieval for "summarize all" queries |
| Formula | 4 | Table/formula element retrieval |
| Section span | 3 | Section expansion provides full context |
| Edge case | 3 | Off-topic rejection, ambiguous queries |

Run: `python scripts/evaluate.py`

---

## Deployment Options

| Option | Guide | Cost |
|--------|-------|------|
| Docker Compose | `DOCKER_README.md` | Local dev |
| AWS EC2 + RDS | `deploy/aws-setup-guide.md` | ~$450-750/mo |
| Direct Python on EC2 | `deploy/direct-python-install.md` | ~$200-400/mo |
| HTTPS setup | `deploy/https-domain-setup.md` | +$0-15/mo |

### Required Services
- **Qdrant Cloud** (or self-hosted) — vector storage
- **Elasticsearch** (or OpenSearch) — BM25 + knowledge graph
- **Redis** — response caching (optional but recommended)
- **OpenAI API** — LLM + embeddings

---

## Configuration

All settings in `config/` directory. Key tuning knobs:

| Setting | File | Default | Purpose |
|---------|------|---------|---------|
| `reranking_enabled` | `mvp_config.yaml` | `true` | Cross-encoder on/off |
| `section_expansion.enabled` | `scoring.yaml` | `true` | Section sibling retrieval |
| `query_decomposition.enabled` | `scoring.yaml` | `true` | Complex query splitting |
| `target_tokens` | `infrastructure.yaml` | `30000` | Context window budget |
| `similarity_threshold` | `scoring.yaml` | `0.30` | Minimum retrieval score |
| `min_keep` | `scoring.yaml` | `5` | Min results after reranking |

---

## Roadmap

### Completed (Phases 0-4)
- Layout-aware ingestion with element-type processors
- Hybrid retrieval with cross-encoder reranking
- Knowledge graph entity resolution
- Section expansion and priority context packing
- Query decomposition with heuristic gating
- Verification pipeline with compliance check
- Eval harness with 30 golden queries
- Dead code cleanup (~4,500 lines removed across phases)

### Next (Phase 5)
- ColPali visual retrieval for chart/diagram understanding
- Agentic orchestration for multi-step analytical queries
- SAML 2.0 SSO completion
- LLM call reduction via response caching layer
- Expanded eval set with expected answer verification

### Future (Phase 6)
- A/B testing infrastructure for retrieval experiments
- Horizontal scaling with Kubernetes
- Multi-language document support
- Teams/Slack integration
- Mobile application

---

## Related Documentation

| Document | Content |
|----------|---------|
| `ARCHITECTURE.md` | Full technical architecture with pipeline stages |
| `CROSS_ENCODER_SOLUTION.md` | Cross-encoder rationale and benchmarks |
| `INTELLIGENCE_ARCHITECTURE.md` | Intelligence subsystems (6 modules) |
| `REBUILD_PLAN.md` | Phase 0-4 rebuild history |
| `DOCKER_README.md` | Docker deployment guide |
| `deploy/aws-setup-guide.md` | AWS production deployment |
