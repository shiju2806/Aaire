# AAIRE Technical Architecture

> Last updated: 2026-03-21 | Covers Phase 4 (post-refactoring)

## Overview

AAIRE (Accounting & Actuarial Insurance Resource Expert) is a domain-specific RAG system built on FastAPI, LlamaIndex, Qdrant, and OpenAI. It processes insurance/accounting documents with layout-aware ingestion and answers natural language queries with citation-backed responses.

**Codebase**: ~24,000 lines across 80 Python files, 11 YAML config files.

---

## System Architecture

```
                          ┌─────────────────────────────────────────┐
                          │              FastAPI Server              │
                          │          main.py (30+ endpoints)         │
                          │    REST + WebSocket + SSE streaming      │
                          └──────────────┬──────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
             ┌──────────┐       ┌──────────────┐      ┌───────────┐
             │ Document │       │ RAG Pipeline │      │ Analytics │
             │ Upload   │       │              │      │ & Feedback│
             └────┬─────┘       └──────┬───────┘      └───────────┘
                  │                    │
                  ▼                    ▼
       ┌──────────────────┐  ┌──────────────────────────────────────┐
       │ Ingestion        │  │         Query Pipeline               │
       │ Pipeline         │  │                                      │
       │                  │  │  1. Topic gate                       │
       │ Layout Parser    │  │  2. Semantic query enhancement       │
       │ → Element Router │  │  3. Query decomposition (gated)      │
       │ → Type Processors│  │  4. Element-aware retrieval          │
       │ → Chunk Schema   │  │  5. Cross-encoder reranking          │
       │ → Embedding      │  │  6. Section expansion                │
       │ → Qdrant + Store │  │  7. Priority context packing         │
       └──────────────────┘  │  8. Verification + compliance        │
                             │  9. Citation extraction               │
                             └──────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
             ┌──────────┐       ┌──────────────┐      ┌───────────┐
             │  Qdrant  │       │ Elasticsearch │      │   Redis   │
             │ (vectors)│       │   (BM25/KG)   │      │  (cache)  │
             └──────────┘       └──────────────┘      └───────────┘
```

---

## Query Pipeline (Detailed)

The core query flow lives in `src/rag_pipeline.py::_run_retrieval_phase()`. Both `process_query()` (buffered) and `process_query_streaming()` (SSE) call this shared method.

### Stage 1: Topic Gate
**File**: `src/rag_modules/query/analyzer.py::classify_query_topic()`

Classifies whether the query falls within AAIRE's domain (finance, insurance, accounting). Off-topic queries get a polite rejection without consuming retrieval resources.

- Keyword check first (zero LLM cost for obvious matches)
- LLM classification for ambiguous queries

### Stage 2: Semantic Query Enhancement
**File**: `src/rag_modules/query/analyzer.py::enhance_query_semantically()`

Single LLM call to expand the query with domain-specific concepts, technical terms, related standards, and alternate phrases. The original query is weighted 5x higher than expansions to preserve BM25 precision.

### Stage 3: Query Decomposition (Gated)
**File**: `src/rag_modules/query/decomposer.py::QueryDecomposer`

Complex queries are detected via lightweight heuristic (no LLM cost):
- Comparison words: "compare", "contrast", "vs", "difference"
- Aggregation phrases: "across", "summarize all", "list all"
- Long queries (>20 words) with question words

If complex → 1 LLM call decomposes into 2-4 sub-questions → multi-pass retrieval with deduplication. Simple queries skip decomposition entirely.

**Config**: `config/scoring.yaml::query_decomposition`

### Stage 4: Element-Aware Retrieval
**File**: `src/retrieval/element_aware_retriever.py::ElementAwareRetriever`

Runs THREE retrieval paths in parallel:
1. **Hybrid search** — vector (semantic) + BM25 (keyword) via `DocumentRetriever`
2. **Graph traversal** — entity resolution in knowledge graph → connected chunks
3. **Targeted element search** — type-filtered queries for tables/formulas when query signals numerical content

Results merged via Reciprocal Rank Fusion (RRF).

**Entity filtering**: Entities extracted from query → optional Qdrant metadata filter. Falls back to unfiltered if too restrictive (<3 results).

### Stage 5: Cross-Encoder Reranking
**File**: `src/rag_modules/services/semantic_similarity.py::rerank_enriched_results()`

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~110ms for 20 docs)

Two-stage scoring:
1. Cross-encoder processes (query, document) pairs together
2. Entity-overlap bonus blended into composite score
3. Relative-gap filtering: keeps min 5 results, drops results >15 points below best

This is the canonical semantic similarity approach (replaces earlier entropy-based disambiguation).

**Config**: `config/scoring.yaml::reranking` and `config/mvp_config.yaml::reranking_enabled`

### Stage 6: Section Expansion
**File**: `src/retrieval/element_aware_retriever.py::_expand_sections()`

For high-scoring chunks, queries Qdrant for sibling chunks from the same `(document_title, section)` pair. Ensures the LLM gets full section context, not just the single best-scoring chunk.

- Expands top 3 sections above score 0.3
- Adds up to 10 sibling chunks at 0.7x parent score
- Deduplicates against existing results

**Config**: `config/scoring.yaml::section_expansion`

### Stage 7: Priority Context Packing
**File**: `src/generation/context_assembler.py::ContextAssembler`

Replaces naive score-order truncation with a 3-pass strategy:

1. **Coverage pass**: One chunk per unique section (highest-scoring)
2. **High-value pass**: All tables, formulas, callouts
3. **Fill pass**: Remaining chunks by score

Properties:
- **Deduplication**: Jaccard word overlap >80% → skip
- **Whole-element policy**: Never truncates mid-table or mid-formula
- **Budget**: 30K tokens (configurable via `config/infrastructure.yaml`)

Element rendering by type:
| Type | Rendering |
|------|-----------|
| text | Display text only |
| table | `[TABLE \| Columns: ... \| N rows]` + original markdown |
| formula | `[FORMULA]` + LaTeX + variable definitions |
| callout | `[IMPORTANT]...[/IMPORTANT]` |
| image | `[DIAGRAM/IMAGE]` + text description |

### Stage 8: Verification + Compliance
**Files**: `src/generation/verification.py`, `src/generation/compliance_check.py`

- **Verification pipeline**: Generate → verify → correct (max 3 iterations, 2-3 LLM calls)
- **Compliance check**: Post-generation scan for tax/legal advice, adds disclaimers
- **Refusal guard**: If LLM response is a refusal ("I couldn't find..."), citations are suppressed

### Stage 9: Citation Extraction
**File**: `src/generation/citation_builder.py`

Extracts inline citations from LLM response, maps `[N]` references back to source documents via the source map built during context assembly. Returns structured citation objects with document title, section, and content snippet.

---

## Ingestion Pipeline

**File**: `src/ingestion/pipeline.py::IngestionPipeline`

```
PDF/DOCX → Layout Parser → Element Router → Type Processors → Chunk Schema → Embed → Qdrant
                                                                    │
                                                            Structured Store
                                                         (original fidelity)
```

### Layout Parser
**File**: `src/ingestion/layout_parser.py` (972 lines)

Uses IBM Docling for layout-aware parsing. Detects element types:
- `text` — paragraphs, headings
- `table` — structured tables with columns/rows
- `formula` — mathematical expressions
- `callout` — boxed/highlighted text
- `image` — diagrams with OCR/description

### Type-Specific Processors
| Processor | File | Output |
|-----------|------|--------|
| TextProcessor | `src/ingestion/processors/text_processor.py` | Chunked text with section tracking |
| TableProcessor | `src/ingestion/processors/table_processor.py` | Markdown table + proposition sentences |
| FormulaProcessor | `src/ingestion/processors/formula_processor.py` | LaTeX + variable extraction |
| CalloutProcessor | `src/ingestion/processors/callout_processor.py` | Tagged important content |
| ImageProcessor | `src/ingestion/processors/image_processor.py` | Vision model description |

### Chunk Schema
**File**: `src/ingestion/chunk_schema.py`

Versioned schema (v2.0) with fields:
- `chunk_id`, `document_id`, `content`, `display_text`
- `element_type`, `section`, `page_number`
- `entities`, `entity_orgs`, `entity_persons`
- `content_hash`, `doc_content_hash` (deduplication)

### Structured Store
Preserves original-fidelity content (full markdown tables, LaTeX formulas) separate from embedding summaries. The LLM receives originals for answering while embeddings use summaries for search.

---

## Knowledge Graph

**Files**: `src/knowledge_graph/`

Elasticsearch-backed entity store:
- **Entity extraction**: NER from chunks during ingestion
- **Entity resolution**: Fuzzy matching + embedding similarity
- **Graph traversal**: Connected chunks via entity relationships
- **Context injection**: Entity context prepended to LLM prompt

Integrated into retrieval as a parallel path (Stage 4).

---

## Provider Abstractions

| Provider | File | Purpose |
|----------|------|---------|
| LLMProvider | `src/providers/llm_provider.py` | Unified LLM interface (OpenAI, extensible) |
| RetrievalProvider | `src/providers/retrieval_provider.py` | Qdrant operations (search, scroll, delete) |
| EmbeddingProvider | `src/providers/embedding_provider.py` | Embedding model abstraction |
| ConfigLoader | `src/providers/config_loader.py` | YAML config loading with caching |

---

## Configuration

All configuration in `config/`:

| File | Purpose |
|------|---------|
| `mvp_config.yaml` | Core settings: LLM model, retrieval thresholds, chunking, compliance |
| `scoring.yaml` | Relevance weights, reranking params, section expansion, query decomposition |
| `llm.yaml` | Model providers (OpenAI, Anthropic stubs, Ollama stubs) |
| `infrastructure.yaml` | Token limits, paths, server settings |
| `ingestion.yaml` | Document processing settings |
| `knowledge_graph.yaml` | Entity extraction, graph traversal config |
| `entity_extraction.yaml` | NER model and retrieval filter strategy |
| `response_generation.yaml` | Follow-up generation, formatting |
| `extraction_config.yaml` | Smart router thresholds |
| `data_sources.yaml` | External API credentials (SEC EDGAR, FRED) |
| `compliance.py` | Compliance rules (Python-based) |

Environment overrides via `.env` for secrets and deployment-specific settings.

---

## API Layer

**File**: `main.py` (1,611 lines, 30+ endpoints)

### Core Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/chat` | Buffered query + response |
| POST | `/api/v1/chat/stream` | SSE streaming response |
| WS | `/api/v1/chat/ws` | WebSocket streaming |
| POST | `/api/v1/upload` | Document upload + ingestion |
| GET | `/api/v1/documents/{job_id}/status` | Processing status |
| DELETE | `/api/v1/documents/{job_id}` | Delete document |

### Analytics & Feedback
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/feedback` | Submit response feedback |
| GET | `/api/v1/feedback/analytics` | Feedback analytics |
| GET | `/api/v1/analytics/summary` | System analytics |
| GET | `/api/v1/analytics/knowledge-gaps` | Knowledge gap detection |
| GET | `/api/v1/knowledge/stats` | Knowledge base stats |

### SEC EDGAR Integration
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/sec/companies/search` | Search SEC companies |
| GET | `/api/v1/sec/filings` | Get company filings |
| POST | `/api/v1/sec/ingest` | Ingest SEC filing |

### Debug / Admin
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/debug/documents` | List stored documents |
| POST | `/api/v1/debug/clear-all-documents` | Clear vector DB |
| POST | `/api/v1/debug/clear-cache` | Clear Redis cache |
| POST | `/api/v1/debug/reset-vector-db` | Reset Qdrant collection |

Full interactive API docs at `http://localhost:8000/docs` (Swagger UI).

---

## Evaluation Harness

**Files**: `scripts/evaluate.py`, `data/evaluation/golden_queries.yaml`

30 golden queries across 6 categories:

| Category | Count | Tests |
|----------|-------|-------|
| simple | 11 | Single-concept, should NOT trigger decomposition |
| comparison | 6 | Should trigger query decomposition |
| aggregation | 3 | Should trigger query decomposition |
| formula | 4 | Should retrieve table/formula elements |
| section_span | 3 | Requires section expansion |
| edge_case | 3 | Off-topic rejection, ambiguous, single-word |

Metrics per query:
- **Doc recall**: Expected documents found in citations or response
- **Concept coverage**: Required concepts present, forbidden concepts absent
- **Citation count**: Meets minimum threshold

```bash
python scripts/evaluate.py                      # All queries
python scripts/evaluate.py --category comparison # One category
python scripts/evaluate.py --save-baseline       # Save baseline
python scripts/evaluate.py --compare             # Check regressions
```

---

## External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| FastAPI | >=0.110.0 | Web framework |
| LlamaIndex | >=0.10.57 | RAG orchestration |
| Qdrant | >=1.8.0 | Vector database |
| OpenAI | >=1.58.0 | LLM + embeddings |
| Docling | >=2.5.0 | Layout-aware PDF parsing |
| sentence-transformers | >=2.2.0 | Cross-encoder reranking |
| Elasticsearch | >=8.12.0 | BM25 search + knowledge graph |
| Redis | >=5.0.0 | Response caching |
| spaCy | (en_core_web_sm) | Named entity recognition |

---

## Key Design Decisions

1. **Cross-encoder over entropy**: Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) replaced entropy-based disambiguation. See `CROSS_ENCODER_SOLUTION.md` for rationale and benchmarks.

2. **Original-fidelity structured store**: Tables and formulas stored separately at full fidelity. Embeddings use summaries (for search), but LLM receives originals (for answering).

3. **Heuristic-gated decomposition**: Query decomposition uses keyword heuristics (zero LLM cost) to gate, only calling LLM for confirmed complex queries. Simple queries are never decomposed.

4. **Priority packing over truncation**: Context assembler ensures coverage across sections and preserves whole elements, rather than naively cutting at a character limit.

5. **Shared retrieval phase**: Both buffered and streaming paths call `_run_retrieval_phase()` — single source of truth for retrieval logic.

6. **Config-driven features**: Section expansion, query decomposition, and reranking are all toggleable via YAML config without code changes.

---

## File Index

| Directory | Key Files | Purpose |
|-----------|-----------|---------|
| `src/` | `rag_pipeline.py` | Main RAG pipeline (1,404 lines) |
| `src/retrieval/` | `element_aware_retriever.py`, `intent_cache.py` | Retrieval layer |
| `src/generation/` | `context_assembler.py`, `verification.py`, `citation_builder.py`, `compliance_check.py` | Generation layer |
| `src/ingestion/` | `pipeline.py`, `layout_parser.py`, `chunk_schema.py`, `processors/` | Document ingestion |
| `src/rag_modules/query/` | `analyzer.py`, `decomposer.py` | Query analysis + decomposition |
| `src/rag_modules/services/` | `retrieval.py`, `generation.py`, `semantic_similarity.py` | Core services |
| `src/rag_modules/cache/` | `manager.py` | Cache management |
| `src/rag_modules/quality/` | Quality metrics + confidence scoring |
| `src/knowledge_graph/` | `graph_store.py`, `audit.py` | Entity graph |
| `src/providers/` | `llm_provider.py`, `retrieval_provider.py`, `embedding_provider.py` | Provider abstractions |
| `config/` | 11 YAML/Python files | All configuration |
| `scripts/` | `evaluate.py` | Eval harness |
| `data/evaluation/` | `golden_queries.yaml` | Golden query set |
| `deploy/` | AWS, Docker, HTTPS guides | Deployment docs |

---

## Related Documentation

- **Product specification**: `PRODUCT.md`
- **Cross-encoder rationale**: `CROSS_ENCODER_SOLUTION.md`
- **Intelligence subsystems**: `INTELLIGENCE_ARCHITECTURE.md`
- **Rebuild history**: `REBUILD_PLAN.md` (Phases 0-4 completed)
- **Deployment**: `deploy/aws-setup-guide.md`, `DOCKER_README.md`
