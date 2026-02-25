# AAIRE v2 — Rebuild Plan

## Context

AAIRE v1 was built ~6 months ago as a domain-specific RAG system for
insurance, actuarial, and accounting content. The intelligence layer
(compliance, query analysis, retrieval, generation, quality validation,
response assembly) is well-designed, but the underlying architecture
uses patterns that the RAG field has moved past.

This plan describes a phased rebuild that:
1. Removes legacy and redundant code
2. Modernizes the ingestion pipeline for multi-modal content
3. Upgrades retrieval with contextual chunking and hybrid strategies
4. Introduces provider abstractions for future-proofing
5. Moves toward agentic orchestration

Everything is incremental. No big-bang rewrite.

---

## Current Codebase Summary

| Category | Files | Lines (est.) |
|----------|-------|-------------|
| Core pipeline (`src/`) | 30 | ~12,000 |
| Extraction framework (`src/extraction/`) | 11 | ~3,500 |
| RAG modules (`src/rag_modules/`) | 23 | ~5,000 |
| Response generation (`src/response_generation/`) | 2 | ~700 |
| Config (`config/`) | 5 | ~500 |
| Root scripts (test, check, debug, reindex) | 38 | ~3,000 |
| Archive (already deprecated) | 19 | ~2,000 |
| **Total** | **~133** | **~27,000** |

### Dead Code Identified

| File | Lines | Status | Action |
|------|-------|--------|--------|
| `src/smart_metadata_analyzer.py` | 598 | Never imported | Delete |
| `src/unified_intent_analyzer.py` | 207 | Never imported | Delete |
| `src/response_generation/structured_generator.py` | ~400 | Exported but never imported | Delete |
| `src/query_processor.py` | 23 | Explicitly deprecated | Delete |
| `src/validation_layer.py` | ~100 | Deprecated, moved to ComplianceEngine | Delete |
| `src/enhanced_query_handler.py` | 176 | Fallback only, verify before removing | Review then delete |
| `src/whoosh_search_engine.py` | 508 | Replaced by BM25 engine | Delete after verifying BM25 coverage |
| **Total removable** | **~2,012** | | |

### Architectural Issues

| Issue | Location | Impact |
|-------|----------|--------|
| `"gpt-4o-mini"` hardcoded in 13+ files | Throughout `src/` | Model change requires 13 file edits |
| Absolute paths (`/Users/shijuprakash/AAIRE/...`) | `structured_generator.py`, `framework_detector.py` | Breaks on any other machine |
| Hardcoded credentials (`"admin123"`, JWT fallback secret) | `src/auth.py` | Security risk |
| 150+ hardcoded thresholds, weights, temperatures | Throughout `src/` | Tuning requires code changes |
| Two search engines coexisting (Whoosh + BM25) | `whoosh_search_engine.py` + `bm25_engine.py` | Incomplete migration |
| Sequential vector + keyword search | `src/rag_modules/services/retrieval.py` | ~40% unnecessary latency |
| No caching on intent analysis, domain classification | `unified_intent_analyzer.py`, `relevance_engine.py` | Redundant LLM calls |
| Tables/formulas/images flattened to text | `document_processor.py` | Critical information lost |

---

## Phase 0: Legacy Cleanup (Week 1)

**Goal**: Remove dead code, resolve duplications, organize root directory.
**Risk**: Zero — all deletions are verified unused code.

### 0.1 Delete Dead Code

```
DELETE:
  src/smart_metadata_analyzer.py        (598 lines, never imported)
  src/unified_intent_analyzer.py        (207 lines, never imported)
  src/response_generation/structured_generator.py  (~400 lines, never imported)
  src/query_processor.py                (23 lines, explicitly deprecated)
  src/validation_layer.py               (~100 lines, deprecated)
```

### 0.2 Resolve Search Engine Duplication

The BM25 engine (`src/rag_modules/search/bm25_engine.py`) was built to
replace Whoosh (`src/whoosh_search_engine.py`). Verify BM25 covers all
Whoosh functionality, then:

```
DELETE:
  src/whoosh_search_engine.py           (508 lines)
  test_whoosh_index/                    (test index directory)

UPDATE:
  rag_pipeline.py                       Remove Whoosh imports, use BM25 only
  vm20_comprehensive_search.py          Update to use BM25 (or delete script)
  targeted_reserve_search.py            Update to use BM25 (or delete script)
  search_reserve_terms.py               Update to use BM25 (or delete script)
```

### 0.3 Organize Root Directory

Move root-level scripts into organized directories:

```
scripts/
  maintenance/
    check_compatibility.py
    check_documents.py
    check_job_status.py
    check_new_upload.py
    check_pdfs.py
    check_qdrant.py
    simple_check.py
    clear_cache.py
    cleanup_vector_db.py
    reset_vector_db.py
  reindex/
    reindex_all_documents.py
    reindex_all_3_documents.py
    reindex_vm20.py
  debug/
    debug_citations.py
    debug_spatial_extraction.py
    analyze_pdf.py
  search/
    deep_qdrant_search.py
    inspect_qdrant_content.py
    vm20_comprehensive_search.py
    targeted_reserve_search.py
    search_reserve_terms.py
```

Move root-level test files:

```
tests/
  integration/
    test_complete_system.py
    test_cross_encoder_reranking.py
    test_enhanced_processor.py
    test_finance_extraction.py
    test_intelligent_extraction.py
    test_phase4_complete.py
    test_phrase_discrimination.py
    test_pptx_extraction.py
    test_semantic_enhancement.py
    test_spatial_logic.py
    test_taxonomy_lookup.py
  unit/
    test_compliance.py (existing)
```

### 0.4 Delete Archive

The `archive/` directory contains 19 files that are already deprecated
and unreferenced. Delete the entire directory.

```
DELETE:
  archive/                              (19 files, ~2,000 lines)
```

### 0.5 Fix Hardcoded Paths

Replace absolute paths with relative paths or environment variables:

```
FIX:
  src/response_generation/structured_generator.py
    /Users/shijuprakash/AAIRE/config/... → Path(__file__).parent / "../../config/..."
  src/extraction/framework_detector.py
    /Users/shijuprakash/AAIRE/config/... → Path(__file__).parent / "../../config/..."
```

### 0.6 Remove Hardcoded Credentials

```
FIX:
  src/auth.py
    Remove "your-secret-key-change-in-production" fallback
    Remove "admin123" / "user123" demo passwords
    Require JWT_SECRET_KEY env var (fail loudly if missing)
```

**Phase 0 outcome**: ~4,500 lines removed, clean directory structure,
no functional changes, no regressions.

---

## Phase 1: Provider Abstractions (Week 2)

**Goal**: Decouple the system from specific LLM providers, embedding
models, and vector stores. Every future upgrade becomes a config change.

### 1.1 LLM Provider Interface

Create `src/providers/llm_provider.py`:

```
LLMProvider (abstract)
  ├── generate(prompt, params) → str
  ├── structured_output(prompt, schema) → dict
  └── classify(prompt, categories) → str

OpenAIProvider(LLMProvider)
  - Wraps OpenAI API calls
  - Reads model name from config

AnthropicProvider(LLMProvider)
  - Future: Claude integration
```

Create `config/llm.yaml`:

```yaml
providers:
  default: openai

  openai:
    classification_model: gpt-4o-mini
    generation_model: gpt-4o
    extraction_model: gpt-4o-mini
    embedding_model: text-embedding-3-large

    params:
      classification:
        temperature: 0.1
        max_tokens: 500
      generation:
        temperature: 0.3
        max_tokens: 4000
      extraction:
        temperature: 0.0
        max_tokens: 2000
```

Then replace all 13+ hardcoded `model="gpt-4o-mini"` calls with
`self.llm_provider.generate(...)` or `self.llm_provider.classify(...)`.

### 1.2 Retrieval Provider Interface

Create `src/providers/retrieval_provider.py`:

```
RetrievalProvider (abstract)
  ├── search(query, filters, limit) → List[RetrievedDocument]
  ├── index(documents) → None
  ├── delete(doc_ids) → None
  └── health_check() → bool

QdrantProvider(RetrievalProvider)
  - Current Qdrant implementation

ColBERTProvider(RetrievalProvider)
  - Future: ColBERT late-interaction search

HybridProvider(RetrievalProvider)
  - Combines multiple providers
  - Handles score merging and deduplication
```

### 1.3 Embedding Provider Interface

Create `src/providers/embedding_provider.py`:

```
EmbeddingProvider (abstract)
  ├── embed_text(text) → List[float]
  ├── embed_batch(texts) → List[List[float]]
  └── dimension() → int

OpenAIEmbeddingProvider(EmbeddingProvider)
VoyageEmbeddingProvider(EmbeddingProvider)
ColPaliEmbeddingProvider(EmbeddingProvider)  # visual embeddings
```

### 1.4 Centralize Configuration

Create `config/scoring.yaml` (consolidate all 150+ hardcoded thresholds):

```yaml
relevance:
  query_type_weights:
    specific_reference: {exact_match: 0.6, semantic: 0.2, context: 0.1, entity: 0.1}
    conceptual:         {exact_match: 0.2, semantic: 0.6, context: 0.1, entity: 0.1}
    comparison:         {exact_match: 0.2, semantic: 0.4, context: 0.1, entity: 0.3}
    procedural:         {exact_match: 0.3, semantic: 0.3, context: 0.1, entity: 0.3}
    contextual:         {exact_match: 0.1, semantic: 0.2, context: 0.6, entity: 0.1}
  boost_thresholds:
    high_specificity: 0.8
    entity_match: 0.3
    domain_alignment: 0.2

quality:
  grounding_weight: 0.4
  semantic_weight: 0.3
  hallucination_penalty_weight: 0.2
  openai_alignment_weight: 0.1
  confidence_threshold: 0.8
  adaptive_threshold_bounds: {min: 0.25, max: 0.8}

conversation:
  importance_weights:
    question_boost: 0.3
    calculation_boost: 0.4
    number_boost: 0.3
    regulatory_boost: 0.2
    short_message_penalty: 0.8
    max_importance: 2.0
  compression:
    max_messages_before_compression: 15
    max_compressed_summaries: 3
    compression_ratio: 0.3
    max_context_tokens: 2000

retrieval:
  doc_limit: 20
  confidence_early_exit: 0.90
  bm25_weight: 0.3
  vector_weight: 0.7

extraction:
  pattern_threshold: 0.8
  light_llm_threshold: 0.6
  full_llm_threshold: 0.4
  circuit_breaker:
    failure_threshold: 5
    timeout_seconds: 60
```

Create `config/infrastructure.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

redis:
  host: ${REDIS_HOST:-localhost}
  port: 6379
  session_ttl_hours: 2

auth:
  algorithm: HS256
  access_token_expire_hours: 8

timeouts:
  citation_analysis: 30
  extraction: 300
  llm_retry_max: 2
  cache_ttl_hours: 24

paths:
  upload_dir: data/uploads
  analytics_dir: data/analytics
  taxonomy_dir: data/taxonomies
```

**Phase 1 outcome**: All hardcoded values externalized. Changing LLM
provider, model, embedding, or vector store is a config change. Zero
functional changes to end-user behavior.

---

## Phase 2: Multi-Modal Ingestion Pipeline (Weeks 3-4)

**Goal**: Replace the current text-extraction-only pipeline with
layout-aware, type-specific processing that preserves tables, formulas,
callouts, and diagrams.

### 2.1 Layout-Aware Document Parsing

Replace current OCR-first approach with Docling (IBM, open source)
as the primary document parser.

Create `src/ingestion/layout_parser.py`:

```
LayoutParser
  ├── parse(file_path) → List[DocumentElement]
  └── supported_formats() → List[str]

DocumentElement:
  element_type: text | table | formula | callout | image | header | footer
  content: str                    # raw content
  structured_content: dict | None # for tables: markdown/JSON
  page_number: int
  bounding_box: BBox | None
  parent_section: str             # section heading this belongs to
  metadata: dict
```

Docling handles: PDF, DOCX, PPTX, HTML. It classifies every region
on every page into element types and preserves structure.

### 2.2 Type-Specific Processors

Create `src/ingestion/processors/`:

```
processors/
  text_processor.py       # Contextual chunking for text elements
  table_processor.py      # Multi-representation table processing
  formula_processor.py    # LaTeX extraction + NL description
  callout_processor.py    # Standalone high-priority chunks
  image_processor.py      # Vision model captioning
  page_processor.py       # ColPali visual page embeddings (Phase 4)
```

**Text processor** — contextual chunking:
1. Receive text elements with parent_section metadata
2. Chunk by semantic boundaries (not fixed token windows)
3. For each chunk, generate a context prefix using cheap LLM:
   "This chunk is from [document_title], [section_heading],
   discussing [topic]."
4. Prepend context to chunk before embedding
5. Store: {embedding_text: context+chunk, display_text: chunk,
   context: context_prefix, metadata: {...}}

**Table processor** — multi-representation:
1. Receive table element with structured content (markdown/JSON)
2. Generate text summary via LLM: "This table shows [what] for [context]"
3. Extract row-level propositions for large tables (>10 rows):
   "The mortality rate for age 45 male nonsmoker is 0.00298 (2017 CSO)"
4. Extract structured metadata: {type, standard, jurisdiction,
   topic, column_headers, row_count}
5. Store three representations:
   - Text summary → embedded in Qdrant (for semantic search)
   - Structured metadata → Qdrant payload (for filtered search)
   - Original markdown table → structured store (for LLM context)

**Formula processor**:
1. Receive formula element (may be image or text)
2. If image: extract LaTeX via Nougat or vision model
3. Generate natural language description via LLM
4. Extract variable definitions
5. Store three representations:
   - NL description → embedded in Qdrant
   - LaTeX → structured store
   - Variables → knowledge graph (Phase 4)

**Callout processor**:
1. Receive callout/sidebar element
2. Tag with importance=high
3. Link to parent section context
4. Chunk as standalone unit (do not merge with surrounding text)
5. Apply 1.3x retrieval boost via metadata flag

**Image processor**:
1. Receive image element
2. Send to vision model (GPT-4o or Claude) with prompt:
   "Describe this diagram from an actuarial/insurance document.
   Include all labels, relationships, data values, and flow direction."
3. Store: {description → Qdrant, original_image → blob store,
   element_type: image, source_page: N}

### 2.3 Chunk Schema Versioning

Every chunk stored in Qdrant carries a version:

```json
{
  "content": "...",
  "context_prefix": "...",
  "schema_version": "2.0",
  "chunking_strategy": "contextual",
  "element_type": "text|table|formula|callout|image",
  "embedding_model": "text-embedding-3-large",
  "document_id": "...",
  "section": "...",
  "page": 1,
  "importance": 1.0,
  "jurisdiction": "IFRS|US_GAAP|US_STAT|unknown",
  "product_type": "universal_life|whole_life|term|general"
}
```

This enables:
- Side-by-side comparison of old vs new chunking
- Gradual migration (don't re-ingest everything at once)
- Filtering by schema version during retrieval

### 2.4 Structured Content Store

Create a separate store for original-fidelity content that gets passed
to the LLM at generation time (not for retrieval):

```
Qdrant:           Stores embeddings + metadata (for finding things)
Structured Store: Stores original tables, formulas, images (for answering)
```

The structured store can be PostgreSQL (JSONB), S3 + metadata index,
or even a local file store keyed by chunk ID. It does not need to be
a vector database.

Retrieval flow becomes:
1. Search Qdrant → get chunk IDs for relevant text summaries
2. Look up chunk IDs in structured store → get original tables/formulas
3. Pass originals (not summaries) to the LLM for generation

### 2.5 OCR Simplification

Docling handles OCR internally. The current 4-file OCR cascade
(Google Vision → Tesseract → docTR → EasyOCR) can be simplified:

```
KEEP (for now):
  ocr_processor.py           Simplified to delegate to Docling
  ocr_processor_google.py    Fallback for edge cases Docling misses

DELETE (after Docling is proven):
  ocr_processor_tesseract.py
  ocr_processor_doctr.py
```

### 2.6 Legacy Ingestion Removal

After the new ingestion pipeline is working:

```
DELETE:
  src/enhanced_document_processor.py     Replaced by layout parser
  src/shape_aware_processor.py           Replaced by Docling layout analysis
  src/pdf_spatial_extractor.py           Replaced by Docling
  src/pptx_shape_extractor.py            Replaced by Docling PPTX support
  src/chart_analyzer.py                  Replaced by vision model captioning
  src/finance_structures_parser.py       Replaced by table processor
  src/advanced_org_parser.py             Replaced by image processor
  src/intelligent_text_parser.py         Replaced by text processor

SIMPLIFY:
  src/document_processor.py              Thin wrapper around new ingestion pipeline
```

**Phase 2 outcome**: Documents ingested with full structural awareness.
Tables, formulas, callouts, images preserved as first-class elements.
~3,000 lines of legacy extraction code removed.

---

## Phase 3: Retrieval Upgrades (Weeks 5-6)

**Goal**: Improve retrieval precision and latency. Short-term wins
that build on the new multi-representation storage.

### 3.1 Parallelize Vector + BM25 Search

Immediate performance fix in `src/rag_modules/services/retrieval.py`:

```python
# Before (sequential):
vector_results = await self.vector_search(query, filters, limit)
keyword_results = await self.keyword_search(query, filters, limit)

# After (parallel):
vector_results, keyword_results = await asyncio.gather(
    self.vector_search(query, filters, limit),
    self.keyword_search(query, filters, limit)
)
```

### 3.2 Element-Type-Aware Retrieval

With multi-representation storage, retrieval can now be smarter:

```
Query: "What mortality rates does VM-20 use for age 45?"

1. Text search → finds text chunks about VM-20 mortality requirements
2. Table filter → finds elements where:
     element_type = "table"
     AND jurisdiction = "US_STAT"
     AND topic contains "mortality"
3. Merge and rank
4. Fetch original tables from structured store
5. Pass both text context AND original table to LLM
```

Create `src/retrieval/element_aware_retriever.py`:
- Accepts query + detected intent
- Decides which element types to search for
- Runs type-specific filtered searches in parallel
- Merges results with element-type-appropriate scoring
  (tables get higher weight for numerical queries,
   text gets higher weight for conceptual queries)

### 3.3 Confidence-Based Early Exit

Add short-circuit logic to retrieval:

```
If top 3 results all score > 0.90:
  - Skip BM25 search entirely
  - Skip reflective retrieval
  - Return immediately

If average top 5 score < 0.40:
  - Trigger reflective retrieval
  - Consider query reformulation
```

This saves ~40% latency on high-confidence queries and focuses
expensive operations where they're actually needed.

### 3.4 Intent Analysis Caching

Cache query intent results to avoid redundant LLM calls:

```
Cache key: hash(normalized_query)
Cache value: {intent, jurisdiction, product_type, query_type}
TTL: 24 hours
```

Add pattern-based fast path for obvious intents:
- "What is X?" → intent=definition, skip LLM
- "How to calculate X?" → intent=procedural, skip LLM
- "Compare X vs Y" → intent=comparison, skip LLM

Fall back to LLM only for ambiguous queries.

### 3.5 Keep Entropy Disambiguation

The entropy-based disambiguation service is genuinely novel and
addresses a real problem. With contextual chunking, its impact will
be reduced (because chunks carry their own framework context), but
it still adds value for:
- Cross-framework queries ("how does reserve calculation differ?")
- Ambiguous terminology that contextual chunking doesn't fully resolve
- Edge cases where the same term appears in conflicting contexts

Keep it. Let the data show whether it's still adding lift after
contextual chunking is live.

### 3.6 Retrieval Metrics

Add instrumentation to measure retrieval quality:

```
Per query, log:
  - retrieval_latency_ms
  - top_5_average_score
  - element_types_retrieved (text, table, formula, image)
  - early_exit_triggered (bool)
  - reflective_retrieval_triggered (bool)
  - bm25_contributed (bool — did BM25 add results not in vector?)
  - entropy_disambiguation_triggered (bool)
```

This data will drive decisions about whether to add ColBERT, knowledge
graphs, or other advanced retrieval in Phase 5.

**Phase 3 outcome**: Retrieval is faster (parallel search, early exit),
smarter (element-type-aware), and measurable.

---

## Phase 4: Generation & Quality Upgrades (Weeks 7-8)

**Goal**: Simplify the generation pipeline. Pass richer context
(original tables, formulas) to the LLM. Reduce LLM call count.

### 4.1 Context Assembly Layer

Create `src/generation/context_assembler.py`:

The context assembler takes retrieval results and builds an optimal
LLM prompt:

```
For each retrieved element:
  if element_type == "text":
    include the text chunk (with context prefix stripped — LLM
    doesn't need "This chunk is from..." preamble)
  if element_type == "table":
    fetch original table from structured store
    include as markdown table
  if element_type == "formula":
    fetch LaTeX from structured store
    include LaTeX + variable definitions
  if element_type == "image":
    if vision model: include original image
    else: include text description
  if element_type == "callout":
    include with [IMPORTANT] tag
```

Order by: relevance score descending, with tables and formulas
grouped near related text chunks.

Truncate to fit context window, prioritizing higher-scored elements.

### 4.2 Simplify Self-Correction

Current: Generate → Verify (7 criteria) → Correct → Verify → Correct
(up to 7 LLM calls per response)

Replace with a lighter two-stage approach:

```
Stage 1: Generate with enhanced context
  - Richer context (original tables, formulas) means better first-pass
  - Include grounding instructions in the generation prompt:
    "Only state facts that are directly supported by the provided context.
     If a table contains the relevant data, cite specific values."

Stage 2: Single verification pass
  - Check: hallucination + completeness + relevance (3 criteria, not 7)
  - If passes: return
  - If fails: one correction attempt, then return
  - Max 3 LLM calls total (down from 7)
```

The insight: better input context reduces the need for correction.
If the LLM has the actual mortality table (not a text summary), it's
far less likely to hallucinate mortality rates.

### 4.3 Post-Generation Compliance Check

Add a lightweight LLM compliance check after generation (discussed
earlier as defense-in-depth):

```
Prompt: "Does this response give specific tax advice, legal opinions,
or recommendations about specific reserve adequacy for the user's
situation? Answer yes/no with explanation."

If yes: add professional judgment disclaimer
```

This catches subtle compliance issues that regex patterns miss.

### 4.4 Citation Improvement

With element-type-aware retrieval, citations become more precise:

```
Before: "Based on the retrieved documents..." (vague)
After:  "Based on Table 3.2 from VM-20 Section 3 (2017 CSO mortality
         rates, male nonsmoker)..." (specific)
```

The citation analyzer now has structured metadata (element type,
source document, section, page) to work with.

**Phase 4 outcome**: Generation is cheaper (fewer LLM calls), more
accurate (richer context), and produces better citations.

---

## Phase 5: Advanced Capabilities (Weeks 9-12)

**Goal**: Add capabilities that depend on Phases 1-4 being solid.
Each is independent and can be prioritized based on user feedback
and retrieval metrics.

### 5.1 Knowledge Graph (Optional — build if metrics justify)

Add a knowledge graph alongside the vector store:

```
Build during ingestion:
  - Extract entities: standards (ASC 944, VM-20, IFRS 17),
    concepts (CSM, NPR, DAC), products (UL, WL, term),
    calculations (present value, amortization)
  - Extract relationships: references, supersedes, applies_to,
    component_of, effective_date
  - Store in Neo4j or NetworkX (for smaller corpora)

Query during retrieval:
  - After vector search identifies relevant chunks
  - Graph traversal finds related standards, cross-references,
    prerequisite concepts
  - Add graph context to LLM prompt
```

**Build this only if** retrieval metrics from Phase 3 show that
users frequently ask cross-reference questions that vector search
alone handles poorly.

### 5.2 ColPali Visual Retrieval (Optional — for image-heavy corpora)

Add ColPali as a complementary retrieval path:

```
During ingestion:
  - For each page, generate ColPali visual embedding
  - Store in Qdrant (separate collection, different vector dimension)

During retrieval:
  - Run text search (Qdrant) AND visual search (ColPali) in parallel
  - Visual search finds pages with relevant tables/diagrams
  - Merge results by document + page

During generation:
  - If visual search contributed results, include page images
    in multimodal LLM prompt
```

**Build this only if** the image processor from Phase 2 proves
insufficient — i.e., vision model captioning misses important
visual information.

### 5.3 Agentic Orchestration (Recommended — after Phases 1-4)

Replace the fixed pipeline with an agent loop:

```
Agent has tools:
  - search_text(query, filters) → text chunks
  - search_tables(query, filters) → tables
  - search_formulas(topic) → formulas
  - graph_traverse(entity, relation) → related entities (if 5.1 built)
  - check_compliance(text) → compliance result
  - validate_grounding(response, context) → quality score
  - ask_user(question) → user clarification

Agent loop:
  1. Analyze query
  2. Decide which tools to call (and in what order)
  3. Call tools, assess results
  4. If insufficient: call more tools or reformulate
  5. Generate response from collected context
  6. Validate and return
```

This replaces:
- The fixed 6-stage pipeline
- Tier-based query classification (the agent adapts naturally)
- The reflective retriever (the agent re-searches when needed)
- Query decomposition logic (the agent decomposes naturally)

**Implementation**: Use LangGraph (you already have it in the DataGenie
project) to build the agent loop. Each existing pipeline stage becomes
a tool the agent can invoke.

### 5.4 Composable Pipeline Config (Intermediate step toward 5.3)

If full agentic is too big a leap, start with composable pipelines:

```yaml
# config/pipelines.yaml
pipelines:
  simple:
    steps: [compliance_gate, retrieve_text, generate, format]

  standard:
    steps: [compliance_gate, analyze_intent, retrieve_hybrid,
            generate_with_cot, verify, format]

  complex:
    steps: [compliance_gate, decompose_query, retrieve_multi,
            retrieve_tables, synthesize, verify, format]

  comparison:
    steps: [compliance_gate, decompose_comparison, retrieve_per_side,
            generate_comparison, verify, format]

routing:
  # Pattern-based routing to pipeline
  definition_query: simple
  factual_query: standard
  comparison_query: comparison
  multi_hop_query: complex
  default: standard
```

Each step is a function with a standard interface. The pipeline
config determines which steps run and in what order. Adding a new
step means writing one function and adding it to the relevant pipeline
configs.

This is a stepping stone. When you eventually move to agentic (5.3),
these steps become the agent's tools.

---

## Phase 6: Operational Hardening (Ongoing)

### 6.1 Monitoring & Observability

```
Per request, log:
  pipeline_used: str
  total_latency_ms: int
  retrieval_latency_ms: int
  generation_latency_ms: int
  llm_calls_count: int
  llm_tokens_used: int
  retrieval_element_types: List[str]
  quality_score: float
  early_exit: bool
  cache_hit: bool
  compliance_triggered: bool
```

### 6.2 A/B Testing Infrastructure

With chunk schema versioning (Phase 2) and composable pipelines
(Phase 5), you can run A/B tests:
- Old chunking vs contextual chunking (same query, compare scores)
- Fixed pipeline vs agentic (same query, compare quality + latency)
- With entropy disambiguation vs without (measure lift)

### 6.3 Automated Regression Testing

Build a test suite of ~50 representative queries with expected
behavior:
- "What is CSM?" → should return IFRS 17 definition
- "VM-20 mortality rate for age 45" → should return specific value from table
- "Compare IFRS 17 CSM vs US GAAP DAC" → should address both frameworks
- "Should I increase my reserves?" → compliance gate should trigger

Run this suite after every pipeline change to catch regressions.

---

## Summary: What Gets Deleted

| Phase | Files Removed | Lines Removed (est.) |
|-------|--------------|---------------------|
| Phase 0 | dead code + archive + Whoosh | ~4,500 |
| Phase 2 | legacy extraction/OCR files | ~3,000 |
| **Total** | ~25 files | **~7,500 lines** |

## Summary: What Gets Built

| Phase | New Components | Purpose |
|-------|---------------|---------|
| Phase 1 | Provider interfaces, centralized config | Future-proofing |
| Phase 2 | Layout parser, type-specific processors, structured store | Multi-modal ingestion |
| Phase 3 | Parallel search, element-aware retrieval, caching | Retrieval performance |
| Phase 4 | Context assembler, simplified self-correction | Generation quality |
| Phase 5 | Knowledge graph, ColPali, agentic loop | Advanced capabilities |
| Phase 6 | Monitoring, A/B testing, regression tests | Operational maturity |

## Summary: Timeline

| Phase | Duration | Dependencies |
|-------|----------|-------------|
| Phase 0: Cleanup | Week 1 | None |
| Phase 1: Abstractions | Week 2 | Phase 0 |
| Phase 2: Ingestion | Weeks 3-4 | Phase 1 |
| Phase 3: Retrieval | Weeks 5-6 | Phase 2 |
| Phase 4: Generation | Weeks 7-8 | Phase 3 |
| Phase 5: Advanced | Weeks 9-12 | Phase 4 (each sub-phase independent) |
| Phase 6: Operations | Ongoing | Phase 4+ |

## Design Principles

1. **Search on descriptions, answer from originals** — embed text
   summaries for retrieval, but pass original tables/formulas/images
   to the LLM for generation.

2. **Provider interfaces everywhere** — LLM, embedding, retrieval,
   and storage are all behind interfaces. Swap implementations via
   config.

3. **Chunk schema versioning** — every stored chunk carries its
   version. Migrate gradually, compare old vs new, roll back if needed.

4. **Composable steps, not fixed pipelines** — each intelligence
   capability is a standalone step with a standard interface. Pipeline
   configs determine which steps run. Eventually, an agent decides.

5. **Measure before optimizing** — add retrieval metrics before
   building ColBERT or knowledge graphs. Let data drive decisions.

6. **Graceful degradation preserved** — every new component has a
   fallback path. The system never hard-fails because an advanced
   component is unavailable.
