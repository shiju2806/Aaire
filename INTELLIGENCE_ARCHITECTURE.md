# AAIRE — Intelligence Architecture

## What This Document Is

This document describes the intelligence layer of AAIRE — the systems
that make it more than a standard RAG application. It is written from
a codebase review, not from aspirational design docs. Everything
described here exists in code.

## System Overview

AAIRE processes a user query through six interconnected intelligence
subsystems before producing a response. Each subsystem can operate
independently and degrades gracefully if unavailable.

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. COMPLIANCE ENGINE                                            │
│     Rule-based filtering + professional judgment detection        │
│     Blocks tax/legal advice. Adds disclaimers where needed.      │
└──────────────────────┬───────────────────────────────────────────┘
                       │ (if not blocked)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. QUERY INTELLIGENCE                                           │
│     ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐  │
│     │ Query Analyzer   │  │ Intelligent Query │  │ NLP Query   │  │
│     │ Topic relevance, │  │ Analyzer          │  │ Processor   │  │
│     │ general vs       │  │ Jurisdiction +    │  │ Query       │  │
│     │ doc-specific,    │  │ product intent    │  │ expansion   │  │
│     │ query expansion  │  │ detection         │  │ with domain │  │
│     └─────────────────┘  └──────────────────┘  │ terms       │  │
│                                                 └─────────────┘  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. RETRIEVAL INTELLIGENCE                                       │
│     ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│     │ Vector Search  │  │ BM25 Keyword  │  │ Entropy          │  │
│     │ (Pinecone)     │──│ Search        │──│ Disambiguation   │  │
│     │ Semantic       │  │ (Whoosh)      │  │ Mutual exclusion │  │
│     │ similarity     │  │ Exact match   │  │ detection        │  │
│     └───────────────┘  └───────────────┘  └──────────────────┘  │
│              │                  │                    │            │
│              ▼                  ▼                    ▼            │
│     ┌──────────────────────────────────────────────────────┐     │
│     │ Hybrid Merge → Relevance Engine → Framework Filter   │     │
│     │ Dedup + score merge   Query-type     LLM-based       │     │
│     │                       adaptive       framework       │     │
│     │                       weighting      detection       │     │
│     └──────────────────────────┬───────────────────────────┘     │
│                                │                                 │
│                                ▼                                 │
│     ┌──────────────────────────────────────────────────────┐     │
│     │ Reflective Retriever                                  │     │
│     │ If semantic alignment fails → evaluate gaps →         │     │
│     │ generate alternative queries → re-retrieve            │     │
│     └──────────────────────────────────────────────────────┘     │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. GENERATION INTELLIGENCE                                      │
│     Chain-of-Thought reasoning with methodology selection         │
│     Multi-pass self-correction loop (max 3 iterations):          │
│                                                                   │
│     Generate (with CoT) → Verify → [if issues] → Correct → ↩    │
│                                                                   │
│     Verification criteria: factual accuracy, completeness,        │
│     logical consistency, hallucination check, context grounding,  │
│     clarity, relevance                                            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. QUALITY VALIDATION                                           │
│     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │
│     │ Grounding       │  │ Semantic        │  │ Unified        │  │
│     │ Validator       │  │ Alignment       │  │ Quality        │  │
│     │ Evidence-based  │  │ Validator       │  │ Validator      │  │
│     │ hallucination   │  │ Intent match    │  │ Weighted       │  │
│     │ detection with  │  │ verification    │  │ composite      │  │
│     │ adaptive        │  │                 │  │ scoring        │  │
│     │ thresholds      │  │                 │  │                │  │
│     └────────────────┘  └────────────────┘  └────────────────┘  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  6. RESPONSE ASSEMBLY                                            │
│     Citation extraction (LLM-based document usage analysis)      │
│     Structured formatting with source attribution                │
│     Follow-up question generation                                │
│     Conversation memory with entity tracking                     │
└──────────────────────────────────────────────────────────────────┘
                       │
                       ▼
                   Response
```


## Subsystem 1: Compliance Engine

**Location**: `src/compliance_engine.py`, `config/compliance.py`

The compliance engine is the first gate. Every query passes through
rule-based pattern matching before any retrieval or generation occurs.

**What it does:**
- Blocks queries requesting tax advice, legal opinions, or other
  out-of-scope content using configurable regex patterns
- Detects queries requiring professional judgment disclaimers
  (e.g., questions about specific reserve adequacy, pricing decisions)
- Logs all compliance events with severity levels for audit trails

**Design decision:** Compliance rules are loaded from `config/compliance.py`,
not hardcoded in the engine. The engine only knows how to match patterns
and produce results — the rules themselves are external configuration.


## Subsystem 2: Query Intelligence

Three components work together to understand what the user is asking
before retrieval begins.

### Query Analyzer
**Location**: `src/rag_modules/query/analyzer.py`

- **Topic classification**: Determines if the query falls within
  AAIRE's domain (insurance, accounting, actuarial, finance). Uses
  keyword matching first, falls back to LLM classification for
  ambiguous queries. Out-of-domain queries receive polite redirects.

- **General vs document-specific detection**: Distinguishes "What is
  revenue recognition?" (general knowledge) from "What does the
  uploaded document say about revenue recognition?" (document-specific).
  Uses regex patterns for document indicators (`the document`,
  `uploaded`, `show me`, ASC/IFRS references) and general knowledge
  patterns (`what is X?`, `define X`).

- **Query expansion**: Maps general terms to domain-specific vocabulary.
  "Capital health" expands to include "LICAT ratio", "core ratio",
  "total ratio", "capital adequacy." This improves retrieval recall
  for users who use colloquial rather than technical language.

- **Follow-up question validation**: Determines whether a suggested
  follow-up question is actually contextual to the conversation or
  generic filler. Checks for specific references to metrics, amounts,
  standards, and entities mentioned in the prior response.

### Intelligent Query Analyzer
**Location**: `src/intelligent_query_analyzer.py`

Extracts two critical metadata signals from the query before retrieval:

- **Jurisdiction detection**: Identifies whether the query relates to
  US Statutory (NAIC, VM-20, statutory reserves), IFRS (IFRS 17, CSM,
  risk adjustment, fulfilment cash flows), US GAAP, or is mixed/unknown.
  Uses configurable regex patterns with confidence scoring.

- **Product type detection**: Identifies insurance product context —
  universal life, whole life, term life, variable life, or general.
  Patterns include product-specific terminology (e.g., "secondary
  guarantee", "no-lapse guarantee" → universal life).

- **Disambiguation flag**: Sets `disambiguation_needed = True` when
  both jurisdiction and product signals are ambiguous, triggering
  the entropy disambiguation service downstream.

These signals are passed to the retrieval layer as `QueryIntent`,
enabling smart filtering of search results by jurisdiction and
product type when confidence exceeds 0.8.

### Query Expansion
**Location**: `src/rag_modules/query/analyzer.py` (expand_query method)

Domain-specific term expansion that adds retrieval-relevant vocabulary
to the user's query. For example, a query about "insurance regulatory"
gets expanded with "OSFI LICAT compliance capital requirements" to
improve recall against documents that use specific regulatory terminology
rather than general language.


## Subsystem 3: Retrieval Intelligence

This is the most complex subsystem. It combines five retrieval
mechanisms into a single ranked result set.

### Hybrid Search
**Location**: `src/rag_modules/services/retrieval.py`

The core retrieval runs **vector search and BM25 keyword search in
parallel** using `asyncio.gather` for ~40% performance improvement
over sequential execution.

**Buffer approach**: Both search methods receive the full document
limit (default 20), not half each. This prevents the problem where
one method returns strong results but is capped, losing relevant
documents. The deduplication step afterward merges scores when a
document appears in both result sets.

**Merge strategy**: When a document appears in both vector and keyword
results, the higher score is kept and the search type is marked as
`hybrid`. The combined set then passes through the relevance engine
for final ranking.

### Dynamic Relevance Engine
**Location**: `src/relevance_engine.py`

Query-type-aware relevance scoring. The engine first classifies the
query into one of five types:

| Query Type | Exact Match | Semantic | Context | Entity Coverage |
|------------|-------------|----------|---------|-----------------|
| Specific Reference (ASC 255-10-50-51) | 0.6 | 0.2 | 0.1 | 0.1 |
| Conceptual ("what is revenue recognition") | 0.2 | 0.6 | 0.1 | 0.1 |
| Comparison ("GAAP vs IFRS") | 0.2 | 0.4 | 0.1 | 0.3 |
| Procedural ("how to calculate reserves") | 0.3 | 0.3 | 0.1 | 0.3 |
| Contextual ("in the attached document") | 0.1 | 0.2 | 0.6 | 0.1 |

This means a specific ASC reference query heavily weights exact string
matching, while a conceptual question relies primarily on semantic
similarity. The engine also supports AI-powered domain classification
via OpenAI when pattern-based detection is insufficient.

### Entropy-Based Disambiguation
**Location**: `src/rag_modules/services/entropy_disambiguation_service.py`

**This is AAIRE's most original technical contribution.**

The problem: Insurance and accounting terminology is heavily overloaded.
"Reserve" means different things in US GAAP, IFRS, and statutory
contexts. "Contract boundary" has distinct definitions across frameworks.
Standard vector similarity search cannot distinguish between these
meanings because the surface-level language is nearly identical.

The solution: Rather than hardcoding disambiguation rules, the system
learns mutual exclusions from the document corpus using statistical
analysis.

**How it works:**

1. **Concept extraction**: KeyBERT extracts key concepts from both
   queries and corpus documents using Maximal Marginal Relevance (MMR)
   for diversity. N-gram range is (1,3), extracting up to 10 concepts
   per text with configurable diversity threshold.

2. **Co-occurrence matrix**: For each pair of extracted concepts, the
   system builds a contingency table tracking how often they appear
   together vs separately across documents.

3. **Statistical testing**: For small samples (<50 documents),
   Fisher's exact test determines if co-occurrence is significantly
   lower than expected. For larger samples, chi-square test is used.
   Significance threshold is p < 0.05.

4. **Exclusion confidence**: Calculated as the ratio between expected
   and actual co-occurrence, adjusted by statistical significance.
   A confidence boost of 1.0 is applied when the Fisher p-value is
   significant; 0.5 otherwise.

5. **Query disambiguation**: When a query contains two concepts with
   high exclusion confidence (i.e., they rarely appear in the same
   document), the system flags the ambiguity and can recommend
   clarification.

6. **Entropy scoring**: Shannon entropy over concept probabilities
   measures overall query ambiguity. Higher entropy = more ambiguous.

**Configuration** (from `quality_validation.yaml`):
```yaml
entropy_disambiguation:
  top_k_concepts: 10
  keyphrase_ngram_range: [1, 3]
  min_concept_frequency: 3
  diversity_threshold: 0.5

statistical_thresholds:
  min_confidence: 0.3
  high_confidence: 0.6
  min_co_occurrence: 5
  fisher_significance: 0.05
  chi_square_threshold: 3.84
```

The system caches corpus analysis results with a 1-hour invalidation
window to avoid re-analyzing on every query.

### LLM Framework Detection
**Location**: `src/rag_modules/filtering/llm_framework_detector.py`

After retrieval, an LLM-based filter analyzes each retrieved chunk to
determine its regulatory framework context (US STAT, IFRS, US GAAP,
Solvency II). This uses GPT-4o-mini with JSON structured output to
detect primary framework, confidence, technical concepts, and regulatory
context for each document.

Retrieved results are then scored for framework alignment with the
query's detected jurisdiction. Documents from mismatched frameworks
are deprioritized rather than removed, preserving recall while
improving precision.

### Reflective Retrieval
**Location**: `src/rag_modules/retrieval/reflective_retriever.py`

When semantic alignment validation fails (i.e., the retrieved documents
don't sufficiently answer the query), the reflective retriever triggers
an iterative improvement loop:

1. **Evaluate**: LLM assesses retrieval quality, producing a quality
   score, list of information gaps, and suggested alternative queries
2. **Re-retrieve**: Executes alternative queries from step 1
3. **Merge**: Combines new results with original results
4. **Re-evaluate**: Checks if quality improved

This loop runs up to a configurable number of rounds. If quality
doesn't improve, the original results are used (graceful degradation).

### Advanced Retrieval Strategies
**Location**: `src/rag_modules/retrieval/advanced_strategies.py`

**Query decomposition**: Complex queries (detected via LLM analysis)
are broken into independent sub-queries with priority ordering and
dependency tracking. Each sub-query is retrieved independently, results
are deduplicated and merged.

**Parent-child chunk expansion**: Retrieved chunks can be expanded
with surrounding context from the same document. If chunk #5 matches,
chunks #3-7 are retrieved and merged to provide fuller context. This
addresses the common RAG problem of relevant information being split
across chunk boundaries.

**Adaptive strategy selection**: An LLM analyzes each query to select
the optimal retrieval strategy (factual, calculation, comparison,
explanation, complex) based on query characteristics, required precision,
and domain specificity.


## Subsystem 4: Generation Intelligence

**Location**: `src/rag_modules/reasoning/self_correction.py`,
`config/self_correction.yaml`

Generation uses a multi-pass self-correction architecture with
three main components.

### Chain of Thought Generator

Before generating a response, the system selects a reasoning
methodology appropriate to the query type:

| Methodology | Use Case |
|-------------|----------|
| General | Step-by-step logical reasoning |
| Analytical | Systematic analysis with evidence evaluation |
| Procedural | Step-by-step procedure or calculation |
| Comparative | Comparative analysis with pros/cons |
| Regulatory | Compliance-focused reasoning with regulations |
| Quantitative | Mathematical and numerical reasoning |

Selection is LLM-driven (GPT-4o-mini, temperature 0.2). The
selected methodology shapes the reasoning prompt, which produces
a structured reasoning chain: understanding → relevant information →
analysis → reasoning → validation → conclusion.

The reasoning chain is then used to generate the final response,
ensuring the output follows from demonstrated logic rather than
being generated in a single pass.

### Self-Verification Module

After generation, the response is verified against seven criteria:

1. **Factual accuracy** — facts verified against provided context
2. **Completeness** — response fully answers the query
3. **Logical consistency** — no internal contradictions
4. **Hallucination check** — no information fabricated beyond context
5. **Context grounding** — response stays within retrieved documents
6. **Clarity** — well-structured and understandable
7. **Relevance** — directly addresses the query

Verification produces per-criterion scores and an overall assessment
with priority issues, missing information, and improvement suggestions.

### Multi-Pass Correction Loop

```
Pass 1: Generate with Chain-of-Thought reasoning
         → Verify against 7 criteria
         → If confidence ≥ 0.8: return response
         → If issues found: continue to Pass 2

Pass 2: Generate correction using:
         - Original query + context
         - Previous response
         - Identified issues + specific feedback
         → Verify again
         → If confidence ≥ 0.8 or max iterations: return

Pass 3: (max) Final correction attempt
         → Return best result regardless of confidence
```

**Fallback architecture**: If self-correction fails entirely, the
system falls back to direct generation without reasoning. If that
also fails, an error response is returned. At no point does a
failure in the enhanced pipeline prevent a response from being
generated.

**Configuration**: Max 3 iterations, confidence threshold 0.8,
45-second timeout per iteration. All prompts are externalized in
`config/self_correction.yaml`.


## Subsystem 5: Quality Validation

Three validators work together to assess response quality. Their
scores are combined by the unified validator using configurable
weights.

### Content Grounding Validator
**Location**: `src/rag_modules/quality/grounding_validator.py`

Validates that the response is grounded in retrieved documents.
Produces five quality signals:

- **Numerical precision**: Are numbers in the response traceable to
  source documents?
- **Concept alignment**: Do the concepts discussed match what's in
  the retrieved context?
- **Contextual consistency**: Is the response internally consistent
  with the context?
- **Source attribution**: Are claims properly attributable to sources?
- **Factual accuracy**: Are stated facts verifiable against context?

Key feature: **Adaptive thresholds**. The grounding validator doesn't
use static thresholds. It maintains a history of validation results
and adjusts its thresholds over time within configurable bounds
(min 0.25, max 0.8). This means the system becomes calibrated to
the actual quality distribution of its responses.

Also includes a semantic alignment validator (OpenAI-powered) that
checks whether the response intent matches the query intent — catching
cases where the response is well-grounded in documents but doesn't
actually answer what was asked.

### Unified Quality Validator
**Location**: `src/rag_modules/quality/unified_validator.py`

Combines all quality signals into a single composite score:

```yaml
weights:
  semantic_score: 0.3
  grounding_score: 0.4
  hallucination_penalty: 0.2
  openai_score: 0.1
```

The grounding score (0.4 weight) is the primary quality signal,
with semantic alignment (0.3) as the secondary. The hallucination
penalty (0.2) is inverted — high hallucination risk reduces the
composite score. OpenAI alignment (0.1) serves as a lightweight
tertiary check.

### Hallucination Detection Patterns
**Location**: `config/quality_validation.yaml`

Hallucination detection uses semantic patterns rather than keyword
lists:

| Pattern | What It Catches |
|---------|-----------------|
| Confidence overstatement | Absolute certainty + financial advice |
| Over-generalization | Universal quantifiers + domain terms |
| Fabricated specificity | Exact numbers + unverifiable precision |
| Regulatory overconfidence | Regulatory claims + unverified mandates |

These patterns are applied semantically, not as string matching. The
LLM evaluates whether the response exhibits these patterns in the
context of the specific query and retrieved documents.


## Subsystem 6: Response Assembly

### Citation Extraction
**Location**: `src/rag_modules/analysis/citations.py`

Citations are not inserted during generation. Instead, after the
response is generated, an LLM-based analysis identifies which of the
retrieved documents were actually used to produce the response.

The citation analyzer:
1. Prepares summaries of each retrieved document (filename, content
   preview, metadata)
2. Sends the response + document summaries to the LLM
3. The LLM identifies which documents contributed to each part of
   the response
4. Citations are attached with confidence scores and relevance
   assessments

This post-hoc approach is more accurate than inline citation
injection because it analyzes the actual response content rather
than relying on the generation model to self-report its sources.

### Conversation Memory
**Location**: `src/conversation_memory.py`

Level 2 intelligent context management backed by Redis:

- **Entity tracking**: An `ActuarialEntityExtractor` identifies
  domain-specific entities (policy types, percentages, years, ASC
  sections, calculations, regulations, monetary amounts, formulas)
  in each message
- **Importance scoring**: Messages are scored based on content type
  (questions get +0.3, specific numbers get +0.2, technical terms
  get +0.1)
- **Semantic compression**: Older conversation segments are summarized
  into `ConversationSummary` objects containing topics, key entities,
  and important facts — preserving context without unbounded token
  growth

### Structured Response Generator
**Location**: `src/response_generation/structured_generator.py`

The final response is assembled with:
- Answer text (from the generation pipeline)
- Confidence level (high/medium/low/none)
- Source summary (from citation analysis)
- Follow-up questions (contextual, not generic)
- Validation results (grounding + semantic alignment metadata)

The generator enforces formatting rules: no internal document
references ("Document 3 says..."), professional tone, and response
structure appropriate to the query type.


## Document Processing Pipeline

**Location**: `src/document_processor.py`, `src/extraction/`

### Ingestion
Supports PDF, DOCX, PPTX, CSV, and XLSX. The processing chain:

1. **OCR with tiered fallback**: Google Cloud Vision (premium) →
   Tesseract → docTR → EasyOCR. The system probes availability
   at startup and selects the best available processor.

2. **Shape-aware processing**: For PDFs containing charts, diagrams,
   and organizational structures, a spatial extractor preserves
   layout information that standard text extraction destroys.

3. **Document deduplication**: Prevents the same document from being
   ingested multiple times into the vector store.

### Extraction Intelligence
**Location**: `src/extraction/smart_router.py`

The extraction pipeline uses a **tiered processing strategy** with
circuit breaker protection:

```
Cache → Pattern-based → Light LLM → Full LLM → Fallback
```

Each tier has a confidence threshold. If a tier produces results above
its threshold, processing stops. This means simple documents are
processed quickly (cache/pattern), while complex documents get LLM
analysis only when needed.

The **circuit breaker** tracks LLM failures. After 5 consecutive
failures, it opens (blocks LLM calls for 60 seconds), preventing
cascading failures during LLM outages. After the timeout, it
transitions to half-open (allows test requests) before fully closing.

### Metadata Extraction
**Location**: `src/extraction/metadata_builder.py`, `src/extraction/framework_detector.py`

During ingestion, each document chunk is enriched with:
- Framework detection (US GAAP, IFRS, US STAT)
- Document fingerprinting for dedup
- Structural metadata (section, page, chunk index)
- Domain classification


## Performance Architecture

### Tiered Query Processing
**Location**: `src/rag_modules/services/performance_optimizer.py`

Queries are classified into tiers before processing:

- **Simple queries** (pattern-matched: "What is X?", "Define X") →
  Lightweight pipeline, skip advanced retrieval and self-correction
- **Complex queries** (detected by indicators: "compared to",
  "difference between", "which is better") → Full pipeline with
  all intelligence subsystems active

This prevents over-processing simple factual lookups while reserving
the full intelligence pipeline for queries that benefit from it.

### Async Background Learning
The performance optimizer maintains a background learning loop:
- Queue-based (max 100 items, batch size 10)
- 30-second processing interval
- Learns response time patterns and cache hit rates
- Persists learned patterns to disk

### Circuit Breakers
Multiple circuit breakers protect against cascading failures:
- Disambiguation service: 5-failure threshold, 60-second timeout
- LLM extraction calls: Independent circuit breaker
- External API calls: Timeout-based protection

### Caching
- Redis for conversation memory and session state
- In-memory caching for entropy disambiguation (1-hour TTL)
- Learned pattern persistence to disk
- Configurable cache TTL across all subsystems


## Domain Knowledge Service

**Location**: `src/rag_modules/services/domain_knowledge_service.py`

Integrates with external authoritative terminology sources:

| Source | Category | Status |
|--------|----------|--------|
| Society of Actuaries | Actuarial terminology | Configured |
| NAIC Insurance Glossary | Insurance terminology | Configured |
| XBRL Insurance Taxonomy | Financial reporting | Configured |
| ACORD Standards | Insurance data standards | Configured |
| SEC Insurance Industry Guide | Regulatory terminology | Configured |

The service fetches and caches terminology with weekly refresh cycles.
Terms are stored as `DomainTerm` objects with category, definition,
source, confidence, aliases, and related terms.

**Note**: These are external web sources. Runtime availability depends
on endpoint stability. The system operates without them if fetches
fail.


## Configuration Architecture

All intelligence behaviors are externalized into YAML configuration:

| File | Controls |
|------|----------|
| `config/mvp_config.yaml` | Core application settings, LLM models, retrieval params |
| `config/quality_validation.yaml` | All quality thresholds, entropy params, hallucination patterns, performance tuning |
| `config/self_correction.yaml` | Multi-pass settings, verification criteria, reasoning prompts, quality thresholds |
| `config/advanced_retrieval.yaml` | Retrieval strategies, decomposition settings, parent-child expansion |
| `config/response_generation.yaml` | Grounding settings, formatting rules, response structure |
| `config/compliance.py` | Compliance rules, professional judgment triggers |
| `config/data_sources.yaml` | External API configuration |
| `config/extraction_config.yaml` | Document extraction settings |

**Design principle**: No intelligence behavior is hardcoded in Python
source files. Thresholds, prompts, weights, strategies, and patterns
are all in configuration. The Python code implements mechanisms; the
YAML files define policies.


## Graceful Degradation Map

Every enhanced component has a fallback path:

| Component | If It Fails | Fallback |
|-----------|-------------|----------|
| Entropy disambiguation | Retrieval continues without disambiguation | Standard hybrid search |
| Reflective retrieval | Base results returned | First-pass vector + keyword results |
| Self-correction loop | First-pass response returned | Direct generation without verification |
| Framework filter | Unfiltered results used | All retrieved docs pass through |
| Chain-of-thought | Direct generation | Single-pass LLM response |
| Semantic validator | No quality gate | Response passes without alignment check |
| Grounding validator | No hallucination check | Response passes without grounding verification |
| OCR processing | Falls through tiers | Google Vision → Tesseract → docTR → EasyOCR → skip |
| LLM extraction | Circuit breaker opens | Pattern-based extraction → raw text fallback |
| Redis (conversation memory) | In-memory fallback | Session-scoped context only |

This means AAIRE can serve responses even when multiple subsystems
are degraded. The quality decreases proportionally, but the system
never hard-fails due to an intelligence component being unavailable.


## Key Technical Decisions

1. **Corpus-driven disambiguation over ontology mapping**: Instead of
   building a taxonomy of "reserve means X in context Y," the system
   learns mutual exclusions from statistical analysis of the document
   corpus. This scales to any domain without manual knowledge engineering.

2. **Post-hoc citation over inline citation**: Citations are extracted
   after generation by analyzing which documents the response actually
   drew from, rather than asking the LLM to cite as it generates. This
   is more accurate and doesn't constrain the generation process.

3. **Adaptive thresholds over static thresholds**: Quality validation
   thresholds adjust based on historical performance within configurable
   bounds. This prevents the system from being overly strict (blocking
   good responses) or overly permissive (passing bad ones) as the
   document corpus and query patterns evolve.

4. **Parallel hybrid search over sequential**: Vector and keyword
   searches run concurrently with full document limits each, then
   merge. This sacrifices some memory for significantly better latency
   and ensures neither search method is artificially constrained.

5. **Multi-tiered processing over uniform processing**: Simple queries
   skip expensive intelligence subsystems. Complex queries get the
   full pipeline. This keeps average response time low while preserving
   quality for hard queries.

6. **LLM-based framework detection over pattern matching**: For
   determining the regulatory framework of a document chunk, an LLM
   call (GPT-4o-mini) provides more accurate detection than regex
   patterns, especially for documents that discuss multiple frameworks
   or use non-standard terminology.
