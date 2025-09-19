# RAG Pipeline Refactoring Plan

## Problem
The `rag_pipeline.py` file has grown to 4,577 lines with 86 methods in a single class, making it difficult to maintain, test, and extend.

## Solution Strategy
Break down the monolithic RAGPipeline class into modular components following the Single Responsibility Principle.

## Phase 1: Structure Setup ✅
- Created modular directory structure
- Established core response module

## Phase 2: Extract Independent Modules (Current)
### 2.1 Citation Module (~250 lines)
- `_extract_citations()`
- `_analyze_document_usage_in_response()`
- `_remove_citations_from_response()`
- `_infer_document_domain()`
- `_check_domain_compatibility()`

### 2.2 Cache Module (~150 lines)
- `_init_cache()`
- `_get_cache_key()`
- `_serialize_response()`
- `_deserialize_response()`
- `clear_cache()`
- `_clear_cache_pattern()`

### 2.3 Formatting Module (~600 lines)
- `_format_response()`
- `_apply_professional_formatting()`
- `_apply_simple_professional_formatting()`
- `_basic_professional_format()`
- `_final_formatting_cleanup()`
- `_normalize_spacing()`
- `_validate_formula_formatting()`
- `_structured_to_markdown()`
- `_generate_structured_response()`
- `_apply_llm_formatting_fix()`
- `_fix_reserve_terminology()`
- `_preserve_formulas()`

### 2.4 Query Analysis Module (~400 lines)
- `_classify_query_topic()`
- `_expand_query()`
- `_is_general_knowledge_query()`
- `_is_organizational_query()`
- `_determine_question_categories()`
- `_is_contextual_question()`

### 2.5 Quality Metrics Module (~200 lines)
- `_calculate_quality_metrics()`
- `_calculate_confidence()`
- `_get_similarity_threshold()`
- `_get_document_limit()`

## Phase 3: Extract Core Services
### 3.1 Retrieval Module (~800 lines)
- `_retrieve_documents()`
- `_vector_search()`
- `_keyword_search()`
- `_combine_search_results()`
- `_get_diverse_context_documents()`
- `_convert_filters_for_whoosh()`

### 3.2 Generation Module (~1000 lines)
- `_generate_response()`
- `_generate_single_pass_response()`
- `_generate_chunked_response()`
- `_generate_enhanced_single_pass()`
- `_generate_organizational_response()`
- `_process_with_chunked_enhancement()`
- `_merge_response_parts()`
- `_completeness_check()`

### 3.3 Follow-up Questions Module (~400 lines)
- `_generate_follow_up_questions()`
- `_generate_extraction_follow_ups()`
- `_generate_response_based_fallback()`
- `_analyze_document_content_for_followups()`
- `_get_category_examples()`

## Phase 4: Extract Storage Management
### 4.1 Document Management (~400 lines)
- `add_documents()`
- `delete_document()`
- `clear_all_documents()`
- `cleanup_orphaned_chunks()`
- `get_all_documents()`

### 4.2 Index Management (~300 lines)
- `_init_qdrant_indexes()`
- `_init_local_index()`
- `_init_hybrid_search()`
- `_update_whoosh_index()`
- `_clear_whoosh_index()`
- `_populate_whoosh_from_existing_documents()`

## Phase 5: Integration
### 5.1 Update Main Pipeline Class
- Keep RAGPipeline as the orchestrator
- Delegate to specialized modules
- Maintain backward compatibility

### 5.2 Update Imports
- Update all files importing from rag_pipeline
- Ensure no breaking changes

## Benefits
1. **Maintainability**: Each module ~200-400 lines
2. **Testability**: Independent unit testing per module
3. **Extensibility**: Easy to add new features
4. **Team Collaboration**: Multiple developers can work on different modules
5. **Performance**: Potential for lazy loading and optimization

## Implementation Order
1. ✅ Create directory structure
2. ✅ Extract Response class
3. ⏳ Extract Citation module
4. Extract Cache module
5. Extract Formatting module
6. Extract Query Analysis module
7. Create backward-compatible wrapper
8. Test thoroughly
9. Continue with remaining modules