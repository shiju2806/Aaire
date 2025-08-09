# Codebase Cleanup Plan

## Files to DELETE (Deprecated/Duplicate):

### Debug Files (Move to archive or delete):
- debug_asc_830.py
- debug_asc_search.py  
- debug_citation_extraction.py
- debug_citation_issue.py
- debug_full_pipeline.py
- debug_imports.py
- debug_llama_index.py
- debug_missing_citations.py
- debug_openai.py
- debug_search.py
- debug_upload.py

### Test Files (Consolidate or delete):
- test_asc_query.py
- test_citation_api.py
- test_citation_filter.py
- test_ocr_issue.py
- test_websocket.py
- test_send_button.html
- test_upload.csv
- test_upload.txt

### Backup/Duplicate Files:
- app.js.bak
- simple_main.py (if main.py is primary)
- start.py (if main.py is primary)

### Multiple OCR Files (Keep only active one):
- ocr_processor_doctr.py (DELETE if not used)
- ocr_processor_google.py (DELETE if not used)  
- ocr_processor_tesseract.py (DELETE if not used)

### Check Scripts (Consolidate):
- check_compatibility.py
- check_documents.py
- check_job_status.py
- check_new_upload.py
- check_pdfs.py
- check_qdrant.py
- simple_check.py

## Files to KEEP:

### Core Application:
- main.py (primary entry point)
- src/ (all core modules)
- templates/
- static/

### Configuration:
- config/
- requirements.txt (primary)
- docker-compose.yml (production)

### Deployment:
- Dockerfile (primary)
- deploy/

## Consolidation Actions:

1. **Create tests/ directory structure**
2. **Create debug/ directory for debugging scripts**  
3. **Create scripts/ directory for utility scripts**
4. **Remove unused OCR processors**
5. **Choose primary entry point (main.py)**

## Estimated Impact:
- Remove ~25 duplicate/deprecated files
- Clean up root directory significantly
- Improve maintainability
- Reduce confusion for developers