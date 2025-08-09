# SAFE CLEANUP ANALYSIS - Do NOT Break Production

## ⚠️ CRITICAL FILES - DO NOT DELETE:

### Entry Points (Production Dependencies):
- ✅ **start.py** - Used by Dockerfile and deployment scripts
- ✅ **main.py** - Primary application (imported by start.py)
- ❌ **simple_main.py** - Testing fallback only, safe to archive

### OCR Processors (Fallback Chain):
- ✅ **ocr_processor_google.py** - Premium option (keep)
- ✅ **ocr_processor_tesseract.py** - Currently active (keep) 
- ✅ **ocr_processor_doctr.py** - Fallback option (keep)
- ✅ **ocr_processor.py** - Final fallback (keep)
- **REASON**: Used in cascading fallback system for reliability

## 📁 SAFE TO CLEAN UP (Phase 1):

### Debug Files (Archive to debug/ folder):
- debug_asc_830.py
- debug_asc_search.py  
- debug_citation_extraction.py
- debug_citation_issue.py
- debug_full_pipeline.py
- debug_missing_citations.py
- debug_openai.py
- debug_search.py

### Test Files (Archive to tests/ folder):
- test_asc_query.py
- test_citation_api.py
- test_citation_filter.py
- test_ocr_issue.py
- test_upload.csv
- test_upload.txt
- test_send_button.html

### Backup Files (Safe to delete):
- app.js.bak

## 🔄 SAFE CLEANUP PROCESS:

### Phase 1: Archive (Not Delete)
1. Create `archive/debug/` folder
2. Create `archive/tests/` folder  
3. Move files (don't delete)
4. Test system still works

### Phase 2: Consolidate Check Scripts
1. Create `scripts/maintenance/` folder
2. Move check_*.py files
3. Update any cron jobs/scripts that reference them

### Phase 3: Final Cleanup
1. Only after confirming system works
2. Delete archived files that are truly unused

## 🎯 ESTIMATED IMPACT:
- Remove ~15 debug files from root (SAFE)
- Archive ~8 test files (SAFE)  
- Keep all production dependencies (NO RISK)
- Improve organization without breaking anything