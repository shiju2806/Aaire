#!/bin/bash
# Script to fix llama-index dependencies on EC2

echo "🔧 Fixing AAIRE dependencies..."

# 1. Uninstall problematic versions
echo "📦 Cleaning old installations..."
pip3 uninstall -y llama-index llama-index-core

# 2. Install specific compatible versions
echo "📦 Installing compatible versions..."
pip3 install llama-index==0.9.48 openai==1.12.0

# 3. Install other required dependencies
echo "📦 Installing additional dependencies..."
pip3 install qdrant-client redis pandas PyPDF2 python-docx

# 4. Verify imports
echo "🧪 Testing imports..."
python3 -c "
try:
    from src.rag_pipeline import RAGPipeline
    print('✅ RAG Pipeline imports successfully!')
except Exception as e:
    print(f'❌ Import error: {e}')
"

echo "✅ Dependency fix complete!"