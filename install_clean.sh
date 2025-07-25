#!/bin/bash
echo "🧹 AAIRE Clean Installation Script"
echo "=================================="

# Uninstall all llama-index related packages
echo "🗑️ Removing existing llama-index packages..."
pip3 uninstall -y llama-index llama-index-core llama-index-legacy \
    llama-index-llms-openai llama-index-embeddings-openai \
    llama-index-vector-stores-qdrant llama-index-readers-file \
    llama-index-indices-managed-llama-cloud \
    llama-index-multi-modal-llms-openai llama-index-program-openai \
    llama-index-question-gen-openai llama-index-agent-openai \
    llama-index-cli 2>/dev/null || true

echo "✅ Cleanup complete"

# Install core requirements first
echo "📦 Installing core requirements..."
pip3 install fastapi==0.104.1 uvicorn[standard]==0.24.0 websockets==12.0 pydantic==2.5.0

# Install llama-index step by step
echo "🦙 Installing llama-index core..."
pip3 install llama-index-core==0.12.7

echo "🦙 Installing llama-index main package..."
pip3 install llama-index==0.12.7

echo "🦙 Installing OpenAI integrations..."
pip3 install llama-index-llms-openai==0.3.0
pip3 install llama-index-embeddings-openai==0.2.5

echo "🦙 Installing file readers..."
pip3 install llama-index-readers-file==0.1.15

echo "🦙 Installing Qdrant vector store..."
pip3 install qdrant-client==1.7.3
pip3 install llama-index-vector-stores-qdrant==0.6.1

echo "🔧 Installing utilities..."
pip3 install openai==1.12.0 PyPDF2==3.0.1 python-docx==1.1.0
pip3 install redis==5.0.1 python-dotenv==1.0.0 pyyaml==6.0.1
pip3 install structlog==23.2.0 PyJWT==2.8.0 httpx==0.25.2

echo "✅ Installation complete!"
echo "🧪 Testing imports..."
python3 -c "
try:
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    print('✅ Core imports successful')
except Exception as e:
    print(f'❌ Import test failed: {e}')
"

echo "🏁 Setup complete!"