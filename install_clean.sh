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

# Install all requirements
echo "📦 Installing all requirements..."
pip3 install -r requirements.txt

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