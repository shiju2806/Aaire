# Fix .env File Format

The issue is in your .env file format. The OpenAI API key line is malformed:

**Current (BROKEN):**
```
OPENAI_API_KEY=sk-proj-hK34lcK_jVShhI16z# Vector Database Configuration (choose one)
```

**Should be (FIXED):**
```
OPENAI_API_KEY=sk-proj-hK34lcK_jVShhI16z[COMPLETE_API_KEY_HERE]

# Vector Database Configuration (choose one)
```

## Steps to Fix:

1. **Complete the API key** - The key appears truncated
2. **Move comment to new line** - Comments can't be on same line as values
3. **Ensure no spaces** around the `=` sign
4. **Restart the application** after fixing

## Expected .env format:
```bash
# AAIRE MVP Environment Configuration

# OpenAI Configuration  
OPENAI_API_KEY=sk-proj-COMPLETE_API_KEY_HERE

# Vector Database Configuration (choose one)
QDRANT_URL=https://ce8b5f05-c0a2-47b1-a761-c2f9e6f73817.europe-west3-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=eyJhbGciOiJIUzI1COMPLETE_API_KEY_HERE
```

Once fixed, the document processing should work and ASC queries will find your uploaded content.