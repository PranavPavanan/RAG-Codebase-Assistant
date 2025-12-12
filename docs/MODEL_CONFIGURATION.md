# Model Configuration Guide

## 🎯 **Easy Model Switching**

To change the model, simply update the `MODEL_PATH` in your `.env` file or environment variables.

## 📁 **Current Model Setup**

- **Active Model**: `codellama-7b-merged-Q4_K_M.gguf` (4-bit, 3.8GB)
- **Location**: `backend/models/`
- **Type**: Complete GGUF model (ready to use)

## 🔧 **How to Change Models**

### Option 1: Environment Variable
```bash
# Set environment variable
export MODEL_PATH=models/your-new-model.gguf

# Or in PowerShell
$env:MODEL_PATH="models/your-new-model.gguf"
```

### Option 2: Create .env file
Create `backend/.env` with:
```env
MODEL_PATH=models/your-new-model.gguf
```

### Option 3: Update config.py
Change the default in `backend/src/config.py`:
```python
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/your-new-model.gguf")
```

## 📋 **Available Models**

| Model File | Size | Type | Status |
|------------|------|------|--------|
| `codellama-7b-merged-Q4_K_M.gguf` | 3.8GB | Complete GGUF | ✅ **Active** |
| `codellama-7b-python-ai-assistant-F16.gguf` | 13.5GB | LoRA Adapter | ❌ Needs base model |

## 🚀 **Testing Your Model**

```bash
cd backend
python -c "import asyncio; from src.services.rag_service import RAGService; asyncio.run(RAGService().initialize())"
```

## ✅ **Success Indicators**
- "SUCCESS: CodeLlama model loaded successfully!"
- No tokenizer errors
- Model loads without AssertionError

## ❌ **Common Issues**
- **LoRA Adapter Error**: Use complete GGUF models, not adapters
- **File Not Found**: Check model path is correct
- **Memory Error**: Use smaller quantized models (Q4_K_M, Q5_K_M)

