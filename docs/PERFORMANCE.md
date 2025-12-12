# Performance Optimization Guide

## Query Latency Target: < 12 seconds

### Current Optimizations

#### 1. Vector Search Optimization
- **FAISS Index**: Using FAISS for fast similarity search
- **Index Type**: Using flat index for exact search (can upgrade to IVF for larger datasets)
- **Batch Processing**: Queries are processed in batches where possible

#### 2. LLM Optimization
- **Model**: CodeLlama-7B Q4_K_M quantized model (reduced memory footprint)
- **Context Window**: Limited to 4000 tokens to reduce processing time
- **Top-K Results**: Retrieve only top 5 most relevant code snippets

#### 3. Code Chunking Strategy
- **Chunk Size**: 1000 characters per chunk (configurable via `CHUNK_SIZE` env var)
- **Overlap**: 200 characters overlap between chunks for context continuity
- **Smart Splitting**: Custom code-aware chunking that preserves functions and classes

#### 4. Caching Strategy
- **Conversation Cache**: Previous messages stored in memory to avoid re-processing
- **Index Cache**: Vectorstore loaded once and reused across queries
- **GitHub API Cache**: Repository metadata cached to reduce API calls

### Performance Monitoring

#### Key Metrics to Track
1. **Query Response Time**: Time from request to response
2. **Vector Search Time**: Time to find relevant code snippets
3. **LLM Generation Time**: Time for model to generate response
4. **Index Size**: Number of vectors and memory usage

#### Recommended Tools
- **Backend**: structlog for structured logging with timing info
- **Frontend**: Browser DevTools Network tab for API latency
- **System**: Monitor CPU/GPU usage during query processing

### Optimization Recommendations

#### If Query Latency > 12s:

1. **Reduce Context Size**
   ```python
   # In .env
   MAX_CONTEXT_LENGTH=2000  # Reduce from 4000
   TOP_K_RESULTS=3  # Reduce from 5
   ```

2. **Upgrade FAISS Index**
   ```python
   # Switch to IVF index for faster search on large datasets
   import faiss
   index = faiss.IndexIVFFlat(quantizer, d, nlist)
   ```

3. **Use Smaller Model**
   - Switch to CodeLlama-7B-Q3 or Q2 quantization
   - Or use TinyLlama for faster inference

4. **Implement Query Caching**
   - Cache common queries and responses
   - Use Redis for distributed caching

5. **Parallel Processing**
   - Process vector search and LLM generation in parallel where possible
   - Use async/await for I/O operations

### Benchmarking

Run performance tests:
```bash
# Backend
cd backend
pytest tests/performance/ -v

# Measure query latency
time curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the main function do?"}'
```

### Expected Performance

- **Vector Search**: < 1s for repositories up to 10,000 files
- **LLM Generation**: 5-10s depending on response length
- **Total Query Time**: 6-11s for typical queries
- **Indexing Time**: 1-2 minutes per 1000 files

### Scaling Considerations

For large repositories (>10,000 files):
1. Use GPU acceleration for FAISS (faiss-gpu)
2. Implement distributed indexing
3. Use streaming responses for better UX
4. Consider model quantization (Q4 → Q3 → Q2)
