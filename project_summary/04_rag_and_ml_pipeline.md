# RAG and ML Pipeline

The heart of the application lies within the `rag_service.py`, which is responsible for the Retrieval-Augmented Generation pipeline.

## 1. Hybrid Search Retrieval
When a query is received, the service attempts to find the most relevant code chunks using a hybrid approach:

- **Semantic Search (FAISS):** The user's query is embedded using `sentence-transformers` (`all-MiniLM-L6-v2`). The `FAISS` library performs a nearest-neighbor search (`IndexFlatIP` - Inner Product) to find semantically similar code chunks, regardless of exact keyword matches.
- **Lexical Search (BM25):** The query is tokenized, and the `rank-bm25` library calculates term frequency/inverse document frequency (TF-IDF) scores. This guarantees that exact variable names or method signatures are not missed.

### Reciprocal Rank Fusion (RRF)
To combine the results from FAISS and BM25, the system uses the RRF algorithm. 
- A constant `k` (standard is 60) is used.
- The score for each document is calculated as: `1 / (k + rank)`.
- The scores from the semantic and lexical lists are weighted (e.g., 50/50 split) and summed. 
- The chunks are then re-sorted by their final RRF score to produce the highest quality context.

## 2. Conversation Context Management
**Nuance:** The backend manages conversation contexts purely in-memory.
```python
self.conversations: dict[str, list[ChatMessage]] = {} 
```
When `app.js` sends a `conversation_id`, `rag_service.py` looks it up in `self.conversations`. It pulls the last 5 messages to assemble a conversational history. This history allows the LLM to understand context (e.g., answering "What does *it* do?").

## 3. LLM Inference
The system uses `llama-cpp-python` for local LLM inference, removing the need for external APIs like OpenAI. 
- **Prompt Assembly:** The service injects the user's query, the conversation context, and the top-ranked code chunks (formatted with file paths and line numbers) into a template.
- **Template Formats:** The code dynamically detects or configures prompt formatting (e.g., Llama 2/3, ChatML, Phi-3) to ensure the local model correctly identifies system instructions, user input, and context.
- **Generation:** The model streams/generates the response which is then cleaned of specific prompt artifacts (like `<|endoftext|>`) before being returned to the frontend.
