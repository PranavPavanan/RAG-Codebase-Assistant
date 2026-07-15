# Known Limitations and Historical Context

When inheriting or working on this codebase, engineers should be aware of several architectural limitations and historical design decisions that impact its functionality.

## The Removal of Session Management
Originally, the system was designed with a complex hierarchical "Session & Conversation Management" feature. 
- A `Session` contained multiple `Conversations`.
- The backend possessed distinct endpoints like `GET /chat/history`, `GET /chat/sessions`, and `DELETE /chat/session/clear`.
- This feature was subsequently stripped out of the codebase to simplify the architecture.

### Implications and In-Memory Constraints
Because the robust session management system was removed without implementing persistent database storage (like PostgreSQL or SQLite):
1. **Frontend Volatility:** The `app.js` stores its multi-threaded conversation tabs locally in an array. Refreshing the browser instantly wipes all chat tabs and history from the UI.
2. **Backend Volatility:** The `RAGService` stores the backend context inside an in-memory python dictionary (`self.conversations`). If the FastAPI server restarts, all contexts are permanently destroyed.
3. **Ghost References (The "session_id" bug):** During the deletion of the session feature, a stray line of code (`self.sessions[session_id].total_messages += 1`) was left behind in `rag_service.py`. Because `self.sessions` no longer existed, this caused runtime crashes whenever a query was processed. This specific bug was identified and removed on December 5, 2025, but engineers should remain vigilant for other orphaned code paths referencing `session_id`.

## Single-Model Loading
Currently, the LLM model is loaded eagerly during the FastAPI `lifespan` event. The system holds the model weights in RAM/VRAM permanently while the server is running. 
- **Constraint:** Switching models requires changing the `.env` configuration and restarting the entire server. Hot-swapping models via the API is not natively supported without causing memory leaks or blocking the event loop.

## Scalability
Since `rag_service.py` and `indexer_service.py` rely heavily on in-memory dictionaries (`self.active_tasks`, `self.conversations`), the application cannot be scaled horizontally across multiple workers or pods natively. Moving to a distributed task queue (like Celery/Redis) and a persistent database would be required for a production-grade multi-tenant deployment.
