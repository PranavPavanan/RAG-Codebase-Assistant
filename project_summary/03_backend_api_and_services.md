# Backend API and Services

The backend is powered by **FastAPI** (`backend/src/main.py`), utilizing Uvicorn as its ASGI server. It acts as the orchestrator for repository cloning, ML model loading, and serving API requests.

## Application Lifespan
In FastAPI, the `lifespan` context manager handles startup and teardown events. On server boot, the application eagerly initializes the `RAGService`. It loads the LLM model weights from disk (`backend/models/`) into memory and attempts to map existing FAISS and BM25 indices to avoid re-indexing previously loaded repositories.

## API Layer
The API routing is split into logical modules under `backend/src/api/`:
- `search.py`: Proxies queries to the `GitHubService`.
- `indexing.py`: Triggers background indexing tasks and provides REST/WebSocket progress tracking.
- `chat.py`: Handles RAG query requests via `rag_service.py`.
- `health.py`: Basic health-checks.

## Core Services

### `github_service.py`
This service utilizes `PyGithub` to interface with the GitHub REST API. 
- **Repository Search**: Uses advanced query qualifiers (e.g., `in:name`, `user:`) to search for repositories. It handles shorthand URLs and explicitly prioritizes exact-name matches.
- **Trending Repositories**: Filters by `stars:>=1000`, languages, and topics to fetch trending repositories.
- **Cloning**: Employs `GitPython` (`git.Repo.clone_from`) to physically download repositories to the `storage/repositories` directory.

### `indexer_service.py`
The `IndexerService` is responsible for processing cloned repositories to make them searchable.
- **Asynchronous Task Management**: Uses `asyncio.create_task` to run indexing in the background so the HTTP request can return immediately with a `task_id`.
- **In-Memory Tracking**: Status, progress, and errors are tracked in the `self.active_tasks` dictionary.
- **Pipeline**:
  1. Clones the repository.
  2. Discovers code files (filtering out binaries, images, and heavy dependencies like `node_modules`).
  3. Chunks the source code into manageable text blocks.
  4. Generates embeddings using `sentence-transformers` and builds the FAISS index.
  5. Tokenizes text for BM25 and serializes both indices.
  6. Writes completion metadata to a JSON file in `storage/metadata/`.
