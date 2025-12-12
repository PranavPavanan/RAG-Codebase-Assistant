# RAG-Powered GitHub Assistant - Technical Documentation

> **Version:** 1.0.0  
> **Last Updated:** December 5, 2025  
> **Author:** PranavPavanan

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Architecture Summary](#3-architecture-summary)
4. [Feature Documentation](#4-feature-documentation)
5. [API / Backend Logic](#5-api--backend-logic)
6. [Frontend Logic](#6-frontend-logic)
7. [Installation & Setup](#7-installation--setup)
8. [Important Algorithms](#8-important-algorithms)
9. [Suggestions for Improvements](#9-suggestions-for-improvements)

---

## 1. Project Overview

### What the Project Does

The **RAG-Powered GitHub Assistant** is an AI-powered tool that enables developers to:
- **Search** GitHub repositories by natural language queries
- **Index** repository code for semantic understanding
- **Query** codebases using natural language with AI-generated answers
- **Reference** exact source code locations with line numbers

### Problem it Solves

1. **Code Understanding**: Developers often struggle to understand large, unfamiliar codebases
2. **Documentation Gaps**: Many repositories lack comprehensive documentation
3. **Time-Consuming Search**: Traditional grep/search is limited to exact keyword matches
4. **Context Loss**: Chat interfaces lose repository context between sessions

### Key Use Cases

| Use Case | Description |
|----------|-------------|
| **Onboarding** | New developers quickly understand codebase architecture |
| **Code Review** | Ask questions about implementation patterns |
| **Documentation** | Generate explanations for undocumented code |
| **Debugging** | Find related code across files for bug investigation |
| **API Discovery** | Find endpoints, functions, and their implementations |

---

## 2. Tech Stack

### Programming Languages

| Language | Usage | Version |
|----------|-------|---------|
| **Python** | Backend API, RAG pipeline, ML models | 3.9+ (tested with 3.13) |
| **JavaScript** | Frontend application (vanilla) | ES6+ |
| **HTML/CSS** | Frontend UI structure and styling | HTML5, CSS3 |

### Frameworks & Libraries

#### Backend (Python)

| Library | Version | Purpose |
|---------|---------|---------|
| **FastAPI** | 0.104.1 | High-performance REST API framework |
| **Uvicorn** | 0.24.0 | ASGI server |
| **Pydantic** | 2.9.0 | Data validation and serialization |
| **FAISS-CPU** | 1.11.0 | Vector similarity search (semantic) |
| **rank-bm25** | 0.2.2 | BM25 keyword search |
| **sentence-transformers** | 2.2.2 | Text embeddings (all-MiniLM-L6-v2) |
| **llama-cpp-python** | 0.3.12 | Local LLM inference |
| **PyGithub** | 2.1.1 | GitHub API integration |
| **GitPython** | 3.1.40 | Git repository operations |
| **httpx** | 0.25.2 | Async HTTP client |
| **python-dotenv** | 1.0.0 | Environment variable management |
| **websockets** | 12.0 | Real-time WebSocket communication |

#### Frontend

| Library | Source | Purpose |
|---------|--------|---------|
| **Lucide Icons** | CDN | SVG icon library |
| **Inter Font** | Google Fonts | Typography |

### Build & Development Tools

| Tool | Purpose |
|------|---------|
| **pytest** | Unit and integration testing |
| **pytest-asyncio** | Async test support |
| **Ruff** | Python linting |
| **Makefile** | Build automation |

### External Integrations

| Service | Purpose |
|---------|---------|
| **GitHub API** | Repository search, metadata, cloning |
| **Hugging Face** | Model downloads (CodeLlama, Phi-3) |

### Database / Storage

| Storage Type | Technology | Location |
|--------------|------------|----------|
| **Vector Store** | FAISS IndexFlatIP | `storage/indices/faiss_index.bin` |
| **BM25 Index** | Pickle serialization | `storage/indices/bm25_index.pkl` |
| **Document Chunks** | Pickle | `storage/indices/document_chunks.pkl` |
| **Metadata** | JSON files | `storage/metadata/*.json` |
| **Repositories** | Git clones | `storage/repositories/` |
| **Sessions** | In-memory (Python dicts) | RAM |

---

## 3. Architecture Summary

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Browser)                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  index.html + styles.css + app.js (Vanilla JavaScript SPA)          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐│   │
│  │  │  Search  │  │  Index   │  │   Chat   │  │   API Client Class   ││   │
│  │  │  Panel   │  │  Panel   │  │  Panel   │  │  (HTTP + WebSocket)  ││   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘│   │
│  └───────┼─────────────┼─────────────┼───────────────────┼────────────┘   │
└──────────┼─────────────┼─────────────┼───────────────────┼────────────────┘
           │             │             │                   │
           │ HTTP/REST   │ HTTP/REST   │ HTTP/REST         │ WebSocket
           ▼             ▼             ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BACKEND (FastAPI Server)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        API Layer (src/api/)                          │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌─────────────────┐   │   │
│  │  │  search.py │ │ indexing.py│ │  chat.py   │ │  websocket.py   │   │   │
│  │  │ /search/*  │ │ /index/*   │ │ /chat/*    │ │   /ws/{task}    │   │   │
│  │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └────────┬────────┘   │   │
│  └────────┼──────────────┼──────────────┼─────────────────┼────────────┘   │
│           │              │              │                 │                 │
│  ┌────────┼──────────────┼──────────────┼─────────────────┼────────────┐   │
│  │        ▼              ▼              ▼                 ▼            │   │
│  │              Service Layer (src/services/)                          │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐  │   │
│  │  │ github_service │ │indexer_service │ │     rag_service        │  │   │
│  │  │  - Search      │ │  - Clone repo  │ │  - Hybrid retrieval    │  │   │
│  │  │  - Validate    │ │  - Discover    │ │  - FAISS + BM25        │  │   │
│  │  │  - Fetch repo  │ │  - Index files │ │  - LLM generation      │  │   │
│  │  └───────┬────────┘ └───────┬────────┘ └───────────┬────────────┘  │   │
│  └──────────┼──────────────────┼──────────────────────┼───────────────┘   │
│             │                  │                      │                    │
│  ┌──────────┼──────────────────┼──────────────────────┼───────────────┐   │
│  │          ▼                  ▼                      ▼               │   │
│  │                    Data Models (src/models/)                       │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────────┐  │   │
│  │  │repository.py│ │  index.py  │  │  query.py  │  │ response.py │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └─────────────┘  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                    │                                       │
└────────────────────────────────────┼───────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Storage Layer (./storage/)                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐    │
│  │  repositories/ │  │    indices/    │  │        metadata/           │    │
│  │  (Git clones)  │  │  - faiss_index │  │    (JSON task files)       │    │
│  │                │  │  - bm25_index  │  │                            │    │
│  │                │  │  - doc_chunks  │  │                            │    │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ML Models (./backend/models/)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LLM: Phi-3 Mini / CodeLlama-7B / Llama-3.1-8B (GGUF format)        │   │
│  │  Embeddings: all-MiniLM-L6-v2 (sentence-transformers)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Folder Structure Explanation

```
coding-assistant/
├── backend/                    # Python FastAPI backend
│   ├── src/                    # Source code
│   │   ├── api/                # REST API endpoints
│   │   │   ├── chat.py         # Chat/query endpoints
│   │   │   ├── health.py       # Health check
│   │   │   ├── indexing.py     # Indexing endpoints
│   │   │   ├── repositories.py # Repository CRUD
│   │   │   ├── search.py       # GitHub search
│   │   │   └── websocket.py    # Real-time updates
│   │   ├── models/             # Pydantic data models
│   │   │   ├── index.py        # Indexing models
│   │   │   ├── query.py        # Chat/query models
│   │   │   ├── repository.py   # Repository models
│   │   │   └── response.py     # Generic responses
│   │   ├── services/           # Business logic
│   │   │   ├── github_service.py    # GitHub API wrapper
│   │   │   ├── indexer_service.py   # Repository indexing
│   │   │   └── rag_service.py       # RAG pipeline + hybrid retrieval
│   │   ├── config.py           # Configuration management
│   │   └── main.py             # FastAPI app entry point
│   ├── models/                 # LLM model files (.gguf)
│   ├── storage/                # Persistent storage
│   │   ├── indices/            # FAISS + BM25 indices
│   │   ├── metadata/           # JSON task metadata
│   │   └── repositories/       # Cloned Git repos
│   ├── tests/                  # Test suite
│   │   ├── unit/               # Unit tests
│   │   ├── integration/        # Integration tests
│   │   └── e2e/                # End-to-end tests
│   ├── requirements.txt        # Python dependencies
│   └── pyproject.toml          # Project config
├── frontend/                   # Vanilla JS frontend
│   ├── index.html              # Main HTML
│   ├── styles.css              # CSS styling
│   ├── app.js                  # Application logic
│   └── package.json            # Metadata
├── docs/                       # Documentation
├── specsnew/                   # OpenAPI spec
│   └── openapi.yaml            # API specification
└── README.md                   # Project readme
```

### Component Interaction Flow

```
User Query Flow:
================
1. User types query in Chat Panel (frontend)
2. app.js sends POST /api/chat/query
3. chat.py receives request, calls rag_service.query()
4. rag_service performs:
   a. _retrieve_relevant_content() → hybrid_search()
   b. semantic_search() → FAISS vector similarity
   c. keyword_search() → BM25 ranking
   d. Reciprocal Rank Fusion combines results
   e. generate_response() → LLM generates answer
5. Response returned with sources and line numbers
6. Frontend displays answer with collapsible source references

Indexing Flow:
==============
1. User selects repository and clicks "Start Indexing"
2. POST /api/index/start triggers indexer_service
3. Background task:
   a. Clones repository (git clone --depth 1)
   b. Discovers files matching patterns
   c. Indexes each file (hash, language detection)
   d. Saves metadata JSON
4. WebSocket sends progress updates to frontend
5. RAG service builds FAISS + BM25 indices
6. Chat panel enabled
```

### Design Patterns Used

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Singleton** | `get_rag_service()`, `get_indexer_service()` | Single instance of services |
| **Repository** | `storage/` directories | Data persistence abstraction |
| **Strategy** | `semantic_search()`, `keyword_search()` | Interchangeable retrieval methods |
| **Factory** | `MODEL_CONFIGS` in config.py | Model configuration factory |
| **Observer** | WebSocket `ConnectionManager` | Real-time event broadcasting |
| **Facade** | `ApiClient` class in app.js | Simplified API interface |

---

## 4. Feature Documentation

### 4.1 Repository Search

**What it does:** Search GitHub repositories by natural language query

**Code Location:** 
- Frontend: `frontend/app.js` → `handleSearch()`
- Backend: `backend/src/api/search.py` → `search_repositories()`
- Service: `backend/src/services/github_service.py` → `search_repositories()`

**How it works:**
1. Frontend sends search query to `/api/search/repositories`
2. `GitHubService` uses PyGithub to search GitHub API
3. Results sorted by stars, limited by `limit` parameter
4. Returns repository metadata (name, description, stars, language)

**Key Classes/Functions:**
```python
# backend/src/services/github_service.py
class GitHubService:
    def search_repositories(self, request: RepositorySearchRequest) -> RepositorySearchResponse
    def validate_repository_url(self, request: RepositoryValidationRequest) -> RepositoryValidationResponse
    def get_repository(self, repo_id: str) -> Optional[Repository]
```

---

### 4.2 Repository Indexing

**What it does:** Clone and index repository code for semantic search

**Code Location:**
- Frontend: `frontend/app.js` → `handleStartIndexing()`
- Backend: `backend/src/api/indexing.py`
- Service: `backend/src/services/indexer_service.py`

**How it works:**
1. Clone repository with `git clone --depth 1`
2. Discover files matching include/exclude patterns
3. For each file:
   - Calculate content hash
   - Detect programming language
   - Create `FileIndexEntry`
4. Save metadata JSON to `storage/metadata/`
5. Trigger hybrid index building

**Key Classes/Functions:**
```python
# backend/src/services/indexer_service.py
class IndexerService:
    def start_indexing(self, request: IndexStartRequest) -> IndexStartResponse
    async def _index_repository_async(self, task_id: str, request: IndexStartRequest)
    def _discover_files(self, repo_path: Path, include_patterns, exclude_patterns) -> List[Path]
    def index_file(self, file_path: Path, repo_root: Path) -> Optional[FileIndexEntry]
```

---

### 4.3 Hybrid Retrieval (FAISS + BM25)

**What it does:** Combines semantic vector search with keyword-based BM25 for optimal retrieval

**Code Location:** `backend/src/services/rag_service.py`

**How it works:**
1. **Semantic Search (FAISS):**
   - Generate query embedding with `all-MiniLM-L6-v2`
   - Search `IndexFlatIP` with normalized vectors (cosine similarity)
   - Return top-k similar chunks

2. **Keyword Search (BM25):**
   - Tokenize query (remove stop words)
   - Score all documents with BM25Okapi
   - Return top-k by BM25 score

3. **Reciprocal Rank Fusion (RRF):**
   - Combine rankings from both methods
   - RRF score = Σ(1 / (k + rank)) weighted by method
   - Default: 50% semantic, 50% keyword

**Key Functions:**
```python
# backend/src/services/rag_service.py
def semantic_search(self, query: str, top_k: int = 5) -> list[SourceReference]
def keyword_search(self, query: str, top_k: int = 5) -> list[SourceReference]
def hybrid_search(self, query: str, top_k: int = 10, semantic_weight: float = 0.5) -> list[SourceReference]
async def _initialize_hybrid_retrieval(self)
async def _build_indices(self)
```

---

### 4.4 Chat / Query Interface

**What it does:** Process natural language questions about indexed code

**Code Location:**
- Frontend: `frontend/app.js` → `handleChatSubmit()`
- Backend: `backend/src/api/chat.py` → `chat_query()`
- Service: `backend/src/services/rag_service.py` → `query()`

**How it works:**
1. Receive user query with optional session/conversation IDs
2. Check response cache for identical queries
3. Retrieve relevant content via `hybrid_search()`
4. Build LLM prompt with:
   - System prompt (code expert persona)
   - Retrieved code context
   - Conversation history
5. Generate response with local LLM
6. Validate and fact-check response
7. Cache response and update conversation history

**Key Classes/Functions:**
```python
# backend/src/services/rag_service.py
class RAGService:
    def query(self, request: QueryRequest) -> QueryResponse
    def generate_response(self, query: str, context: list[SourceReference], conversation_context: str = "") -> str
    def _build_conversation_context(self, conversation_id: str) -> str
```

---

### 4.5 Session & Conversation Management

**What it does:** Track user sessions and conversation history

**Code Location:** `backend/src/services/rag_service.py`

**How it works:**
- **Sessions:** Group multiple conversations for a user
- **Conversations:** Individual chat threads with message history
- **Storage:** In-memory dictionaries (MVP design)

**Data Structures:**
```python
self.sessions: dict[str, SessionInfo] = {}  # session_id -> SessionInfo
self.conversations: dict[str, list[ChatMessage]] = {}  # conversation_id -> messages
self.session_conversations: dict[str, set[str]] = {}  # session_id -> conversation_ids
```

---

### 4.6 WebSocket Real-time Updates

**What it does:** Push indexing progress to frontend in real-time

**Code Location:** `backend/src/api/websocket.py`

**How it works:**
1. Client connects to `/ws/{task_id}`
2. `ConnectionManager` tracks active connections by task
3. Indexer calls `send_indexing_update()` during processing
4. Updates broadcast to all clients monitoring that task

**Key Classes:**
```python
# backend/src/api/websocket.py
class ConnectionManager:
    async def connect(self, websocket: WebSocket, task_id: str = "general")
    def disconnect(self, websocket: WebSocket, task_id: str = "general")
    async def send_message(self, message: dict, task_id: str = "general")
```

---

## 5. API / Backend Logic

### API Endpoints Summary

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| `GET` | `/api/health` | Health check | - |
| `POST` | `/api/search/repositories` | Search GitHub repos | `{ query, limit }` |
| `POST` | `/api/validate/url` | Validate repo URL | `{ url }` |
| `GET` | `/api/repositories/{repo_id}` | Get repo details | - |
| `POST` | `/api/index/start` | Start indexing | `{ repository_url, branch }` |
| `GET` | `/api/index/status/{task_id}` | Get indexing status | - |
| `DELETE` | `/api/index/current` | Clear index | - |
| `GET` | `/api/index/stats` | Get index statistics | - |
| `POST` | `/api/chat/query` | Submit chat query | `{ query, conversation_id, session_id }` |
| `GET` | `/api/chat/history` | Get chat history | Query: `conversation_id, limit` |
| `GET` | `/api/chat/context` | Get conversation context | Query: `conversation_id` |
| `DELETE` | `/api/chat/history` | Clear history | Query: `conversation_id` |
| `POST` | `/api/chat/session/clear` | Clear session | `{ session_id, clear_all }` |
| `WS` | `/ws` | General WebSocket | - |
| `WS` | `/ws/{task_id}` | Task-specific WebSocket | - |

### Request/Response Examples

#### Chat Query
```http
POST /api/chat/query
Content-Type: application/json

{
  "query": "How does the chunker work?",
  "conversation_id": "conv-123",
  "session_id": "sess-456",
  "max_sources": 5
}
```

**Response:**
```json
{
  "response": "The chunker in this repository...",
  "sources": [
    {
      "file": "src/chunker.py",
      "content": "def chunk_text(text, size=500)...",
      "score": 0.85,
      "line_start": 15,
      "line_end": 45,
      "type": "code"
    }
  ],
  "conversation_id": "conv-123",
  "session_id": "sess-456",
  "model": "phi3-mini"
}
```

### Middleware

| Middleware | Purpose |
|------------|---------|
| **CORSMiddleware** | Allow cross-origin requests from frontend |

```python
# backend/src/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Data Models / Schemas

#### Core Models (`backend/src/models/`)

| Model | File | Purpose |
|-------|------|---------|
| `QueryRequest` | query.py | Chat query input |
| `QueryResponse` | query.py | Chat response with sources |
| `SourceReference` | query.py | Code reference with line numbers |
| `ChatMessage` | query.py | Individual chat message |
| `SessionInfo` | query.py | Session metadata |
| `ConversationInfo` | query.py | Conversation metadata |
| `IndexStartRequest` | index.py | Start indexing request |
| `IndexStatusResponse` | index.py | Indexing progress |
| `IndexStats` | index.py | Index statistics |
| `Repository` | repository.py | Full repository model |
| `RepositoryBase` | repository.py | Basic repo info |
| `RepositorySearchResponse` | repository.py | Search results |

---

## 6. Frontend Logic

### Component Hierarchy

```
RAGApp (Main Application Class)
├── ApiClient (HTTP/WebSocket Client)
├── Search Panel
│   ├── Search Form
│   ├── Search Results List
│   └── Selected Repository Card
├── Index Panel
│   ├── Repository URL Input
│   ├── Start Indexing Button
│   ├── Progress Bar
│   └── Current Index Stats
├── Chat Panel
│   ├── Messages Container
│   ├── Welcome Message
│   ├── Chat Input Form
│   └── Source References (collapsible)
└── Sidebar
    ├── Navigation Buttons
    ├── Status Chip
    └── Theme Toggle
```

### State Management

The frontend uses a **simple state object** pattern (no external library):

```javascript
// frontend/app.js
this.state = {
  // Repository & index
  selectedRepository: null,
  searchResults: [],
  isSearching: false,
  indexStats: null,
  currentIndexingTask: null,
  isIndexing: false,
  
  // Chat
  messages: [],
  conversationId: null,
  sessionId: null,
  currentConversation: null,
  conversations: [],
  isQuerying: false,
  
  // UI
  route: 'search',  // Current panel
  elapsedMs: 0,
  elapsedTimer: null,
  
  // Theme
  theme: 'system'  // 'light' | 'dark' | 'system'
};
```

### Key Utilities

| Function | Purpose |
|----------|---------|
| `switchPanel(route)` | Navigate between Search/Index/Chat panels |
| `handleSearch()` | Execute GitHub repository search |
| `selectRepository(repo)` | Select repo for indexing |
| `handleStartIndexing()` | Begin repository indexing |
| `startIndexingPolling()` | Poll for indexing progress |
| `handleChatSubmit()` | Send chat query |
| `displayMessages()` | Render chat messages |
| `toggleTheme()` | Toggle light/dark theme |
| `isIndexed()` | Check if repository is indexed |

### API Client Class

```javascript
// frontend/app.js
class ApiClient {
  constructor(baseUrl) { this.baseUrl = baseUrl; }
  
  async req(path, opts = {}) { /* Fetch wrapper with error handling */ }
  
  health() { return this.req('/health'); }
  searchRepositories(query, language, minStars) { /* ... */ }
  startIndexing(repository_url, branch) { /* ... */ }
  indexStatus(taskId) { /* ... */ }
  indexStats() { /* ... */ }
  clearIndex() { /* ... */ }
  chatQuery(query, conversation_id, session_id) { /* ... */ }
}
```

---

## 7. Installation & Setup

### Prerequisites

- **Python 3.9+** (tested with 3.13)
- **Node.js 18+** (optional, for npm scripts)
- **Git** (for repository cloning)
- **8GB+ RAM** (16GB recommended for LLM)

### Environment Variables

#### Backend (`backend/.env`)

```env
# GitHub Personal Access Token (optional, increases API rate limits)
GITHUB_TOKEN=your_github_token_here

# Model Selection (phi3-mini, codellama-7b, llama3.1-8b)
MODEL_NAME=phi3-mini

# Indexing Configuration
MAX_FILE_SIZE=1048576
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG Configuration
TOP_K_RESULTS=5
MAX_CONTEXT_LENGTH=4000

# Server
HOST=0.0.0.0
PORT=8000
```

#### Frontend (`frontend/.env`)

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### How to Run Locally

#### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download model (place in backend/models/)
# Option 1: Phi-3 Mini (recommended)
# Download Q4_K_M-00001-of-00001.gguf from HuggingFace

# Option 2: CodeLlama-7B
# Download codellama-7b-instruct.Q4_K_M.gguf

# Run server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Frontend Setup

```bash
cd frontend

# Serve with Python
python -m http.server 5173

# Or with npm
npm run dev
```

#### 3. Access Application

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs (Swagger UI)

### Build for Production

#### Backend
```bash
# Install production dependencies
pip install -r requirements.txt

# Run with production server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Frontend
```bash
# No build step needed (vanilla HTML/CSS/JS)
# Serve static files with any web server (nginx, Apache, etc.)
```

---

## 8. Important Algorithms

### 8.1 Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

The core retrieval algorithm combines semantic and keyword search using RRF:

```python
# backend/src/services/rag_service.py

def hybrid_search(self, query: str, top_k: int = 10, semantic_weight: float = 0.5) -> list[SourceReference]:
    """
    Combines FAISS (semantic) and BM25 (keyword) search using RRF.
    
    RRF Formula: score(d) = Σ (1 / (k + rank(d))) for each ranking list
    
    Args:
        query: Search query
        top_k: Number of results
        semantic_weight: Weight for semantic search (0-1)
    """
    # Get results from both methods
    semantic_results = self.semantic_search(query, top_k * 2)
    keyword_results = self.keyword_search(query, top_k * 2)
    
    # RRF constant (standard value)
    k_constant = 60
    
    # Build RRF scores
    rrf_scores = {}
    
    # Add semantic scores
    for rank, src in enumerate(semantic_results):
        key = f"{src.file}:{src.line_start}"
        rrf_scores[key] = semantic_weight * (1.0 / (k_constant + rank + 1))
    
    # Add keyword scores
    keyword_weight = 1.0 - semantic_weight
    for rank, src in enumerate(keyword_results):
        key = f"{src.file}:{src.line_start}"
        rrf_scores[key] = rrf_scores.get(key, 0) + keyword_weight * (1.0 / (k_constant + rank + 1))
    
    # Sort by combined score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    
    return [chunk_map[key] for key in sorted_keys]
```

**Why RRF?**
- Robust to score normalization differences between FAISS and BM25
- Documents appearing in both lists get boosted
- Simple yet effective for combining heterogeneous rankings

---

### 8.2 Semantic Search with FAISS

```python
# backend/src/services/rag_service.py

def semantic_search(self, query: str, top_k: int = 5) -> list[SourceReference]:
    """
    Vector similarity search using FAISS IndexFlatIP (Inner Product).
    
    Process:
    1. Encode query with sentence-transformers (all-MiniLM-L6-v2)
    2. Normalize vectors for cosine similarity
    3. Search FAISS index
    4. Return top-k similar chunks
    """
    # Generate query embedding (384-dimensional vector)
    query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search index (Inner Product on normalized vectors = Cosine Similarity)
    distances, indices = self.faiss_index.search(query_embedding, k)
    
    # Build results
    sources = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:  # Valid result
            chunk = self.document_chunks[idx]
            sources.append(SourceReference(
                file=chunk['file_path'],
                content=chunk['content'],
                score=max(0.0, min(1.0, float(dist))),  # Clamp to [0,1]
                line_start=chunk['line_start'],
                line_end=chunk['line_end']
            ))
    
    return sources
```

---

### 8.3 BM25 Keyword Search

```python
# backend/src/services/rag_service.py

def keyword_search(self, query: str, top_k: int = 5) -> list[SourceReference]:
    """
    BM25 (Best Matching 25) keyword-based search.
    
    BM25 Formula:
    score(q, d) = Σ IDF(qi) * (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * |d|/avgdl))
    
    Where:
    - qi = query terms
    - f(qi, d) = term frequency in document
    - |d| = document length
    - avgdl = average document length
    - k1, b = tuning parameters (typically k1=1.5, b=0.75)
    """
    # Tokenize query
    query_tokens = self._tokenize(query)  # Removes stop words
    
    # Get BM25 scores for all documents
    scores = self.bm25_index.get_scores(query_tokens)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # Build results with normalized scores
    max_score = max(scores) if max(scores) > 0 else 1.0
    sources = []
    for idx in top_indices:
        if scores[idx] > 0:
            chunk = self.document_chunks[idx]
            sources.append(SourceReference(
                file=chunk['file_path'],
                content=chunk['content'],
                score=float(scores[idx]) / max_score,
                line_start=chunk['line_start'],
                line_end=chunk['line_end']
            ))
    
    return sources
```

---

### 8.4 Python Code Chunking

```python
# backend/src/services/rag_service.py

def _chunk_python_code(self, content: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Chunk Python code by logical units (classes, functions).
    
    Strategy:
    - Split on class/function definitions at module level
    - Keep related code together
    - Preserve line number information
    """
    import re
    chunks = []
    lines = content.split('\n')
    
    # Pattern for top-level definitions
    pattern = r'^(class\s+\w+|def\s+\w+|async\s+def\s+\w+)'
    
    current_chunk_start = 0
    current_chunk_lines = []
    
    for i, line in enumerate(lines):
        # New definition at module level (not indented)
        if re.match(pattern, line) and not line.startswith(' '):
            # Save previous chunk
            if current_chunk_lines:
                chunks.append({
                    'file_path': file_path,
                    'content': '\n'.join(current_chunk_lines),
                    'line_start': current_chunk_start + 1,
                    'line_end': i,
                    'type': 'code'
                })
            
            current_chunk_start = i
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)
    
    # Add final chunk
    if current_chunk_lines:
        chunks.append({
            'file_path': file_path,
            'content': '\n'.join(current_chunk_lines),
            'line_start': current_chunk_start + 1,
            'line_end': len(lines),
            'type': 'code'
        })
    
    return chunks if chunks else self._chunk_generic(content, file_path, 500, 100)
```

---

## 9. Suggestions for Improvements

### Code Quality Improvements

| Area | Current State | Recommendation |
|------|---------------|----------------|
| **Type Hints** | Partial coverage | Add comprehensive type hints throughout |
| **Error Handling** | Basic try/catch | Implement custom exception classes |
| **Logging** | Inconsistent | Use structured logging (structlog) |
| **Tests** | Unit tests exist | Add integration tests for hybrid retrieval |
| **Documentation** | Inline comments | Add docstrings to all public methods |

### Architectural Improvements

| Area | Current State | Recommendation |
|------|---------------|----------------|
| **Session Storage** | In-memory dicts | Use Redis for persistence and scaling |
| **Index Storage** | File-based pickle | Consider Milvus/Pinecone for production |
| **Task Queue** | Asyncio background tasks | Use Celery for distributed task processing |
| **Caching** | Simple dict cache | Use Redis with TTL for response caching |
| **Model Loading** | Single model at startup | Support model hot-swapping |
| **Frontend State** | Manual state object | Consider using Zustand or similar |

### Missing Documentation

| Item | Status | Action |
|------|--------|--------|
| ~~Technical Architecture~~ | ✅ Now complete | This document |
| API Rate Limiting | ❌ Missing | Document GitHub API limits |
| Error Codes | ❌ Missing | Create error code reference |
| Deployment Guide | ❌ Missing | Add Docker + cloud deployment docs |
| Contributing Guide | ❌ Missing | Add CONTRIBUTING.md |

### Refactor Suggestions

1. **Extract Retrieval Module**
   ```
   backend/src/retrieval/
   ├── __init__.py
   ├── faiss_retriever.py
   ├── bm25_retriever.py
   ├── hybrid_retriever.py
   └── base.py (abstract interface)
   ```

2. **Separate LLM Module**
   ```
   backend/src/llm/
   ├── __init__.py
   ├── llama_provider.py
   ├── prompt_builder.py
   └── response_validator.py
   ```

3. **Add Configuration Validation**
   ```python
   # Use pydantic-settings for validated configuration
   class Settings(BaseSettings):
       model_name: Literal["phi3-mini", "codellama-7b", "llama3.1-8b"]
       github_token: Optional[SecretStr] = None
       
       class Config:
           env_file = ".env"
   ```

4. **Implement Proper Dependency Injection**
   ```python
   # Use FastAPI's Depends for service injection
   from fastapi import Depends
   
   def get_rag_service(
       config: Settings = Depends(get_settings),
       index: VectorIndex = Depends(get_index)
   ) -> RAGService:
       return RAGService(config, index)
   ```

### Performance Optimizations

| Area | Suggestion |
|------|------------|
| **Embedding Generation** | Batch queries for embedding model |
| **Index Loading** | Lazy-load indices on first query |
| **LLM Inference** | Consider vLLM for faster inference |
| **File Discovery** | Parallelize file discovery with asyncio |
| **Response Streaming** | Add SSE for streaming LLM responses |

---

## Appendix: Configuration Reference

### Model Configurations

```python
# backend/src/config.py
MODEL_CONFIGS = {
    "phi3-mini": {
        "filename": "Q4_K_M-00001-of-00001.gguf",
        "context_length": 8192,
        "max_tokens": 512,
        "temperature": 0.7,
        "prompt_format": "phi3"
    },
    "codellama-7b": {
        "filename": "codellama-7b-merged-Q4_K_M.gguf",
        "context_length": 2048,
        "max_tokens": 128,
        "prompt_format": "llama2"
    },
    "llama3.1-8b": {
        "filename": "llama-3.1-8b-instruct-q4_k_m.gguf",
        "context_length": 8192,
        "max_tokens": 256,
        "prompt_format": "llama3"
    }
}
```

### File Patterns

**Default Include Patterns:**
```
*.py, *.md, *.txt, *.json, *.yaml, *.yml
```

**Default Exclude Patterns:**
```
.git/**, __pycache__/**, *.pyc, venv/**, node_modules/**, *.log
```

---

*End of Technical Documentation*
