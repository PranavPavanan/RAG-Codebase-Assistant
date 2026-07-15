# RAG-Powered GitHub Assistant 🤖

An AI-powered tool for querying and understanding GitHub repository code using Retrieval-Augmented Generation (RAG).

## Features ✨

- **Repository Search**: Search and discover GitHub repositories
- **Code Indexing**: Index repository code for semantic search
- **AI-Powered Queries**: Ask natural language questions about the codebase
- **Conversational Interface**: Maintain context across multiple queries
- **Real-time Updates**: WebSocket support for live indexing progress
- **Source References**: Get exact file locations and line numbers for answers

## Architecture 🏗️

### Backend (FastAPI + Python)
- **FastAPI** for high-performance REST API
- **FAISS** for vector similarity search
- **CodeLlama-7B** for code understanding and generation
- **PyGithub** for GitHub API integration
- **sentence-transformers** for code embeddings

### Frontend (React + TypeScript)
- **React 18** with hooks and functional components
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Zustand** for state management
- **Axios** for API communication
- **WebSocket** for real-time updates

## Prerequisites 📋

- **Python 3.10 to 3.12** (Note: Python 3.13+ or 3.14 may cause compilation errors for ML dependencies)
- **Node.js 18+** (for frontend)
- **Git** (for repository cloning)
- **8GB+ RAM** (16GB recommended for LLM inference)

## Quick Start 🚀

### 1. Setup from GitHub

Clone this repository from GitHub (HTTPS or SSH):

```bash
git clone https://github.com/PranavPavanan/coding-assistant.git
# or
# git clone git@github.com:PranavPavanan/coding-assistant.git
cd coding-assistant
```

If you plan to contribute, create your feature branch now:

```bash
git checkout -b feature/your-feature-name
```

### 2. Backend Setup

```bash
# First, create virtual environment in the project root
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Note for Windows users: If `llama-cpp-python` fails to build due to missing C++ tools, install its pre-compiled wheel using:
# pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Move to backend directory to run the server
cd backend

# Create .env file (optional)
copy .env.example .env   # Windows (PowerShell/CMD)
# cp .env.example .env   # Mac/Linux
# Edit .env and add your GitHub token if you have one

# Download the Qwen 2.5 Coder Model
# VERY IMPORTANT: Create the models directory inside the `backend` folder!
mkdir models
# Download `qwen2.5-coder-3b-instruct-q4_k_m.gguf` from Hugging Face
# Ensure you place the file precisely at: backend/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf

# Run backend
uvicorn src.main:app --reload
```

Backend will be available at **http://localhost:8000**

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env file
copy .env.example .env   # Windows (PowerShell/CMD)
# cp .env.example .env   # Mac/Linux
# Default values should work for local development

# Run frontend
npm run dev
```

Frontend will be available at **http://localhost:3000**

## Usage Guide 📖

### 1. Search for a Repository
- Enter keywords in the search box (e.g., "react hooks", "python web framework")
- Browse results and click on a repository to select it

### 2. Index the Repository
- Click "Start Indexing" to analyze the repository code
- Watch the progress bar as files are processed
- Indexing creates vector embeddings for semantic search

### 3. Query the Code
- Once indexing is complete, switch to the "Query" tab
- Ask questions in natural language:
  - "What does the main function do?"
  - "How is authentication implemented?"
  - "Explain the database schema"
- View answers with source code references

### 4. Conversational Queries
- Follow-up questions maintain context
- Ask clarifying questions
- Reference previous parts of the conversation

## Environment Variables ⚙️

### Backend (.env)

```env
# GitHub Personal Access Token (optional, increases rate limits)
GITHUB_TOKEN=your_github_token_here

# Model Configuration
MODEL_NAME=qwen2.5-coder-3b
MODEL_PATH=./models/qwen2.5-coder-3b-instruct-q4_k_m.gguf

# Indexing Configuration
MAX_FILE_SIZE=1048576  # 1MB in bytes
CHUNK_SIZE=1000  # Characters per chunk
CHUNK_OVERLAP=200  # Characters overlap

# RAG Configuration
TOP_K_RESULTS=5  # Number of relevant snippets to retrieve
MAX_CONTEXT_LENGTH=4000  # Maximum context for LLM

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### Frontend (.env)

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## API Documentation 📚

### Endpoints

#### Health Check
```http
GET /api/health
```

#### Search Repositories
```http
POST /api/search/repositories
Content-Type: application/json

{
  "query": "python web framework",
  "limit": 10
}
```

#### Start Indexing
```http
POST /api/index/start
Content-Type: application/json

{
  "repository_url": "https://github.com/owner/repo"
}
```

#### Chat Query
```http
POST /api/chat/query
Content-Type: application/json

{
  "query": "What does the main function do?",
  "conversation_id": "optional-conversation-id"
}
```

Full API documentation available at **http://localhost:8000/docs** (Swagger UI)

## Testing 🧪

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only contract tests
pytest tests/contract/ -v

# Run only unit tests
pytest tests/unit/ -v
```

### Frontend Tests

```bash
cd frontend

# Run tests (requires vitest and testing-library packages)
npm test

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage
```

## Performance 🚀

- **Query Latency Target**: < 12 seconds
- **Vector Search**: < 1s for repositories up to 10,000 files
- **LLM Generation**: 5-10s depending on response length
- **Indexing Time**: 1-2 minutes per 1000 files

See [PERFORMANCE.md](docs/PERFORMANCE.md) for optimization tips.

## Accessibility ♿

The application follows WCAG 2.1 Level AA guidelines:
- Keyboard navigation support
- Screen reader compatible
- High contrast support
- Semantic HTML

See [ACCESSIBILITY.md](docs/ACCESSIBILITY.md) for details.

## Project Structure 📁

```
coding-assistant/
├── backend/
│   ├── src/
│   │   ├── api/          # FastAPI endpoints
│   │   ├── models/       # Pydantic data models
│   │   ├── services/     # Business logic
│   │   └── main.py       # Application entry point
│   ├── tests/            # Test suite
│   └── requirements.txt  # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── services/     # API and WebSocket clients
│   │   ├── store/        # Zustand state management
│   │   └── App.tsx       # Main application
│   └── package.json      # Node dependencies
│
├── docs/                 # Documentation
└── specs/                # Feature specifications
```

## Troubleshooting 🔧

### Backend Issues

**Model not loading:**
- Ensure the model file is strictly in `backend/models/` (NOT `backend/src/models/` or root `models/`)
- Check the file name matches `MODEL_NAME` configuration in .env
- Verify you have enough RAM (8GB minimum)

**GitHub API rate limit:**
- Add a GitHub Personal Access Token to .env
- Token increases rate limit from 60 to 5000 requests/hour

**Slow queries:**
- Reduce `MAX_CONTEXT_LENGTH` in .env
- Reduce `TOP_K_RESULTS` to retrieve fewer snippets
- Use a smaller quantized model (Q3 or Q2)

### Frontend Issues

**Cannot connect to backend:**
- Verify backend is running on port 8000
- Check `VITE_API_URL` in frontend/.env
- Look for CORS errors in browser console

**Components not updating:**
- Check browser console for errors
- Verify WebSocket connection (Network tab)
- Clear browser cache and reload

## Contributing 🤝

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- **CodeLlama** by Meta AI for the language model
- **FAISS** by Meta Research for vector search
- **FastAPI** for the excellent Python web framework
- **React** and **Vite** for the frontend stack
- **Tailwind CSS** and **Radix UI** for UI components

## Contact 📧

**Pranav Pavanan**
- GitHub: [@PranavPavanan](https://github.com/PranavPavanan)
- Repository: [codingAssist](https://github.com/PranavPavanan/codingAssist)

## Status 🚦

**Current Version**: 0.1.0 (MVP)
**Status**: Active Development
**Last Updated**: October 2025

---

Built with ❤️ for developers by developers
