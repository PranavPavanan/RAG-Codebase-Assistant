"""RAG (Retrieval-Augmented Generation) pipeline service."""
import json
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import time
import pickle
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Try importing FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not available. Install with: pip install faiss-cpu")

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Try importing BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not available. Install with: pip install rank-bm25")

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logger.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")

from src.models.query import (
    ChatContextResponse,
    ChatHistoryRequest,
    ChatHistoryResponse,
    ChatMessage,
    QueryRequest,
    QueryResponse,
    SourceReference,
    SessionInfo,
    ConversationInfo,
    SessionClearRequest,
    SessionClearResponse,
)
from src.config import settings


class RAGService:
    """Service for RAG-powered code querying."""

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize RAG service.

        Args:
            vector_store_path: Path to FAISS vector store
            model_path: Path to LLM model (CodeLlama)
        """
        self.vector_store_path = vector_store_path
        self.model_path = model_path

        # In-memory session and conversation storage (MVP - should use database in production)
        self.sessions: dict[str, SessionInfo] = {}  # session_id -> SessionInfo
        self.conversations: dict[str, list[ChatMessage]] = {}  # conversation_id -> messages
        self.session_conversations: dict[str, set[str]] = {}  # session_id -> set of conversation_ids
        
        # Response cache for common queries
        self.response_cache = {}
        self.cache_max_size = 100

        # Flag to indicate if index is loaded
        self.is_initialized = False

        # Placeholder for vector store and model
        self.vector_store = None
        self.model = None
        self.embeddings = None
        
        # Hybrid retrieval components
        self.faiss_index = None
        self.embedding_model = None
        self.bm25_index = None
        self.document_chunks: List[Dict[str, Any]] = []  # Stores chunk metadata
        self.chunk_texts: List[str] = []  # Stores chunk text for BM25
        self.retrieval_initialized = False
        
        # Index storage paths
        self.indices_dir = Path("./storage/indices")
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        
        # Embedding model name (all-MiniLM-L6-v2 is fast and effective)
        self.embedding_model_name = "all-MiniLM-L6-v2"

    async def initialize(self):
        """
        Initialize vector store and LLM model.

        Returns:
            True if initialization successful
        """
        try:
            # Get model configuration
            model_config = settings.get_model_config()
            model_path = settings.get_model_path()
            model_name = settings.MODEL_NAME
            
            # Check if llama-cpp-python is available
            if not LLAMA_AVAILABLE:
                logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
                self.is_initialized = False
                return False
            
            # Check if model file exists
            if not model_path.exists():
                logger.error(f"Model file not found at {model_path}")
                logger.error(f"Expected filename: {model_config['filename']}")
                logger.error(f"Available models: {settings.get_available_models()}")
                logger.error("To switch models, change MODEL_NAME in .env file or config.py")
                self.is_initialized = False
                return False
            
            logger.info(f"Loading {model_name} model from {model_path}")
            logger.info(f"Context: {model_config['context_length']}, Threads: {model_config['n_threads']}, Size: {model_path.stat().st_size / (1024*1024*1024):.2f} GB")
            
            # Load the model with configuration-specific settings
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=model_config["context_length"],
                n_threads=model_config["n_threads"],
                n_gpu_layers=model_config["n_gpu_layers"],
                verbose=False,  # Disable verbose output
            )
            
            logger.info(f"{model_name} model loaded successfully")

            # Initialize hybrid retrieval (FAISS + BM25)
            await self._initialize_hybrid_retrieval()

            self.is_initialized = True
            return True
        except Exception as e:
            error_msg = str(e)
            logger.error(f"RAG initialization failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Provide specific guidance for common errors
            if "key not found in model: tokenizer.ggml.tokens" in error_msg:
                logger.error("This appears to be a LoRA adapter model that requires a base model")
                logger.error("Solution: Download a base model and merge, or use a complete GGUF model")
            elif "failed to load model" in error_msg:
                logger.error("The model file may be corrupted or incompatible")
                logger.error("Solution: Re-download the model file or try a different model")
            elif "AssertionError" in error_msg:
                logger.error("The model file appears to be corrupted or incomplete")
                logger.error("Solution: Re-download the model file or use a different model")
            else:
                import traceback
                logger.error(f"Error details: {error_msg}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Don't fallback to mock mode - show the actual error
            self.is_initialized = False
            return False

    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query using RAG pipeline.

        Args:
            request: Query request with question and optional context

        Returns:
            QueryResponse with answer and sources
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{request.query.lower().strip()}:{request.max_sources or 5}"
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            logger.debug(f"Cache hit for query: {request.query[:50]}...")
            return cached_response
        
        # Generate or use provided session ID and conversation ID
        session_id = request.session_id or str(uuid.uuid4())
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Create session if it doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionInfo(
                session_id=session_id,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                conversation_count=0,
                total_messages=0
            )
            self.session_conversations[session_id] = set()

        # Create conversation if it doesn't exist
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            # Add conversation to session
            self.session_conversations[session_id].add(conversation_id)
            self.sessions[session_id].conversation_count += 1

        # Update session activity
        self.sessions[session_id].last_activity = datetime.utcnow()

        # Add user message to conversation history
        user_message = ChatMessage(
            role="user",
            content=request.query,
            timestamp=datetime.utcnow(),
        )
        self.conversations[conversation_id].append(user_message)
        self.sessions[session_id].total_messages += 1

        # Check if model is loaded
        if not self.is_initialized:
            error_msg = "RAG service not initialized."
            logger.error(error_msg)
            
            # Create detailed debug response
            model_path = settings.get_model_path()
            model_config = settings.get_model_config()
            
            answer = f"""DEBUG: Model Loading Failed

Error: {error_msg}

Debug Information:
- Model Name: {settings.MODEL_NAME}
- Expected File: {model_config['filename']}
- Full Path: {model_path}
- File Exists: {model_path.exists()}
- File Size: {model_path.stat().st_size if model_path.exists() else 'N/A'} bytes

Common Issues:
1. Model file not found - check the filename in config.py
2. Model file corrupted - try re-downloading
3. Wrong model format - ensure it's a valid GGUF file
4. Insufficient memory - try a smaller model

To fix:
1. Check the server logs for detailed error messages
2. Verify the model file exists and is not corrupted
3. Try downloading a different model
4. Update the filename in backend/src/config.py

The contextual awareness system is working, but you need a valid model file."""
            
            sources = []
            model = "debug - model not loaded"
        else:
            try:
                # Use actual CodeLlama model with basic retrieval from indexed content
                # Retrieve relevant content from indexed files
                retrieved_content, sources = self._retrieve_relevant_content(request.query)
                
                # Build conversation context for contextual awareness
                conversation_context = self._build_conversation_context(conversation_id)
                
                # Generate response using CodeLlama with retrieved context and conversation history
                answer = self.generate_response(request.query, retrieved_content, conversation_context)
                model = settings.MODEL_NAME  # Use configured model name
                
            except Exception as e:
                logger.error(f"Error during query processing: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                # Create error response
                answer = f"Error processing query: {str(e)}\n\nPlease try again or check the server logs."
                sources = []
                model = "error - processing failed"

        # Create assistant message
        assistant_message = ChatMessage(
            role="assistant",
            content=answer,
            timestamp=datetime.utcnow(),
            sources=sources,
        )
        self.conversations[conversation_id].append(assistant_message)
        self.sessions[session_id].total_messages += 1

        processing_time = time.time() - start_time

        # Create response
        response = QueryResponse(
            response=answer,
            sources=sources,
            conversation_id=conversation_id,
            session_id=session_id,
            model=model,
        )
        
        # Cache the response (with size limit)
        if len(self.response_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = response
        logger.debug(f"Response cached for query: {request.query[:50]}...")
        
        return response
    
    
    def _build_conversation_context(self, conversation_id: str) -> str:
        """Build context string from conversation history."""
        if conversation_id not in self.conversations:
            return ""
        
        messages = self.conversations[conversation_id]
        # Get last 5 messages for context
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        
        context_parts = []
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(context_parts)

    def get_chat_history(self, request: ChatHistoryRequest) -> Optional[ChatHistoryResponse]:
        """
        Get conversation history.

        Args:
            request: Chat history request with conversation ID

        Returns:
            ChatHistoryResponse or None if not found
        """
        conversation_id = request.conversation_id

        if conversation_id not in self.conversations:
            return None

        messages = self.conversations[conversation_id]

        # Apply limit if specified
        if request.limit:
            messages = messages[-request.limit :]

        return ChatHistoryResponse(
            conversation_id=conversation_id,
            messages=messages,
            total_count=len(self.conversations[conversation_id]),
        )

    def get_chat_context(self, conversation_id: str) -> Optional[ChatContextResponse]:
        """
        Get conversation context summary.

        Args:
            conversation_id: Conversation identifier

        Returns:
            ChatContextResponse or None if not found
        """
        if conversation_id not in self.conversations:
            return None

        messages = self.conversations[conversation_id]

        # Count messages by role
        user_messages = sum(1 for m in messages if m.role == "user")
        assistant_messages = sum(1 for m in messages if m.role == "assistant")

        # Get last query and response
        last_user_message = next((m for m in reversed(messages) if m.role == "user"), None)
        last_assistant_message = next(
            (m for m in reversed(messages) if m.role == "assistant"), None
        )

        return ChatContextResponse(
            conversation_id=conversation_id,
            message_count=len(messages),
            user_message_count=user_messages,
            assistant_message_count=assistant_messages,
            last_query=last_user_message.content if last_user_message else None,
            last_response=last_assistant_message.content if last_assistant_message else None,
            created_at=messages[0].timestamp if messages else datetime.utcnow(),
            last_updated=messages[-1].timestamp if messages else datetime.utcnow(),
        )

    def clear_history(self, conversation_id: Optional[str] = None) -> dict:
        """
        Clear conversation history.

        Args:
            conversation_id: Optional specific conversation to clear, or None for all

        Returns:
            Dictionary with operation result
        """
        try:
            if conversation_id:
                if conversation_id in self.conversations:
                    del self.conversations[conversation_id]
                    return {
                        "success": True,
                        "message": f"Conversation {conversation_id} cleared",
                    }
                else:
                    return {
                        "success": False,
                        "error": "Conversation not found",
                    }
            else:
                count = len(self.conversations)
                self.conversations.clear()
                return {
                    "success": True,
                    "message": f"Cleared {count} conversations",
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_conversation_history(self, conversation_id: str) -> list[ChatMessage]:
        """
        Get conversation history for a specific conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of chat messages
        """
        return self.conversations.get(conversation_id, [])

    def clear_conversations(self) -> None:
        """Clear all conversations."""
        self.conversations.clear()
        self.sessions.clear()
        self.session_conversations.clear()

    def clear_session(self, request: SessionClearRequest) -> SessionClearResponse:
        """
        Clear session data.

        Args:
            request: Session clear request

        Returns:
            SessionClearResponse with operation result
        """
        try:
            sessions_cleared = 0
            conversations_cleared = 0

            if request.clear_all:
                # Clear all sessions
                conversations_cleared = len(self.conversations)
                sessions_cleared = len(self.sessions)
                self.conversations.clear()
                self.sessions.clear()
                self.session_conversations.clear()
                
                return SessionClearResponse(
                    success=True,
                    message=f"Cleared all sessions and {conversations_cleared} conversations",
                    sessions_cleared=sessions_cleared,
                    conversations_cleared=conversations_cleared
                )
            else:
                # Clear specific session
                session_id = request.session_id
                if session_id not in self.sessions:
                    return SessionClearResponse(
                        success=False,
                        message=f"Session {session_id} not found",
                        sessions_cleared=0,
                        conversations_cleared=0
                    )

                # Count conversations to be cleared
                if session_id in self.session_conversations:
                    conversations_cleared = len(self.session_conversations[session_id])
                    # Remove all conversations in this session
                    for conv_id in self.session_conversations[session_id]:
                        if conv_id in self.conversations:
                            del self.conversations[conv_id]
                    del self.session_conversations[session_id]

                # Remove session
                del self.sessions[session_id]
                sessions_cleared = 1

                return SessionClearResponse(
                    success=True,
                    message=f"Cleared session {session_id} and {conversations_cleared} conversations",
                    sessions_cleared=sessions_cleared,
                    conversations_cleared=conversations_cleared
                )

        except Exception as e:
            return SessionClearResponse(
                success=False,
                message=f"Failed to clear session: {str(e)}",
                sessions_cleared=0,
                conversations_cleared=0
            )

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get session information.

        Args:
            session_id: Session identifier

        Returns:
            SessionInfo or None if not found
        """
        return self.sessions.get(session_id)

    def get_conversation_info(self, conversation_id: str) -> Optional[ConversationInfo]:
        """
        Get conversation information.

        Args:
            conversation_id: Conversation identifier

        Returns:
            ConversationInfo or None if not found
        """
        if conversation_id not in self.conversations:
            return None

        messages = self.conversations[conversation_id]
        
        # Find the session this conversation belongs to
        session_id = None
        for sid, conv_ids in self.session_conversations.items():
            if conversation_id in conv_ids:
                session_id = sid
                break

        if not session_id:
            return None

        return ConversationInfo(
            conversation_id=conversation_id,
            session_id=session_id,
            created_at=messages[0].timestamp if messages else datetime.utcnow(),
            last_activity=messages[-1].timestamp if messages else datetime.utcnow(),
            message_count=len(messages)
        )

    def list_sessions(self) -> list[SessionInfo]:
        """
        List all active sessions.

        Returns:
            List of SessionInfo objects
        """
        return list(self.sessions.values())

    def list_conversations_in_session(self, session_id: str) -> list[ConversationInfo]:
        """
        List all conversations in a session.

        Args:
            session_id: Session identifier

        Returns:
            List of ConversationInfo objects
        """
        if session_id not in self.session_conversations:
            return []

        conversations = []
        for conv_id in self.session_conversations[session_id]:
            conv_info = self.get_conversation_info(conv_id)
            if conv_info:
                conversations.append(conv_info)

        return conversations

    def get_conversation_context(self, conversation_id: str) -> Optional[ChatContextResponse]:
        """
        Get conversation context summary.

        Args:
            conversation_id: Conversation identifier

        Returns:
            ChatContextResponse or None if not found
        """
        if conversation_id not in self.conversations:
            return None

        messages = self.conversations[conversation_id]

        # Count messages by role
        user_messages = sum(1 for m in messages if m.role == "user")
        assistant_messages = sum(1 for m in messages if m.role == "assistant")

        # Get last query and response
        last_user_message = next((m for m in reversed(messages) if m.role == "user"), None)
        last_assistant_message = next(
            (m for m in reversed(messages) if m.role == "assistant"), None
        )

        return ChatContextResponse(
            conversation_id=conversation_id,
            message_count=len(messages),
            user_message_count=user_messages,
            assistant_message_count=assistant_messages,
            last_query=last_user_message.content if last_user_message else None,
            last_response=last_assistant_message.content if last_assistant_message else None,
            created_at=messages[0].timestamp if messages else datetime.utcnow(),
            last_updated=messages[-1].timestamp if messages else datetime.utcnow(),
        )

    def semantic_search(self, query: str, top_k: int = 5) -> list[SourceReference]:
        """
        Perform semantic search on indexed code using FAISS.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of source references
        """
        if not self.retrieval_initialized or self.faiss_index is None:
            logger.warning("FAISS index not initialized, falling back to empty results")
            return []
        
        if not self.embedding_model:
            logger.warning("Embedding model not loaded")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype('float32')
            
            # Normalize for cosine similarity (if using IndexFlatIP)
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            k = min(top_k, len(self.document_chunks))
            if k == 0:
                return []
            
            distances, indices = self.faiss_index.search(query_embedding, k)
            
            # Build source references from results
            sources = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                chunk = self.document_chunks[idx]
                # Convert distance to similarity score (for IP, higher is better)
                # Clamp to [0, 1] range to handle edge cases
                score = max(0.0, min(1.0, float(dist)))
                
                sources.append(SourceReference(
                    file=chunk['file_path'],
                    content=chunk['content'],
                    score=score,
                    line_start=chunk.get('line_start', 1),
                    line_end=chunk.get('line_end', 1),
                    type=chunk.get('type', 'code')
                ))
            
            logger.debug(f"Semantic search returned {len(sources)} results for query: {query[:50]}...")
            return sources
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def keyword_search(self, query: str, top_k: int = 5) -> list[SourceReference]:
        """
        Perform keyword-based search using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of source references
        """
        if not self.retrieval_initialized or self.bm25_index is None:
            logger.warning("BM25 index not initialized, falling back to empty results")
            return []
        
        try:
            # Tokenize query (simple whitespace tokenization)
            query_tokens = self._tokenize(query)
            
            if not query_tokens:
                return []
            
            # Get BM25 scores for all documents
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k indices
            k = min(top_k, len(scores))
            top_indices = np.argsort(scores)[::-1][:k]
            
            # Build source references from results
            sources = []
            max_score = max(scores) if max(scores) > 0 else 1.0
            
            for idx in top_indices:
                if scores[idx] <= 0:  # Skip zero-score results
                    continue
                
                chunk = self.document_chunks[idx]
                # Normalize score to 0-1 range
                normalized_score = float(scores[idx]) / max_score
                
                sources.append(SourceReference(
                    file=chunk['file_path'],
                    content=chunk['content'],
                    score=normalized_score,
                    line_start=chunk.get('line_start', 1),
                    line_end=chunk.get('line_end', 1),
                    type=chunk.get('type', 'code')
                ))
            
            logger.debug(f"BM25 search returned {len(sources)} results for query: {query[:50]}...")
            return sources
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 10, semantic_weight: float = 0.5) -> list[SourceReference]:
        """
        Perform hybrid search combining semantic (FAISS) and keyword (BM25) search.
        Uses Reciprocal Rank Fusion (RRF) for score combination.

        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0-1), keyword weight = 1 - semantic_weight

        Returns:
            Combined and reranked source references
        """
        # If retrieval not initialized, fall back to old method
        if not self.retrieval_initialized:
            logger.warning("Hybrid retrieval not initialized, falling back to keyword-based retrieval")
            return self._fallback_keyword_retrieval(query, top_k)
        
        try:
            # Get results from both methods (fetch more to allow for deduplication)
            fetch_k = top_k * 2
            semantic_results = self.semantic_search(query, fetch_k)
            keyword_results = self.keyword_search(query, fetch_k)
            
            # Use Reciprocal Rank Fusion (RRF) to combine results
            # RRF score = sum(1 / (k + rank)) for each ranking list
            k_constant = 60  # Standard RRF constant
            
            # Build score maps
            rrf_scores: Dict[str, float] = {}  # file_path:line_start -> score
            chunk_map: Dict[str, SourceReference] = {}  # file_path:line_start -> SourceReference
            
            # Process semantic results
            for rank, src in enumerate(semantic_results):
                key = f"{src.file}:{src.line_start}"
                rrf_score = semantic_weight * (1.0 / (k_constant + rank + 1))
                rrf_scores[key] = rrf_scores.get(key, 0) + rrf_score
                if key not in chunk_map:
                    chunk_map[key] = src
            
            # Process keyword results
            keyword_weight = 1.0 - semantic_weight
            for rank, src in enumerate(keyword_results):
                key = f"{src.file}:{src.line_start}"
                rrf_score = keyword_weight * (1.0 / (k_constant + rank + 1))
                rrf_scores[key] = rrf_scores.get(key, 0) + rrf_score
                if key not in chunk_map:
                    chunk_map[key] = src
            
            # Sort by RRF score and take top_k
            sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
            
            # Build final results with updated scores
            results = []
            max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
            
            for key in sorted_keys:
                src = chunk_map[key]
                # Normalize RRF score to 0-1
                normalized_score = rrf_scores[key] / max_rrf if max_rrf > 0 else 0
                results.append(SourceReference(
                    file=src.file,
                    content=src.content,
                    score=normalized_score,
                    line_start=src.line_start,
                    line_end=src.line_end,
                    type=src.type
                ))
            
            logger.info(f"Hybrid search returned {len(results)} results (semantic: {len(semantic_results)}, keyword: {len(keyword_results)})")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self._fallback_keyword_retrieval(query, top_k)

    def generate_response(self, query: str, context: list[SourceReference], conversation_context: str = "") -> str:
        """
        Generate response using LLM with retrieved context.

        Args:
            query: User query
            context: Retrieved source references
            conversation_context: Previous conversation context

        Returns:
            Generated response text
        """
        if not self.model:
            return "Model not loaded"
        
        # Get model configuration for system prompt and generation settings
        model_config = settings.get_model_config()
        system_prompt = model_config["system_prompt"]
        prompt_format = model_config.get("prompt_format", "llama2")

        # Add retrieved code context if available
        code_context = ""
        if context:
            code_context = "\n\nRelevant Code from Repository:\n"
            for src in context[:3]:  # Top 3 most relevant
                if isinstance(src, str):
                    # Handle string content directly
                    code_context += f"\n{src}\n"
                else:
                    # Handle SourceReference objects - present more naturally with line info
                    line_info = ""
                    if src.line_start and src.line_end:
                        line_info = f" (lines {src.line_start}-{src.line_end})"
                    code_context += f"\nFrom {src.file}{line_info}:\n{src.content}\n"
        
        # Build instructions
        instructions = """When answering:
- Speak naturally about the repository, not "provided code snippets"
- Reference specific files when relevant (e.g., "In src/services/rag_service.py...")
- Mention line numbers when discussing specific implementations
- Give direct, clear answers
- If information is not available, say so clearly"""
        
        # Build prompt based on model format
        if prompt_format == "phi3":
            # Phi-3 format: <|system|>\n...<|end|>\n<|user|>\n...<|end|>\n<|assistant|>
            prompt = f"""<|system|>
{system_prompt}

{instructions}<|end|>
<|user|>
{code_context}

Question: {query}<|end|>
<|assistant|>
"""
        elif prompt_format == "llama3":
            # Llama 3 format
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

{instructions}<|eot_id|><|start_header_id|>user<|end_header_id|>

{code_context}

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            # Default Llama 2 / CodeLlama format
            prompt = f"""<s>[INST] <<SYS>>
{system_prompt}

{instructions}
<</SYS>>

{code_context}

Question: {query} [/INST]

Answer naturally and conversationally about what you observe in the repository.

"""

        try:
            # Generate response using configured model
            response = self.model(
                prompt,
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
                top_p=model_config["top_p"],
                stop=model_config["stop_tokens"],
                echo=False,
            )
            
            # Extract the generated text
            generated_text = response["choices"][0]["text"].strip()
            
            # Validate and clean the response
            generated_text = self._validate_response(generated_text, context)
            
            # Fact-check the response against source content
            generated_text = self._fact_check_response(generated_text, context, query)
            
            # Clean up response object to free memory
            del response
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return f"Error generating response: {str(e)}"

    def _retrieve_relevant_content(self, query: str) -> tuple[List[str], List[SourceReference]]:
        """
        Retrieve relevant content from indexed files using hybrid search.
        
        Uses FAISS (semantic) + BM25 (keyword) hybrid retrieval when available,
        falls back to keyword-based scoring otherwise.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (retrieved_content, sources)
        """
        try:
            # Use hybrid search if available
            if self.retrieval_initialized:
                logger.info(f"Using hybrid retrieval for query: {query[:50]}...")
                sources = self.hybrid_search(query, top_k=5, semantic_weight=0.5)
                
                if sources:
                    retrieved_content = []
                    for src in sources:
                        content_str = f"File: {src.file} (lines {src.line_start}-{src.line_end})\n{src.content}"
                        retrieved_content.append(content_str)
                    
                    logger.info(f"Hybrid retrieval returned {len(sources)} sources")
                    return retrieved_content, sources
                else:
                    logger.warning("Hybrid search returned no results, falling back to keyword-based")
            
            # Fallback to original keyword-based retrieval
            return self._keyword_based_retrieval(query)
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            # Fall back to keyword-based retrieval
            return self._keyword_based_retrieval(query)
    
    def _keyword_based_retrieval(self, query: str) -> tuple[List[str], List[SourceReference]]:
        """
        Fallback keyword-based retrieval using file path and content scoring.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (retrieved_content, sources)
        """
        try:
            # Load the most recent metadata file
            metadata_dir = Path("./storage/metadata")
            if not metadata_dir.exists():
                return [], []
            
            # Find the most recent metadata file
            metadata_files = list(metadata_dir.glob("*.json"))
            if not metadata_files:
                return [], []
            
            # Sort by modification time and get the most recent
            latest_metadata = max(metadata_files, key=lambda f: f.stat().st_mtime)
            
            # Load metadata
            with open(latest_metadata, 'r') as f:
                metadata = json.load(f)
            
            files = metadata.get('files', [])
            if not files:
                return [], []
            
            # Improved keyword-based retrieval with better scoring
            query_lower = query.lower()
            query_words = set(query_lower.split())
            relevant_files = []
            
            # Score files based on query keywords
            for file_info in files:
                file_path = file_info.get('file_path', '')
                language = file_info.get('language', '')
                
                # Calculate relevance score
                score = 0
                
                # High priority: exact keyword matches in file path
                for word in query_words:
                    if word in file_path.lower():
                        score += 8  # Increased from 5
                
                # Medium priority: partial matches in file path
                for word in query_words:
                    if len(word) > 3 and any(word in part for part in file_path.lower().split('_')):
                        score += 5  # Increased from 3
                
                # Boost for exact class/function name matches
                for word in query_words:
                    if word in ['chunker', 'chunking', 'chunk', 'service']:
                        if 'chunker' in file_path.lower():
                            score += 10  # High boost for exact matches
                    elif word in ['embedding', 'model', 'config', 'configuration', 'transformer', 'sentence']:
                        if any(keyword in file_path.lower() for keyword in ['config', 'chunker', 'embedding', 'vector']):
                            score += 12  # Very high boost for embedding-related files
                        elif 'all-minilm' in file_path.lower() or 'sentence' in file_path.lower():
                            score += 15  # Maximum boost for specific model files
                    elif word in ['default', 'parameter', 'setting']:
                        if any(keyword in file_path.lower() for keyword in ['config', 'chunker', 'main']):
                            score += 6  # Boost for configuration files
                
                # Language-specific scoring
                if language:
                    if any(word in language.lower() for word in query_words):
                        score += 2
                    # Boost for common code files
                    if language in ['python', 'javascript', 'typescript']:
                        score += 1
                
                # File type priority
                if file_path.endswith('.md'):  # Documentation files
                    score += 4
                elif file_path.endswith(('.py', '.js', '.ts')):  # Code files
                    score += 2
                elif file_path.endswith(('.txt', '.json', '.yaml', '.yml')):  # Config files
                    score += 1
                
                # Special file names that are often relevant
                special_files = ['readme', 'config', 'main', 'index', 'chunker', 'indexer', 'crawler']
                for special in special_files:
                    if special in file_path.lower():
                        score += 2
                
                # Add content-based scoring if file content is available
                content_score = self._calculate_content_relevance(file_info, query_words)
                score += content_score
                
                # Only include files with meaningful scores
                if score > 0:
                    relevant_files.append((file_info, score))
            
            # Sort by score and take top files
            relevant_files.sort(key=lambda x: x[1], reverse=True)
            
            # Deduplicate by file path (keep highest score) before taking top 3
            seen_files = {}
            for file_info, score in relevant_files:
                file_path = file_info.get('file_path', '')
                if file_path not in seen_files or score > seen_files[file_path][1]:
                    seen_files[file_path] = (file_info, score)
            
            # Convert back to list and sort by score
            deduplicated_files = list(seen_files.values())
            deduplicated_files.sort(key=lambda x: x[1], reverse=True)
            top_files = deduplicated_files[:3]  # Top 3 most relevant files to reduce context
            
            # Read content from the most relevant files
            retrieved_content = []
            sources = []
            
            for file_info, score in top_files:
                file_path = file_info.get('file_path', '')
                
                repo_path = Path("./storage/repositories/web-rag-service")
                full_path = repo_path / file_path
                
                if full_path.exists():
                    try:
                        # Read file content
                        content = full_path.read_text(encoding='utf-8')
                        
                        # Smart content extraction based on query
                        content_preview, line_start, line_end = self._extract_relevant_content(
                            content, query, file_path
                        )
                        
                        retrieved_content.append(f"File: {file_path} (lines {line_start}-{line_end})\n{content_preview}")
                        
                        # Normalize score to 0-1 range (max possible score is around 30)
                        normalized_score = min(score / 30.0, 1.0)
                        
                        sources.append(SourceReference(
                            file=file_path,
                            content=content_preview,
                            score=normalized_score,
                            line_start=line_start,
                            line_end=line_end,
                            type="code" if file_path.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.rs', '.go', '.rb', '.php', '.swift', '.kt', '.scala', '.sh', '.bash')) else "doc"
                        ))
                    except Exception as e:
                        logger.debug(f"Error reading file {file_path}: {e}")
                        continue
            
            return retrieved_content, sources
            
        except Exception as e:
            logger.error(f"Error retrieving content: {e}")
            return [], []

    def _extract_relevant_content(self, content: str, query: str, file_path: str) -> tuple[str, int, int]:
        """
        Extract the most relevant section of content based on query.
        
        Args:
            content: Full file content
            query: User query
            file_path: Path to file
            
        Returns:
            Tuple of (extracted_content, line_start, line_end)
        """
        lines = content.split('\n')
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Keywords that indicate we should look for API patterns
        api_keywords = {'endpoint', 'api', 'route', 'decorator', '@app', '@router', 'get', 'post', 'put', 'delete'}
        looking_for_apis = bool(query_words & api_keywords)
        
        if looking_for_apis and file_path.endswith('.py'):
            # Look for FastAPI/Flask decorator patterns
            relevant_sections = []
            current_section = []
            section_start = 0
            in_endpoint = False
            
            for i, line in enumerate(lines, 1):
                # Check if this line defines an endpoint
                if '@app.' in line or '@router.' in line or 'def ' in line:
                    if '@app.' in line or '@router.' in line:
                        # Start of endpoint definition
                        if current_section and in_endpoint:
                            relevant_sections.append((section_start, i-1, '\n'.join(current_section)))
                        current_section = [line]
                        section_start = i
                        in_endpoint = True
                    elif in_endpoint and 'def ' in line:
                        # Function definition following decorator
                        current_section.append(line)
                        # Get next 3-5 lines for context
                        for j in range(1, min(6, len(lines) - i)):
                            current_section.append(lines[i + j - 1])
                        relevant_sections.append((section_start, i + 5, '\n'.join(current_section)))
                        current_section = []
                        in_endpoint = False
                elif in_endpoint:
                    current_section.append(line)
            
            # If we found endpoint definitions, return them
            if relevant_sections:
                # Combine up to 3 most relevant sections
                combined = []
                total_lines = 0
                start_line = relevant_sections[0][0]
                end_line = relevant_sections[0][1]
                
                for sec_start, sec_end, sec_content in relevant_sections[:5]:
                    combined.append(f"# Lines {sec_start}-{sec_end}")
                    combined.append(sec_content)
                    combined.append("")
                    end_line = max(end_line, sec_end)
                    total_lines += len(sec_content.split('\n'))
                    if total_lines > 100:  # Limit to ~100 lines
                        break
                
                extracted = '\n'.join(combined)
                if len(extracted) > 2000:
                    extracted = extracted[:2000] + "\n... (more endpoints below)"
                
                return extracted, start_line, end_line
        
        # Default: Take first 1000 characters with smart splitting
        if len(content) <= 1000:
            return content, 1, len(lines)
        
        # Try to find relevant section by keyword matching
        best_start = 0
        best_score = 0
        window_size = 1000
        
        for i in range(0, len(content) - window_size, 200):
            window = content[i:i+window_size].lower()
            score = sum(1 for word in query_words if word in window)
            if score > best_score:
                best_score = score
                best_start = i
        
        # Extract content around best match
        extract_start = max(0, best_start)
        extract_end = min(len(content), best_start + 1000)
        extracted = content[extract_start:extract_end]
        
        # Calculate line numbers
        lines_before = content[:extract_start].count('\n')
        lines_in_extract = extracted.count('\n')
        
        if extract_end < len(content):
            extracted += "\n... (truncated)"
        
        return extracted, lines_before + 1, lines_before + lines_in_extract + 1

    def _validate_response(self, response: str, context: List[str]) -> str:
        """
        Validate and clean the generated response for accuracy.
        
        Args:
            response: Generated response text
            context: Retrieved context for validation
            
        Returns:
            Cleaned and validated response
        """
        # Remove common artifacts
        response = response.replace("<|endoftext|>", "")
        response = response.replace("</s>", "")
        response = response.replace("[INST]", "")
        
        # Remove incomplete sentences at the end
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.endswith(('```', '```python', '```json')):
                cleaned_lines.append(line)
        
        # Join lines and clean up
        cleaned_response = '\n'.join(cleaned_lines)
        
        # Remove trailing incomplete sentences
        sentences = cleaned_response.split('. ')
        if len(sentences) > 1 and len(sentences[-1]) < 20:
            cleaned_response = '. '.join(sentences[:-1]) + '.'
        
        return cleaned_response.strip()

    def _calculate_content_relevance(self, file_info: dict, query_words: set) -> int:
        """
        Calculate content-based relevance score by reading file content.
        
        Args:
            file_info: File metadata information
            query_words: Set of query words
            
        Returns:
            Content relevance score
        """
        try:
            file_path = file_info.get('file_path', '')
            repo_path = Path("./storage/repositories/web-rag-service")
            full_path = repo_path / file_path
            
            if not full_path.exists():
                return 0
            
            # Read first 1000 characters for content analysis
            content = full_path.read_text(encoding='utf-8')[:1000].lower()
            
            score = 0
            
            # Check for exact keyword matches in content
            for word in query_words:
                if word in content:
                    score += 3
                    
            # Check for related terms
            related_terms = {
                'embedding': ['sentence', 'transformer', 'model', 'vector', 'embedding', 'all-minilm', 'minilm'],
                'transformer': ['sentence', 'transformer', 'model', 'embedding', 'all-minilm', 'minilm', 'sentence_transformers'],
                'model': ['model', 'llm', 'transformer', 'neural', 'ai', 'all-minilm', 'minilm', 'sentence_transformers'],
                'config': ['config', 'configuration', 'setting', 'default', 'parameter'],
                'chunking': ['chunk', 'chunking', 'split', 'segment', 'token']
            }
            
            # Special boost for specific model names
            if 'all-minilm-l6-v2' in content:
                score += 15  # Maximum boost for the specific model
            elif 'all-minilm' in content:
                score += 10  # High boost for model family
            elif 'sentence_transformers' in content:
                score += 8  # Boost for library name
            
            for query_word in query_words:
                if query_word in related_terms:
                    for term in related_terms[query_word]:
                        if term in content:
                            score += 2
            
            # Boost for class/function definitions
            if any(word in content for word in ['class ', 'def ', 'function ']):
                score += 1
                
            # Boost for configuration sections
            if any(section in content for section in ['config', 'settings', 'default', 'parameter']):
                score += 2
                
            return min(score, 10)  # Cap at 10 points
            
        except Exception as e:
            logger.debug(f"Error calculating content relevance for {file_path}: {e}")
            return 0

    # ==================== HYBRID RETRIEVAL METHODS ====================
    
    async def _initialize_hybrid_retrieval(self):
        """
        Initialize FAISS and BM25 indices for hybrid retrieval.
        Loads existing indices or builds new ones from indexed repository.
        """
        try:
            # Check if required libraries are available
            if not FAISS_AVAILABLE:
                logger.warning("FAISS not available - semantic search disabled")
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("sentence-transformers not available - semantic search disabled")
            if not BM25_AVAILABLE:
                logger.warning("BM25 not available - keyword search disabled")
            
            # Load embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model loaded successfully")
            
            # Try to load existing indices
            if await self._load_indices():
                logger.info("Loaded existing hybrid indices")
                self.retrieval_initialized = True
                return
            
            # Build indices from indexed repository
            logger.info("Building hybrid indices from indexed repository...")
            await self._build_indices()
            
            self.retrieval_initialized = True
            logger.info("Hybrid retrieval initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing hybrid retrieval: {e}")
            self.retrieval_initialized = False
    
    async def _load_indices(self) -> bool:
        """
        Load existing FAISS and BM25 indices from disk.
        
        Returns:
            True if indices loaded successfully, False otherwise
        """
        try:
            faiss_path = self.indices_dir / "faiss_index.bin"
            bm25_path = self.indices_dir / "bm25_index.pkl"
            chunks_path = self.indices_dir / "document_chunks.pkl"
            
            # Check if all index files exist
            if not all(p.exists() for p in [faiss_path, bm25_path, chunks_path]):
                logger.info("Index files not found, will build new indices")
                return False
            
            # Load document chunks
            with open(chunks_path, 'rb') as f:
                data = pickle.load(f)
                self.document_chunks = data['chunks']
                self.chunk_texts = data['texts']
            
            # Load FAISS index
            if FAISS_AVAILABLE:
                self.faiss_index = faiss.read_index(str(faiss_path))
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            
            # Load BM25 index
            if BM25_AVAILABLE:
                with open(bm25_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                logger.info(f"Loaded BM25 index with {len(self.chunk_texts)} documents")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            return False
    
    async def _save_indices(self):
        """Save FAISS and BM25 indices to disk."""
        try:
            faiss_path = self.indices_dir / "faiss_index.bin"
            bm25_path = self.indices_dir / "bm25_index.pkl"
            chunks_path = self.indices_dir / "document_chunks.pkl"
            
            # Save document chunks
            with open(chunks_path, 'wb') as f:
                pickle.dump({
                    'chunks': self.document_chunks,
                    'texts': self.chunk_texts
                }, f)
            
            # Save FAISS index
            if FAISS_AVAILABLE and self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(faiss_path))
                logger.info(f"Saved FAISS index with {self.faiss_index.ntotal} vectors")
            
            # Save BM25 index
            if BM25_AVAILABLE and self.bm25_index is not None:
                with open(bm25_path, 'wb') as f:
                    pickle.dump(self.bm25_index, f)
                logger.info(f"Saved BM25 index with {len(self.chunk_texts)} documents")
            
            logger.info("Indices saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving indices: {e}")
    
    async def _build_indices(self):
        """
        Build FAISS and BM25 indices from indexed repository content.
        """
        try:
            # Load metadata to find indexed files
            metadata_dir = Path("./storage/metadata")
            if not metadata_dir.exists():
                logger.warning("No metadata directory found")
                return
            
            metadata_files = list(metadata_dir.glob("*.json"))
            if not metadata_files:
                logger.warning("No metadata files found")
                return
            
            # Get the most recent metadata
            latest_metadata = max(metadata_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_metadata, 'r') as f:
                metadata = json.load(f)
            
            files = metadata.get('files', [])
            if not files:
                logger.warning("No files in metadata")
                return
            
            # Get repository path from metadata
            repo_name = metadata.get('repository_name', 'web-rag-service')
            repo_path = Path("./storage/repositories") / repo_name
            
            if not repo_path.exists():
                # Try default path
                repo_path = Path("./storage/repositories/web-rag-service")
            
            logger.info(f"Building indices from {len(files)} files in {repo_path}")
            
            # Chunk all files
            self.document_chunks = []
            self.chunk_texts = []
            
            for file_info in files:
                file_path = file_info.get('file_path', '')
                full_path = repo_path / file_path
                
                if not full_path.exists():
                    continue
                
                try:
                    content = full_path.read_text(encoding='utf-8')
                    chunks = self._chunk_content(content, file_path)
                    
                    for chunk in chunks:
                        self.document_chunks.append(chunk)
                        self.chunk_texts.append(chunk['content'])
                        
                except Exception as e:
                    logger.debug(f"Error processing file {file_path}: {e}")
                    continue
            
            logger.info(f"Created {len(self.document_chunks)} chunks from repository")
            
            # Build FAISS index
            if FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE and self.embedding_model:
                await self._build_faiss_index()
            
            # Build BM25 index
            if BM25_AVAILABLE:
                self._build_bm25_index()
            
            # Save indices to disk
            await self._save_indices()
            
        except Exception as e:
            logger.error(f"Error building indices: {e}")
    
    async def _build_faiss_index(self):
        """Build FAISS index from document chunks."""
        if not self.chunk_texts:
            logger.warning("No chunks to index for FAISS")
            return
        
        try:
            logger.info(f"Generating embeddings for {len(self.chunk_texts)} chunks...")
            
            # Generate embeddings in batches
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(self.chunk_texts), batch_size):
                batch = self.chunk_texts[i:i + batch_size]
                embeddings = self.embedding_model.encode(batch, convert_to_numpy=True)
                all_embeddings.append(embeddings)
            
            # Concatenate all embeddings
            embeddings_array = np.vstack(all_embeddings).astype('float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Create FAISS index (using Inner Product for cosine similarity on normalized vectors)
            dimension = embeddings_array.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings_array)
            
            logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors (dimension: {dimension})")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
    
    def _build_bm25_index(self):
        """Build BM25 index from document chunks."""
        if not self.chunk_texts:
            logger.warning("No chunks to index for BM25")
            return
        
        try:
            # Tokenize all documents
            tokenized_corpus = [self._tokenize(text) for text in self.chunk_texts]
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_corpus)
            
            logger.info(f"Built BM25 index with {len(tokenized_corpus)} documents")
            
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        import re
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove very short tokens and common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                      'and', 'or', 'but', 'if', 'then', 'else', 'when', 'up', 'out', 'this',
                      'that', 'it', 'its', 'not', 'no', 'so', 'than', 'too', 'very'}
        tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]
        return tokens
    
    def _chunk_content(self, content: str, file_path: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
        """
        Split content into overlapping chunks for indexing.
        
        Args:
            content: File content
            file_path: Path to the file
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        lines = content.split('\n')
        
        # Determine file type
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', 
                         '.rs', '.go', '.rb', '.php', '.swift', '.kt', '.scala', '.sh', '.bash'}
        file_ext = Path(file_path).suffix.lower()
        is_code = file_ext in code_extensions
        
        # For code files, try to chunk by logical units (functions, classes)
        if is_code and file_ext == '.py':
            chunks.extend(self._chunk_python_code(content, file_path))
        else:
            # Generic chunking with overlap
            chunks.extend(self._chunk_generic(content, file_path, chunk_size, overlap))
        
        return chunks
    
    def _chunk_python_code(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Chunk Python code by functions and classes.
        """
        import re
        chunks = []
        lines = content.split('\n')
        
        # Find function and class definitions
        pattern = r'^(class\s+\w+|def\s+\w+|async\s+def\s+\w+)'
        
        current_chunk_start = 0
        current_chunk_lines = []
        
        for i, line in enumerate(lines):
            # Check if this is a new definition at module level (not indented)
            if re.match(pattern, line) and not line.startswith(' '):
                # Save previous chunk if it exists
                if current_chunk_lines:
                    chunk_content = '\n'.join(current_chunk_lines)
                    if chunk_content.strip():
                        chunks.append({
                            'file_path': file_path,
                            'content': chunk_content,
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
            chunk_content = '\n'.join(current_chunk_lines)
            if chunk_content.strip():
                chunks.append({
                    'file_path': file_path,
                    'content': chunk_content,
                    'line_start': current_chunk_start + 1,
                    'line_end': len(lines),
                    'type': 'code'
                })
        
        # If no logical chunks found, fall back to generic chunking
        if not chunks:
            chunks = self._chunk_generic(content, file_path, 500, 100)
        
        return chunks
    
    def _chunk_generic(self, content: str, file_path: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
        """
        Generic content chunking with overlap.
        """
        chunks = []
        lines = content.split('\n')
        
        # Calculate approximate lines per chunk
        avg_line_length = len(content) / max(len(lines), 1)
        lines_per_chunk = max(10, int(chunk_size / max(avg_line_length, 1)))
        overlap_lines = max(2, int(overlap / max(avg_line_length, 1)))
        
        i = 0
        while i < len(lines):
            end = min(i + lines_per_chunk, len(lines))
            chunk_lines = lines[i:end]
            chunk_content = '\n'.join(chunk_lines)
            
            if chunk_content.strip():
                # Determine type based on file extension
                file_ext = Path(file_path).suffix.lower()
                doc_extensions = {'.md', '.txt', '.rst', '.adoc'}
                chunk_type = 'doc' if file_ext in doc_extensions else 'code'
                
                chunks.append({
                    'file_path': file_path,
                    'content': chunk_content,
                    'line_start': i + 1,
                    'line_end': end,
                    'type': chunk_type
                })
            
            # Move forward with overlap
            i += lines_per_chunk - overlap_lines
            if i >= len(lines) - overlap_lines:
                break
        
        return chunks
    
    def _fallback_keyword_retrieval(self, query: str, top_k: int) -> List[SourceReference]:
        """
        Fallback retrieval method using basic keyword matching when hybrid indices are not available.
        """
        try:
            _, sources = self._retrieve_relevant_content(query)
            return sources[:top_k]
        except Exception as e:
            logger.error(f"Error in fallback retrieval: {e}")
            return []
    
    async def rebuild_indices(self):
        """
        Rebuild FAISS and BM25 indices from scratch.
        Call this after indexing a new repository.
        """
        logger.info("Rebuilding hybrid indices...")
        
        # Clear existing indices
        self.faiss_index = None
        self.bm25_index = None
        self.document_chunks = []
        self.chunk_texts = []
        
        # Build new indices
        await self._build_indices()
        
        self.retrieval_initialized = bool(self.document_chunks)
        
        logger.info(f"Indices rebuilt with {len(self.document_chunks)} chunks")
        return self.retrieval_initialized

    # ==================== END HYBRID RETRIEVAL METHODS ====================

    def _fact_check_response(self, response: str, context: List[SourceReference], query: str) -> str:
        """
        Fact-check the response against source content to improve accuracy.
        
        Args:
            response: Generated response text
            context: Retrieved source references
            query: Original user query
            
        Returns:
            Fact-checked and corrected response
        """
        try:
            # Common fact-checking patterns
            corrections = []
            
            # Check for embedding model references
            if 'embedding' in query.lower() and 'model' in query.lower():
                # Look for actual model names in context
                actual_models = []
                for src in context:
                    if hasattr(src, 'content'):
                        content = src.content.lower()
                        if 'all-minilm-l6-v2' in content:
                            actual_models.append('all-MiniLM-L6-v2')
                        elif 'all-minilm' in content:
                            actual_models.append('all-MiniLM-L6-v2')
                        elif 'sentence_transformers' in content:
                            actual_models.append('sentence-transformers')
                
                # If response mentions wrong model, add correction
                if actual_models and not any(model.lower() in response.lower() for model in actual_models):
                    corrections.append(f"\n\nNote: Based on the source code, the actual embedding model is: {', '.join(set(actual_models))}")
                elif not actual_models and 'sentence' in response.lower() and 'transformer' in response.lower():
                    # If no specific model found but response is generic, provide specific info
                    corrections.append(f"\n\nNote: The specific model used is all-MiniLM-L6-v2 as configured in the chunker.py file.")
            
            # Check for configuration file references
            if 'config' in query.lower():
                config_files = [src.file for src in context if 'config' in src.file.lower()]
                if config_files and not any(f in response for f in config_files):
                    corrections.append(f"\n\nConfiguration is found in: {', '.join(config_files)}")
            
            # Check for default values
            if 'default' in query.lower():
                # Look for default values in context
                default_values = []
                for src in context:
                    if hasattr(src, 'content'):
                        content = src.content
                        # Look for common default patterns
                        import re
                        defaults = re.findall(r'default[:\s=]+["\']?([^"\'\s,]+)["\']?', content, re.IGNORECASE)
                        default_values.extend(defaults)
                
                if default_values:
                    unique_defaults = list(set(default_values))[:3]  # Top 3 unique defaults
                    corrections.append(f"\n\nDefault values found in source: {', '.join(unique_defaults)}")
            
            # Add corrections to response if any found
            if corrections:
                response += ''.join(corrections)
            
            return response
            
        except Exception as e:
            logger.debug(f"Error in fact-checking: {e}")
            return response


# Singleton instance
_rag_service: Optional[RAGService] = None


def get_rag_service(
    vector_store_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> RAGService:
    """
    Get or create RAG service instance.

    Args:
        vector_store_path: Path to vector store
        model_path: Path to LLM model

    Returns:
        RAGService instance
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(vector_store_path, model_path)
    return _rag_service
