"""Main FastAPI application."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import chat, health, indexing, search
from src.config import settings
from src.services.rag_service import get_rag_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - runs on startup and shutdown."""
    # Startup: Initialize RAG service
    logger.info("="*60)
    logger.info("Starting RAG GitHub Assistant")
    logger.info("="*60)
    
    rag_service = get_rag_service(
        model_path=str(settings.get_model_path())
    )
    
    logger.info(f"Model: {settings.MODEL_NAME}")
    logger.info(f"Path: {settings.get_model_path()}")
    
    # Try to initialize the model
    if await rag_service.initialize():
        logger.info("RAG Service initialized successfully - Model ready for queries")
    else:
        logger.error("RAG Service initialization failed - Check logs for details")
        logger.warning("Server will start but queries will return errors until fixed")
    
    logger.info("="*60)
    
    yield
    
    # Shutdown: Cleanup if needed
    logger.info("Shutting down RAG GitHub Assistant")


app = FastAPI(
    title="RAG GitHub Assistant",
    version="0.1.0",
    description="RAG-powered GitHub repository code assistant",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with /api prefix
app.include_router(health.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(indexing.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
