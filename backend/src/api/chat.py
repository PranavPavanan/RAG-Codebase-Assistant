"""Chat and query endpoints."""
from fastapi import APIRouter, HTTPException, Query

from src.models.query import (
    QueryRequest,
    QueryResponse,
)
from src.models.response import ErrorResponse, NotFoundResponse
from src.services import get_rag_service

router = APIRouter()


@router.post(
    "/chat/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["chat"],
)
async def chat_query(request: QueryRequest) -> QueryResponse:
    """
    Process a chat query using RAG pipeline.

    Args:
        request: Query request with question and optional context

    Returns:
        QueryResponse with answer and sources

    Raises:
        HTTPException: If query processing fails
    """
    try:
        rag_service = get_rag_service()
        return rag_service.query(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}") from e


