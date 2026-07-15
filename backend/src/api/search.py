"""Repository search and validation endpoints."""
from fastapi import APIRouter, HTTPException

from src.models.repository import (
    RepositorySearchRequest,
    RepositorySearchResponse,
    RepositoryTrendingRequest,
    RepositoryValidationRequest,
    RepositoryValidationResponse,
)
from src.models.response import ErrorResponse
from src.services import get_github_service

router = APIRouter()


@router.post(
    "/search/repositories",
    response_model=RepositorySearchResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["search"],
)
async def search_repositories(
    request: RepositorySearchRequest,
) -> RepositorySearchResponse:
    """
    Search for GitHub repositories by exact name or owner/name.
    """
    try:
        github_service = get_github_service()
        return github_service.search_repositories(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@router.post(
    "/search/trending",
    response_model=RepositorySearchResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["search"],
)
async def search_trending_repositories(
    request: RepositoryTrendingRequest,
) -> RepositorySearchResponse:
    """
    Search for trending GitHub repositories (sorted by stars).
    """
    try:
        github_service = get_github_service()
        return github_service.search_trending_repositories(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trending search failed: {str(e)}") from e



@router.post(
    "/validate/url",
    response_model=RepositoryValidationResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["search"],
)
async def validate_repository_url(
    request: RepositoryValidationRequest,
) -> RepositoryValidationResponse:
    """
    Validate a GitHub repository URL.

    Args:
        request: Validation request with repository URL

    Returns:
        RepositoryValidationResponse with validation result

    Raises:
        HTTPException: If validation fails
    """
    try:
        github_service = get_github_service()
        return github_service.validate_repository_url(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}") from e
