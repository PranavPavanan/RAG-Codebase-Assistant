from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional

class RepositoryBase(BaseModel):
    id: str
    name: str
    full_name: str
    description: Optional[str] = None
    url: str
    html_url: str
    stars: int
    stargazers_count: int
    forks: int
    language: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    owner: str
    default_branch: str
    size: int
    updated_at: Optional[str] = None
    created_at: Optional[str] = None

    @field_validator('updated_at', 'created_at', mode='before')
    @classmethod
    def coerce_datetime_to_str(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        # Handle datetime objects returned by PyGithub
        return v.isoformat()

class RepositorySearchRequest(BaseModel):
    query: str
    username: Optional[str] = None
    limit: int = 10

class RepositoryTrendingRequest(BaseModel):
    keyword: Optional[str] = None
    language: Optional[str] = None
    topic: Optional[str] = None
    min_stars: Optional[int] = 1000
    limit: int = 10

class RepositorySearchResponse(BaseModel):
    repositories: List[RepositoryBase]
    total_count: int
    page: int

