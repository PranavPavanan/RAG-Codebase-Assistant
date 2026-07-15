from pydantic import BaseModel, Field
from typing import List, Optional

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
    updated_at: str
    created_at: str

class Repository(RepositoryBase):
    clone_url: str
    ssh_url: str
    open_issues: int
    watchers: int
    license: str
    is_private: bool
    is_fork: bool
    has_wiki: bool
    has_issues: bool

class RepositorySearchRequest(BaseModel):
    query: str
    limit: int = 10

class RepositorySearchResponse(BaseModel):
    repositories: List[RepositoryBase]
    total_count: int
    page: int

class RepositoryValidationRequest(BaseModel):
    url: str

class RepositoryValidationResponse(BaseModel):
    valid: bool
    message: str
    repository_info: Optional[RepositoryBase] = None
