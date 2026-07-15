from enum import Enum
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class IndexingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class IndexStartRequest(BaseModel):
    repository_url: str
    branch: Optional[str] = None

class IndexStartResponse(BaseModel):
    task_id: str
    status: IndexingStatus
    message: str
    repository_url: str
    estimated_time: int
    created_at: datetime

class IndexProgressInfo(BaseModel):
    files_processed: int
    total_files: int
    percentage: float

class IndexStatusResponse(BaseModel):
    task_id: str
    status: IndexingStatus
    message: str
    progress: IndexProgressInfo
    percentage: float
    repository_url: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[dict] = None

class IndexStats(BaseModel):
    is_indexed: bool
    repository_name: Optional[str] = None
    file_count: int
    total_size: int
    vector_count: int
    last_updated: Optional[datetime] = None
    created_at: Optional[datetime] = None

class FileIndexEntry(BaseModel):
    file_path: str
    content_hash: str
    size: int
    language: str
    chunk_count: int
    indexed_at: datetime

class IndexClearResponse(BaseModel):
    success: bool
    message: str
    files_removed: int
    space_freed: int
