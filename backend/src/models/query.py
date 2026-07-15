from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime

class SourceReference(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float
    type: Optional[str] = None

    @field_validator('score')
    @classmethod
    def check_score(cls, v):
        if not (0 <= v <= 1.0):
            raise ValueError('Score must be between 0 and 1')
        return v

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    max_sources: Optional[int] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    model_config = {'protected_namespaces': ()}

    response: str
    sources: List[SourceReference]
    conversation_id: str
    confidence: float
    processing_time: float
    model_used: str

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime
    sources: Optional[List[SourceReference]] = None

    @field_validator('role')
    @classmethod
    def check_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Invalid role')
        return v

