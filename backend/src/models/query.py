from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime

class SourceReference(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float

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

class ChatHistoryRequest(BaseModel):
    conversation_id: str
    limit: Optional[int] = None

class ChatHistoryResponse(BaseModel):
    conversation_id: str
    messages: List[ChatMessage]
    total_messages: int
    created_at: datetime
    total_count: Optional[int] = None

class ChatContextResponse(BaseModel):
    conversation_id: str
    message_count: int
    user_message_count: int
    assistant_message_count: int
    last_query: Optional[str] = None
    last_response: Optional[str] = None
    created_at: datetime
    last_updated: datetime

class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    last_activity: datetime
    conversation_count: int
    total_messages: int

class ConversationInfo(BaseModel):
    conversation_id: str
    session_id: str
    created_at: datetime
    message_count: int

class SessionClearRequest(BaseModel):
    session_id: str

class SessionClearResponse(BaseModel):
    success: bool
    cleared_count: int
