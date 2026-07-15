from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    message: str

class ErrorResponse(BaseModel):
    detail: str

class NotFoundResponse(BaseModel):
    detail: str
