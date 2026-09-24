from pydantic import BaseModel
from typing import Literal


class HealthResponse(BaseModel):
    status: str


class UploadResponse(BaseModel):
    document_id: int
    filename: str
    chunking_strategy: Literal["fixed", "recursive"]
    chunk_count: int


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str