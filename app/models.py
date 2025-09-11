from pydantic import BaseModel
from typing import List, Dict, Any


class AddDocumentRequest(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]


class ChatRequest(BaseModel):
    question: str
    max_results: int = 30


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    model_used: str
    tokens_used: int