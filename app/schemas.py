"""Typed API contracts (Pydantic) — validation + auto-generated docs."""
from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source: str
    page_number: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
