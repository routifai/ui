"""
Request Models - Pydantic models for API requests
"""

from typing import Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Request model for general chat queries"""
    message: str = Field(..., description="User message to send to LLM", min_length=1, max_length=1000)
    temperature: Optional[float] = Field(default=0.7, description="Temperature for LLM response", ge=0.0, le=2.0)
    model: Optional[str] = Field(default="gpt-4o-mini", description="OpenAI model to use")

class QueryChunksRequest(BaseModel):
    """Request model for searching document chunks"""
    query: str = Field(..., description="Query to search for relevant chunks")
    top_k: int = Field(default=5, description="Number of chunks to retrieve", ge=1, le=20)

class RAGQuestionRequest(BaseModel):
    """Request model for RAG questions"""
    query: str = Field(..., description="Question to ask about uploaded documents") 