"""
Response Models - Pydantic models for API responses
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class QueryResponse(BaseModel):
    """Response model for chat queries"""
    response: str = Field(..., description="LLM response")
    model: str = Field(..., description="Model used for the response")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    context_type: Optional[str] = Field(None, description="Type of context used: 'document' or 'general'")

class DocumentChunk(BaseModel):
    """Model for document chunks"""
    content: str = Field(..., description="Document chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    similarity_score: Optional[float] = Field(None, description="Similarity score")

class QueryChunksResponse(BaseModel):
    """Response model for chunk queries"""
    chunks: List[DocumentChunk] = Field(..., description="Relevant document chunks")
    query: str = Field(..., description="Original query")
    total_chunks: int = Field(..., description="Total number of chunks returned")

class ProcessingStatus(BaseModel):
    """Response model for processing status"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Processing status: pending, processing, completed, failed")
    message: Optional[str] = Field(None, description="Status message")
    document_id: Optional[str] = Field(None, description="Document ID if completed")
    chunks_count: Optional[int] = Field(None, description="Number of chunks created") 