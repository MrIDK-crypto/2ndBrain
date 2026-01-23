"""
Search and RAG schemas for request/response validation
"""

from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class SearchRequest(BaseModel):
    """Schema for search/RAG queries"""
    query: str = Field(min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    source_types: Optional[List[str]] = Field(None, description="Filter by source types")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    include_content: bool = Field(default=True, description="Include document content")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "project timeline Q1",
                "top_k": 10,
                "threshold": 0.7,
                "source_types": ["gmail", "slack"]
            }
        }
    )


class SearchResult(BaseModel):
    """Single search result"""
    document_id: UUID
    title: Optional[str]
    content_snippet: str = Field(description="Relevant content snippet")
    source_type: str
    source_url: Optional[str]
    similarity_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any]
    created_at: datetime


class SearchResponse(BaseModel):
    """Search results response"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: int = Field(description="Search time in milliseconds")


class QuestionRequest(BaseModel):
    """Schema for Q&A queries"""
    question: str = Field(min_length=1, max_length=2000, description="Question to answer")
    context_docs: int = Field(default=5, ge=1, le=20, description="Number of context documents")
    include_sources: bool = Field(default=True, description="Include source documents")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What are the main project milestones?",
                "context_docs": 5
            }
        }
    )


class AnswerResponse(BaseModel):
    """Q&A response"""
    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[SearchResult] = Field(description="Source documents used")
    processing_time_ms: int


class QuestionGenerateRequest(BaseModel):
    """Schema for generating questions from documents"""
    document_ids: Optional[List[UUID]] = Field(None, max_length=10)
    topic: Optional[str] = Field(None, max_length=500)
    count: int = Field(default=5, ge=1, le=20)


class GeneratedQuestion(BaseModel):
    """A generated question"""
    question: str
    category: Optional[str]
    difficulty: Optional[str]
    source_document_id: Optional[UUID]


class QuestionGenerateResponse(BaseModel):
    """Generated questions response"""
    questions: List[GeneratedQuestion]
    topic: Optional[str]


class StakeholderQueryRequest(BaseModel):
    """Schema for stakeholder queries"""
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=50)


class StakeholderResult(BaseModel):
    """Stakeholder search result"""
    name: str
    email: Optional[str]
    role: Optional[str]
    expertise: List[str]
    relevance_score: float
    related_documents: int


class StakeholderQueryResponse(BaseModel):
    """Stakeholder query response"""
    query: str
    stakeholders: List[StakeholderResult]
    total: int
