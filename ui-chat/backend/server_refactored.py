"""
Refactored FastAPI Server - Clean separation of concerns
Uses service layer for business logic and models for data validation
"""

import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from instructor import patch
from dotenv import load_dotenv
from pathlib import Path

# Import our services and models
from services.rag_service import RAGService
from services.chat_service import ChatService
from models.requests import QueryRequest, QueryChunksRequest, RAGQuestionRequest
from models.responses import QueryResponse, QueryChunksResponse, ProcessingStatus

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Chat API",
    description="A FastAPI server that handles user queries and returns LLM responses",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client with instructor
try:
    client = patch(OpenAI())
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None

# Initialize services
rag_service = None
chat_service = None

def initialize_services():
    """Initialize service instances"""
    global rag_service, chat_service
    if client:
        rag_service = RAGService(client)
        chat_service = ChatService(client)
        logger.info("Services initialized successfully")
    else:
        logger.error("Failed to initialize services - OpenAI client not available")

# Dependency to check if services are available
def get_services():
    if not rag_service or not chat_service:
        raise HTTPException(status_code=503, detail="Services not available")
    return rag_service, chat_service

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    initialize_services()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "services": {
        "rag_service": rag_service is not None,
        "chat_service": chat_service is not None
    }}

# Chat endpoints
@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    services: tuple = Depends(get_services)
):
    """
    Process a user query and return LLM response.
    
    Args:
        request: QueryRequest containing the user message and optional parameters
        services: Tuple of (rag_service, chat_service)
    
    Returns:
        QueryResponse containing the LLM response and metadata
    """
    try:
        rag_service, chat_service = services
        logger.info(f"Processing query: {request.message[:50]}...")
        
        # Generate response using chat service
        result = chat_service.generate_response(
            message=request.message,
            temperature=request.temperature,
            model=request.model
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# RAG endpoints
@app.post("/rag/upload-pdf", response_model=ProcessingStatus)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    services: tuple = Depends(get_services)
):
    """
    Upload a PDF file for processing.
    
    Process:
    1. Llama Parse extracts text from PDF
    2. Text is chunked into smaller pieces
    3. OpenAI generates embeddings for each chunk
    4. Chunks are stored for semantic search
    
    Args:
        file: PDF file to upload
        background_tasks: FastAPI background tasks
        services: Tuple of (rag_service, chat_service)
    
    Returns:
        ProcessingStatus with job ID and initial status
    """
    try:
        rag_service, chat_service = services
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create temporary file path
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Create job and save file
        job_id = rag_service.create_job(file.filename, "")
        file_path = temp_dir / f"{job_id}_{file.filename}"
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Update job with file path
        rag_service.processing_jobs[job_id]["file_path"] = str(file_path)
        
        # Start background processing
        background_tasks.add_task(rag_service.process_pdf, job_id, str(file_path), file.filename)
        
        logger.info(f"Started PDF processing job: {job_id} for file: {file.filename}")
        
        return ProcessingStatus(
            job_id=job_id,
            status="pending",
            message="PDF upload accepted, processing started"
        )
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

@app.get("/rag/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str, services: tuple = Depends(get_services)):
    """
    Get the processing status of a PDF upload job.
    
    Args:
        job_id: Unique job identifier
        services: Tuple of (rag_service, chat_service)
    
    Returns:
        ProcessingStatus with current status and details
    """
    rag_service, chat_service = services
    job = rag_service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ProcessingStatus(
        job_id=job_id,
        status=job["status"],
        message=job.get("message"),
        document_id=job.get("document_id"),
        chunks_count=job.get("chunks_count")
    )

@app.post("/rag/query-chunks", response_model=QueryChunksResponse)
async def query_chunks(
    request: QueryChunksRequest,
    services: tuple = Depends(get_services)
):
    """
    Search for relevant document chunks based on a query.
    
    Args:
        request: QueryChunksRequest with search query and parameters
        services: Tuple of (rag_service, chat_service)
    
    Returns:
        QueryChunksResponse with relevant chunks and similarity scores
    """
    try:
        rag_service, chat_service = services
        
        if not rag_service.documents:
            raise HTTPException(status_code=404, detail="No documents available for search")
        
        # Search for relevant chunks
        chunks_data = rag_service.search_chunks(request.query, request.top_k)
        
        # Convert to response format
        chunks = [
            DocumentChunk(
                content=chunk["content"],
                metadata=chunk["metadata"],
                similarity_score=chunk["similarity_score"]
            )
            for chunk in chunks_data
        ]
        
        logger.info(f"Retrieved {len(chunks)} chunks for query: {request.query[:50]}...")
        
        return QueryChunksResponse(
            chunks=chunks,
            query=request.query,
            total_chunks=len(chunks)
        )
        
    except Exception as e:
        logger.error(f"Error querying chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying chunks: {str(e)}")

@app.post("/rag/ask", response_model=QueryResponse)
async def ask_rag_question(
    request: RAGQuestionRequest,
    services: tuple = Depends(get_services)
):
    """
    Ask a question about uploaded documents using RAG.
    
    Process:
    1. Retrieve relevant chunks using semantic search
    2. Pass chunks as context to LLM
    3. Generate answer based on document content
    
    Args:
        request: RAGQuestionRequest with the question
        services: Tuple of (rag_service, chat_service)
    
    Returns:
        QueryResponse with answer based on document context
    """
    try:
        rag_service, chat_service = services
        
        # Get relevant chunks
        relevant_chunks = rag_service.get_relevant_chunks(request.query)
        
        # Generate RAG response
        result = chat_service.generate_rag_response(request.query, relevant_chunks)
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in RAG question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing RAG question: {str(e)}")

@app.get("/rag/documents")
async def list_documents(services: tuple = Depends(get_services)):
    """
    List all processed documents in the RAG system.
    
    Args:
        services: Tuple of (rag_service, chat_service)
    
    Returns:
        List of document metadata
    """
    rag_service, chat_service = services
    documents = rag_service.list_documents()
    
    return {"documents": documents, "total": len(documents)}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LLM Chat API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"General Exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        exit(1)
    
    # Run the server
    uvicorn.run(
        "server_refactored:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
        log_level="info"
    ) 