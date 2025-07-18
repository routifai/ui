import os
import logging
import uuid
import hashlib
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from instructor import patch
from dotenv import load_dotenv
from pathlib import Path
import json
import asyncio

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

# Pydantic models for request/response
class QueryRequest(BaseModel):
    message: str = Field(..., description="User message to send to LLM", min_length=1, max_length=1000)
    temperature: Optional[float] = Field(default=0.7, description="Temperature for LLM response", ge=0.0, le=2.0)
    model: Optional[str] = Field(default="gpt-4o-mini", description="OpenAI model to use")

class QueryResponse(BaseModel):
    response: str = Field(..., description="LLM response")
    model: str = Field(..., description="Model used for the response")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    context_type: Optional[str] = Field(None, description="Type of context used: 'document' or 'general'")

# RAG-specific models
class DocumentChunk(BaseModel):
    content: str = Field(..., description="Document chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    similarity_score: Optional[float] = Field(None, description="Similarity score")

class QueryChunksResponse(BaseModel):
    chunks: List[DocumentChunk] = Field(..., description="Relevant document chunks")
    query: str = Field(..., description="Original query")
    total_chunks: int = Field(..., description="Total number of chunks returned")

class ProcessingStatus(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Processing status: pending, processing, completed, failed")
    message: Optional[str] = Field(None, description="Status message")
    document_id: Optional[str] = Field(None, description="Document ID if completed")
    chunks_count: Optional[int] = Field(None, description="Number of chunks created")

# Global storage for RAG system
rag_documents: Dict[str, Dict[str, Any]] = {}
processing_jobs: Dict[str, Dict[str, Any]] = {}

# Dependency to check if OpenAI client is available
def get_openai_client():
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not available")
    return client

# RAG System Functions
def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

async def process_pdf_with_llama(job_id: str, file_path: str, filename: str):
    """Process PDF file with Llama Parse and generate embeddings"""
    try:
        # Update job status
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["message"] = "Step 1: Parsing document with Llama Parse..."
        
        # Step 1: Parse document with Llama Parse
        try:
            from llama_parse import LlamaParse
            parser = LlamaParse(api_key="llx-7mGfU1RM0lDW271oEGQlv6pYSPpWf6N9kpDGQKBjvnCkHSkq")
            
            # Llama Parse extracts text from PDF
            parsed_documents = parser.load_data(file_path)
            
            if parsed_documents:
                # Extract clean text from Llama Parse results
                full_text = "\n\n".join([doc.text for doc in parsed_documents])
                
                # Update status
                processing_jobs[job_id]["message"] = "Step 2: Chunking extracted text..."
                
                # Step 2: Chunk the extracted text
                chunks = chunk_text(full_text)
                
                # Update status
                processing_jobs[job_id]["message"] = "Step 3: Generating embeddings with OpenAI..."
                
                # Step 3: Generate embeddings for chunks using OpenAI
                document_id = hashlib.md5(full_text.encode()).hexdigest()
                
                rag_documents[document_id] = {
                    "content": full_text,
                    "chunks": chunks,
                    "embeddings": [],
                    "metadata": {
                        "filename": filename,
                        "source": "pdf_upload",
                        "chunks_count": len(chunks),
                        "job_id": job_id,
                        "parsed_with": "llama_parse"
                    }
                }
                
                # Generate OpenAI embeddings for each chunk
                for i, chunk in enumerate(chunks):
                    embedding = generate_embedding(chunk)
                    rag_documents[document_id]["embeddings"].append(embedding)
                    
                    # Update progress
                    if i % 5 == 0:  # Update every 5 chunks
                        processing_jobs[job_id]["message"] = f"Step 3: Generated embeddings for {i+1}/{len(chunks)} chunks..."
                
                # Update job status
                processing_jobs[job_id]["status"] = "completed"
                processing_jobs[job_id]["message"] = f"Successfully processed: {len(chunks)} chunks extracted and embedded"
                processing_jobs[job_id]["document_id"] = document_id
                processing_jobs[job_id]["chunks_count"] = len(chunks)
                
                logger.info(f"Successfully processed PDF: {filename} -> {len(chunks)} chunks with embeddings")
                
            else:
                raise Exception("No content extracted from PDF by Llama Parse")
                
        except ImportError:
            raise Exception("Llama Parse not available - install with: pip install llama-parse")
            
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["message"] = f"Error: {str(e)}"
    
    finally:
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "LLM Chat API"}

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    openai_client: OpenAI = Depends(get_openai_client)
):
    """
    Process a user query and return LLM response.
    
    Args:
        request: QueryRequest containing the user message and optional parameters
        openai_client: OpenAI client dependency
    
    Returns:
        QueryResponse containing the LLM response and metadata
    """
    try:
        logger.info(f"Processing query: {request.message[:50]}...")
        
        # Send message to OpenAI
        response = openai_client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "user", "content": request.message}
            ],
            temperature=request.temperature
        )
        
        # Extract response
        llm_response = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else None
        
        logger.info(f"Successfully processed query with {tokens_used} tokens")
        
        return QueryResponse(
            response=llm_response,
            model=request.model,
            tokens_used=tokens_used,
            context_type="general"
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# RAG Endpoints

@app.post("/rag/upload-pdf", response_model=ProcessingStatus)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    openai_client: OpenAI = Depends(get_openai_client)
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
        openai_client: OpenAI client dependency
    
    Returns:
        ProcessingStatus with job ID and initial status
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create temporary file path
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / f"{job_id}_{file.filename}"
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize job status
        processing_jobs[job_id] = {
            "status": "pending",
            "message": "Job created, starting processing...",
            "filename": file.filename,
            "file_path": str(file_path)
        }
        
        # Start background processing
        background_tasks.add_task(process_pdf_with_llama, job_id, str(file_path), file.filename)
        
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
async def get_processing_status(job_id: str):
    """
    Get the processing status of a PDF upload job.
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        ProcessingStatus with current status and details
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    return ProcessingStatus(
        job_id=job_id,
        status=job["status"],
        message=job.get("message"),
        document_id=job.get("document_id"),
        chunks_count=job.get("chunks_count")
    )

class QueryChunksRequest(BaseModel):
    query: str = Field(..., description="Query to search for relevant chunks")
    top_k: int = Field(default=5, description="Number of chunks to retrieve", ge=1, le=20)

class RAGQuestionRequest(BaseModel):
    query: str = Field(..., description="Question to ask about uploaded documents")

@app.post("/rag/query-chunks", response_model=QueryChunksResponse)
async def query_chunks(
    request: QueryChunksRequest,
    openai_client: OpenAI = Depends(get_openai_client)
):
    """
    Search for relevant document chunks based on a query.
    
    Args:
        query: Search query
        top_k: Number of chunks to retrieve
        openai_client: OpenAI client dependency
    
    Returns:
        QueryChunksResponse with relevant chunks and similarity scores
    """
    try:
        if not rag_documents:
            raise HTTPException(status_code=404, detail="No documents available for search")
        
        # Generate embedding for query
        query_embedding = generate_embedding(request.query)
        
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        # Search through all documents and chunks
        similarities = []
        
        for doc_id, doc_data in rag_documents.items():
            chunks = doc_data["chunks"]
            embeddings = doc_data["embeddings"]
            metadata = doc_data["metadata"]
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if embedding:  # Only process chunks with valid embeddings
                    similarity = cosine_similarity(query_embedding, embedding)
                    similarities.append({
                        "content": chunk,
                        "metadata": {**metadata, "chunk_index": i, "document_id": doc_id},
                        "similarity_score": similarity
                    })
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_chunks = similarities[:request.top_k]
        
        # Convert to response format
        chunks = [
            DocumentChunk(
                content=chunk["content"],
                metadata=chunk["metadata"],
                similarity_score=chunk["similarity_score"]
            )
            for chunk in top_chunks
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
    openai_client: OpenAI = Depends(get_openai_client)
):
    """
    Ask a question about uploaded documents using RAG.
    
    Process:
    1. Retrieve relevant chunks using semantic search
    2. Pass chunks as context to LLM
    3. Generate answer based on document content
    
    Args:
        query: Question to ask
        openai_client: OpenAI client dependency
    
    Returns:
        QueryResponse with answer based on document context
    """
    try:
        if not rag_documents:
            # No documents available - answer with general knowledge
            logger.info(f"No documents available for query: {request.query[:50]}... - using general knowledge")
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the user's question using your general knowledge. Be informative and helpful."},
                    {"role": "user", "content": request.query}
                ],
                temperature=0.7
            )
            
            llm_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            return QueryResponse(
                response=llm_response,
                model="gpt-4o-mini",
                tokens_used=tokens_used,
                context_type="general"
            )
        
        # Step 1: Retrieve relevant chunks
        query_embedding = generate_embedding(request.query)
        
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        # Search through all documents and chunks
        similarities = []
        
        for doc_id, doc_data in rag_documents.items():
            chunks = doc_data["chunks"]
            embeddings = doc_data["embeddings"]
            metadata = doc_data["metadata"]
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if embedding:
                    similarity = cosine_similarity(query_embedding, embedding)
                    similarities.append({
                        "content": chunk,
                        "metadata": {**metadata, "chunk_index": i, "document_id": doc_id},
                        "similarity_score": similarity
                    })
        
        # Get top 3 most relevant chunks
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_chunks = similarities[:3]
        
        # Check if we have relevant chunks (similarity threshold)
        relevant_chunks = [chunk for chunk in top_chunks if chunk["similarity_score"] > 0.3]
        
        if not relevant_chunks:
            # No relevant document context found - answer with general knowledge
            logger.info(f"No relevant document context found for query: {request.query[:50]}... - using general knowledge")
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the user's question using your general knowledge. Be informative and helpful."},
                    {"role": "user", "content": request.query}
                ],
                temperature=0.7
            )
            
            llm_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            return QueryResponse(
                response=llm_response,
                model="gpt-4o-mini",
                tokens_used=tokens_used,
                context_type="general"
            )
        
        # Step 2: Build context from relevant chunks
        context_parts = []
        for i, chunk_data in enumerate(relevant_chunks, 1):
            context_parts.append(f"Document chunk {i}:\n{chunk_data['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Create RAG prompt with fallback instruction
        rag_prompt = f"""Based on the following document context, answer the user's question. 
        If the context doesn't contain enough information to answer the question completely, 
        you can supplement with your general knowledge, but prioritize information from the documents.
        Provide a comprehensive and accurate answer.

        Context:
        {context}

        Question: {request.query}

        Answer:"""
        
        # Step 4: Generate response using LLM
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document context. Always provide accurate and relevant information."},
                {"role": "user", "content": rag_prompt}
            ],
            temperature=0.7
        )
        
        llm_response = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else None
        
        logger.info(f"Generated RAG response for query: {request.query[:50]}... with {tokens_used} tokens")
        
        return QueryResponse(
            response=llm_response,
            model="gpt-4o-mini",
            tokens_used=tokens_used,
            context_type="document"
        )
        
    except Exception as e:
        logger.error(f"Error in RAG question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing RAG question: {str(e)}")

@app.get("/rag/documents")
async def list_documents():
    """
    List all processed documents in the RAG system.
    
    Returns:
        List of document metadata
    """
    documents = []
    
    for doc_id, doc_data in rag_documents.items():
        documents.append({
            "document_id": doc_id,
            "filename": doc_data["metadata"]["filename"],
            "chunks_count": doc_data["metadata"]["chunks_count"],
            "source": doc_data["metadata"]["source"],
            "upload_time": doc_data["metadata"].get("upload_time")
        })
    
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
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"General Exception: {exc}")
    from fastapi.responses import JSONResponse
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
        "server:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
        log_level="info"
    ) 