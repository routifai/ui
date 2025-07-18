"""
RAG Service - Handles all RAG-related business logic
Separates business logic from API endpoints for better maintainability
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai import OpenAI
from llama_parse import LlamaParse

logger = logging.getLogger(__name__)

class RAGService:
    """Service class for RAG (Retrieval-Augmented Generation) operations"""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.processing_jobs: Dict[str, Dict[str, Any]] = {}
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    async def process_pdf(self, job_id: str, file_path: str, filename: str):
        """Process PDF file with Llama Parse and generate embeddings"""
        try:
            # Update job status
            self.processing_jobs[job_id]["status"] = "processing"
            self.processing_jobs[job_id]["message"] = "Step 1: Parsing document with Llama Parse..."
            
            # Step 1: Parse document with Llama Parse
            try:
                parser = LlamaParse(api_key="llx-7mGfU1RM0lDW271oEGQlv6pYSPpWf6N9kpDGQKBjvnCkHSkq")
                
                # Llama Parse extracts text from PDF
                parsed_documents = parser.load_data(file_path)
                
                if parsed_documents:
                    # Extract clean text from Llama Parse results
                    full_text = "\n\n".join([doc.text for doc in parsed_documents])
                    
                    # Update status
                    self.processing_jobs[job_id]["message"] = "Step 2: Chunking extracted text..."
                    
                    # Step 2: Chunk the extracted text
                    chunks = self.chunk_text(full_text)
                    
                    # Update status
                    self.processing_jobs[job_id]["message"] = "Step 3: Generating embeddings with OpenAI..."
                    
                    # Step 3: Generate embeddings for chunks using OpenAI
                    document_id = hashlib.md5(full_text.encode()).hexdigest()
                    
                    self.documents[document_id] = {
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
                        embedding = self.generate_embedding(chunk)
                        self.documents[document_id]["embeddings"].append(embedding)
                        
                        # Update progress
                        if i % 5 == 0:  # Update every 5 chunks
                            self.processing_jobs[job_id]["message"] = f"Step 3: Generated embeddings for {i+1}/{len(chunks)} chunks..."
                    
                    # Update job status
                    self.processing_jobs[job_id]["status"] = "completed"
                    self.processing_jobs[job_id]["message"] = f"Successfully processed: {len(chunks)} chunks extracted and embedded"
                    self.processing_jobs[job_id]["document_id"] = document_id
                    self.processing_jobs[job_id]["chunks_count"] = len(chunks)
                    
                    logger.info(f"Successfully processed PDF: {filename} -> {len(chunks)} chunks with embeddings")
                    
                else:
                    raise Exception("No content extracted from PDF by Llama Parse")
                    
            except ImportError:
                raise Exception("Llama Parse not available - install with: pip install llama-parse")
                
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            self.processing_jobs[job_id]["status"] = "failed"
            self.processing_jobs[job_id]["message"] = f"Error: {str(e)}"
        
        finally:
            # Clean up temporary file
            try:
                Path(file_path).unlink()
            except:
                pass
    
    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks based on a query"""
        if not self.documents:
            return []
        
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        
        if not query_embedding:
            return []
        
        # Search through all documents and chunks
        similarities = []
        
        for doc_id, doc_data in self.documents.items():
            chunks = doc_data["chunks"]
            embeddings = doc_data["embeddings"]
            metadata = doc_data["metadata"]
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if embedding:
                    similarity = self.cosine_similarity(query_embedding, embedding)
                    similarities.append({
                        "content": chunk,
                        "metadata": {**metadata, "chunk_index": i, "document_id": doc_id},
                        "similarity_score": similarity
                    })
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarities[:top_k]
    
    def get_relevant_chunks(self, query: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Get relevant chunks above similarity threshold"""
        chunks = self.search_chunks(query, top_k=3)
        return [chunk for chunk in chunks if chunk["similarity_score"] > threshold]
    
    def create_job(self, filename: str, file_path: str) -> str:
        """Create a new processing job"""
        import uuid
        job_id = str(uuid.uuid4())
        
        self.processing_jobs[job_id] = {
            "status": "pending",
            "message": "Job created, starting processing...",
            "filename": filename,
            "file_path": file_path
        }
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a processing job"""
        return self.processing_jobs.get(job_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents"""
        documents = []
        
        for doc_id, doc_data in self.documents.items():
            documents.append({
                "document_id": doc_id,
                "filename": doc_data["metadata"]["filename"],
                "chunks_count": doc_data["metadata"]["chunks_count"],
                "source": doc_data["metadata"]["source"],
                "upload_time": doc_data["metadata"].get("upload_time")
            })
        
        return documents 