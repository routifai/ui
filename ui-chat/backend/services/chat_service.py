"""
Chat Service - Handles all chat-related business logic
Separates business logic from API endpoints for better maintainability
"""

import logging
from typing import Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class ChatService:
    """Service class for chat operations"""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def generate_response(self, message: str, temperature: float = 0.7, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Generate a chat response using OpenAI"""
        try:
            logger.info(f"Generating chat response for: {message[:50]}...")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": message}
                ],
                temperature=temperature
            )
            
            llm_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            logger.info(f"Successfully generated response with {tokens_used} tokens")
            
            return {
                "response": llm_response,
                "model": model,
                "tokens_used": tokens_used,
                "context_type": "general"
            }
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            raise Exception(f"Error generating chat response: {str(e)}")
    
    def generate_rag_response(self, query: str, relevant_chunks: list, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate a RAG response using document context"""
        try:
            if not relevant_chunks:
                # No relevant document context found - answer with general knowledge
                logger.info(f"No relevant document context found for query: {query[:50]}... - using general knowledge")
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer the user's question using your general knowledge. Be informative and helpful."},
                        {"role": "user", "content": query}
                    ],
                    temperature=temperature
                )
                
                llm_response = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else None
                
                return {
                    "response": llm_response,
                    "model": "gpt-4o-mini",
                    "tokens_used": tokens_used,
                    "context_type": "general"
                }
            
            # Build context from relevant chunks
            context_parts = []
            for i, chunk_data in enumerate(relevant_chunks, 1):
                context_parts.append(f"Document chunk {i}:\n{chunk_data['content']}")
            
            context = "\n\n".join(context_parts)
            
            # Create RAG prompt with fallback instruction
            rag_prompt = f"""Based on the following document context, answer the user's question. 
            If the context doesn't contain enough information to answer the question completely, 
            you can supplement with your general knowledge, but prioritize information from the documents.
            Provide a comprehensive and accurate answer.

            Context:
            {context}

            Question: {query}

            Answer:"""
            
            # Generate response using LLM
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document context. Always provide accurate and relevant information."},
                    {"role": "user", "content": rag_prompt}
                ],
                temperature=temperature
            )
            
            llm_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            logger.info(f"Generated RAG response for query: {query[:50]}... with {tokens_used} tokens")
            
            return {
                "response": llm_response,
                "model": "gpt-4o-mini",
                "tokens_used": tokens_used,
                "context_type": "document"
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            raise Exception(f"Error generating RAG response: {str(e)}") 