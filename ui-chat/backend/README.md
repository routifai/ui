# LLM Chat API - Refactored Architecture

A FastAPI server with clean separation of concerns, making it easy for developers to understand and maintain.

## 📁 Project Structure

```
backend/
├── services/           # Business logic layer
│   ├── __init__.py
│   ├── rag_service.py  # RAG operations (PDF processing, embeddings, search)
│   └── chat_service.py # Chat operations (LLM responses)
├── models/             # Data validation layer
│   ├── __init__.py
│   ├── requests.py     # API request models
│   └── responses.py    # API response models
├── server.py           # Original monolithic server
├── server_refactored.py # New modular server
├── requirements.txt    # Dependencies
└── temp_uploads/      # Temporary file storage
```

## 🏗️ Architecture Overview

### **Service Layer** (`services/`)
- **RAGService**: Handles PDF processing, text chunking, embedding generation, and semantic search
- **ChatService**: Handles LLM interactions for both general chat and RAG responses

### **Model Layer** (`models/`)
- **Request Models**: Validate incoming API requests
- **Response Models**: Structure API responses

### **API Layer** (`server_refactored.py`)
- Clean endpoints that delegate to services
- Proper error handling and logging
- Dependency injection for services

## 🔄 How It Works

### **Document Processing Flow:**
1. **Upload PDF** → `RAGService.process_pdf()`
2. **LlamaParse** → Extracts text from PDF
3. **Chunking** → Splits text into manageable pieces
4. **OpenAI Embeddings** → Generates vector embeddings for each chunk
5. **Storage** → Chunks and embeddings stored in memory

### **Question Answering Flow:**
1. **User Question** → `RAGService.get_relevant_chunks()`
2. **Semantic Search** → Finds most similar document chunks
3. **Context Building** → Combines relevant chunks
4. **LLM Response** → `ChatService.generate_rag_response()`
5. **Answer** → Returns response with context type indicator

## 🚀 Benefits of Refactoring

### **For Developers:**
- ✅ **Clear separation** - Business logic separate from API endpoints
- ✅ **Easy to understand** - Each file has a single responsibility
- ✅ **Easy to test** - Services can be tested independently
- ✅ **Easy to extend** - Add new services without touching API layer
- ✅ **Type safety** - Pydantic models ensure data validation

### **For Maintenance:**
- ✅ **Modular design** - Change one component without affecting others
- ✅ **Clear documentation** - Each service and model is well-documented
- ✅ **Error handling** - Centralized error handling in services
- ✅ **Logging** - Comprehensive logging for debugging

## 🔧 Usage

### **Start the server:**
```bash
python server_refactored.py
```

### **API Endpoints:**
- `POST /query` - General chat
- `POST /rag/upload-pdf` - Upload PDF for processing
- `GET /rag/status/{job_id}` - Check processing status
- `POST /rag/ask` - Ask questions about uploaded documents
- `GET /rag/documents` - List processed documents

## 📝 Key Clarifications

### **LlamaParse vs Embeddings:**
- **LlamaParse**: Only extracts text from PDFs (no embeddings)
- **OpenAI Embeddings**: Generated separately for semantic search
- **Our Architecture**: Combines both for complete RAG pipeline

### **Service Responsibilities:**
- **RAGService**: Document processing, embeddings, search
- **ChatService**: LLM interactions, response generation
- **Models**: Data validation and structure

This refactored architecture makes the codebase much more maintainable and easier for other developers to understand and extend! 