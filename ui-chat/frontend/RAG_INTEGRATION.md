# RAG Integration Guide

## 🎯 **What We Built**

A complete RAG (Retrieval-Augmented Generation) system that:

1. **Upload PDF** → Llama Parse extracts text
2. **Process** → Chunk text and generate embeddings
3. **Ask Questions** → Retrieve relevant chunks and generate answers

## 📁 **Files Created**

### Backend (`backend/`)
- `server.py` - Updated with RAG endpoints
- `test_complete_rag.py` - Complete flow testing
- `RAG_FLOW.md` - Architecture documentation

### Frontend (`frontend/src/components/`)
- `RAGChat.tsx` - Complete RAG chat interface

## 🚀 **API Endpoints**

### 1. **Upload PDF** (`POST /rag/upload-pdf`)
```typescript
const formData = new FormData();
formData.append('file', pdfFile);

const response = await fetch('http://localhost:8010/rag/upload-pdf', {
  method: 'POST',
  body: formData,
});

const result = await response.json();
// { job_id: "uuid", status: "pending", message: "..." }
```

### 2. **Check Status** (`GET /rag/status/{job_id}`)
```typescript
const response = await fetch(`http://localhost:8010/rag/status/${jobId}`);
const status = await response.json();
// { status: "completed", chunks_count: 15, ... }
```

### 3. **Ask Question** (`POST /rag/ask`)
```typescript
const response = await fetch('http://localhost:8010/rag/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: "What is machine learning?" }),
});

const result = await response.json();
// { response: "Machine learning is...", model: "gpt-4o-mini", tokens_used: 150 }
```

## 🔧 **Integration Steps**

### Step 1: Add RAG Component to Your App

```tsx
// In your main App.tsx or page component
import RAGChat from './components/RAGChat';

function App() {
  return (
    <div>
      {/* Your existing components */}
      <RAGChat />
    </div>
  );
}
```

### Step 2: Add Route (if using React Router)

```tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import RAGChat from './components/RAGChat';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<YourExistingComponent />} />
        <Route path="/rag" element={<RAGChat />} />
      </Routes>
    </BrowserRouter>
  );
}
```

### Step 3: Add Navigation

```tsx
// Add to your navigation menu
<Link to="/rag" className="nav-link">
  📄 RAG Chat
</Link>
```

## 🎨 **Customization Options**

### Modify the RAGChat Component

```tsx
// Customize styling
const RAGChat: React.FC = () => {
  // ... existing code ...
  
  return (
    <div className="your-custom-styles">
      {/* Your custom layout */}
    </div>
  );
};
```

### Add to Existing Chat Interface

```tsx
// Integrate RAG into existing chat
const [isRAGMode, setIsRAGMode] = useState(false);
const [uploadedDocuments, setUploadedDocuments] = useState<string[]>([]);

// Add RAG toggle
<button onClick={() => setIsRAGMode(!isRAGMode)}>
  {isRAGMode ? 'Regular Chat' : 'RAG Chat'}
</button>

// Modify send message function
const handleSendMessage = async () => {
  if (isRAGMode && uploadedDocuments.length > 0) {
    // Use RAG endpoint
    const response = await fetch('/rag/ask', {
      method: 'POST',
      body: JSON.stringify({ query: inputText }),
    });
  } else {
    // Use regular chat endpoint
    const response = await fetch('/query', {
      method: 'POST',
      body: JSON.stringify({ message: inputText }),
    });
  }
};
```

## 🧪 **Testing**

### 1. Start the Backend Server
```bash
cd backend
python server.py
```

### 2. Test the API
```bash
python test_complete_rag.py
```

### 3. Test the Frontend
```bash
cd frontend
npm run dev
```

## 📱 **User Flow**

1. **User uploads PDF** → Shows processing status
2. **Processing completes** → Shows "Ready for questions"
3. **User asks question** → Gets contextual answer
4. **User can ask more questions** → About the same document

## 🔍 **Features**

- ✅ **File Upload**: PDF upload with progress tracking
- ✅ **Processing Status**: Real-time status updates
- ✅ **Chat Interface**: Clean, responsive chat UI
- ✅ **Error Handling**: Graceful error messages
- ✅ **Loading States**: Visual feedback during processing
- ✅ **Responsive Design**: Works on mobile and desktop

## 🎯 **Next Steps**

1. **Integrate into your existing UI**
2. **Add more file types** (DOCX, TXT, etc.)
3. **Add document management** (list, delete documents)
4. **Add conversation history** (save chat sessions)
5. **Add multiple document support** (query across multiple PDFs)

## 🚀 **Ready to Use**

The RAG system is fully functional and ready to integrate into your frontend. Just add the `RAGChat` component and you'll have a complete document Q&A system! 