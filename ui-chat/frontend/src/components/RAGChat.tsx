import React, { useState, useRef } from 'react';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

interface ProcessingJob {
  job_id: string;
  status: string;
  message: string;
  document_id?: string;
  chunks_count?: number;
}

const RAGChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [processingJob, setProcessingJob] = useState<ProcessingJob | null>(null);
  const [uploadedDocuments, setUploadedDocuments] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_BASE = 'http://localhost:8010';

  // Upload PDF file
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please upload a PDF file');
      return;
    }

    setIsLoading(true);
    setUploadStatus('Uploading PDF...');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE}/rag/upload-pdf`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result: ProcessingJob = await response.json();
      setProcessingJob(result);
      setUploadStatus(`Processing started: ${result.message}`);

      // Poll for status updates
      pollProcessingStatus(result.job_id);

    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus(`Upload failed: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Poll processing status
  const pollProcessingStatus = async (jobId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/rag/status/${jobId}`);
        if (!response.ok) throw new Error('Status check failed');

        const status: ProcessingJob = await response.json();
        setProcessingJob(status);

        if (status.status === 'completed') {
          setUploadStatus(`✅ Processing completed: ${status.chunks_count} chunks created`);
          setUploadedDocuments(prev => [...prev, status.document_id || '']);
          clearInterval(pollInterval);
        } else if (status.status === 'failed') {
          setUploadStatus(`❌ Processing failed: ${status.message}`);
          clearInterval(pollInterval);
        } else {
          setUploadStatus(`⏳ ${status.message}`);
        }
      } catch (error) {
        console.error('Status check error:', error);
        clearInterval(pollInterval);
      }
    }, 2000); // Poll every 2 seconds
  };

  // Send message and get RAG response
  const handleSendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/rag/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: inputText }),
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: result.response,
        isUser: false,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Send message error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: `Error: ${error}`,
        isUser: false,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Enter key
  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b p-4">
        <h1 className="text-xl font-semibold text-gray-800">RAG Chat</h1>
        <p className="text-sm text-gray-600">Upload a PDF and ask questions about it</p>
      </div>

      {/* Upload Section */}
      <div className="bg-white border-b p-4">
        <div className="flex items-center space-x-4">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {isLoading ? 'Uploading...' : 'Upload PDF'}
          </button>
          {uploadStatus && (
            <span className="text-sm text-gray-600">{uploadStatus}</span>
          )}
        </div>
        {uploadedDocuments.length > 0 && (
          <div className="mt-2 text-sm text-green-600">
            ✅ {uploadedDocuments.length} document(s) ready for questions
          </div>
        )}
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-8">
            <p>Upload a PDF and start asking questions!</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  message.isUser
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-800 border'
                }`}
              >
                <p className="text-sm">{message.text}</p>
                <p className="text-xs opacity-70 mt-1">
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white text-gray-800 border px-4 py-2 rounded-lg">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span className="text-sm">Thinking...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Section */}
      <div className="bg-white border-t p-4">
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about your document..."
            disabled={isLoading || uploadedDocuments.length === 0}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || !inputText.trim() || uploadedDocuments.length === 0}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            Send
          </button>
        </div>
        {uploadedDocuments.length === 0 && (
          <p className="text-xs text-gray-500 mt-2">
            Please upload a PDF document first
          </p>
        )}
      </div>
    </div>
  );
};

export default RAGChat; 