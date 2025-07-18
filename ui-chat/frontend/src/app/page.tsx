'use client'

import { useState, useEffect, useRef } from 'react'
import { Send, Bot, User, RotateCcw, Plus, Paperclip, Sparkles, FileText } from 'lucide-react'

interface Message {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
  contextType?: 'document' | 'general'
}

const API_BASE_URL = 'http://localhost:8010'

export default function MyAssistant() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploadStatus, setUploadStatus] = useState<string>('')
  const [uploadedDocuments, setUploadedDocuments] = useState<string[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '52px'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  const clearChat = () => {
    setMessages([])
    setInput('')
    setError(null)
  }



  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  // File upload functionality
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

      const response = await fetch(`${API_BASE_URL}/rag/upload-pdf`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
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
        const response = await fetch(`${API_BASE_URL}/rag/status/${jobId}`);
        if (!response.ok) throw new Error('Status check failed');

        const status = await response.json();

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

  // Modified submit handler to use RAG when documents are available
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input.trim(),
      role: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    const currentInput = input.trim()
    setInput('')
    setIsLoading(true)
    setError(null)

    try {
      let response;
      
      // Use RAG endpoint if documents are uploaded, otherwise use regular chat
      if (uploadedDocuments.length > 0) {
        response = await fetch(`${API_BASE_URL}/rag/ask`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: currentInput
          }),
        });
      } else {
        response = await fetch(`${API_BASE_URL}/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: currentInput,
            temperature: 0.7,
            model: 'gpt-4o-mini'
          }),
        });
      }

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response,
        role: 'assistant',
        timestamp: new Date(),
        contextType: data.context_type
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred'
      setError(errorMessage)
      console.error('Error sending message:', err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex h-screen animated-gradient">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        onChange={handleFileUpload}
        className="hidden"
      />
      
      {/* Sidebar */}
      <div className="w-[260px] glass border-r border-white/20 flex flex-col">
        {/* New Chat Button */}
        <div className="p-3">
          <button 
            onClick={clearChat}
            className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-white hover:bg-white/20 rounded-lg transition-all duration-200 hover-lift border border-white/30"
          >
            <Plus className="w-4 h-4" />
            New chat
          </button>
        </div>

        {/* Chat History - Empty for now */}
        <div className="flex-1 px-3">
          {/* This would contain chat history */}
        </div>

        {/* Bottom Section */}
        <div className="p-3 border-t border-white/20">
          <div className="text-xs text-white/70 px-2 flex items-center gap-2">
            <Sparkles className="w-3 h-3" />
            MyAssistant
          </div>
          
          {/* Document Status */}
          {uploadedDocuments.length > 0 && (
            <div className="mt-3 px-3 py-2 bg-green-500/20 rounded-lg border border-green-400/30 shadow-md">
              <div className="flex items-center gap-2 text-sm font-medium text-green-200">
                <FileText className="w-4 h-4" />
                <span>{uploadedDocuments.length} document{uploadedDocuments.length > 1 ? 's' : ''} loaded</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="h-14 border-b border-white/20 flex items-center justify-between px-4 glass">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-400 to-pink-400 rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-lg font-medium text-white">MyAssistant</h1>
            
            {/* Document indicator */}
            {uploadedDocuments.length > 0 && (
              <div className="flex items-center gap-2 ml-4 px-3 py-2 bg-green-500/30 rounded-lg border border-green-400/40 shadow-lg">
                <FileText className="w-4 h-4 text-green-300" />
                <span className="text-sm font-semibold text-green-200">
                  {uploadedDocuments.length} doc{uploadedDocuments.length > 1 ? 's' : ''}
                </span>
              </div>
            )}
          </div>
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto">
          {/* Upload Status */}
          {uploadStatus && (
            <div className="px-4 py-2 bg-blue-500/20 backdrop-blur-sm border-b border-white/20">
              <div className="max-w-3xl mx-auto">
                <div className="text-sm text-blue-200 flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                  {uploadStatus}
                </div>
              </div>
            </div>
          )}
          
          {messages.length === 0 ? (
            /* Initial State - Centered Input */
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-md mx-auto px-4 w-full">
                <div className="w-16 h-16 glass-strong rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg hover-lift">
                  <Bot className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-2xl font-semibold text-white mb-3">
                  How can I help you today?
                </h2>
                <p className="text-white/70 text-sm mb-8">
                  Ask me anything and I'll assist you with a helpful response.
                </p>
                
                {/* Centered Input Box */}
                <div className="max-w-2xl mx-auto">
                  <div className="relative">

                    
                    <form onSubmit={handleSubmit} className="relative">
                      <div className="flex items-end gap-2 glass-strong rounded-xl shadow-lg hover-lift">
                        {/* Attachment button */}
                        <button
                          type="button"
                          onClick={() => fileInputRef.current?.click()}
                          className="flex-shrink-0 p-2 text-white/70 hover:text-white transition-colors"
                        >
                          <Paperclip className="w-5 h-5" />
                        </button>

                        {/* Text input */}
                        <textarea
                          ref={textareaRef}
                          value={input}
                          onChange={(e) => setInput(e.target.value)}
                          onKeyDown={handleKeyDown}
                          placeholder="Message MyAssistant..."
                          className="flex-1 resize-none bg-transparent border-none outline-none text-white placeholder-white/60 py-3 px-0 max-h-[200px] min-h-[52px] leading-6"
                          rows={1}
                          disabled={isLoading}
                        />

                        {/* Send button */}
                        <div className="flex-shrink-0 p-2">
                          <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className="w-8 h-8 bg-white/30 hover:bg-white/40 disabled:bg-white/10 disabled:cursor-not-allowed rounded-lg flex items-center justify-center transition-all duration-200 hover:scale-105"
                          >
                            <Send className="w-4 h-4 text-white" />
                          </button>
                        </div>
                      </div>
                    </form>

                    {/* Footer text */}
                    <div className="text-xs text-white/60 text-center mt-3 flex items-center justify-center gap-1">
                      <Sparkles className="w-3 h-3" />
                      MyAssistant can make mistakes. Check important info.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Messages */
            <div className="max-w-3xl mx-auto w-full">
              {messages.map((message, index) => (
                <div
                  key={message.id}
                  className={`group w-full message-animate ${
                    message.role === 'assistant' 
                      ? 'flex justify-start' 
                      : 'flex justify-end'
                  }`}
                >
                  <div className={`max-w-2xl mx-4 my-2 ${
                    message.role === 'assistant' 
                      ? 'flex gap-3' 
                      : 'flex gap-3 flex-row-reverse'
                  }`}>
                    {/* Avatar */}
                    <div className="flex-shrink-0">
                      {message.role === 'assistant' ? (
                        <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-green-600 rounded-full flex items-center justify-center shadow-lg">
                          <Bot className="w-5 h-5 text-white" />
                        </div>
                      ) : (
                        <div className="w-8 h-8 bg-gradient-to-r from-purple-400 to-purple-600 rounded-full flex items-center justify-center shadow-lg">
                          <User className="w-5 h-5 text-white" />
                        </div>
                      )}
                    </div>

                    {/* Message Bubble */}
                    <div className={`flex-1 min-w-0 ${
                      message.role === 'assistant' 
                        ? 'bg-white/10 backdrop-blur-sm rounded-2xl rounded-tl-md px-4 py-3 shadow-lg border border-white/20' 
                        : 'bg-gradient-to-r from-purple-500/20 to-pink-500/20 backdrop-blur-sm rounded-2xl rounded-tr-md px-4 py-3 shadow-lg border border-purple-400/30'
                    }`}>
                      <div className="whitespace-pre-wrap text-white leading-6">
                        {message.content}
                      </div>
                      
                      {/* Context indicator for assistant messages when documents are loaded */}
                      {message.role === 'assistant' && uploadedDocuments.length > 0 && message.contextType && (
                        <div className={`mt-2 flex items-center gap-1 text-xs ${
                          message.contextType === 'document' 
                            ? 'text-green-300/70' 
                            : 'text-blue-300/70'
                        }`}>
                          {message.contextType === 'document' ? (
                            <>
                              <FileText className="w-3 h-3" />
                              <span>Generated with document context</span>
                            </>
                          ) : (
                            <>
                              <Sparkles className="w-3 h-3" />
                              <span>Generated with general knowledge</span>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {/* Loading indicator */}
              {isLoading && (
                <div className="group w-full message-animate flex justify-start">
                  <div className="max-w-2xl mx-4 my-2 flex gap-3">
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-green-600 rounded-full flex items-center justify-center shadow-lg">
                        <Bot className="w-5 h-5 text-white" />
                      </div>
                    </div>
                    <div className="flex-1 min-w-0 bg-white/10 backdrop-blur-sm rounded-2xl rounded-tl-md px-4 py-3 shadow-lg border border-white/20">
                      <div className="flex items-center gap-1 text-white/70">
                        <div className="w-2 h-2 bg-white/60 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-white/60 rounded-full animate-pulse delay-100"></div>
                        <div className="w-2 h-2 bg-white/60 rounded-full animate-pulse delay-200"></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Error message */}
              {error && (
                <div className="group w-full message-animate flex justify-start">
                  <div className="max-w-2xl mx-4 my-2 flex gap-3">
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center shadow-lg">
                        <Bot className="w-5 h-5 text-white" />
                      </div>
                    </div>
                    <div className="flex-1 min-w-0 bg-red-500/20 backdrop-blur-sm rounded-2xl rounded-tl-md px-4 py-3 shadow-lg border border-red-400/30">
                      <div className="text-red-200">
                        <strong>Error:</strong> {error}
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Bottom Input Area - Only show when there are messages */}
        {messages.length > 0 && (
          <div className="border-t border-white/20 glass">
            <div className="max-w-3xl mx-auto px-4 py-4">
              <div className="relative">

                
                <form onSubmit={handleSubmit} className="relative">
                  <div className="flex items-end gap-2 glass-strong rounded-xl shadow-lg hover-lift">
                    {/* Attachment button */}
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="flex-shrink-0 p-2 text-white/70 hover:text-white transition-colors"
                    >
                      <Paperclip className="w-5 h-5" />
                    </button>

                    {/* Text input */}
                    <textarea
                      ref={textareaRef}
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Message MyAssistant..."
                      className="flex-1 resize-none bg-transparent border-none outline-none text-white placeholder-white/60 py-3 px-0 max-h-[200px] min-h-[52px] leading-6"
                      rows={1}
                      disabled={isLoading}
                    />

                    {/* Send button */}
                    <div className="flex-shrink-0 p-2">
                      <button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="w-8 h-8 bg-white/30 hover:bg-white/40 disabled:bg-white/10 disabled:cursor-not-allowed rounded-lg flex items-center justify-center transition-all duration-200 hover:scale-105"
                      >
                        <Send className="w-4 h-4 text-white" />
                      </button>
                    </div>
                  </div>
                </form>

                {/* Footer text */}
                <div className="text-xs text-white/60 text-center mt-3 flex items-center justify-center gap-1">
                  <Sparkles className="w-3 h-3" />
                  MyAssistant can make mistakes. Check important info.
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}