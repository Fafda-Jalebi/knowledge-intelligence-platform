import { useState, useEffect, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../lib/api'
import { Send, Loader2, Copy, Check, FileText, Sparkles } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import clsx from 'clsx'

interface Citation {
  marker: number
  chunk_id: string
  document_id: string
  document_label: string
  label: string
  text: string
  page_start: number | null
  page_end: number | null
  section_path: string[]
  score: number
  truncated: boolean
  count: number
}

interface Usage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

interface StageInfo {
  stage: string
  ms: number
  detail: Record<string, unknown>
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  groundedness?: number
  refused?: boolean
  refusal_reason?: string
  explanation?: string
  usage?: Usage
  stages?: StageInfo[]
  total_ms?: number
}

export function Chat() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  const [showSources, setShowSources] = useState<number | null>(null)
  const [copiedCitation, setCopiedCitation] = useState<number | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const docParam = searchParams.get('doc')
  useEffect(() => {
    if (docParam) {
      setSelectedDocIds([docParam])
      setSearchParams({})
    }
  }, [docParam, setSearchParams])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input
    setInput('')
    setLoading(true)

    const newMessages: ChatMessage[] = [...messages, { role: 'user', content: userMessage }]
    setMessages(newMessages)

    try {
      const response = await api.post('/chat/ask', {
        question: userMessage,
        document_ids: selectedDocIds,
      })
      setMessages([...newMessages, response.data as ChatMessage])
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } }
      setMessages([
        ...newMessages,
        {
          role: 'assistant' as const,
          content: 'Sorry, an error occurred. Please try again.',
          refused: true,
          refusal_reason: 'error',
          explanation: axiosError.response?.data?.detail || 'Unknown error',
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleCopy = (text: string, marker: number) => {
    navigator.clipboard.writeText(text)
    setCopiedCitation(marker)
    setTimeout(() => setCopiedCitation(null), 2000)
  }

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-[var(--color-border)] bg-[var(--color-surface)]">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-text)]">Chat</h1>
          <p className="text-sm text-[var(--color-text-muted)]">
            Ask questions about your documents. Answers are grounded with citations.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {selectedDocIds.length > 0 && (
            <span className="badge badge-info">
              <FileText size={12} />
              <span>{selectedDocIds.length} document(s) selected</span>
            </span>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center text-[var(--color-text-muted)]">
            <Sparkles size={64} className="mb-4 opacity-30" />
            <h3 className="text-lg font-medium text-[var(--color-text)] mb-2">Start a conversation</h3>
            <p className="max-w-md">Ask a question about your documents. I'll search through them and provide a grounded answer with citations.</p>
            <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-md">
              {[
                'What are the main topics in my documents?',
                'Summarize the key findings',
                'What methods are described?',
                'Are there any limitations mentioned?',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => {
                    setInput(suggestion)
                    inputRef.current?.focus()
                  }}
                  className="text-left p-3 border border-[var(--color-border)] rounded-lg hover:bg-[var(--color-background)] transition-colors text-sm"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className="flex gap-3 max-w-4xl mx-auto w-full">
            <div
              className={clsx(
                'w-8 flex-shrink-0 flex items-center justify-center text-sm font-medium rounded-full',
                message.role === 'user'
                  ? 'bg-[var(--color-primary)] text-white'
                  : 'bg-[var(--color-primary-light)] text-[var(--color-primary)]'
              )}
            >
              {message.role === 'user' ? 'You' : 'KIP'}
            </div>
            <div className="flex-1 min-w-0">
              <div className={clsx('prose prose-sm max-w-none markdown-content', message.refused && 'opacity-70')}>
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>

              {/* Citations */}
              {message.citations && message.citations.length > 0 && (
                <div className="mt-3 space-y-2">
                  {message.citations.map((citation, citeIndex) => (
                    <div
                      key={citeIndex}
                      className={clsx(
                        'source-passage cursor-pointer',
                        showSources === citeIndex ? 'ring-2 ring-[var(--color-primary)]' : ''
                      )}
                      onClick={() => setShowSources(showSources === citeIndex ? null : citeIndex)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="source-passage-marker">[{citation.marker}]</span>
                          <span className="font-medium text-sm">{citation.document_label}</span>
                          {citation.page_start && (
                            <span className="text-xs text-[var(--color-text-muted)]">
                              p. {citation.page_start}{citation.page_end && citation.page_end !== citation.page_start ? `-${citation.page_end}` : ''}
                            </span>
                          )}
                          {citation.section_path.length > 0 && (
                            <span className="text-xs text-[var(--color-text-muted)]">
                              {' > '}{citation.section_path.join(' > ')}
                            </span>
                          )}
                        </div>
                        <button
                          onClick={() => handleCopy(citation.text, citation.marker)}
                          className="p-1.5 text-[var(--color-text-muted)] hover:text-[var(--color-primary)] rounded transition-colors"
                          title={copiedCitation === citation.marker ? 'Copied!' : 'Copy passage'}
                        >
                          {copiedCitation === citation.marker ? <Check size={16} /> : <Copy size={16} />}
                        </button>
                      </div>
                      {showSources === citeIndex && (
                        <div className="mt-2 p-3 bg-[var(--color-background)] rounded-md text-sm border border-[var(--color-border)]">
                          {citation.text}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Metadata */}
              {!message.refused && (message.groundedness !== undefined || message.total_ms) && (
                <div className="mt-3 flex items-center gap-4 text-xs text-[var(--color-text-muted)]">
                  {message.groundedness !== undefined && (
                    <span className="flex items-center gap-1">
                      <Sparkles size={12} />
                      Groundedness: {(message.groundedness * 100).toFixed(0)}%
                    </span>
                  )}
                  {message.total_ms && (
                    <span>Response time: {formatTime(message.total_ms)}</span>
                  )}
                  {message.usage && (
                    <span>Tokens: {message.usage.total_tokens || message.usage.prompt_tokens + message.usage.completion_tokens}</span>
                  )}
                </div>
              )}

              {/* Refusal */}
              {message.refused && message.explanation && (
                <div className="mt-3 p-3 bg-[var(--color-error)]/10 border border-[var(--color-error)]/20 rounded-lg text-[var(--color-error)] text-sm">
                  <p className="font-medium mb-1">Could not answer</p>
                  <p>{message.explanation}</p>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSend} className="p-4 border-t border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-end gap-3 max-w-4xl mx-auto w-full">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={selectedDocIds.length > 0 ? `Ask about selected documents...` : 'Ask a question...'}
            className="flex-1 input resize-none min-h-[48px] max-h-40"
            rows={1}
            disabled={loading}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend(e)
              }
            }}
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="btn btn-primary h-[48px] px-6 flex-shrink-0"
          >
            {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
          </button>
        </div>
        <p className="text-xs text-[var(--color-text-muted)] mt-2 text-center">
          Press <kbd className="px-1.5 py-0.5 bg-[var(--color-background)] border border-[var(--color-border)] rounded">Enter</kbd> to send, <kbd className="px-1.5 py-0.5 bg-[var(--color-background)] border border-[var(--color-border)] rounded">Shift+Enter</kbd> for new line
        </p>
      </form>
    </div>
  )
}
