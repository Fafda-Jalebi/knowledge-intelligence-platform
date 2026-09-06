import { useState, useEffect, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../lib/api'
import { Send, Loader2, Copy, Check, FileText, Sparkles, ShieldCheck, Clock, Layers, ChevronDown, ChevronUp, AlertCircle, Bot } from 'lucide-react'
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
  const [expandedCitation, setExpandedCitation] = useState<{ msgIdx: number; citeIdx: number } | null>(null)
  const [copiedCitation, setCopiedCitation] = useState<string | null>(null)
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

    const userMessage = input.trim()
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
          content: 'I encountered an issue generating a grounded response. Please verify that documents are indexed and try again.',
          refused: true,
          refusal_reason: 'error',
          explanation: axiosError.response?.data?.detail || 'Network or server error during retrieval.',
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleCopy = (text: string, citeId: string) => {
    navigator.clipboard.writeText(text)
    setCopiedCitation(citeId)
    setTimeout(() => setCopiedCitation(null), 2000)
  }

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  const toggleCitation = (msgIdx: number, citeIdx: number) => {
    if (expandedCitation?.msgIdx === msgIdx && expandedCitation?.citeIdx === citeIdx) {
      setExpandedCitation(null)
    } else {
      setExpandedCitation({ msgIdx, citeIdx })
    }
  }

  return (
    <div className="chat-card">
      {/* Chat Header */}
      <div className="flex items-center justify-between px-6 lg:px-8 py-4 border-b border-[var(--color-border)] bg-slate-50/70 flex-shrink-0">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold text-slate-900 tracking-tight">Grounded Assistant</h1>
            <span className="badge badge-info text-[10px] py-0.5">
              <ShieldCheck size={11} className="mr-0.5" /> Citation Verified
            </span>
          </div>
          <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
            Hybrid search with dense embeddings, BM25 keyword matching, and evidence gating.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {selectedDocIds.length > 0 ? (
            <div className="flex items-center gap-1.5 px-2.5 py-1 bg-indigo-50 border border-indigo-200 text-indigo-700 rounded-lg text-xs font-medium">
              <FileText size={13} />
              <span>Targeting {selectedDocIds.length} doc{selectedDocIds.length === 1 ? '' : 's'}</span>
              <button
                type="button"
                onClick={() => setSelectedDocIds([])}
                className="ml-1 text-indigo-400 hover:text-indigo-700 text-xs font-bold"
                title="Clear filter"
              >
                ×
              </button>
            </div>
          ) : (
            <span className="badge badge-neutral text-xs py-1">
              <span>All Indexed Documents</span>
            </span>
          )}
        </div>
      </div>

      {/* Empty State Welcome & Suggestions */}
      {messages.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center p-6 lg:p-10 text-center">
          <div className="w-14 h-14 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 mb-4 shadow-xs">
            <Sparkles size={28} />
          </div>
          <h2 className="text-2xl lg:text-3xl font-extrabold text-slate-900 mb-2.5 tracking-tight">
            Ask your Knowledge Base
          </h2>
          <p className="text-xs sm:text-sm text-[var(--color-text-muted)] max-w-xl mb-8 leading-relaxed">
            Every answer is synthesized directly from verified passages in your documents with strict passage-level citations and similarity checks.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 w-full max-w-3xl text-left">
            {[
              'What are the main topics across my documents?',
              'Summarize the core technical findings and results',
              'What methodologies and architecture are proposed?',
              'What constraints or limitations are highlighted?',
            ].map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => {
                  setInput(suggestion)
                  inputRef.current?.focus()
                }}
                className="flex items-center min-h-[64px] p-4 px-5 bg-slate-50/80 hover:bg-indigo-50/60 border border-slate-200 hover:border-indigo-300 rounded-xl transition-all duration-150 text-xs sm:text-sm text-slate-700 font-medium group shadow-2xs hover:shadow-xs"
              >
                <span className="group-hover:text-indigo-600 transition-colors leading-snug">{suggestion}</span>
              </button>
            ))}
          </div>
        </div>
      ) : (
        /* Messages Stream */
        <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 space-y-6">
          {messages.map((message, msgIdx) => (
            <div
              key={msgIdx}
              className={clsx(
                'flex gap-3.5 max-w-5xl mx-auto w-full transition-opacity',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              {message.role === 'assistant' && (
                <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center flex-shrink-0 shadow-xs mt-0.5">
                  <Bot size={16} />
                </div>
              )}

              <div
                className={clsx(
                  'min-w-0',
                  message.role === 'user' ? 'max-w-2xl' : 'flex-1'
                )}
              >
                {message.role === 'user' ? (
                  <div className="p-3.5 px-4 bg-slate-900 text-white rounded-2xl rounded-tr-xs text-sm shadow-xs leading-relaxed font-normal">
                    {message.content}
                  </div>
                ) : (
                  <div className="card p-5 border border-slate-200 bg-[var(--color-surface)] shadow-xs rounded-2xl rounded-tl-xs space-y-3">
                    {/* Markdown Answer */}
                    <div className="markdown-content text-sm">
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>

                    {/* Refusal Explanations */}
                    {message.refused && message.explanation && (
                      <div className="p-3.5 bg-amber-50 border border-amber-200 rounded-xl text-amber-800 text-xs">
                        <div className="flex items-center gap-1.5 font-semibold text-amber-900 mb-1">
                          <AlertCircle size={14} className="text-amber-600" />
                          <span>Evidence Gate Notice</span>
                        </div>
                        <p className="leading-relaxed">{message.explanation}</p>
                      </div>
                    )}

                    {/* Citations & Verified Sources */}
                    {message.citations && message.citations.length > 0 && (
                      <div className="pt-3 border-t border-[var(--color-border)] space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
                            <Layers size={13} className="text-indigo-600" />
                            <span>Sources & Citations ({message.citations.length})</span>
                          </span>
                        </div>

                        <div className="space-y-2">
                          {message.citations.map((citation, citeIdx) => {
                            const isExpanded =
                              expandedCitation?.msgIdx === msgIdx && expandedCitation?.citeIdx === citeIdx
                            const citeKey = `${msgIdx}-${citation.marker}`
                            const isCopied = copiedCitation === citeKey

                            return (
                              <div
                                key={citeIdx}
                                className={clsx(
                                  'border rounded-xl transition-all overflow-hidden',
                                  isExpanded
                                    ? 'border-indigo-300 bg-indigo-50/20 shadow-xs'
                                    : 'border-slate-200 bg-slate-50/60 hover:bg-slate-50'
                                )}
                              >
                                <div
                                  className="flex items-center justify-between p-2.5 px-3 cursor-pointer select-none"
                                  onClick={() => toggleCitation(msgIdx, citeIdx)}
                                >
                                  <div className="flex items-center gap-2 min-w-0">
                                    <span className="source-passage-marker">
                                      [{citation.marker}]
                                    </span>
                                    <span className="font-semibold text-xs text-slate-900 truncate">
                                      {citation.document_label}
                                    </span>
                                    {citation.page_start && (
                                      <span className="badge badge-neutral text-[10px] py-0 px-1.5">
                                        p. {citation.page_start}
                                        {citation.page_end && citation.page_end !== citation.page_start
                                          ? `-${citation.page_end}`
                                          : ''}
                                      </span>
                                    )}
                                    {citation.section_path && citation.section_path.length > 0 && (
                                      <span className="hidden sm:inline-block text-[11px] text-[var(--color-text-muted)] truncate max-w-xs">
                                        › {citation.section_path.join(' › ')}
                                      </span>
                                    )}
                                  </div>

                                  <div className="flex items-center gap-1.5 flex-shrink-0">
                                    {citation.score !== undefined && (
                                      <span className="text-[10px] font-mono text-slate-500 bg-white border border-slate-200 px-1.5 py-0.5 rounded">
                                        {(citation.score * 100).toFixed(0)}% match
                                      </span>
                                    )}
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        handleCopy(citation.text, citeKey)
                                      }}
                                      className="p-1 text-slate-400 hover:text-indigo-600 rounded transition-colors"
                                      title={isCopied ? 'Copied!' : 'Copy passage'}
                                      aria-label="Copy citation text"
                                    >
                                      {isCopied ? <Check size={14} className="text-emerald-600" /> : <Copy size={14} />}
                                    </button>
                                    {isExpanded ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                                  </div>
                                </div>

                                {isExpanded && (
                                  <div className="p-3 pt-0 border-t border-indigo-100 bg-white text-xs leading-relaxed text-slate-700 font-sans mt-2">
                                    <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-200 font-mono text-[11px] text-slate-800 whitespace-pre-wrap max-h-48 overflow-y-auto">
                                      {citation.text}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}

                    {/* Metadata Footer */}
                    {!message.refused && (message.groundedness !== undefined || message.total_ms || message.usage) && (
                      <div className="flex flex-wrap items-center gap-3 pt-2 text-[11px] text-[var(--color-text-muted)] border-t border-[var(--color-border-subtle)]">
                        {message.groundedness !== undefined && (
                          <span className="inline-flex items-center gap-1 text-emerald-700 font-medium bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-md">
                            <Sparkles size={11} className="text-emerald-600" />
                            <span>Groundedness: {(message.groundedness * 100).toFixed(0)}%</span>
                          </span>
                        )}
                        {message.total_ms && (
                          <span className="inline-flex items-center gap-1">
                            <Clock size={11} />
                            <span>Latency: {formatTime(message.total_ms)}</span>
                          </span>
                        )}
                        {message.usage && (
                          <span className="inline-flex items-center gap-1">
                            <Layers size={11} />
                            <span>
                              Tokens: {message.usage.total_tokens || message.usage.prompt_tokens + message.usage.completion_tokens}
                            </span>
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      )}

      {/* Input Area */}
      <div className="p-4 sm:p-5 lg:p-6 border-t border-[var(--color-border)] bg-[var(--color-surface)] flex-shrink-0">
        <form onSubmit={handleSend} className="w-full max-w-4xl mx-auto">
          <div className="flex items-end gap-2.5 sm:gap-3 bg-slate-50 border border-slate-300 rounded-2xl p-2.5 focus-within:border-[var(--color-primary)] focus-within:ring-2 focus-within:ring-indigo-100 transition-all shadow-2xs">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                selectedDocIds.length > 0
                  ? `Ask question about ${selectedDocIds.length} selected document(s)...`
                  : 'Ask any question about your indexed knowledge base...'
              }
              className="flex-1 bg-transparent border-0 resize-none min-h-[46px] max-h-36 p-1.5 px-3 text-xs sm:text-sm text-slate-900 focus:outline-none placeholder:text-slate-400"
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
              className={clsx(
                'h-10 sm:h-11 w-10 sm:w-11 rounded-xl flex-shrink-0 flex items-center justify-center transition-all duration-150',
                !input.trim() || loading
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  : 'bg-[var(--color-primary)] text-white hover:bg-[var(--color-primary-hover)] shadow-xs hover:shadow-sm active:scale-95'
              )}
              aria-label="Send message"
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} className="translate-x-0.5" />}
            </button>
          </div>
          <div className="flex items-center justify-between mt-2 px-1 text-[11px] sm:text-xs text-[var(--color-text-muted)]">
            <span>
              Press <kbd className="px-1 py-0.5 bg-slate-100 border border-slate-200 rounded text-[10px] font-mono">Enter</kbd> to ask, <kbd className="px-1 py-0.5 bg-slate-100 border border-slate-200 rounded text-[10px] font-mono">Shift + Enter</kbd> for newline
            </span>
            <span className="hidden sm:inline">Attributed with citation markers</span>
          </div>
        </form>
      </div>
    </div>
  )
}
