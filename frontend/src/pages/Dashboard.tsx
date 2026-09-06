import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { FileText, MessageSquare, Upload, Layers, ArrowUpRight, CheckCircle2, Search, Database } from 'lucide-react'
import { api } from '../lib/api'

interface DocumentItem {
  id: string
  chunk_count: number
}

interface ConversationItem {
  id: number
  message_count: number
}

export function Dashboard() {
  const [docCount, setDocCount] = useState<number>(0)
  const [chunkCount, setChunkCount] = useState<number>(0)
  const [convCount, setConvCount] = useState<number>(0)
  const [queryCount, setQueryCount] = useState<number>(0)

  useEffect(() => {
    let isMounted = true

    const loadStats = async () => {
      try {
        const [docsRes, convsRes] = await Promise.all([
          api.get('/documents'),
          api.get('/chat/conversations'),
        ])
        if (!isMounted) return

        const docs: DocumentItem[] = docsRes.data.documents || []
        const totalDocs = docsRes.data.total ?? docs.length
        const totalChunks = docs.reduce((sum, d) => sum + (d.chunk_count || 0), 0)
        const convs: ConversationItem[] = convsRes.data || []
        const totalQueries = convs.reduce((sum, c) => sum + (c.message_count || 0), 0)

        setDocCount(totalDocs)
        setChunkCount(totalChunks)
        setConvCount(convs.length)
        setQueryCount(totalQueries)
      } catch {
        // Fallback silently if not loaded yet
      }
    }

    loadStats()
    return () => {
      isMounted = false
    }
  }, [])

  const stats = [
    {
      label: 'Indexed Documents',
      value: String(docCount),
      subtext: 'Active knowledge base files',
      icon: FileText,
      color: 'bg-indigo-50 text-indigo-600 border-indigo-100',
      href: '/documents',
    },
    {
      label: 'Vector Chunks',
      value: String(chunkCount),
      subtext: 'Embedded text passages',
      icon: Layers,
      color: 'bg-purple-50 text-purple-600 border-purple-100',
      href: '/documents',
    },
    {
      label: 'Conversations',
      value: String(convCount),
      subtext: 'Active query sessions',
      icon: MessageSquare,
      color: 'bg-emerald-50 text-emerald-600 border-emerald-100',
      href: '/chat',
    },
    {
      label: 'Grounded Queries',
      value: String(queryCount),
      subtext: 'Answered with citations',
      icon: CheckCircle2,
      color: 'bg-blue-50 text-blue-600 border-blue-100',
      href: '/chat',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-[var(--color-border)]">
        <div>
          <h1 className="text-2xl lg:text-3xl font-bold text-slate-900 tracking-tight">Intelligence Dashboard</h1>
          <p className="text-sm text-[var(--color-text-muted)] mt-1">
            Enterprise RAG overview, indexed document statistics, and retrieval metrics.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/chat" className="btn btn-secondary text-xs sm:text-sm font-medium">
            <MessageSquare size={16} />
            <span>Open Chat</span>
          </Link>
          <Link to="/documents" className="btn btn-primary text-xs sm:text-sm font-medium">
            <Upload size={16} />
            <span>Upload Document</span>
          </Link>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Link
            key={stat.label}
            to={stat.href}
            className="card p-5 card-interactive group flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">
                  {stat.label}
                </span>
                <div className={`p-2 rounded-lg border ${stat.color}`}>
                  <stat.icon size={18} />
                </div>
              </div>
              <p className="text-3xl font-bold text-slate-900 tracking-tight">{stat.value}</p>
            </div>
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-[var(--color-border-subtle)] text-xs text-[var(--color-text-muted)]">
              <span>{stat.subtext}</span>
              <ArrowUpRight size={14} className="text-slate-400 group-hover:text-[var(--color-primary)] transition-colors" />
            </div>
          </Link>
        ))}
      </div>

      {/* Quick Actions & Knowledge Access */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card p-6">
          <h2 className="text-base font-semibold text-slate-900 mb-1">Knowledge Operations</h2>
          <p className="text-xs text-[var(--color-text-muted)] mb-4">Quickly expand or query your knowledge base</p>
          
          <div className="space-y-3">
            <Link
              to="/documents"
              className="flex items-center justify-between p-3.5 rounded-xl border border-[var(--color-border)] hover:border-indigo-200 hover:bg-indigo-50/40 transition-all group"
            >
              <div className="flex items-center gap-3.5">
                <div className="w-10 h-10 rounded-lg bg-indigo-50 border border-indigo-100 text-indigo-600 flex items-center justify-center flex-shrink-0">
                  <Upload size={20} />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-900 group-hover:text-indigo-600 transition-colors">
                    Upload & Ingest Documents
                  </p>
                  <p className="text-xs text-[var(--color-text-muted)]">
                    Extract, chunk, and embed PDF, DOCX, TXT, or Markdown files
                  </p>
                </div>
              </div>
              <ArrowUpRight size={16} className="text-slate-400 group-hover:text-indigo-600 transition-colors" />
            </Link>

            <Link
              to="/chat"
              className="flex items-center justify-between p-3.5 rounded-xl border border-[var(--color-border)] hover:border-emerald-200 hover:bg-emerald-50/40 transition-all group"
            >
              <div className="flex items-center gap-3.5">
                <div className="w-10 h-10 rounded-lg bg-emerald-50 border border-emerald-100 text-emerald-600 flex items-center justify-center flex-shrink-0">
                  <MessageSquare size={20} />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-900 group-hover:text-emerald-600 transition-colors">
                    Ask Grounded Questions
                  </p>
                  <p className="text-xs text-[var(--color-text-muted)]">
                    Query documents with strict citation verification and evidence gates
                  </p>
                </div>
              </div>
              <ArrowUpRight size={16} className="text-slate-400 group-hover:text-emerald-600 transition-colors" />
            </Link>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between mb-1">
            <h2 className="text-base font-semibold text-slate-900">System Pipeline</h2>
            <span className="badge badge-info text-[10px]">Hybrid Search Active</span>
          </div>
          <p className="text-xs text-[var(--color-text-muted)] mb-4">Multi-stage retrieval & evidence gating architecture</p>

          <div className="space-y-3 text-xs">
            <div className="flex items-start gap-3 p-3 bg-slate-50 border border-[var(--color-border)] rounded-lg">
              <div className="w-6 h-6 rounded-full bg-slate-200 text-slate-700 flex items-center justify-center font-bold text-xs flex-shrink-0 mt-0.5">
                1
              </div>
              <div>
                <p className="font-semibold text-slate-800">Extraction & Chunking</p>
                <p className="text-[var(--color-text-muted)] mt-0.5">Structured text parsing with metadata indexing</p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-slate-50 border border-[var(--color-border)] rounded-lg">
              <div className="w-6 h-6 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center font-bold text-xs flex-shrink-0 mt-0.5">
                2
              </div>
              <div>
                <p className="font-semibold text-slate-800">Hybrid Search & RRF Reranking</p>
                <p className="text-[var(--color-text-muted)] mt-0.5">Dense semantic vector search + BM25 keyword matching</p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-slate-50 border border-[var(--color-border)] rounded-lg">
              <div className="w-6 h-6 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center font-bold text-xs flex-shrink-0 mt-0.5">
                3
              </div>
              <div>
                <p className="font-semibold text-slate-800">Evidence Gate & Citation Verification</p>
                <p className="text-[var(--color-text-muted)] mt-0.5">Grounded answers with passage-level attribution</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Architecture Overview Card */}
      <div className="card p-6 bg-gradient-to-r from-slate-900 to-slate-800 text-white border-0 shadow-lg">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-indigo-500/20 text-indigo-300 text-xs font-medium border border-indigo-400/20">
              <Database size={13} />
              <span>Production RAG Architecture</span>
            </div>
            <h3 className="text-lg font-bold text-white tracking-tight">
              Verified Knowledge Intelligence with Zero-Hallucination Guardrails
            </h3>
            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              Every synthesized response is verified against retrieved chunks. If relevance scores fall below safety thresholds, the platform refuses to speculate, guaranteeing high-precision enterprise compliance.
            </p>
          </div>
          <div className="flex items-center gap-3 flex-shrink-0">
            <Link to="/chat" className="btn btn-primary text-xs font-semibold px-4 py-2 bg-indigo-600 hover:bg-indigo-500 border-0 text-white">
              <Search size={14} />
              <span>Start Exploring</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
