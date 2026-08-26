import { Link } from 'react-router-dom'
import { FileText, MessageSquare, Upload, BarChart2, ExternalLink } from 'lucide-react'

export function Dashboard() {
  const stats = [
    { label: 'Documents', value: '0', icon: FileText, color: 'bg-blue-100 text-blue-600', href: '/documents' },
    { label: 'Conversations', value: '0', icon: MessageSquare, color: 'bg-green-100 text-green-600', href: '/chat' },
    { label: 'Total Chunks', value: '0', icon: BarChart2, color: 'bg-purple-100 text-purple-600', href: '/documents' },
    { label: 'Queries Today', value: '0', icon: ExternalLink, color: 'bg-orange-100 text-orange-600', href: '/chat' },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text)]">Dashboard</h1>
          <p className="text-[var(--color-text-muted)]">Overview of your knowledge base</p>
        </div>
        <Link to="/documents" className="btn btn-primary">
          <Upload size={18} />
          <span>Upload Document</span>
        </Link>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Link
            key={stat.label}
            to={stat.href}
            className="card p-5 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-[var(--color-text-muted)]">{stat.label}</p>
                <p className="text-3xl font-bold text-[var(--color-text)] mt-1">{stat.value}</p>
              </div>
              <div className={`${stat.color} p-3 rounded-lg`}>
                <stat.icon size={24} />
              </div>
            </div>
          </Link>
        ))}
      </div>

      <div className="card">
        <div className="p-5 border-b border-[var(--color-border)]">
          <h2 className="text-lg font-semibold">Quick Actions</h2>
        </div>
        <div className="p-5 space-y-3">
          <Link to="/documents" className="flex items-center gap-3 p-3 rounded-lg hover:bg-[var(--color-background)] transition-colors">
            <div className="bg-blue-100 text-blue-600 p-2 rounded-lg">
              <Upload size={20} />
            </div>
            <div>
              <p className="font-medium">Upload your first document</p>
              <p className="text-sm text-[var(--color-text-muted)]">PDF, DOCX, TXT, or Markdown</p>
            </div>
          </Link>
          <Link to="/chat" className="flex items-center gap-3 p-3 rounded-lg hover:bg-[var(--color-background)] transition-colors">
            <div className="bg-green-100 text-green-600 p-2 rounded-lg">
              <MessageSquare size={20} />
            </div>
            <div>
              <p className="font-medium">Ask a question</p>
              <p className="text-sm text-[var(--color-text-muted)]">Get grounded answers with citations</p>
            </div>
          </Link>
        </div>
      </div>

      <div className="card">
        <div className="p-5 border-b border-[var(--color-border)]">
          <h2 className="text-lg font-semibold">How it works</h2>
        </div>
        <div className="p-5 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="space-y-2">
            <h3 className="font-medium">1. Ingest</h3>
            <p className="text-[var(--color-text-muted)]">Upload documents - they're extracted, chunked, and embedded automatically</p>
          </div>
          <div className="space-y-2">
            <h3 className="font-medium">2. Retrieve</h3>
            <p className="text-[var(--color-text-muted)]">Hybrid search combines semantic and keyword retrieval for best coverage</p>
          </div>
          <div className="space-y-2">
            <h3 className="font-medium">3. Answer</h3>
            <p className="text-[var(--color-text-muted)]">Grounded answers with citations - every claim traced to source passages</p>
          </div>
        </div>
      </div>
    </div>
  )
}
