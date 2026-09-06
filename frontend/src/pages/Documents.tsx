import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import { FileText, Trash2, Loader2, AlertCircle, MessageSquare, X, UploadCloud, Layers, BookOpen, Clock } from 'lucide-react'
import clsx from 'clsx'

interface Document {
  id: string
  filename: string
  title: string | null
  page_count: number
  chunk_count: number
  status: string
  created_at: string
  size_bytes: number
  warnings: string[]
}

export function Documents() {
  const navigate = useNavigate()
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState(false)

  const fetchDocuments = async () => {
    try {
      const response = await api.get('/documents')
      setDocuments(response.data.documents || [])
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } }
      setError(axiosError.response?.data?.detail || 'Failed to load documents')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDocuments()
  }, [])

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setSelectedFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return
    setUploading(true)
    setError('')
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      await api.post('/documents/upload', formData)
      setSelectedFile(null)
      await fetchDocuments()
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } }
      setError(axiosError.response?.data?.detail || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const handleDelete = async (docId: string) => {
    if (!confirm('Are you sure you want to delete this document? This cannot be undone.')) return
    try {
      await api.delete(`/documents/${docId}`)
      await fetchDocuments()
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } }
      setError(axiosError.response?.data?.detail || 'Delete failed')
    }
  }

  const formatSize = (bytes: number) => {
    if (!bytes && bytes !== 0) return '0 B'
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatDate = (dateStr: string) => {
    if (!dateStr) return '—'
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const getFileBadge = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase()
    switch (ext) {
      case 'pdf':
        return { label: 'PDF', bg: 'bg-red-50 text-red-600 border-red-200' }
      case 'docx':
      case 'doc':
        return { label: 'DOCX', bg: 'bg-blue-50 text-blue-600 border-blue-200' }
      case 'txt':
        return { label: 'TXT', bg: 'bg-emerald-50 text-emerald-600 border-emerald-200' }
      case 'md':
        return { label: 'MD', bg: 'bg-purple-50 text-purple-600 border-purple-200' }
      default:
        return { label: ext?.toUpperCase() || 'FILE', bg: 'bg-slate-50 text-slate-600 border-slate-200' }
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-[var(--color-border)]">
        <div>
          <h1 className="text-2xl lg:text-3xl font-bold text-slate-900 tracking-tight">Document Library</h1>
          <p className="text-sm text-[var(--color-text-muted)] mt-1">
            Upload, chunk, and manage documents indexed in the vector store.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-600 bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200">
          <Layers size={15} className="text-indigo-600" />
          <span>{documents.length} Indexed Document{documents.length === 1 ? '' : 's'}</span>
        </div>
      </div>

      {/* Upload Zone */}
      <div
        className={clsx(
          'card p-6 border-2 border-dashed transition-all',
          dragActive
            ? 'border-[var(--color-primary)] bg-indigo-50/50 shadow-sm'
            : 'border-[var(--color-border)] hover:border-slate-300'
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-upload"
          accept=".pdf,.docx,.txt,.md"
          onChange={handleFileChange}
          className="hidden"
          disabled={uploading}
        />

        <div className="text-center max-w-lg mx-auto">
          {selectedFile ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3.5 bg-slate-50 border border-slate-200 rounded-xl shadow-xs">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-10 h-10 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center flex-shrink-0 border border-indigo-100">
                    <FileText size={20} />
                  </div>
                  <div className="text-left min-w-0">
                    <p className="font-semibold text-sm text-slate-900 truncate">{selectedFile.name}</p>
                    <p className="text-xs text-[var(--color-text-muted)]">{formatSize(selectedFile.size)}</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                  aria-label="Remove selected file"
                  disabled={uploading}
                >
                  <X size={18} />
                </button>
              </div>

              <div className="flex items-center justify-center gap-3">
                <button
                  onClick={() => setSelectedFile(null)}
                  className="btn btn-secondary text-xs"
                  disabled={uploading}
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpload}
                  className="btn btn-primary text-xs font-semibold px-5"
                  disabled={uploading}
                >
                  {uploading ? (
                    <>
                      <Loader2 className="animate-spin mr-1.5" size={16} />
                      <span>Extracting & Chunking...</span>
                    </>
                  ) : (
                    <>
                      <UploadCloud size={16} className="mr-1.5" />
                      <span>Upload & Ingest</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="w-12 h-12 rounded-xl bg-indigo-50 border border-indigo-100 text-indigo-600 flex items-center justify-center mx-auto mb-3">
                <UploadCloud size={24} />
              </div>
              <p className="text-base font-semibold text-slate-900 mb-1">
                Drop your document here, or{' '}
                <label htmlFor="file-upload" className="text-indigo-600 hover:underline cursor-pointer">
                  browse
                </label>
              </p>
              <p className="text-xs text-[var(--color-text-muted)] mb-4">
                Supported formats: PDF, DOCX, TXT, Markdown (Max file size: 25MB)
              </p>
              <div className="flex items-center justify-center gap-2">
                <span className="badge badge-neutral text-[10px]">PDF</span>
                <span className="badge badge-neutral text-[10px]">DOCX</span>
                <span className="badge badge-neutral text-[10px]">TXT</span>
                <span className="badge badge-neutral text-[10px]">Markdown</span>
              </div>
            </>
          )}

          {error && (
            <div className="mt-4 flex items-start gap-2.5 p-3.5 bg-red-50 border border-red-200 text-red-700 rounded-lg text-xs text-left">
              <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>

      {/* Documents Table or Empty State */}
      {loading ? (
        <div className="card p-12 text-center">
          <Loader2 size={32} className="mx-auto animate-spin text-[var(--color-primary)]" />
          <p className="mt-3 text-sm text-[var(--color-text-muted)]">Loading document index...</p>
        </div>
      ) : documents.length === 0 ? (
        <div className="card p-12 text-center">
          <div className="w-14 h-14 rounded-2xl bg-slate-100 flex items-center justify-center mx-auto mb-3 text-slate-400">
            <FileText size={28} />
          </div>
          <h3 className="text-base font-semibold text-slate-900 mb-1">No documents indexed yet</h3>
          <p className="text-xs text-[var(--color-text-muted)] max-w-sm mx-auto mb-4">
            Upload your technical documentation, research papers, or knowledge files to enable grounded AI Q&A.
          </p>
          <label htmlFor="file-upload" className="btn btn-primary text-xs cursor-pointer">
            <UploadCloud size={15} />
            <span>Upload Document</span>
          </label>
        </div>
      ) : (
        <div className="card overflow-hidden shadow-card border border-[var(--color-border)]">
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-[var(--color-border)] bg-slate-50/80 text-xs font-semibold text-slate-600">
                  <th className="px-5 py-3.5">Document</th>
                  <th className="px-4 py-3.5">Format</th>
                  <th className="px-4 py-3.5">Pages</th>
                  <th className="px-4 py-3.5">Chunks</th>
                  <th className="px-4 py-3.5">Size</th>
                  <th className="px-4 py-3.5">Status</th>
                  <th className="px-4 py-3.5">Indexed At</th>
                  <th className="px-5 py-3.5 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--color-border)] text-xs">
                {documents.map((doc) => {
                  const badge = getFileBadge(doc.filename)
                  return (
                    <tr key={doc.id} className="hover:bg-slate-50/60 transition-colors">
                      <td className="px-5 py-3.5">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-lg bg-slate-100 border border-slate-200 flex items-center justify-center flex-shrink-0 text-slate-600">
                            <FileText size={16} />
                          </div>
                          <div className="min-w-0">
                            <p className="font-semibold text-slate-900 truncate max-w-xs">{doc.filename}</p>
                            {doc.title && (
                              <p className="text-[11px] text-[var(--color-text-muted)] truncate max-w-xs">{doc.title}</p>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3.5">
                        <span className={clsx('badge text-[10px] font-mono border', badge.bg)}>
                          {badge.label}
                        </span>
                      </td>
                      <td className="px-4 py-3.5 text-slate-700 font-medium">
                        <div className="flex items-center gap-1">
                          <BookOpen size={13} className="text-slate-400" />
                          <span>{doc.page_count || 1}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3.5 text-slate-700 font-medium">
                        <div className="flex items-center gap-1">
                          <Layers size={13} className="text-slate-400" />
                          <span>{doc.chunk_count || 0}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3.5 text-[var(--color-text-muted)] font-mono text-[11px]">
                        {formatSize(doc.size_bytes)}
                      </td>
                      <td className="px-4 py-3.5">
                        {doc.status === 'ready' && (
                          <span className="badge badge-success text-[10px]">
                            <span className="status-dot bg-emerald-500 mr-0.5" /> Ready
                          </span>
                        )}
                        {doc.status === 'processing' && (
                          <span className="badge badge-info text-[10px]">
                            <Loader2 size={11} className="animate-spin mr-0.5" /> Processing
                          </span>
                        )}
                        {doc.status === 'pending' && (
                          <span className="badge badge-warning text-[10px]">
                            <span className="status-dot bg-amber-500 mr-0.5" /> Pending
                          </span>
                        )}
                        {doc.status === 'failed' && (
                          <span className="badge badge-error text-[10px]">
                            <span className="status-dot bg-red-500 mr-0.5" /> Failed
                          </span>
                        )}
                        {!['ready', 'processing', 'pending', 'failed'].includes(doc.status) && (
                          <span className="badge badge-neutral text-[10px]">
                            {doc.status}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3.5 text-[var(--color-text-muted)]">
                        <div className="flex items-center gap-1 text-[11px]">
                          <Clock size={12} className="text-slate-400" />
                          <span>{formatDate(doc.created_at)}</span>
                        </div>
                      </td>
                      <td className="px-5 py-3.5 text-right">
                        <div className="flex items-center justify-end gap-1">
                          <button
                            onClick={() => navigate(`/chat?doc=${doc.id}`)}
                            className="p-1.5 text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 rounded-md transition-colors"
                            title="Chat with this document"
                            aria-label="Chat with document"
                          >
                            <MessageSquare size={16} />
                          </button>
                          <button
                            onClick={() => handleDelete(doc.id)}
                            className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-md transition-colors"
                            title="Delete document"
                            aria-label="Delete document"
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
