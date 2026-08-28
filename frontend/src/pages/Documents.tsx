import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import { FileText, Trash2, Loader2, AlertCircle, MessageSquare, ExternalLink } from 'lucide-react'
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
      setDocuments(response.data.documents)
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
    if (e.dataTransfer.files.length > 0) {
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
      await api.post('/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
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
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const statusColors: Record<string, string> = {
    ready: 'bg-green-100 text-green-700',
    pending: 'bg-yellow-100 text-yellow-700',
    failed: 'bg-red-100 text-red-700',
    processing: 'bg-blue-100 text-blue-700',
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text)]">Documents</h1>
          <p className="text-[var(--color-text-muted)]">Manage your document library</p>
        </div>
      </div>

      {/* Upload zone */}
      <div
        className={clsx(
          'card p-6 border-2 border-dashed transition-colors',
          dragActive ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/5' : 'border-[var(--color-border)]'
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
        <div className="text-center">
          {selectedFile ? (
            <div className="flex items-center justify-between p-4 bg-[var(--color-background)] rounded-lg">
              <div className="flex items-center gap-3">
                <FileText className={clsx('text-[var(--color-text-muted)]', selectedFile.type.includes('pdf') && 'text-red-500', selectedFile.type.includes('word') && 'text-blue-500')} size={24} />
                <div>
                  <p className="font-medium">{selectedFile.name}</p>
                  <p className="text-sm text-[var(--color-text-muted)]">{formatSize(selectedFile.size)}</p>
                </div>
              </div>
              <button
                onClick={() => setSelectedFile(null)}
                className="text-[var(--color-text-muted)] hover:text-[var(--color-error)]"
              >
                <ExternalLink size={20} />
              </button>
            </div>
          ) : (
            <>
              <FileText size={48} className="mx-auto text-[var(--color-text-muted)] mb-4" />
              <p className="text-lg font-medium text-[var(--color-text)] mb-1">Drag & drop a document or click to browse</p>
              <p className="text-[var(--color-text-muted)] mb-4">Supports PDF, DOCX, TXT, MD (max 25MB)</p>
              <label htmlFor="file-upload" className="btn btn-primary cursor-pointer">
                Choose File
              </label>
            </>
          )}

          {error && (
            <div className="mt-4 flex items-center gap-2 p-3 bg-[var(--color-error)]/10 border border-[var(--color-error)]/20 text-[var(--color-error)] rounded-lg text-sm">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          {selectedFile && !uploading && (
            <button onClick={handleUpload} className="btn btn-primary mt-4 w-full sm:w-auto" disabled={uploading}>
              <Loader2 className={clsx(uploading && 'animate-spin')} size={18} />
              <span>{uploading ? 'Uploading...' : 'Upload & Process'}</span>
            </button>
          )}
        </div>
      </div>

      {/* Documents list */}
      {loading ? (
        <div className="card p-12 text-center">
          <Loader2 size={32} className="mx-auto animate-spin text-[var(--color-primary)]" />
          <p className="mt-2 text-[var(--color-text-muted)]">Loading documents...</p>
        </div>
      ) : documents.length === 0 ? (
        <div className="card p-12 text-center">
          <FileText size={48} className="mx-auto text-[var(--color-text-muted)] mb-4 opacity-50" />
          <h3 className="text-lg font-medium text-[var(--color-text)] mb-1">No documents yet</h3>
          <p className="text-[var(--color-text-muted)]">Upload your first document to get started</p>
        </div>
      ) : (
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--color-border)] bg-[var(--color-background)]">
                  <th className="px-4 py-3 text-left text-sm font-medium text-[var(--color-text-muted)]">Document</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-[var(--color-text-muted)]">Pages</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-[var(--color-text-muted)]">Chunks</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-[var(--color-text-muted)]">Size</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-[var(--color-text-muted)]">Status</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-[var(--color-text-muted)]">Added</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-[var(--color-text-muted)]">Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.id} className="border-b border-[var(--color-border)] hover:bg-[var(--color-background)]">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <FileText className={clsx('text-[var(--color-text-muted)]', doc.filename.endsWith('.pdf') && 'text-red-500', doc.filename.endsWith('.docx') && 'text-blue-500')} size={20} />
                        <div>
                          <p className="font-medium truncate max-w-xs">{doc.filename}</p>
                          {doc.title && <p className="text-sm text-[var(--color-text-muted)] truncate max-w-xs">{doc.title}</p>}
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm text-[var(--color-text)]">{doc.page_count}</td>
                    <td className="px-4 py-3 text-sm text-[var(--color-text)]">{doc.chunk_count}</td>
                    <td className="px-4 py-3 text-sm text-[var(--color-text-muted)]">{formatSize(doc.size_bytes)}</td>
                    <td className="px-4 py-3">
                      <span className={clsx('badge', statusColors[doc.status] || 'bg-gray-100 text-gray-700')}>
                        {doc.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-[var(--color-text-muted)]">{formatDate(doc.created_at)}</td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => navigate(`/chat?doc=${doc.id}`)}
                          className="p-2 text-[var(--color-text-muted)] hover:text-[var(--color-primary)] hover:bg-[var(--color-primary)]/10 rounded-lg transition-colors"
                          title="Chat with this document"
                        >
                          <MessageSquare size={18} />
                        </button>
                        <button
                          onClick={() => handleDelete(doc.id)}
                          className="p-2 text-[var(--color-text-muted)] hover:text-[var(--color-error)] hover:bg-[var(--color-error)]/10 rounded-lg transition-colors"
                          title="Delete"
                        >
                          <Trash2 size={18} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
