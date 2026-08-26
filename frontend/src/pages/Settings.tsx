import { useState, useEffect } from 'react'
import { api } from '../lib/api'
import { User, Database, Brain, Zap, Save, Loader2, AlertCircle, CheckCircle, RotateCcw } from 'lucide-react'
import clsx from 'clsx'

interface SettingsData {
  embedding_provider: string
  embedding_model: string
  llm_provider: string
  llm_model: string
  retrieval_mode: string
  reranker: string
  grounding_min_score: number
  vector_store: string
  keyword_index: string
  app_env: string
}

interface ProviderOption {
  name: string
  note: string
}

export function Settings() {
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [passwordData, setPasswordData] = useState({ current: '', new: '', confirm: '' })

  const providers = {
    embedding: [] as ProviderOption[],
    llm: [] as ProviderOption[],
    reranker: [] as ProviderOption[],
    vectorStore: [] as ProviderOption[],
    keywordIndex: [] as ProviderOption[],
  }

  useEffect(() => {
    loadSettings()
    loadProviders()
  }, [])

  const loadSettings = async () => {
    try {
      const response = await api.get('/settings')
      setSettings(response.data)
    } catch (err) {
      setMessage({ type: 'error', text: 'Failed to load settings' })
    } finally {
      setLoading(false)
    }
  }

  const loadProviders = async () => {
    try {
      const [embedding, llm, reranker, vectorStore, keywordIndex] = await Promise.all([
        api.get('/settings/embedding-providers'),
        api.get('/settings/llm-providers'),
        api.get('/settings/rerankers'),
        api.get('/settings/vector-stores'),
        api.get('/settings/keyword-indexes'),
      ])
      providers.embedding = embedding.data
      providers.llm = llm.data
      providers.reranker = reranker.data
      providers.vectorStore = vectorStore.data
      providers.keywordIndex = keywordIndex.data
    } catch (err) {
      console.error('Failed to load providers', err)
    }
  }

  const handleChange = (key: keyof SettingsData, value: string | number) => {
    setSettings((prev) => (prev ? { ...prev, [key]: value } : null))
  }

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault()
    if (passwordData.new !== passwordData.confirm) {
      setMessage({ type: 'error', text: 'New passwords do not match' })
      return
    }
    if (passwordData.new.length < 10) {
      setMessage({ type: 'error', text: 'Password must be at least 10 characters' })
      return
    }
    setSaving(true)
    try {
      await api.post('/auth/change-password', {
        current_password: passwordData.current,
        new_password: passwordData.new,
      })
      setPasswordData({ current: '', new: '', confirm: '' })
      setMessage({ type: 'success', text: 'Password changed successfully' })
    } catch (err: any) {
      setMessage({ type: 'error', text: err.response?.data?.detail || 'Failed to change password' })
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={32} className="animate-spin text-[var(--color-primary)]" />
      </div>
    )
  }

  const selectOptions: Record<keyof SettingsData | 'embedding_provider' | 'llm_provider' | 'reranker' | 'vector_store' | 'keyword_index' | 'retrieval_mode', string[]> = {
    embedding_provider: providers.embedding.map((p) => p.name),
    llm_provider: providers.llm.map((p) => p.name),
    reranker: providers.reranker.map((p) => p.name),
    vector_store: providers.vectorStore.map((p) => p.name),
    keyword_index: providers.keywordIndex.map((p) => p.name),
    retrieval_mode: ['hybrid', 'dense', 'keyword'],
    embedding_model: [],
    llm_model: [],
    grounding_min_score: [],
    app_env: [],
  }

  const renderSelect = (key: keyof typeof selectOptions, label: string, icon: React.ReactNode) => {
    const options = selectOptions[key] || []
    const currentValue = settings?.[key] || ''
    let providerInfo: ProviderOption | undefined

    if (key === 'embedding_provider') {
      providerInfo = providers.embedding.find((p) => p.name === currentValue)
    } else if (key === 'llm_provider') {
      providerInfo = providers.llm.find((p) => p.name === currentValue)
    } else if (key === 'reranker') {
      providerInfo = providers.reranker.find((p) => p.name === currentValue)
    } else if (key === 'vector_store') {
      providerInfo = providers.vectorStore.find((p) => p.name === currentValue)
    } else if (key === 'keyword_index') {
      providerInfo = providers.keywordIndex.find((p) => p.name === currentValue)
    }

    return (
      <div className="card p-5">
        <div className="flex items-center gap-2 mb-3">
          {icon}
          <label className="label">{label}</label>
        </div>
        <select
          value={currentValue}
          onChange={(e) => handleChange(key, e.target.value)}
          className="input"
        >
          {options.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
        {providerInfo && (
          <p className="mt-2 text-sm text-[var(--color-text-muted)]">{providerInfo.note}</p>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold text-[var(--color-text)]">Settings</h1>
        <p className="text-[var(--color-text-muted)]">Configure your Knowledge Intelligence Platform</p>
      </div>

      {message && (
        <div className={clsx('flex items-center gap-2 p-3 rounded-lg', message.type === 'success' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700')}>
          {message.type === 'success' ? <CheckCircle size={18} /> : <AlertCircle size={18} />}
          <span>{message.text}</span>
        </div>
      )}

      {/* Account */}
      <section className="card p-5">
        <h2 className="flex items-center gap-2 text-lg font-semibold mb-4">
          <User size={22} />
          Account
        </h2>
        <form onSubmit={handlePasswordChange} className="space-y-4 max-w-md">
          <div>
            <label className="label">Current Password</label>
            <input
              type="password"
              value={passwordData.current}
              onChange={(e) => setPasswordData({ ...passwordData, current: e.target.value })}
              className="input"
              placeholder="••••••••"
            />
          </div>
          <div>
            <label className="label">New Password</label>
            <input
              type="password"
              value={passwordData.new}
              onChange={(e) => setPasswordData({ ...passwordData, new: e.target.value })}
              className="input"
              placeholder="•••••••••• (min 10 chars)"
              minLength={10}
            />
          </div>
          <div>
            <label className="label">Confirm New Password</label>
            <input
              type="password"
              value={passwordData.confirm}
              onChange={(e) => setPasswordData({ ...passwordData, confirm: e.target.value })}
              className="input"
              placeholder="••••••••••"
            />
          </div>
          <button type="submit" className="btn btn-primary" disabled={saving}>
            {saving ? <Loader2 size={18} className="animate-spin" /> : <Save size={18} />}
            <span>Change Password</span>
          </button>
        </form>
      </section>

      {/* RAG Pipeline */}
      <section className="card p-5">
        <h2 className="flex items-center gap-2 text-lg font-semibold mb-4">
          <Brain size={22} />
          RAG Pipeline
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {renderSelect('embedding_provider', 'Embedding Provider', <Database size={18} />)}
          {renderSelect('llm_provider', 'LLM Provider', <Zap size={18} />)}
          {renderSelect('reranker', 'Reranker', <RotateCcw size={18} />)}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
          {renderSelect('retrieval_mode', 'Retrieval Mode', <Brain size={18} />)}
          {renderSelect('vector_store', 'Vector Store', <Database size={18} />)}
          {renderSelect('keyword_index', 'Keyword Index', <Brain size={18} />)}
        </div>
      </section>

      {/* Grounding Thresholds */}
      <section className="card p-5">
        <h2 className="flex items-center gap-2 text-lg font-semibold mb-4">
          <Zap size={22} />
          Grounding Thresholds
        </h2>
        <p className="text-sm text-[var(--color-text-muted)] mb-4">
          These thresholds control when the system refuses to answer. Higher values = more conservative.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="label">Minimum Similarity Score</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={settings?.grounding_min_score || 0.16}
              onChange={(e) => handleChange('grounding_min_score', parseFloat(e.target.value))}
              className="w-full"
            />
            <p className="text-sm text-[var(--color-text-muted)] mt-1">
              Current: {(settings?.grounding_min_score || 0.16).toFixed(2)}
            </p>
          </div>
        </div>
      </section>

      {/* Environment Info */}
      <section className="card p-5">
        <h2 className="flex items-center gap-2 text-lg font-semibold mb-4">
          <Database size={22} />
          Environment
        </h2>
        <dl className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <dt className="text-[var(--color-text-muted)]">Environment</dt>
            <dd className="font-medium">{settings?.app_env || 'development'}</dd>
          </div>
          <div>
            <dt className="text-[var(--color-text-muted)]">Embedding Model</dt>
            <dd className="font-medium">{settings?.embedding_model || 'kip-hashing-v1'}</dd>
          </div>
          <div>
            <dt className="text-[var(--color-text-muted)]">LLM Model</dt>
            <dd className="font-medium">{settings?.llm_model || 'kip-extractive-v1'}</dd>
          </div>
        </dl>
      </section>
    </div>
  )
}
