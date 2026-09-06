import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from '../lib/api'
import { User, Database, Brain, Zap, Save, Loader2, AlertCircle, CheckCircle2, RotateCcw, Sliders, Server, Lock } from 'lucide-react'
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

  const providersRef = useRef({
    embedding: [] as ProviderOption[],
    llm: [] as ProviderOption[],
    reranker: [] as ProviderOption[],
    vectorStore: [] as ProviderOption[],
    keywordIndex: [] as ProviderOption[],
  })

  const providers = providersRef.current

  const loadSettings = useCallback(async () => {
    try {
      const response = await api.get('/settings')
      setSettings(response.data)
    } catch {
      setMessage({ type: 'error', text: 'Failed to load system settings' })
    } finally {
      setLoading(false)
    }
  }, [])

  const loadProviders = useCallback(async () => {
    try {
      const [embedding, llm, reranker, vectorStore, keywordIndex] = await Promise.all([
        api.get('/settings/embedding-providers'),
        api.get('/settings/llm-providers'),
        api.get('/settings/rerankers'),
        api.get('/settings/vector-stores'),
        api.get('/settings/keyword-indexes'),
      ])
      providersRef.current.embedding = embedding.data || []
      providersRef.current.llm = llm.data || []
      providersRef.current.reranker = reranker.data || []
      providersRef.current.vectorStore = vectorStore.data || []
      providersRef.current.keywordIndex = keywordIndex.data || []
    } catch (err) {
      console.error('Failed to load providers', err)
    }
  }, [])

  useEffect(() => {
    loadSettings()
    loadProviders()
  }, [loadSettings, loadProviders])

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
      setMessage({ type: 'success', text: 'Password updated successfully' })
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } }
      setMessage({ type: 'error', text: axiosError.response?.data?.detail || 'Failed to update password' })
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[300px] gap-3">
        <Loader2 size={32} className="animate-spin text-[var(--color-primary)]" />
        <p className="text-xs text-[var(--color-text-muted)]">Loading configuration...</p>
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
      <div className="p-4 bg-slate-50/80 border border-slate-200 rounded-xl space-y-2">
        <div className="flex items-center gap-2">
          <div className="text-slate-500">{icon}</div>
          <label className="label m-0 text-xs font-semibold text-slate-900">{label}</label>
        </div>
        <select
          value={currentValue}
          onChange={(e) => handleChange(key, e.target.value)}
          className="input text-xs bg-white"
        >
          {options.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
        {providerInfo && (
          <p className="text-[11px] text-[var(--color-text-muted)] leading-tight">{providerInfo.note}</p>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div className="pb-2 border-b border-[var(--color-border)]">
        <h1 className="text-2xl lg:text-3xl font-bold text-slate-900 tracking-tight">System Settings</h1>
        <p className="text-sm text-[var(--color-text-muted)] mt-1">
          Manage account security, RAG pipeline providers, and grounding thresholds.
        </p>
      </div>

      {message && (
        <div
          className={clsx(
            'flex items-center gap-3 p-3.5 rounded-xl text-xs font-medium border',
            message.type === 'success'
              ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
              : 'bg-red-50 text-red-800 border-red-200'
          )}
        >
          {message.type === 'success' ? <CheckCircle2 size={16} className="text-emerald-600" /> : <AlertCircle size={16} className="text-red-600" />}
          <span>{message.text}</span>
        </div>
      )}

      {/* Account Security */}
      <section className="card p-6 shadow-card">
        <div className="flex items-center gap-2.5 mb-1">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center">
            <User size={18} />
          </div>
          <h2 className="text-base font-bold text-slate-900">Account Security</h2>
        </div>
        <p className="text-xs text-[var(--color-text-muted)] mb-5">
          Update your platform credentials and password
        </p>

        <form onSubmit={handlePasswordChange} className="space-y-4 max-w-md">
          <div>
            <label className="label">Current Password</label>
            <div className="relative">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={15} />
              <input
                type="password"
                value={passwordData.current}
                onChange={(e) => setPasswordData({ ...passwordData, current: e.target.value })}
                className="input pl-10 text-xs"
                placeholder="••••••••••••"
                required
              />
            </div>
          </div>

          <div>
            <label className="label">New Password</label>
            <div className="relative">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={15} />
              <input
                type="password"
                value={passwordData.new}
                onChange={(e) => setPasswordData({ ...passwordData, new: e.target.value })}
                className="input pl-10 text-xs"
                placeholder="•••••••••••• (min 10 characters)"
                minLength={10}
                required
              />
            </div>
          </div>

          <div>
            <label className="label">Confirm New Password</label>
            <div className="relative">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={15} />
              <input
                type="password"
                value={passwordData.confirm}
                onChange={(e) => setPasswordData({ ...passwordData, confirm: e.target.value })}
                className="input pl-10 text-xs"
                placeholder="••••••••••••"
                required
              />
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary text-xs font-semibold px-4 py-2"
            disabled={saving}
          >
            {saving ? (
              <>
                <Loader2 size={15} className="animate-spin mr-1.5" />
                <span>Updating Password...</span>
              </>
            ) : (
              <>
                <Save size={15} className="mr-1.5" />
                <span>Update Password</span>
              </>
            )}
          </button>
        </form>
      </section>

      {/* RAG Pipeline Settings */}
      <section className="card p-6 shadow-card">
        <div className="flex items-center gap-2.5 mb-1">
          <div className="w-8 h-8 rounded-lg bg-purple-50 text-purple-600 flex items-center justify-center">
            <Brain size={18} />
          </div>
          <h2 className="text-base font-bold text-slate-900">RAG Pipeline Architecture</h2>
        </div>
        <p className="text-xs text-[var(--color-text-muted)] mb-5">
          Configured embedding models, hybrid search methods, and rerankers
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {renderSelect('embedding_provider', 'Embedding Engine', <Database size={16} />)}
          {renderSelect('llm_provider', 'Synthesis LLM', <Zap size={16} />)}
          {renderSelect('reranker', 'Reranking Algorithm', <RotateCcw size={16} />)}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          {renderSelect('retrieval_mode', 'Retrieval Strategy', <Brain size={16} />)}
          {renderSelect('vector_store', 'Dense Vector Store', <Database size={16} />)}
          {renderSelect('keyword_index', 'Lexical BM25 Index', <Server size={16} />)}
        </div>
      </section>

      {/* Grounding & Evidence Guardrails */}
      <section className="card p-6 shadow-card">
        <div className="flex items-center gap-2.5 mb-1">
          <div className="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
            <Sliders size={18} />
          </div>
          <h2 className="text-base font-bold text-slate-900">Evidence Gate & Grounding Threshold</h2>
        </div>
        <p className="text-xs text-[var(--color-text-muted)] mb-5">
          Controls the minimum similarity cutoff for citation acceptance. Higher scores enforce stricter factual verification.
        </p>

        <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl max-w-lg space-y-3">
          <div className="flex items-center justify-between">
            <label className="label m-0 text-xs font-semibold text-slate-900">
              Minimum Similarity Threshold
            </label>
            <span className="badge badge-info text-xs font-mono font-bold">
              {(settings?.grounding_min_score || 0.16).toFixed(2)}
            </span>
          </div>

          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={settings?.grounding_min_score || 0.16}
            onChange={(e) => handleChange('grounding_min_score', parseFloat(e.target.value))}
            className="w-full accent-indigo-600 cursor-pointer"
          />

          <div className="flex items-center justify-between text-[11px] text-[var(--color-text-muted)] font-medium">
            <span>Permissive (0.00)</span>
            <span>Balanced (0.16)</span>
            <span>Strict (1.00)</span>
          </div>
        </div>
      </section>

      {/* Environment Info */}
      <section className="card p-6 shadow-card">
        <div className="flex items-center gap-2.5 mb-1">
          <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
            <Server size={18} />
          </div>
          <h2 className="text-base font-bold text-slate-900">Environment & Models</h2>
        </div>
        <p className="text-xs text-[var(--color-text-muted)] mb-4">
          Runtime environment and active AI model configurations
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl">
            <p className="text-[11px] font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">Deployment Environment</p>
            <p className="text-xs font-bold text-slate-900 mt-1 capitalize font-mono">{settings?.app_env || 'production'}</p>
          </div>
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl">
            <p className="text-[11px] font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">Embedding Identifier</p>
            <p className="text-xs font-bold text-slate-900 mt-1 font-mono truncate">{settings?.embedding_model || 'kip-hashing-v1'}</p>
          </div>
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl">
            <p className="text-[11px] font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">Synthesis Engine</p>
            <p className="text-xs font-bold text-slate-900 mt-1 font-mono truncate">{settings?.llm_model || 'kip-extractive-v1'}</p>
          </div>
        </div>
      </section>
    </div>
  )
}
