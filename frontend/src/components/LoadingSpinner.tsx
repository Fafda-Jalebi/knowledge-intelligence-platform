import { Loader2 } from 'lucide-react'

export function LoadingSpinner({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[200px] gap-3 p-6">
      <div className="relative flex items-center justify-center">
        <Loader2 size={32} className="animate-spin text-[var(--color-primary)]" />
      </div>
      {label && <p className="text-sm font-medium text-[var(--color-text-muted)]">{label}</p>}
    </div>
  )
}
