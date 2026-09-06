import { Outlet, Link, useLocation, NavLink } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import { Menu, X, LogOut, Settings, FileText, MessageSquare, LayoutDashboard, Shield, Activity } from 'lucide-react'
import { useState } from 'react'
import clsx from 'clsx'

export function Layout() {
  const { user, logout } = useAuth()
  const location = useLocation()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/documents', label: 'Documents', icon: FileText },
    { path: '/chat', label: 'Chat & Citations', icon: MessageSquare },
    { path: '/settings', label: 'Settings', icon: Settings },
  ]

  const userInitial = user?.email ? user.email[0].toUpperCase() : 'U'

  return (
    <div className="min-h-screen bg-[var(--color-background)]">
      {/* Mobile sidebar overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden transition-opacity duration-300"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={clsx(
          'fixed inset-y-0 left-0 z-50 w-64 bg-[var(--color-surface)] border-r border-[var(--color-border)] transition-transform duration-300 ease-out lg:translate-x-0 flex flex-col',
          mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between h-16 px-5 border-b border-[var(--color-border)]">
          <Link to="/" className="flex items-center gap-3 font-semibold text-lg text-[var(--color-text)] tracking-tight">
            <div className="w-8 h-8 rounded-lg bg-[var(--color-primary)] flex items-center justify-center text-white shadow-sm flex-shrink-0">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-base leading-none text-slate-900">KIP</span>
              <span className="text-[10px] uppercase font-semibold tracking-wider text-[var(--color-text-muted)] mt-1">Intelligence Platform</span>
            </div>
          </Link>
          <button
            className="lg:hidden p-1.5 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-muted)] transition-colors"
            onClick={() => setMobileMenuOpen(false)}
            aria-label="Close menu"
          >
            <X size={20} />
          </button>
        </div>

        {/* Navigation */}
        <div className="px-3 pt-4 pb-2">
          <p className="px-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-subtle)] mb-2">
            Navigation
          </p>
          <nav className="space-y-1">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all',
                    isActive
                      ? 'bg-[var(--color-primary-light)] text-[var(--color-primary)] font-semibold shadow-xs'
                      : 'text-[var(--color-text-muted)] hover:bg-[var(--color-surface-muted)] hover:text-[var(--color-text)]'
                  )
                }
                onClick={() => setMobileMenuOpen(false)}
              >
                <item.icon size={18} className="flex-shrink-0" />
                <span>{item.label}</span>
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Engine Status pill */}
        <div className="mx-4 my-2 p-3 bg-slate-50 border border-[var(--color-border)] rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="status-dot status-dot-pulse bg-emerald-500" />
              <span className="text-xs font-medium text-slate-700">RAG Engine</span>
            </div>
            <span className="badge badge-success text-[10px] py-0.5">Online</span>
          </div>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* User profile footer */}
        <div className="p-3 border-t border-[var(--color-border)] bg-[var(--color-surface-subtle)]">
          <div className="flex items-center gap-3 p-2 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] mb-2 shadow-xs">
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-600 to-blue-500 bg-[var(--color-primary)] text-white flex items-center justify-center font-bold text-xs flex-shrink-0 shadow-xs">
              {userInitial}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-[var(--color-text)] truncate">{user?.email || 'User'}</p>
              <div className="flex items-center gap-1 mt-0.5">
                {user?.is_superuser ? (
                  <span className="inline-flex items-center gap-0.5 text-[10px] font-medium text-purple-600">
                    <Shield size={10} /> Admin
                  </span>
                ) : (
                  <span className="text-[10px] text-[var(--color-text-muted)]">Active Workspace</span>
                )}
              </div>
            </div>
          </div>
          <button
            onClick={logout}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-[var(--color-text-muted)] hover:text-[var(--color-error)] hover:bg-[var(--color-error-bg)] rounded-lg transition-colors border border-transparent hover:border-[var(--color-error-border)]"
          >
            <LogOut size={15} />
            <span>Sign out</span>
          </button>
        </div>
      </aside>

      {/* Main content wrapper */}
      <div className="lg:pl-64 flex flex-col min-h-screen">
        {/* Top bar */}
        <header className="sticky top-0 z-30 h-16 bg-[var(--color-surface)]/90 backdrop-blur-md border-b border-[var(--color-border)] transition-colors">
          <div className="flex items-center justify-between h-full px-4 lg:px-8">
            <div className="flex items-center gap-3">
              <button
                className="lg:hidden p-2 rounded-lg text-[var(--color-text-muted)] hover:bg-[var(--color-surface-muted)] hover:text-[var(--color-text)] transition-colors"
                onClick={() => setMobileMenuOpen(true)}
                aria-label="Open menu"
              >
                <Menu size={22} />
              </button>

              <div className="flex items-center gap-2">
                <span className="hidden sm:inline-block text-xs font-medium text-[var(--color-text-muted)]">KIP</span>
                <span className="hidden sm:inline-block text-xs text-[var(--color-text-subtle)]">/</span>
                <h1 className="text-sm lg:text-base font-semibold text-[var(--color-text)] truncate">
                  {navItems.find((i) => i.path === location.pathname)?.label || 'Overview'}
                </h1>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-xs text-slate-600 font-medium">
                <Activity size={12} className="text-emerald-500" />
                <span>API Connected</span>
              </div>
              <div className="text-xs text-[var(--color-text-muted)] hidden md:block">
                {user?.email}
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 p-4 lg:p-8 max-w-7xl w-full mx-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
