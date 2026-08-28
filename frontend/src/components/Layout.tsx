import { Outlet, Link, useLocation, NavLink } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import { Menu, X, LogOut, Settings, FileText, MessageSquare, Home } from 'lucide-react'
import { useState } from 'react'
import clsx from 'clsx'

export function Layout() {
  const { user, logout } = useAuth()
  const location = useLocation()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/documents', label: 'Documents', icon: FileText },
    { path: '/chat', label: 'Chat', icon: MessageSquare },
    { path: '/settings', label: 'Settings', icon: Settings },
  ]

  return (
    <div className="min-h-screen bg-[var(--color-background)]">
      {/* Mobile sidebar overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={clsx(
          'fixed inset-y-0 left-0 z-50 w-64 bg-[var(--color-surface)] border-r border-[var(--color-border)] transition-transform duration-300 ease-out lg:translate-x-0',
          mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between h-16 px-4 border-b border-[var(--color-border)]">
            <Link to="/" className="flex items-center gap-2 font-semibold text-lg text-[var(--color-primary)]">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
              <span className="hidden sm:inline">KIP</span>
            </Link>
            <button
              className="lg:hidden p-2 rounded-md hover:bg-[var(--color-background)]"
              onClick={() => setMobileMenuOpen(false)}
            >
              <X size={20} />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-[var(--color-primary-light)] text-[var(--color-primary)]'
                      : 'text-[var(--color-text-muted)] hover:bg-[var(--color-background)] hover:text-[var(--color-text)]'
                  )
                }
                onClick={() => setMobileMenuOpen(false)}
              >
                <item.icon size={20} />
                <span>{item.label}</span>
              </NavLink>
            ))}
          </nav>

          {/* User section */}
          <div className="p-4 border-t border-[var(--color-border)]">
            <div className="flex items-center gap-3 px-3 py-2 mb-2">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{user?.email}</p>
                <p className="text-xs text-[var(--color-text-muted)]">User</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium text-[var(--color-error)] hover:bg-[var(--color-error)]/10 rounded-lg transition-colors"
            >
              <LogOut size={18} />
              <span>Sign out</span>
            </button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className='lg:pl-64 transition-all duration-300'>
        {/* Top bar */}
        <header className="sticky top-0 z-30 h-16 bg-[var(--color-surface)] border-b border-[var(--color-border)]">
          <div className="flex items-center justify-between h-full px-4 lg:px-6">
            <button
              className="lg:hidden p-2 rounded-md hover:bg-[var(--color-background)]"
              onClick={() => setMobileMenuOpen(true)}
            >
              <Menu size={24} />
            </button>

            <div className="flex-1 lg:hidden">
              <h1 className="text-lg font-semibold truncate">
                {navItems.find((i) => i.path === location.pathname)?.label || 'KIP'}
              </h1>
            </div>

            <div className="flex items-center gap-4">
              <div className="hidden sm:block text-sm text-[var(--color-text-muted)]">
                {user?.email}
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="p-4 lg:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
