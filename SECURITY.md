# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly:

1. **Do not open a public issue** - Security vulnerabilities should be reported privately
2. **Email us** at security@fafda-jalebi.dev (or create a private GitHub Security Advisory)
3. **Include details**: Description, steps to reproduce, potential impact, and any suggested fixes

We will acknowledge receipt within 48 hours and provide a timeline for investigation and remediation.

## Security Best Practices

### For Users

- **Always change the default `JWT_SECRET`** in production (generate with: `python -c "import secrets; print(secrets.token_urlsafe(48))"`)
- **Use strong database passwords** - Change `POSTGRES_PASSWORD` from the default
- **Enable HTTPS** - Use a reverse proxy (nginx, Traefik, Caddy) with TLS termination
- **Restrict `CORS_ORIGINS`** to your specific domain(s) in production
- **Set `ALLOW_REGISTRATION=false`** if you don't need public registration
- **Keep dependencies updated** - Run `pip-audit` and `npm audit` regularly

### For Developers

- **Never commit secrets** - Use `.env` files (already in `.gitignore`)
- **Run security checks** - `pip-audit` for Python, `npm audit` for Node.js
- **Follow secure coding practices** - Input validation, parameterized queries, proper error handling
- **Review dependencies** - Check for known vulnerabilities before adding new packages

## Security Features

- **JWT Authentication** - Stateless auth with configurable expiration
- **Password Hashing** - bcrypt with configurable rounds (default 12)
- **File Upload Validation** - MIME type checking, size limits, extension allowlist
- **SQL Injection Protection** - SQLAlchemy ORM with parameterized queries
- **CORS Configuration** - Restrictable origins
- **Rate Limiting Ready** - Can be added via middleware (e.g., slowapi)

## Disclosure Policy

We follow responsible disclosure. Once a vulnerability is confirmed and patched:
1. We will release a security update
2. We will publish a Security Advisory on GitHub
3. We will credit the reporter (unless they prefer anonymity)

Thank you for helping keep KIP secure!