# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create var directory for SQLite/default storage
RUN mkdir -p /app/var /app/storage

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "kip.api:app", "--host", "0.0.0.0", "--port", "8000"]
