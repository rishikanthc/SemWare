# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Install build essentials (required for many PyPI packages)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential curl && \
#     rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --upgrade pip && pip install uv

# Set working directory
WORKDIR /app

# Copy only dependency files first for better caching (edit if using requirements.txt/pyproject.toml/requirements.lock)
COPY pyproject.toml uv.lock ./

# Sync dependencies (edit the command if you use pyproject.toml vs requirements.txt)
RUN uv sync --native-tls

# Copy the rest of the app code
COPY . .

# Create /db and mount as volume (for LanceDB persistence)
VOLUME ["/db"]

# Set environment variable so your DB code finds the persistent LanceDB folder
# ENV LANCEDB_URI=/db

# Expose FastAPI port
EXPOSE 8000

# Entrypoint: Run FastAPI with hot reload (edit --reload if not for dev)
CMD ["uv", "run", "uvicorn", "zendown_ai.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]

