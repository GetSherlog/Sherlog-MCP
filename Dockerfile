# syntax=docker/dockerfile:1

FROM python:3.11-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
COPY logai_mcp ./logai_mcp
COPY logai_mcp_server.py ./

RUN uv pip install --system .

# Download NLTK data during build to avoid runtime issues
RUN python -m nltk.downloader punkt wordnet averaged_perceptron_tagger

FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /root/nltk_data /root/nltk_data

WORKDIR /app
COPY logai_mcp ./logai_mcp
COPY logai_mcp_server.py ./

# Create data directory for session persistence
RUN mkdir -p /app/data

# Expose port for streamable-http transport
EXPOSE 8000

ENTRYPOINT ["uv", "run", "-m", "logai_mcp.server"] 