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

FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local


WORKDIR /app
COPY logai_mcp ./logai_mcp
COPY logai_mcp_server.py ./

ENTRYPOINT ["uv", "run", "-m", "logai_mcp.server"] 