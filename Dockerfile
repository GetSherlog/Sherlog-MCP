# syntax=docker/dockerfile:1

FROM python:3.11-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
COPY sherlog_mcp ./sherlog_mcp
COPY sherlog_mcp_server.py ./

RUN uv pip install --system .

FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
        nodejs \
        npm \
        gh \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp \
    && curl -L -o gwcli-linux-amd64 "https://github.com/GetSherlog/google-workspace-cli/releases/download/v1.0.0/gwcli-linux-amd64" \
    && echo "addaf11913b1c647f6d8e7cca734235d3bbc44c149c69cbc34850f010cc20cea  gwcli-linux-amd64" | sha256sum -c - \
    && chmod +x gwcli-linux-amd64 \
    && mv gwcli-linux-amd64 /usr/local/bin/gwcli \
    && rm -rf /tmp/*

COPY --from=builder /usr/local /usr/local

WORKDIR /app
COPY sherlog_mcp ./sherlog_mcp
COPY sherlog_mcp_server.py ./

RUN mkdir -p /app/data

EXPOSE 8000

ENTRYPOINT ["uv", "run", "-m", "sherlog_mcp.main"] 