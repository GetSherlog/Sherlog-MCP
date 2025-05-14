# syntax=docker/dockerfile:1

############################
# Stage 1 – Build / Install
############################
FROM python:3.11-bookworm AS builder

# Install system packages needed for building wheels / git-based deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (dependency resolver / runner)
RUN pip install --no-cache-dir uv

# Put application code under /app
WORKDIR /app

# Copy only the files that affect dependency resolution first (leverages Docker cache)
COPY pyproject.toml logai_mcp_server.py ./

# Install ALL runtime dependencies into the *system* site-packages layer
# uv will read pyproject.toml, resolve pins, create a lock, and `pip install` them.
RUN uv pip install --system -r pyproject.toml

############################
# Stage 2 – Runtime image
############################
FROM python:3.11-slim-bookworm AS runtime

# Copy the pre-installed Python environment from the builder layer
COPY --from=builder /usr/local /usr/local

# Re-create workdir and copy the full source tree (for hot-reloading / edits)
WORKDIR /app
COPY . .

# Expose no network ports – MCP uses stdin/stdout for transport
# CMD is executed through uv so that the in-file deps block is honored if changed later.
ENTRYPOINT ["uv", "run", "logai_mcp_server.py"] 