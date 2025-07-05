# Sherlog MCP Server

A powerful Model Context Protocol (MCP) server that provides a persistent IPython workspace for data analysis, log processing, and multi-agent collaboration.

## Overview

Sherlog MCP Server transforms Claude Desktop into a stateful data analysis powerhouse by providing:

- **Persistent IPython Shell**: A living workspace where all operations persist across tool calls
- **DataFrame-Centric Architecture**: Every operation returns DataFrames, creating a unified data model
- **Shared Blackboard**: Perfect for multi-agent workflows where data needs to be passed between operations
- **MCP Proxy**: Seamlessly integrates any external MCP server, executing all operations within the same IPython context

Think of it as giving Claude a persistent Python notebook that never forgets, where every piece of data is immediately available for the next operation.

## Key Features

### 🐍 Persistent IPython Workspace
- **Stateful Execution**: Variables, imports, and results persist across all tool calls
- **Automatic Memory Management**: Smart cleanup after 50 operations to prevent bloat
- **Session Preservation**: Workspace saves on shutdown and restores on startup

### 📊 DataFrame-First Design
- **Unified Data Model**: All tools return pandas/polars DataFrames
- **Seamless Integration**: Results from any tool become inputs for others
- **Smart Conversions**: Automatically converts various formats to DataFrames

### 🔗 MCP Proxy Capabilities
- **Dynamic Tool Integration**: Connect any MCP server and use its tools within the IPython context
- **Unified Namespace**: External tools' results become DataFrames in the shared workspace
- **Zero Configuration**: Just add external MCPs to your environment

## Installation

### Prerequisites
- Docker Desktop
- Claude Desktop

### Quick Start with Docker

1. Add to Claude Desktop configuration:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sherlog": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--volume=/var/run/docker.sock:/var/run/docker.sock",
        "--mount=type=bind,src=/Users/username/,dst=/Users/username/,ro",
        "-e", "MCP_TRANSPORT=stdio",
        "ghcr.io/navneet-mkr/sherlog-mcp:latest"
      ]
    }
  }
}
```

2. Restart Claude Desktop

### Configuration with Environment Variables

For a fully configured setup with external integrations:

```json
{
  "mcpServers": {
    "sherlog": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--volume=/var/run/docker.sock:/var/run/docker.sock",
        "--mount=type=bind,src=/Users/username/,dst=/Users/username/,ro",
        "-e", "MCP_TRANSPORT=stdio",
        "-e", "GITHUB_TOKEN=your_github_token",
        "-e", "AWS_ACCESS_KEY_ID=your_aws_key",
        "-e", "AWS_SECRET_ACCESS_KEY=your_aws_secret",
        "-e", "EXTERNAL_MCPS_JSON={\"filesystem\":{\"command\":\"npx\",\"args\":[\"-y\",\"@modelcontextprotocol/server-filesystem\",\"/Users/username/data\"]}}",
        "ghcr.io/navneet-mkr/sherlog-mcp:latest"
      ]
    }
  }
}
```

**Important Notes:**
- Replace `/Users/username/` with your actual home directory path
- The mount path must match between source and destination for file access to work
- Add read-only (`,ro`) to mounts for security unless write access is needed
- Docker must be running before starting Claude Desktop

### Volume Mounts Explained

The Docker configuration includes two types of mounts:

1. **Docker Socket** (for Docker tools):
   ```
   --volume=/var/run/docker.sock:/var/run/docker.sock
   ```
   Allows the MCP server to manage Docker containers (if using Docker tools)

2. **File System Access**:
   ```
   --mount=type=bind,src=/Users/username/,dst=/Users/username/,ro
   ```
   Grants read-only access to your files. Adjust the path to limit access:
   - For specific project: `src=/Users/username/projects,dst=/Users/username/projects`
   - For data folder only: `src=/Users/username/data,dst=/Users/username/data`

### Common Configurations

**Minimal Setup** (no external integrations):
```json
{
  "mcpServers": {
    "sherlog": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-e", "MCP_TRANSPORT=stdio",
        "ghcr.io/navneet-mkr/sherlog-mcp:latest"
      ]
    }
  }
}
```

**With File Access** (for log analysis):
```json
{
  "mcpServers": {
    "sherlog": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--mount=type=bind,src=/path/to/logs,dst=/data/logs,ro",
        "-e", "MCP_TRANSPORT=stdio",
        "ghcr.io/navneet-mkr/sherlog-mcp:latest"
      ]
    }
  }
}
```

## Configuration

### Core Settings

```bash
# Session Management
export MCP_AUTO_RESET_THRESHOLD=50      # Operations before auto-cleanup
export MCP_AUTO_RESET_ENABLED=true      # Enable automatic memory management
export MCP_MAX_OUTPUT_SIZE=50000        # Max output size per buffer (default: 50KB)

# Logging
export LOG_LEVEL=INFO
```

### External Integrations

Configure API keys for built-in integrations:

```bash
# AWS
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1

# GitHub
export GITHUB_TOKEN=your_token

# Grafana
export GRAFANA_URL=https://your-instance.grafana.net
export GRAFANA_API_KEY=your_key
```

### External MCP Servers

Connect any MCP server to execute within the IPython workspace:

```bash
export EXTERNAL_MCPS_JSON='{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
  },
  "postgres": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-postgres"],
    "env": {
      "DATABASE_URL": "$DATABASE_URL"
    }
  }
}'
```

## Core Concepts

### The IPython Workspace

Every tool execution happens within a persistent IPython shell. This means:

- **Variables Persist**: Create `df` in one tool call, use it in the next
- **Imports Stay**: Import once, use everywhere
- **State Accumulates**: Build complex analyses step by step

### DataFrame as Universal Currency

All tools follow a simple pattern:
1. Execute operation
2. Store result as DataFrame in IPython namespace
3. Return reference for next operation

This creates a powerful chain of operations where each step builds on the last.

### Multi-Agent Blackboard

In multi-agent scenarios, the IPython workspace acts as a shared blackboard:
- Agent A loads data → stores as `raw_data`
- Agent B processes it → creates `processed_data`
- Agent C analyzes results → uses both previous DataFrames

## Available Tools

Sherlog MCP provides a comprehensive set of native tools optimized for the IPython workspace, with the ability to extend functionality through external MCP servers.

### Native Tools (Built-in)

Our native tools are designed to work seamlessly with the DataFrame-centric architecture:

#### Session Management
- `execute_python_code`: Run arbitrary Python code in the workspace
- `list_shell_variables`: See all available DataFrames and variables
- `session_memory_status`: Monitor memory usage and auto-reset status
- `reset_session_now`: Manually trigger a session cleanup

#### Data Sources & Loading
- **Local Files**: `load_file_log_data`, `read_file`, `write_file`
- **AWS S3**: `s3_list_files`, `s3_download_file`, `s3_upload_file`
- **GitHub**: `github_fetch_issues`, `github_fetch_pull_requests`, `github_fetch_commits`
- **Grafana**: `grafana_query_prometheus`, `grafana_query_loki`

#### Log Analysis (Powered by LogAI)
- `detect_anomalies`: Time-series and semantic anomaly detection
- `cluster_logs`: Group similar log entries using various algorithms
- `extract_features`: Generate ML features from log text
- `parse_logs`: Extract structured data from unstructured logs
- `vectorize_logs`: Convert logs to numerical representations

#### Development Tools
- **Docker**: `docker_list_containers`, `docker_logs`, `docker_exec`
- **Kubernetes**: `k8s_get_pods`, `k8s_get_logs`, `k8s_describe_resource`
- **Code Analysis**: `analyze_code_structure`, `search_codebase`

### External MCP Integration

While Sherlog MCP includes many tools natively, you can connect any MCP server to extend functionality. External tools are automatically integrated into the IPython workspace:

#### How It Works
1. External tools are prefixed: `external_[server]_[tool]`
2. Results automatically convert to DataFrames
3. Full access to the same IPython namespace

#### Popular External MCPs
- **Filesystem**: Advanced file operations beyond our built-in tools
- **PostgreSQL/MySQL**: Direct database queries
- **Weather**: Real-time weather data
- **Slack**: Send messages and read channels
- **Google Sheets**: Spreadsheet operations

#### Adding External MCPs
```json
"-e", "EXTERNAL_MCPS_JSON={\"postgres\":{\"command\":\"npx\",\"args\":[\"-y\",\"@modelcontextprotocol/server-postgres\"],\"env\":{\"DATABASE_URL\":\"$DATABASE_URL\"}}}"
```

### Why Native Tools Are Better

Native tools in Sherlog MCP offer advantages over external MCPs:
- **DataFrame Integration**: Results are automatically structured as DataFrames
- **Session Awareness**: Tools can access and modify the IPython namespace
- **Optimized Performance**: No subprocess overhead
- **Unified Error Handling**: Consistent error messages and recovery
- **Cross-Tool State**: Results from one tool are immediately available to others

### Tool Discovery

To see all available tools in your session:
1. Native tools: Check the list above
2. External tools: Use `list_external_tools()` to see connected MCP servers
3. In Claude: Ask "What tools do you have available?"

## Architecture

```
Claude Desktop
     ↓
Sherlog MCP Server (stdio)
     ↓
IPython Shell (persistent workspace)
     ├── Built-in Tools (return DataFrames)
     ├── External MCP Tools (via proxy)
     └── User Code (execute_python_code)
```

## Advanced Usage

### Session Memory Management

The server automatically manages memory to prevent bloat:
- Monitors execution count
- After 50 operations with DataFrames, triggers smart cleanup
- Preserves imports and recent DataFrames
- Configurable via environment variables

### Working with External MCPs

External MCP tools integrate seamlessly:
1. Results automatically convert to DataFrames
2. Stored in IPython namespace with tool name
3. Available for subsequent operations

Example flow:
- PostgreSQL MCP queries database → result stored as DataFrame
- LogAI tools analyze the data → create new DataFrames
- Custom Python code combines results → final analysis

## Development

### Building from Source

If you want to build and run locally:

1. Clone the repository:
```bash
git clone https://github.com/navneet-mkr/sherlog-mcp.git
cd sherlog-mcp
```

2. Build the Docker image:
```bash
./build.sh
```

3. Use your local image in Claude Desktop by replacing the image name:
```json
"ghcr.io/navneet-mkr/sherlog-mcp:latest"
```
with:
```json
"ghcr.io/navneet-mkr/sherlog-mcp:your-version"
```

### Running Tests
```bash
pytest tests/
```

### Debug Mode
```bash
docker run --rm -it \
  -e LOG_LEVEL=DEBUG \
  -e MCP_TRANSPORT=stdio \
  ghcr.io/navneet-mkr/sherlog-mcp:latest
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built on:
- [LogAI](https://github.com/salesforce/logai) by Salesforce
- [Model Context Protocol](https://modelcontextprotocol.io) by Anthropic
- [IPython](https://ipython.org) for the persistent shell