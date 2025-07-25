![Sherlog Logo](images/sherlog.png)

# Sherlog Log AI MCP



A powerful Model Context Protocol (MCP) server that provides a persistent IPython shell per session. 
The server only exposes a limited number of tools and at the core of it just exposes the following 2 -

1. `call_cli`
2. `execute_python_code`

And the rest of the tools are more for supporting the LLM to write the code or call the appropriate cli to complete the task. 
There are some tools for code retrieval based on tree sitter but thats pretty much it. 

For example, installing packages or installing cli. Or doing introspection of variables in the ipython shell. 

The server also supports adding external mcp servers. The idea here is you can run mcp servers inside the shell that allows you to persist results in the shell with variables and results etc. 

One of the primary things I have noticed is that LLMs mess up when provided with a lot of tools and also as is quite evident by now. There are multiple articles and videos (written and shared by people way more qualified than me) and they main theme across them is the problems with tools overload and context limitation. 

This server is my attempt at "solving" that problem.

**How you ask ?** 

> **Note:** The core of this approach is:
> 1. All tool calls are persisted to a variable in the shell and the LLM would then write code to inspect the variable, slice/dice and get the required part out of a "gigantic" payload. My idea was similar to when us engineers work with large data we get them into dataframes and then slice/dice to get the required part out.
> 2. CLI calls are composable and that helps the LLM to get only the required content
> 
> Both are ways of not polluting the context with useless info.

### Overview

Sherlog MCP Server transforms Claude Desktop into a stateful data analysis powerhouse by providing:

- **Session-Aware IPython Shells**: Isolated workspaces per session with automatic persistence
- **DataFrame-Centric Architecture**: Every operation returns DataFrames, creating a unified data model
- **Multi-Session Support**: Handle up to 4 concurrent sessions with automatic lifecycle management
- **MCP Proxy**: Seamlessly integrates any external MCP server, executing all operations within the same IPython context

Think of it as giving Claude a persistent Python notebook that maintains separate workspaces for different conversations, where every piece of data is immediately available for the next operation.


## Architecture & Design

Sherlog MCP Server supports different deployment scenarios through specialized Docker containers:

### Container Variants

#### **Vanilla Container**
A lightweight, general-purpose environment optimized for data analysis and development workflows.

**Pre-installed Tools:**
- **GitHub CLI (`gh`)** - Complete GitHub integration for repository management, issues, PRs, and workflows
- **Python Ecosystem** - Full scientific computing stack (pandas, numpy, matplotlib, etc.)
- **System Utilities** - Essential command-line tools for file operations and system management

**Best For:** Data analysis, web scraping, API integrations, general development tasks

#### **Android Development Container**
A specialized environment for Android development and testing workflows.

**Pre-installed Tools:**
- **Android SDK & Build Tools** - Complete Android development environment
- **Java Development Kit (JDK)** - Required Java runtime and development tools
- **GitHub CLI (`gh`)** - Version control and repository management
- **BrowserStack CLI** - Real device testing and debugging capabilities
- **ADB & Fastboot** - Android device communication tools

**Best For:** Android app development, device testing, mobile automation, CI/CD pipelines

### Google OAuth Integration

Sherlog MCP Server includes built-in Google OAuth 2.0 support for accessing Google Workspace services (Gmail, Drive, Calendar) directly within IPython sessions. OAuth tokens are securely stored with encryption and automatically refreshed as needed.

Configure with `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` environment variables. When running with HTTP transport, OAuth endpoints are available at `/auth/google/*` for authentication flow.

### Design Principles

- **Environment Isolation**: Each container provides a complete, reproducible environment
- **Tool Integration**: All CLI tools are accessible through the unified `call_cli` interface
- **Persistent State**: Session data persists across container restarts
- **Extensibility**: Easy to add new tools or create custom container variants

## Installation

### Prerequisites
- Docker Desktop

### Remote Connection Support

Sherlog MCP supports connecting from remote Claude instances via HTTP transport. See [Remote Connection Guide](docs/remote-connection.md) for detailed setup instructions.

## Configuration

### Core Settings

```bash
# Session Management
export MCP_AUTO_RESET_THRESHOLD=200     # Operations before auto-cleanup (default: 200)
export MCP_AUTO_RESET_ENABLED=true      # Enable automatic memory management
export MCP_MAX_OUTPUT_SIZE=50000        # Max output size per buffer (default: 50KB)
export MCP_MAX_SESSIONS=4               # Maximum concurrent sessions (default: 4)

# Logging
export LOG_LEVEL=INFO
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

### External MCP Integration

While Sherlog MCP includes many tools natively, you can connect any MCP server to extend functionality. External tools are automatically integrated into the IPython workspace:

#### How It Works
1. External tools are prefixed: `external_[server]_[tool]`
2. Results automatically convert to DataFrames
3. Full access to the same IPython namespace

#### Adding External MCPs
```json
"-e", "EXTERNAL_MCPS_JSON={\"postgres\":{\"command\":\"npx\",\"args\":[\"-y\",\"@modelcontextprotocol/server-postgres\"],\"env\":{\"DATABASE_URL\":\"$DATABASE_URL\"}}}"
```

## Railway Deployment

This MCP server is designed to be deployed on Railway with persistent session storage.

### Persistent Sessions

The server automatically persists IPython shell sessions across container restarts using:

- **Session State**: Individual session files stored in `/app/data/sessions/`
- **Session Metadata**: Tracked in `session_metadata.json` for session timing and state
- **Active Session Registry**: Maintained in `session_registry.json` to restore shells on startup

### Railway Volume Configuration

When deploying to Railway, the `/app/data` directory is automatically persisted through Railway's persistent storage. This ensures that:

- User sessions survive container restarts
- IPython shell state (variables, imports, etc.) is maintained 
- Session metadata persists for proper session management

No additional configuration is needed for Railway deployment - the persistent volume is automatically mounted.

## Architecture

```
Claude Desktop
     ↓
Sherlog MCP Server (http)
     ↓
Session Middleware (manages shells)
     ↓
IPython Shells (one per session)
     ├── Built-in Tools (return DataFrames)
     ├── External MCP Tools (via proxy)
     └── User Code (execute_python_code)
     └── User CLI (call_cli)
```

## Advanced Usage

### Working with External MCPs

External MCP tools integrate seamlessly:
1. Results automatically convert to DataFrames
2. Stored in IPython namespace with tool name
3. Available for subsequent operations

Example flow:
- PostgreSQL MCP queries database → result stored as DataFrame
- LogAI tools analyze the data → create new DataFrames
- Custom Python code combines results → final analysis

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Talks and articles that inspired me
[Armin's article](https://lucumr.pocoo.org/2025/7/3/tools/)
[How to fix your context](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html)
[MCPs are Boring](https://lucumr.pocoo.org/2025/7/3/tools/#:~:text=Manuel%20Odendahl%27s%20excellent%20%E2%80%9C-,MCPs%20are%20Boring,-%E2%80%9D%20talk%20from%20AI) - Recommended using eval as the only tool
[Alita paper](https://arxiv.org/abs/2505.20286) - This paper kind of influenced me as well