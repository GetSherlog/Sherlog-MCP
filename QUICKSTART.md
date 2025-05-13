# LogAI MCP Server Quick Start Guide

This guide will help you quickly set up and start using the LogAI MCP server with Claude desktop.

## 1. Install the Server

### Automatic Installation

Run the installation script:

```bash
./install.sh
```

This script will:
- Create a virtual environment
- Install all required dependencies
- Set up the LogAI package
- Make necessary scripts executable

### Manual Installation

If the automatic installation doesn't work, follow these steps:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install mcp fastmcp
pip install scikit-learn pandas numpy nltk Cython
pip install -e ./logai
python -m nltk.downloader punkt

# Make scripts executable
chmod +x logai_mcp_server.py test_logai_mcp.py
```

## 2. Start the Server

```bash
# Activate the virtual environment (if not already activated)
source venv/bin/activate

# Start the LogAI MCP server
./logai_mcp_server.py
```

The server will start running at http://localhost:8080.

## 3. Test the Server (Optional)

In a new terminal window:

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the test script
./test_logai_mcp.py
```

This will test all the server endpoints to make sure everything is working correctly.

## 4. Install in Claude Desktop

### Using the MCP CLI

```bash
# Activate the virtual environment
source venv/bin/activate

# Install the server in Claude desktop
mcp install logai_mcp_server.py --name "LogAI Analytics"
```

### Manual Installation in Claude Desktop

1. Open Claude desktop
2. Go to Settings → MCP Servers
3. Click "Add Server"
4. Enter:
   - Name: LogAI Analytics
   - URL: http://localhost:8080
5. Click "Add Server"

## 5. Use with Claude

Try these example prompts with Claude:

### List available datasets:
```
What log datasets are available in LogAI?
```

### Run anomaly detection:
```
Using the LogAI Analytics server, please run anomaly detection on the HDFS_2000 dataset with the isolation_forest algorithm.
```

### Cluster logs:
```
Can you use LogAI Analytics to cluster the BGL_2000 logs into 5 groups using kmeans?
```

### Analyze custom logs:
```
I have some server logs I'd like to analyze for anomalies. Can you help me upload and analyze them using LogAI Analytics?
```

## Troubleshooting

- If the server doesn't start, check if you're in the virtual environment (`source venv/bin/activate`)
- If dependencies fail to install, try installing them individually
- If Claude can't find the server, make sure it's running and properly installed in Claude desktop

For more detailed information, see the full README.md file.