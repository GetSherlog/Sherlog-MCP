# LogAI MCP Server

This project exposes [LogAI](https://github.com/salesforce/logai) functionality as an MCP (Model Context Protocol) server for use with Claude desktop.

## What is LogAI?

LogAI is a one-stop open source library for log analytics and intelligence developed by Salesforce. It supports various log analytics and log intelligence tasks such as:

- Log parsing and summarization
- Log clustering
- Log anomaly detection
- Deep learning-based log analysis

## What is MCP?

The Model Context Protocol (MCP) is an open-source standard developed by Anthropic for connecting AI assistants like Claude to systems where data lives. It enables Claude to access external tools, APIs, and data repositories.

## Getting Started

### Prerequisites

- Python 3.10+
- Claude desktop app (latest version)
- LogAI dependencies
- FastMCP and MCP SDK

### Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/sherlog-log-ai-mcp.git
cd sherlog-log-ai-mcp
```

2. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:

```bash
# Install MCP packages
pip install mcp fastmcp

# Install core dependencies
pip install scikit-learn pandas numpy nltk Cython

# Install LogAI (development mode)
pip install -e ./logai
```

4. If you encounter any issues with the LogAI installation, try installing optional dependencies:

```bash
# Install full dependencies
pip install "logai[all]"

# Download NLTK data
python -m nltk.downloader punkt
```

## Running the Server

Start the LogAI MCP server:

```bash
python logai_mcp_server.py
```

The server will be available at http://localhost:8080.

You can test if the server is working properly using the included test script:

```bash
python test_logai_mcp.py
```

## Installing the Server in Claude Desktop

### Method 1: Using the MCP CLI

1. Install the MCP command-line tool (if not already installed):

```bash
pip install mcp
```

2. Make sure your LogAI MCP server is running locally.

3. Install the server in Claude desktop:

```bash
mcp install logai_mcp_server.py --name "LogAI Analytics"
```

### Method 2: Manual Installation in Claude Desktop

1. Make sure your LogAI MCP server is running locally.

2. Open Claude desktop application.

3. Click on your profile picture in the bottom-left corner to open settings.

4. Select "MCP Servers" from the settings menu.

5. Click the "Add Server" button.

6. Enter the following details:
   - Name: LogAI Analytics
   - URL: http://localhost:8080
   - (No authentication required for local development)

7. Click "Add Server" to save.

## Using LogAI with Claude Desktop

Once the LogAI MCP server is installed, you can interact with it through Claude using natural language. Here are some examples:

### Listing Available Resources

Ask Claude about the available datasets or algorithms:

```
What log datasets are available in LogAI?
```

```
What anomaly detection algorithms are available in LogAI?
```

Claude will use the MCP server to retrieve and display this information.

### Running Anomaly Detection

Ask Claude to perform anomaly detection on a sample dataset:

```
Can you run anomaly detection on the HDFS_2000 dataset using the isolation_forest algorithm?
```

```
I'd like to analyze the BGL_5000 log dataset for anomalies using one_class_svm.
```

You can specify parameters:

```
Please run anomaly detection on HealthApp_2000 with the following settings:
- Parser: drain
- Vectorizer: tfidf
- Anomaly detector: isolation_forest
```

### Running Log Clustering

Ask Claude to cluster log data:

```
Can you cluster the HDFS_5000 logs into 5 groups using kmeans?
```

```
I'd like to see the log patterns in BGL_2000 clustered using dbscan.
```

### Analyzing Custom Logs

You can upload your own log files for analysis:

1. Prepare a log file on your computer.

2. Ask Claude to analyze it:

```
I have some server logs I'd like to analyze for anomalies. Can you help me?
```

3. Claude will prompt you to upload your file.

4. After uploading, specify the analysis type:

```
Please use the drain parser and one_class_svm detector to find anomalies in these logs.
```

### Advanced Usage

You can customize algorithm parameters:

```
Run anomaly detection on HDFS_2000 with the following configuration:
- Parser: drain with similarity threshold 0.6
- Use word2vec vectorization
- Apply isolation_forest with 100 estimators and auto contamination
```

## Troubleshooting

### Server Connection Issues

If Claude can't connect to your LogAI MCP server:

1. Ensure the server is running (http://localhost:8080)
2. Check if the test script (`test_logai_mcp.py`) works properly
3. Verify the server is correctly installed in Claude desktop
4. Try restarting Claude desktop

### LogAI Installation Issues

If you have issues installing LogAI dependencies:

1. Make sure your Python version is compatible (3.10+)
2. Try installing dependencies individually:
   ```bash
   pip install schema salesforce-merlion Cython nltk gensim scikit-learn pandas numpy spacy
   ```
3. Check the LogAI documentation for specific environment requirements

### Claude Can't Find Tools or Resources

If Claude doesn't recognize the LogAI tools:

1. Make sure Claude recognizes the server is installed (ask "What MCP servers do I have installed?")
2. Try explicitly asking Claude to use the LogAI server (e.g., "Use LogAI Analytics to...")
3. Restart the conversation with Claude

## Example Prompts for Claude

Here are some complete example prompts to get you started:

### Basic Anomaly Detection

```
Using the LogAI Analytics MCP server, could you perform anomaly detection on the HDFS_2000 dataset? Please use the drain parser and one_class_svm detector. After running the analysis, explain what anomalies were found and what they might indicate about the system's health.
```

### Advanced Clustering Analysis

```
I'd like to understand the different types of logs in the BGL_5000 dataset. Using the LogAI Analytics server, please:
1. Run log clustering with 7 clusters using kmeans
2. Show me samples from each cluster
3. Analyze what each cluster might represent in terms of system behavior or events
4. Recommend which clusters might be worth investigating further
```

### Custom Log Analysis Workflow

```
I have some web server logs I need to analyze. I suspect there might be some unusual patterns that could indicate potential security issues. I'd like to:

1. First upload my logs for analysis
2. Run both clustering (to see the main patterns) and anomaly detection
3. Compare the results of different anomaly detection algorithms
4. Get recommendations on what the anomalies might indicate

Can you guide me through this process using the LogAI Analytics server?
```

## Contributing

Contributions to improve the LogAI MCP server are welcome! Please follow the standard GitHub pull request process.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.