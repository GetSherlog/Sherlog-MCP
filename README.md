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

## Grafana Integration

This MCP server also includes comprehensive Grafana integration, allowing you to interact with Grafana instances, query Prometheus metrics, and analyze Loki logs directly through Claude. The Grafana tools include:

### Prometheus Tools
- **query_prometheus**: Execute PromQL queries against Prometheus datasources
- **list_prometheus_metric_metadata**: List metric metadata from Prometheus
- **list_prometheus_metric_names**: List available metric names
- **list_prometheus_label_names**: List label names matching a selector
- **list_prometheus_label_values**: List values for a specific label

### Loki Tools
- **query_loki_logs**: Query and retrieve logs using LogQL (either log or metric queries)
- **list_loki_label_names**: List all available label names in logs
- **list_loki_label_values**: List values for a specific log label
- **query_loki_stats**: Get statistics about log streams

### Configuration

To use the Grafana tools, you need to set the following environment variables:

```bash
export GRAFANA_URL="http://localhost:3000"  # Your Grafana instance URL
export GRAFANA_API_KEY="your_api_key_here"  # Your Grafana API key
```

You can create a Grafana API key by:
1. Going to Administration → Users and access → Service accounts in your Grafana instance
2. Creating a new service account with appropriate permissions
3. Generating a token for that service account

### Example Grafana Usage with Claude

Once configured, you can ask Claude to interact with your Grafana instance:

```
Query my Prometheus datasource for CPU usage: rate(cpu_usage_total[5m])
```

```
Show me the available metrics in my Prometheus datasource with UID "prometheus-uid"
```

```
Get the last 100 error logs from my Loki datasource using the query: {job="myapp"} |= "ERROR"
```

```
List all the label names available in my Loki logs for the past hour
```

## GitHub Integration

This MCP server includes comprehensive GitHub integration, allowing you to interact with GitHub repositories, analyze issues, pull requests, and commits directly through Claude. The GitHub tools include:

### Issue Tools
- **get_issue**: Get details of a specific issue from a repository
- **search_issues**: Search for issues in a repository with filters

### Pull Request Tools
- **get_pull_request**: Get details of a specific pull request
- **list_pull_requests**: List and filter pull requests in a repository
- **get_pull_request_files**: Get the list of files changed in a pull request
- **get_pull_request_comments**: Get review comments on a pull request
- **get_pull_request_reviews**: Get reviews on a pull request

### Commit Tools
- **list_commits**: List commits in a repository with optional filters

### Configuration

To use the GitHub tools, you need to set your GitHub Personal Access Token:

```bash
export GITHUB_PAT_TOKEN="your_github_token_here"  # Your GitHub Personal Access Token
```

You can create a GitHub Personal Access Token by:
1. Going to Settings → Developer settings → Personal access tokens in your GitHub account
2. Generating a new token with appropriate repository permissions
3. For public repositories, you need `public_repo` scope
4. For private repositories, you need `repo` scope

### Example GitHub Usage with Claude

Once configured, you can ask Claude to interact with GitHub repositories:

```
Get issue #123 from the microsoft/vscode repository
```

```
List all open pull requests in the facebook/react repository
```

```
Show me the files changed in pull request #456 from the tensorflow/tensorflow repository
```

```
Get all commits from the main branch of the nodejs/node repository from the last week
```

```
Search for open issues labeled "bug" in the python/cpython repository
```

```
Get all review comments for pull request #789 in the kubernetes/kubernetes repository
```

## AWS S3 Integration

This MCP server includes comprehensive AWS S3 integration, allowing you to interact with Amazon S3 buckets and objects directly through Claude. The S3 tools include:

### Bucket Operations
- **list_s3_buckets**: List all S3 buckets in your AWS account
- **create_s3_bucket**: Create new S3 buckets with regional configuration
- **delete_s3_bucket**: Delete S3 buckets (with optional force delete of all contents)

### Object Operations
- **list_s3_objects**: List objects in S3 buckets with prefix filtering
- **upload_s3_object**: Upload files to S3 with automatic content type detection
- **download_s3_object**: Download objects from S3 to local storage
- **delete_s3_object**: Delete specific objects from S3 buckets
- **get_s3_object_info**: Get detailed metadata about S3 objects
- **read_s3_object_content**: Read text content from S3 objects (with size limits for safety)

### Configuration

To use the S3 tools, you need to configure AWS credentials using one of these methods:

**Method 1: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID="your_access_key_here"
export AWS_SECRET_ACCESS_KEY="your_secret_key_here"
export AWS_REGION="us-east-1"  # or your preferred region
```

**Method 2: AWS CLI Configuration**
```bash
aws configure
```

**Method 3: AWS Credentials File**
Create `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your_access_key_here
aws_secret_access_key = your_secret_key_here
region = us-east-1
```

**Method 4: IAM Roles**
If running on EC2, you can use IAM roles for automatic credential management.

### Required Permissions

Your AWS credentials need the following S3 permissions:
- `s3:ListBucket` - List bucket contents
- `s3:GetObject` - Download/read objects
- `s3:PutObject` - Upload objects  
- `s3:DeleteObject` - Delete objects
- `s3:CreateBucket` - Create buckets
- `s3:DeleteBucket` - Delete buckets
- `s3:ListAllMyBuckets` - List all buckets

### Example S3 Usage with Claude

Once configured, you can ask Claude to interact with your S3 storage:

```
List all my S3 buckets and show their creation dates
```

```
Create a new S3 bucket called "my-data-bucket" in us-west-2 region
```

```
Upload the file "/tmp/logfile.txt" to bucket "my-logs" with key "2024/01/logfile.txt"
```

```
List all objects in bucket "my-data" that start with "logs/"
```

```
Download the object "data/report.csv" from bucket "my-reports" to "/tmp/report.csv"
```

```
Read the content of the small text file "config/settings.txt" from bucket "my-config"
```

```
Get detailed information about the object "images/photo.jpg" in bucket "my-media"
```

```
Delete the object "temp/old-file.txt" from bucket "my-temp-storage"
```

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

### Grafana Monitoring and Metrics Analysis

```
I need to analyze the performance of my web application. Please use the Grafana tools to:
1. Query my Prometheus datasource (UID: "prometheus-main") for HTTP request rate over the last hour: rate(http_requests_total[5m])
2. Check for any error rate spikes: rate(http_requests_total{status=~"5.."}[5m])
3. Analyze the results and tell me if there are any concerning patterns
```

### Log Analysis with Grafana Loki

```
I'm investigating an incident that occurred between 2024-01-15T10:00:00Z and 2024-01-15T11:00:00Z. Using my Loki datasource (UID: "loki-main"), please:
1. Query for all error logs during that time: {job="webapp"} |= "ERROR"
2. Get statistics about the log streams for that period
3. List the most common error patterns
4. Help me understand what might have caused the incident
```

### Infrastructure Monitoring Deep Dive

```
I want to do a comprehensive health check of my infrastructure. Please help me by:
1. Getting all available Prometheus metric names from datasource "prometheus-infrastructure"
2. Querying CPU usage across all hosts: avg by (instance) (rate(cpu_usage_total[5m]))
3. Checking memory usage: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100
4. Looking at disk usage: df_bytes{mountpoint="/"} / df_size_bytes{mountpoint="/"} * 100
5. Analyzing the results to identify any systems that need attention
```

### GitHub Repository Analysis and Issue Investigation

```
I'm investigating a performance issue in our application. Please help me by:
1. Getting issue #234 from our main repository owner/myapp
2. Listing recent commits from the last 48 hours to see what might have caused the issue: since="2024-01-15T00:00:00Z"
3. Checking if there are any open pull requests that might be related to performance
4. If there are related PRs, get the files changed to understand the scope of modifications
5. Combine this with LogAI analysis - can you run anomaly detection on our recent application logs to correlate with the code changes?
```

### Comprehensive Development Workflow Analysis

```
I want to analyze our development workflow and code quality. Please:
1. List all open pull requests in our repository owner/myproject 
2. For each PR, get the review comments and current review status
3. Get the files changed in the most recent 5 pull requests
4. List commits from the main branch for the last week
5. Search for any open issues labeled "bug" or "critical"
6. Use LogAI to analyze any error patterns in our CI/CD logs and correlate them with the recent code changes
```

### Security and Quality Analysis

```
I need to do a security and quality review of our repository. Can you:
1. Search for issues labeled "security" in owner/myrepo
2. Get details on any critical security issues that are currently open
3. List all commits from the last month by author "security-bot" or containing "security" in the message
4. Check recent pull request reviews to see if any security concerns were raised
5. Run LogAI clustering on our application logs to identify unusual patterns that might indicate security issues
6. Cross-reference the LogAI findings with the GitHub security issues to identify potential correlations
```

### Release and Deployment Correlation

```
We're preparing for a release and need to understand the current state. Please help by:
1. Listing all merged pull requests since our last release tag
2. Getting all commits from the release branch for the past 2 weeks  
3. Checking if there are any open issues that should block the release
4. For any blocking issues, get the full details and any related pull requests
5. Use LogAI to analyze production logs from the current version to identify any anomalies
6. Compare the LogAI anomaly patterns with the changes planned for the new release
7. Provide a risk assessment based on the correlation between code changes and log anomalies
```

### S3 + LogAI Integration Examples

These examples show how to combine S3 storage with LogAI analytics for comprehensive log analysis workflows:

#### Log Storage and Analysis Pipeline

```
I have application logs stored in my S3 bucket "production-logs". Please help me analyze them:
1. List all log files in the "application-logs/2024/01/" prefix in my S3 bucket
2. Download the most recent log file to analyze locally
3. Run LogAI anomaly detection on the downloaded logs using isolation_forest
4. If anomalies are found, upload a summary report back to S3 under "analysis-reports/"
5. Provide insights on the detected anomalies and their potential impact
```

#### Multi-Source Log Correlation

```
I want to correlate logs from different sources using S3 and LogAI:
1. Download web server logs from s3://my-logs/nginx/access.log
2. Download application logs from s3://my-logs/app/application.log  
3. Download database logs from s3://my-logs/db/postgres.log
4. Use LogAI to cluster each log type separately to understand normal patterns
5. Run anomaly detection across all log sources to find correlations
6. Upload the correlation analysis results to s3://my-logs/analysis/multi-source-report.json
```

#### Automated Log Processing Workflow

```
Set up an automated log analysis workflow:
1. List all unprocessed log files in S3 bucket "incoming-logs" (files without ".processed" suffix)
2. For each unprocessed file:
   - Download the log file
   - Run LogAI parsing using the drain parser
   - Perform anomaly detection with one_class_svm
   - Generate a summary report
   - Upload the report to "processed-logs/" with timestamp
   - Create a marker file with ".processed" suffix
3. Provide a summary of all processed files and key findings
```

#### Historical Log Trend Analysis

```
I want to analyze log trends over time using S3 storage:
1. List all daily log files in s3://historical-logs/ for the past 30 days
2. Download a sample from each day (e.g., one file per day)
3. Use LogAI clustering to identify the main log patterns for each day
4. Compare clustering results across days to identify trending issues
5. Run anomaly detection to find days with unusual patterns
6. Create a trend analysis report and store it in s3://analytics/trends/monthly-report.json
```

#### Incident Investigation with S3 Logs

```
I'm investigating an incident that occurred on 2024-01-15. Help me analyze using S3 and LogAI:
1. Search S3 bucket "incident-logs" for all files from 2024-01-15
2. Download logs from critical systems: web-servers, databases, load-balancers
3. Use LogAI to parse and extract error patterns from each log source
4. Run anomaly detection focusing on the incident timeframe (10:00-12:00 UTC)
5. Cross-correlate anomalies across different log sources
6. Generate an incident analysis report with timeline and root cause analysis
7. Upload the complete investigation report to s3://incident-reports/2024-01-15/
```

#### Log Backup and Compliance Analysis

```
Help me with log backup validation and compliance analysis:
1. List all log files in our compliance bucket s3://compliance-logs/
2. Verify the integrity and completeness of stored logs for the past quarter
3. Download a representative sample of logs from each month
4. Use LogAI to validate log format consistency and detect any data quality issues
5. Check for any gaps in log coverage that might affect compliance
6. Generate a compliance report with recommendations
7. Store the compliance validation report in s3://compliance-reports/
```

#### Performance Log Analysis with S3

```
Analyze application performance using S3-stored logs:
1. Download performance logs from s3://perf-logs/application/ for the past week
2. Use LogAI to extract performance metrics and response times from logs
3. Run clustering to identify different performance patterns (fast, slow, error conditions)
4. Detect performance anomalies using LogAI anomaly detection
5. Correlate performance issues with application deployment times
6. Generate performance insights and recommendations
7. Upload performance analysis dashboard data to s3://dashboards/performance/
```

#### Security Log Analysis Pipeline

```
Set up a security-focused log analysis using S3 and LogAI:
1. Access security logs from s3://security-logs/ including access logs, auth logs, and firewall logs
2. Download logs from the past 24 hours for real-time threat analysis
3. Use LogAI to parse and normalize different security log formats
4. Run anomaly detection specifically tuned for security events
5. Cluster security events to identify attack patterns or suspicious behavior
6. Cross-reference findings with threat intelligence (if available in other S3 buckets)
7. Generate security alerts and store them in s3://security-alerts/
8. Provide actionable security recommendations based on the analysis
```

## Contributing

Contributions to improve the LogAI MCP server are welcome! Please follow the standard GitHub pull request process.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.