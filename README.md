# LogAI MCP Server

This project exposes [LogAI](https://github.com/salesforce/logai) functionality as an MCP (Model Context Protocol) server for use with AI-powered investigation platforms like [Sherlog Canvas](https://github.com/GetSherlog/Canvas) and Claude Desktop.

## 🔥 Sherlog Canvas Integration

**This MCP server is specifically designed to work as the primary data analysis engine for [Sherlog Canvas](https://github.com/GetSherlog/Canvas)** - an AI-powered notebook interface for investigations.

### Quick Start for Sherlog Canvas

1. **Start the LogAI MCP server:**
   ```bash
   python logai_mcp_server.py
   ```
   
   This starts the server with:
   - **Streamable HTTP transport** for web compatibility
   - **Port 8000** by default (configurable)
   - **Stateless HTTP mode** for multi-node deployments

2. **Add to Sherlog Canvas:**
   - Open Sherlog Canvas data connections
   - Add new MCP server connection:
     - **URL**: `http://localhost:8000`
     - **Transport**: `streamable-http`
     - **Name**: `LogAI Analytics`

3. **Start investigating!** Ask Sherlog Canvas to:
   - "Analyze logs for anomalies in the payment service"
   - "Cluster similar error messages from the last hour"
   - "Query Prometheus metrics for CPU usage spikes"
   - "Search GitHub issues related to authentication failures"

### Why Streamable HTTP Transport?

Unlike stdio transport (used for Claude Desktop), **streamable HTTP transport is essential for web applications** because:
- ✅ **HTTP-based**: Compatible with web backends
- ✅ **REST API compatible**: Works with standard HTTP clients
- ✅ **Firewall-friendly**: Uses standard HTTP ports
- ✅ **Scalable**: Stateless mode supports multiple instances

### Available Analysis Capabilities

When integrated with Sherlog Canvas, you get access to:

- 🤖 **AI-Powered Log Analysis** (LogAI algorithms)
- 📊 **Grafana & Prometheus Integration**
- 🐙 **GitHub Repository Analysis**
- ☁️ **AWS CloudWatch & S3 Operations**
- 🚨 **Sentry Error Tracking**
- 📈 **Mixpanel Analytics**
- 🐳 **Docker Container Management**
- 📁 **File System Operations**

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
- **get_commit_details**: Get detailed information about a specific commit, including files changed and diffs
- **analyze_file_commits_around_issue**: Analyze commits to specific files around the time an issue was created to identify what changes might be responsible

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

```
Get detailed information about commit abc123def456 including all file changes and diffs from the microsoft/vscode repository
```

```
Analyze commits around issue #456 from the tensorflow/tensorflow repository to see what file changes might have caused the issue. Look 3 days before and 1 day after the issue was created.
```

```
Check commits to specific files (src/main.py,tests/test_main.py) around issue #789 from the python/myproject repository to correlate file changes with the reported bug
```

```
I suspect a recent change to our authentication system caused login issues (issue #789). Can you:
1. Get issue #789 details to see exactly when users started reporting problems
2. Analyze commits around that time specifically for files matching: auth.py,login.py,session.py
3. Get detailed diffs for any commits that modified these authentication-related files
4. Show me the timeline: when was the issue reported vs when were the auth files last modified
5. Help me determine if there's a clear correlation between the code changes and the reported issue
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

## External MCP Integration

LogAI MCP can dynamically integrate with any other MCP server, making their tools available as native LogAI tools with automatic DataFrame conversion. This allows you to combine LogAI's powerful analytics with data from any MCP-compatible source.

### Configuration

Create an `mcp.json` file in your project directory:

```json
{
  "mcpServers": {
    "google-sheets": {
      "command": "uvx",
      "args": ["mcp-google-sheets@latest"],
      "env": {
        "SERVICE_ACCOUNT_PATH": "/path/to/service-account.json",
        "DRIVE_FOLDER_ID": "optional-folder-id"
      }
    }
  }
}
```

The server looks for `mcp.json` in:
1. Current working directory
2. `~/.logai-mcp/mcp.json`
3. LogAI MCP installation directory
4. Custom path via `MCP_CONFIG_PATH` environment variable

### How It Works

1. **Automatic Discovery**: On startup, LogAI MCP connects to configured external MCPs and discovers their tools
2. **Dynamic Registration**: Each external tool is registered with a prefixed name (e.g., `google-sheets_read_sheet_data`)
3. **DataFrame Integration**: Results are automatically converted to pandas DataFrames when possible
4. **IPython Shell Storage**: All results are stored in the persistent IPython shell for further analysis

### Complete Google Sheets Setup Example

#### Step 1: Create Google Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Sheets API:
   - Navigate to "APIs & Services" → "Library"
   - Search for "Google Sheets API"
   - Click and enable it
4. Create Service Account:
   - Go to "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"
   - Give it a name (e.g., "logai-mcp-sheets")
   - Grant it "Editor" role (or customize permissions)
   - Click "Create Key" → JSON format
   - Save the downloaded JSON file securely

#### Step 2: Configure mcp.json

Create `mcp.json` in your LogAI MCP directory:

```json
{
  "mcpServers": {
    "google-sheets": {
      "command": "uvx",
      "args": ["mcp-google-sheets@latest"],
      "env": {
        "SERVICE_ACCOUNT_PATH": "/absolute/path/to/your-service-account-key.json",
        "DRIVE_FOLDER_ID": "1xlNdWFfQjb4pyvK3WLD2Xp0Lxt8SSMlk"
      }
    }
  }
}
```

**Note**: The `DRIVE_FOLDER_ID` is optional. If provided, it limits access to sheets in that specific folder. You can find the folder ID in the Google Drive URL: `https://drive.google.com/drive/folders/[FOLDER_ID_HERE]`

#### Step 3: Share Sheets with Service Account

For the service account to access your sheets:
1. Open the Google Sheet you want to access
2. Click "Share" button
3. Add the service account email (found in your JSON key file)
4. Grant "Editor" or "Viewer" permissions

#### Step 4: Using Google Sheets with LogAI

Once configured, the Google Sheets tools are automatically available:

```python
# List all accessible spreadsheets
list_external_tools()  # Shows all available Google Sheets tools

# List spreadsheets
google-sheets_list_spreadsheets(save_as="my_sheets")

# Read data from a specific sheet
google-sheets_get_sheet_data(
    spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    range="Sheet1!A1:E100",
    save_as="sales_data"
)

# Update cells in a sheet
google-sheets_update_cells(
    spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    range="Sheet1!A1",
    values=[["Updated", "Data"], ["Row", "2"]],
    save_as="update_result"
)

# Now combine with LogAI analytics
detect_anomalies(sales_data, save_as="anomalies")
cluster_logs(sales_data['descriptions'], save_as="clusters")
```

### Available Google Sheets Tools

When configured, you'll have access to these Google Sheets operations:

- `google-sheets_list_spreadsheets` - List all accessible spreadsheets
- `google-sheets_get_spreadsheet` - Get spreadsheet metadata
- `google-sheets_list_sheets` - List sheets within a spreadsheet
- `google-sheets_get_sheet_data` - Read data from a sheet
- `google-sheets_update_cells` - Update cell values
- `google-sheets_append_data` - Append rows to a sheet
- `google-sheets_clear_sheet` - Clear sheet data
- `google-sheets_create_spreadsheet` - Create new spreadsheet
- `google-sheets_create_sheet` - Add new sheet to spreadsheet
- `google-sheets_delete_sheet` - Delete a sheet
- `google-sheets_batch_update` - Perform batch operations
- And more...

### Troubleshooting Google Sheets Integration

1. **Authentication Error**: Ensure service account JSON path is absolute and file exists
2. **Permission Denied**: Share the sheet with service account email
3. **API Not Enabled**: Enable Google Sheets API in Cloud Console
4. **Tool Not Found**: Check logs for MCP registration errors
5. **Rate Limits**: Google Sheets API has quotas; implement delays for bulk operations

### Adding Other External MCPs

You can add multiple MCP servers to `mcp.json`:

```json
{
  "mcpServers": {
    "google-sheets": {
      "command": "uvx",
      "args": ["mcp-google-sheets@latest"],
      "env": {
        "SERVICE_ACCOUNT_PATH": "/path/to/service-account.json"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:password@localhost/dbname"
      }
    },
    "weather": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-weather"]
    },
    "slack": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-slack"],
      "env": {
        "SLACK_BOT_TOKEN": "xoxb-your-token",
        "SLACK_TEAM_ID": "T1234567890"
      }
    }
  }
}
```

### Benefits

- **No Code Changes**: Add any MCP server through configuration alone
- **Unified Workflow**: External data automatically flows into LogAI's analysis pipeline
- **Type Conversion**: Smart conversion of various data formats to DataFrames
- **Persistent State**: Results remain in the IPython shell for complex multi-step analysis
- **Tool Namespacing**: External tools are prefixed to avoid conflicts

### Popular External MCPs

Some popular MCP servers you can integrate:

- **Google Sheets**: Spreadsheet operations and data analysis
- **PostgreSQL**: Database queries and management
- **SQLite**: Local database operations
- **Weather**: Real-time weather data
- **GitHub**: Repository operations (different from built-in tools)
- **Slack**: Message and channel operations
- **Filesystem**: Advanced file operations
- **Time**: Time and timezone utilities
- **Fetch**: HTTP requests and web scraping
- And any other MCP-compatible server!

## Contributing

Contributions to improve the LogAI MCP server are welcome! Please follow the standard GitHub pull request process.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.