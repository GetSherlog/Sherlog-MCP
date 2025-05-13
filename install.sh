#!/bin/bash
# Installation script for LogAI MCP Server

# Exit immediately if a command exits with a non-zero status
set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================${NC}"
echo -e "${BLUE}LogAI MCP Server Installation Script${NC}"
echo -e "${BLUE}====================================${NC}"
echo

# Check if Python 3.10+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10 or higher is required. You have Python $python_version.${NC}"
    exit 1
fi

echo -e "${GREEN}Found Python $python_version${NC}"

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate the virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}Virtual environment activated.${NC}"

# Update pip
echo -e "${BLUE}Updating pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}Pip updated.${NC}"

# Install MCP packages
echo -e "${BLUE}Installing MCP packages...${NC}"
pip install mcp fastmcp
echo -e "${GREEN}MCP packages installed.${NC}"

# Install core dependencies
echo -e "${BLUE}Installing core dependencies...${NC}"
pip install scikit-learn pandas numpy nltk Cython
echo -e "${GREEN}Core dependencies installed.${NC}"

# Install LogAI
echo -e "${BLUE}Installing LogAI (this may take a few minutes)...${NC}"
pip install -e ./logai
echo -e "${GREEN}LogAI installed.${NC}"

# Download NLTK data
echo -e "${BLUE}Downloading NLTK data...${NC}"
python -m nltk.downloader punkt
echo -e "${GREEN}NLTK data downloaded.${NC}"

# Make Python scripts executable
echo -e "${BLUE}Making scripts executable...${NC}"
chmod +x logai_mcp_server.py test_logai_mcp.py
echo -e "${GREEN}Scripts are now executable.${NC}"

echo
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${GREEN}====================================${NC}"
echo
echo -e "To start the LogAI MCP server:"
echo -e "  1. Activate the virtual environment:"
echo -e "     ${BLUE}source venv/bin/activate${NC}"
echo -e "  2. Run the server:"
echo -e "     ${BLUE}./logai_mcp_server.py${NC}"
echo
echo -e "To test the server:"
echo -e "     ${BLUE}./test_logai_mcp.py${NC}"
echo
echo -e "To install the server in Claude desktop:"
echo -e "     ${BLUE}mcp install logai_mcp_server.py --name \"LogAI Analytics\"${NC}"
echo
echo -e "For more information, see the README.md file."
echo