#!/bin/bash

# GitHub CLI Setup Script for Sherlog MCP
# This script helps users set up GitHub CLI in the MCP container

set -e

echo "🧙 GitHub CLI Setup for Sherlog MCP"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running inside container
if [ ! -f /.dockerenv ] && [ ! -f /run/.containerenv ]; then
    echo -e "${YELLOW}⚠️  Warning: This script should be run inside the MCP container${NC}"
    echo "Run: docker exec -it <container-name> /bin/bash"
    echo ""
fi

# Check if gh is already installed
if command -v gh &> /dev/null; then
    echo -e "${GREEN}✅ GitHub CLI is already installed${NC}"
    
    # Check authentication status
    if gh auth status &> /dev/null; then
        echo -e "${GREEN}✅ GitHub CLI is already authenticated!${NC}"
        echo ""
        echo "You're all set! The LLM can now use GitHub CLI commands like:"
        echo "  - gh repo list --json name,description"
        echo "  - gh pr create --title 'Fix' --body 'Details'"
        echo "  - gh issue list --json number,title,state"
        exit 0
    else
        echo -e "${YELLOW}🔐 GitHub CLI needs authentication${NC}"
    fi
else
    # Install GitHub CLI
    echo -e "${YELLOW}📦 Installing GitHub CLI...${NC}"
    
    # Detect OS
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        echo "Detected Debian/Ubuntu system"
        
        # Add GitHub CLI repository
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
        chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
        
        # Update and install
        apt-get update -qq
        apt-get install -y gh
        
    elif [ -f /etc/redhat-release ]; then
        # RHEL/CentOS/Fedora
        echo "Detected Red Hat system"
        dnf install -y 'dnf-command(config-manager)'
        dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
        dnf install -y gh
        
    elif [ -f /etc/alpine-release ]; then
        # Alpine Linux
        echo "Detected Alpine Linux"
        apk add github-cli
        
    else
        echo -e "${RED}❌ Unsupported operating system${NC}"
        echo "Please install GitHub CLI manually:"
        echo "https://github.com/cli/cli#installation"
        exit 1
    fi
    
    echo -e "${GREEN}✅ GitHub CLI installed successfully${NC}"
fi

# Authentication instructions
echo ""
echo -e "${YELLOW}🔐 Now authenticate GitHub CLI:${NC}"
echo ""
echo "Run the following command:"
echo -e "${GREEN}gh auth login${NC}"
echo ""
echo "When prompted, select:"
echo "  1. GitHub.com"
echo "  2. HTTPS"
echo "  3. Login with a web browser"
echo ""
echo "The CLI will show you a code and open a browser (or give you a URL)."
echo "Complete the authentication in your browser."
echo ""
echo -e "${YELLOW}📝 Note:${NC} If you're in a headless environment, you can use:"
echo "  - Personal Access Token authentication instead"
echo "  - Run: gh auth login --with-token < your-token-file.txt"
echo ""

# Quick test command
echo "After authentication, test with:"
echo -e "${GREEN}gh auth status${NC}"
echo ""
echo "Happy coding! 🚀"