#!/bin/bash

# Build and push Sherlog MCP Docker image
# Usage: ./build.sh [version]
# If no version is provided, it will increment the patch version

set -e

# Default values
REGISTRY="ghcr.io"
NAMESPACE="navneet-mkr"
IMAGE_NAME="sherlog-mcp"
FULL_IMAGE_NAME="$REGISTRY/$NAMESPACE/$IMAGE_NAME"

# Function to get the latest version from git tags or default to 0.1.0
get_latest_version() {
    # Try to get the latest version tag from git
    latest_tag=$(git tag -l | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n1)
    
    if [ -z "$latest_tag" ]; then
        echo "0.1.0"
    else
        echo "$latest_tag"
    fi
}

# Function to increment patch version
increment_patch_version() {
    local version=$1
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)
    local patch=$(echo $version | cut -d. -f3)
    
    patch=$((patch + 1))
    echo "$major.$minor.$patch"
}

# Determine version to build
if [ $# -eq 1 ]; then
    NEW_VERSION=$1
else
    LATEST_VERSION=$(get_latest_version)
    NEW_VERSION=$(increment_patch_version $LATEST_VERSION)
    echo "No version specified, incrementing from $LATEST_VERSION to $NEW_VERSION"
fi

echo "Building Docker image: $FULL_IMAGE_NAME:$NEW_VERSION"

# Build the image
echo "🔨 Building Docker image..."
docker build -t "$FULL_IMAGE_NAME:$NEW_VERSION" .

# Tag as latest
echo "🏷️  Tagging as latest..."
docker tag "$FULL_IMAGE_NAME:$NEW_VERSION" "$FULL_IMAGE_NAME:latest"

# Test the image
echo "🧪 Testing the image..."
TEST_OUTPUT=$(docker run --rm "$FULL_IMAGE_NAME:$NEW_VERSION" --help 2>&1)
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "❌ Image test failed! Error details:"
    echo "Exit code: $TEST_EXIT_CODE"
    echo "Output:"
    echo "$TEST_OUTPUT"
    echo ""
    echo "Not pushing the image due to test failure."
    exit 1
fi

echo "✅ Image test passed!"