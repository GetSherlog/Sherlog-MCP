#!/bin/bash

# Integration test runner for LogAI MCP with Google Sheets

set -e

echo "Starting LogAI MCP integration tests..."

# Clean up any existing containers
echo "Cleaning up existing containers..."
docker-compose -f docker-compose.test.yml down -v

# Build and start services
echo "Building test containers..."
docker-compose -f docker-compose.test.yml build

# Run tests
echo "Running integration tests..."
docker-compose -f docker-compose.test.yml up --abort-on-container-exit --exit-code-from test-runner 2>&1 | tee test-output.log

# Capture exit code
TEST_EXIT_CODE=$?

# Clean up
echo "Cleaning up..."
docker-compose -f docker-compose.test.yml down -v

# Exit with test status
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ All integration tests passed!"
else
    echo "❌ Integration tests failed with exit code: $TEST_EXIT_CODE"
fi

exit $TEST_EXIT_CODE