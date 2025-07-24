#!/bin/bash

# Function to check if a port is available
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

# Function to kill existing server processes
kill_existing_servers() {
    echo "🔍 Checking for existing server processes..."
    pkill -f "uvicorn api:app" 2>/dev/null || true
    sleep 2
}

# Find an available port
find_available_port() {
    local start_port=${1:-8080}
    local port=$start_port
    
    while ! check_port $port; do
        echo "⚠️  Port $port is in use, trying next port..."
        port=$((port + 1))
        if [ $port -gt 8090 ]; then
            echo "❌ Could not find available port between $start_port and 8090"
            exit 1
        fi
    done
    
    echo $port
}

# Main script
echo "🚀 Starting fact-checking server..."

# Kill any existing servers
kill_existing_servers

# Find available port
PORT=$(find_available_port 8080)
echo "✅ Using port $PORT"

# Start the server
echo "🌟 Starting server on http://localhost:$PORT"
python -m uvicorn api:app --host 0.0.0.0 --port $PORT --reload

echo "🛑 Server stopped" 