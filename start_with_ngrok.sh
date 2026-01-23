#!/bin/bash
# Automated script to start 2nd Brain backend + ngrok tunnel
# Usage: ./start_with_ngrok.sh [custom-subdomain]

set -e

cd "$(dirname "$0")"

echo "🚀 UCLA 2nd Brain - Ngrok Deployment"
echo "======================================"
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok is not installed!"
    echo ""
    echo "Install with: brew install ngrok/ngrok/ngrok"
    echo "Or visit: https://ngrok.com/download"
    exit 1
fi

echo "✅ ngrok is installed"

# Check if authtoken is configured
if ! ngrok config check &> /dev/null; then
    echo "⚠️  ngrok authtoken not configured"
    echo ""
    echo "1. Sign up at: https://dashboard.ngrok.com/signup"
    echo "2. Get your token: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "3. Run: ngrok config add-authtoken YOUR_TOKEN"
    echo ""
    read -p "Press Enter after configuring authtoken..."
fi

# Set environment variables
export DATABASE_URL="sqlite:///./2ndbrain_ucla.db"
export FLASK_ENV="production"

# Load API keys from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Start backend in background
echo ""
echo "🔧 Starting backend server..."
./venv/bin/python -m backend.api.app > backend.log 2>&1 &
BACKEND_PID=$!

# Save PID for cleanup
echo $BACKEND_PID > .backend.pid

echo "✅ Backend started (PID: $BACKEND_PID)"
echo "📝 Logs: tail -f backend.log"

# Wait for backend to initialize
echo "⏳ Waiting 10 seconds for backend to initialize..."
sleep 10

# Check if backend is responding
if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed!"
    echo "Check backend.log for errors"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "🌐 Starting ngrok tunnel..."
echo ""

# Start ngrok (with custom subdomain if provided)
if [ -n "$1" ]; then
    echo "Using custom subdomain: $1"
    ngrok http 5000 --domain="$1.ngrok.app"
else
    ngrok http 5000
fi

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down..."

    if [ -f .backend.pid ]; then
        BACKEND_PID=$(cat .backend.pid)
        kill $BACKEND_PID 2>/dev/null || true
        rm .backend.pid
        echo "✅ Backend stopped"
    fi

    echo "✅ Cleanup complete"
}

# Register cleanup on exit
trap cleanup EXIT INT TERM
