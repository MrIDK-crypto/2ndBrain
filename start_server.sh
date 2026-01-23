#!/bin/bash
# Start 2nd Brain backend server with UCLA data

cd "$(dirname "$0")"

export DATABASE_URL="sqlite:///./2ndbrain_ucla.db"
# Load API keys from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "🚀 Starting 2nd Brain backend..."
echo "📊 Database: 2ndbrain_ucla.db"
echo "🔐 Login: admin@2ndbrain.local / admin123"
echo ""

./venv/bin/python -m backend.api.app
