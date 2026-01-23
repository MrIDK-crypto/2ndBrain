#!/bin/bash
# One-command setup for ngrok deployment
# This script will guide you through the entire process

set -e

cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════╗"
echo "║   UCLA 2nd Brain - Ngrok Setup Wizard              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check ngrok installation
echo "Step 1/5: Checking ngrok installation..."
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok is not installed"
    echo ""
    echo "Installing ngrok via Homebrew..."

    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew is not installed either!"
        echo ""
        echo "Option 1: Install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo ""
        echo "Option 2: Download ngrok manually:"
        echo "  https://ngrok.com/download"
        exit 1
    fi

    brew install ngrok/ngrok/ngrok
    echo "✅ ngrok installed"
else
    echo "✅ ngrok is already installed ($(ngrok version))"
fi

echo ""

# Step 2: Check ngrok authtoken
echo "Step 2/5: Checking ngrok authentication..."

if ngrok config check &> /dev/null 2>&1; then
    echo "✅ ngrok authtoken is configured"
else
    echo "⚠️  ngrok authtoken not configured"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to: https://dashboard.ngrok.com/signup"
    echo "2. Sign up (free)"
    echo "3. Copy your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo ""
    read -p "Paste your authtoken here: " AUTHTOKEN

    if [ -z "$AUTHTOKEN" ]; then
        echo "❌ No authtoken provided"
        exit 1
    fi

    ngrok config add-authtoken "$AUTHTOKEN"
    echo "✅ Authtoken configured"
fi

echo ""

# Step 3: Create demo user
echo "Step 3/5: Creating demo user for sharing..."

if [ -f "./venv/bin/python" ]; then
    chmod +x create_demo_user.py
    ./venv/bin/python create_demo_user.py
else
    echo "❌ Python virtual environment not found"
    echo "Please run: python -m venv venv && ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo ""

# Step 4: Update CORS settings
echo "Step 4/5: Updating CORS settings..."

if ! grep -q "CORS_ORIGINS" .env 2>/dev/null; then
    echo "CORS_ORIGINS=*" >> .env
    echo "✅ CORS configured to accept all origins"
else
    echo "✅ CORS already configured"
fi

echo ""

# Step 5: Test backend
echo "Step 5/5: Testing backend server..."

export DATABASE_URL="sqlite:///./2ndbrain_ucla.db"

# Start backend temporarily
./venv/bin/python -m backend.api.app > /tmp/backend_test.log 2>&1 &
TEST_PID=$!

echo "⏳ Waiting for backend to start..."
sleep 8

if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
    kill $TEST_PID 2>/dev/null || true
else
    echo "❌ Backend health check failed"
    echo "Check /tmp/backend_test.log for errors"
    kill $TEST_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ✅ Setup Complete!                                 ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "📋 What's Ready:"
echo "   ✅ ngrok installed and configured"
echo "   ✅ Demo user created (demo@ucla.beat / DemoUCLA2024)"
echo "   ✅ CORS configured for public access"
echo "   ✅ Backend tested and working"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Start the server + ngrok:"
echo "   ./start_with_ngrok.sh"
echo ""
echo "2. Share the ngrok URL and credentials from COLLABORATOR_GUIDE.md"
echo ""
echo "3. Monitor traffic at: http://localhost:4040"
echo ""
echo "📖 Documentation:"
echo "   - Full guide: NGROK_DEPLOYMENT_PLAN.md"
echo "   - Collaborator guide: COLLABORATOR_GUIDE.md"
echo ""
echo "💡 Tip: For a custom subdomain (no random URLs), upgrade to ngrok Basic ($8/mo)"
echo ""
