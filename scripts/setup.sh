#!/bin/bash
# Larpy Setup Script

set -e

echo "🏷️ Setting up Larpy..."

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d. -f1,2)
REQUIRED_VERSION="3.10"

if (( $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc -l) )); then
    echo "✅ Python version OK: $PYTHON_VERSION"
else
    echo "❌ Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate and install
echo "📥 Installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install in editable mode
pip install -e .

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start Redis: redis-server"
echo "  2. Start worker: python -m larpy.workers.worker"
echo "  3. Start API: python -m larpy.api.server"
echo "  4. Start frontend: streamlit run larpy/frontend/app.py"
echo ""
echo "Or use Docker: docker-compose -f docker/docker-compose.yml up"
