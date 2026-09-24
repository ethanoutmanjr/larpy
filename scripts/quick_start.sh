#!/bin/bash
# Quick start all services

echo "🚀 Starting all Larpy services..."

# Start Redis
redis-server --daemonize yes
echo "✅ Redis started"

# Start API server in background
python -m larpy.api.server &
echo "✅ API server starting on port 8000"

# Start RQ worker in background
python -m larpy.workers.worker &
echo "✅ RQ worker starting"

# Wait for services to initialize
sleep 3

# Start Streamlit
streamlit run larpy/frontend/app.py &
echo "✅ Streamlit frontend starting on port 8501"

echo ""
echo "🎉 All services running! Open http://localhost:8501"
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait
