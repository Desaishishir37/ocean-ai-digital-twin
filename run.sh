#!/usr/bin/env bash
# Launcher for Ocean-AI 3D Digital Twin

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================================="
echo "🌊 Ocean-AI: 3D Subsurface Twin & What-If Studio"
echo "=========================================================="

# Check if the always-on server is already active
if curl -s http://localhost:8000/api/v1/ocean/status >/dev/null 2>&1; then
  echo "✅ Ocean-AI server is ALREADY ACTIVE and running 24/7 at:"
  echo "   👉 http://localhost:8000"
  echo "=========================================================="
  exit 0
fi

# If LaunchAgent exists, load it
PLIST="$HOME/Library/LaunchAgents/com.oceansim.server.plist"
if [ -f "$PLIST" ]; then
  echo "🚀 Starting background LaunchAgent service..."
  launchctl load -w "$PLIST" 2>/dev/null || true
  sleep 2
  if curl -s http://localhost:8000/api/v1/ocean/status >/dev/null 2>&1; then
    echo "✅ Server started successfully at http://localhost:8000"
    exit 0
  fi
fi

# Fallback: run directly in foreground
export PYTHONPATH="$DIR/backend"
echo "Starting Uvicorn Server at http://localhost:8000"
echo "Open your browser to http://localhost:8000 to view the 3D Ocean Twin"

exec .venv/bin/python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
