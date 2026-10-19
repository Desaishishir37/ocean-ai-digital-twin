#!/bin/bash
# ==============================================================================
# Ocean-AI Digital Twin - Instant 1-Click Offline Desktop Launcher for macOS
# ==============================================================================

DIR="/Users/shashank/Desktop/OCEAN SIM"
cd "$DIR" || exit 1

# Check if port 8000 is already active
if ! lsof -i :8000 > /dev/null 2>&1; then
    echo "Starting Ocean-AI Digital Twin local server..."
    nohup "$DIR/.venv/bin/python" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 > /dev/null 2>&1 &
    
    # Wait up to 5 seconds for the server to be ready
    for i in {1..10}; do
        if lsof -i :8000 > /dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done
fi

# Open in Google Chrome if installed, otherwise default browser
if [ -d "/Applications/Google Chrome.app" ]; then
    open -a "Google Chrome" "http://localhost:8000"
else
    open "http://localhost:8000"
fi

# Display a native macOS notification
osascript -e 'display notification "Ocean-AI Digital Twin is running on http://localhost:8000" with title "Ocean-AI Digital Twin" sound name "Glass"' > /dev/null 2>&1 &

# Exit cleanly
exit 0
