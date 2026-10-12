#!/usr/bin/env bash
# Ocean-AI Background Service Manager (macOS LaunchAgent)

PLIST="$HOME/Library/LaunchAgents/com.oceansim.server.plist"

case "$1" in
  status)
    echo "🔍 Checking Ocean-AI server status..."
    if curl -s http://localhost:8000/api/v1/ocean/status >/dev/null 2>&1; then
      echo "✅ Server is ACTIVE and responding at http://localhost:8000"
      curl -s http://localhost:8000/api/v1/ocean/status | grep -o '"status":"[^"]*"'
    else
      echo "❌ Server is NOT responding."
    fi
    ;;
  restart)
    echo "🔄 Restarting Ocean-AI background service..."
    launchctl unload "$PLIST" 2>/dev/null || true
    sleep 1
    launchctl load -w "$PLIST"
    sleep 2
    $0 status
    ;;
  stop)
    echo "🛑 Stopping Ocean-AI background service..."
    launchctl unload "$PLIST" 2>/dev/null || true
    echo "Service stopped."
    ;;
  start)
    echo "🚀 Starting Ocean-AI background service..."
    launchctl load -w "$PLIST" 2>/dev/null || true
    sleep 2
    $0 status
    ;;
  *)
    echo "Usage: $0 {status|restart|start|stop}"
    exit 1
    ;;
esac
