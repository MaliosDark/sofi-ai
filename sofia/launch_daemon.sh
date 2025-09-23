#!/bin/bash
# SOFIA Daemon Launcher
# Launches SOFIA daemon in background

echo "🚀 Launching SOFIA daemon in background..."

# Check if daemon is already running
if pgrep -f "sofia_daemon.py" > /dev/null; then
    echo "⚠️  SOFIA daemon is already running"
    exit 1
fi

# Launch daemon in background
cd "$(dirname "$0")"
python sofia_daemon.py > sofia_daemon.log 2>&1 &

# Wait a moment for daemon to start
sleep 2

# Check if daemon started successfully
if pgrep -f "sofia_daemon.py" > /dev/null; then
    echo "✅ SOFIA daemon started successfully (PID: $!)"
    echo "📊 Check status with: make status"
    echo "📜 View logs with: tail -f sofia/sofia_daemon.log"
else
    echo "❌ Failed to start SOFIA daemon"
    echo "📜 Check logs: sofia/sofia_daemon.log"
    exit 1
fi
