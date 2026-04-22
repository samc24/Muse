#!/usr/bin/env bash
# run_demo.sh -- launch the Streamlit app and the sidecar video server together.
# Ctrl+C stops both. Bind to 0.0.0.0 so the phone can reach it over Wi-Fi.

set -euo pipefail

cd "$(dirname "$0")"

PY=/Users/Sameer/anaconda3/envs/eyeball/bin/python3
STREAMLIT=/Users/Sameer/anaconda3/envs/eyeball/bin/streamlit

# Kill any leftovers from a previous run
pkill -f "serve_videos.py" 2>/dev/null || true
pkill -f "streamlit run app.py" 2>/dev/null || true
sleep 1

echo "Starting video server on :8502..."
"$PY" serve_videos.py --host 0.0.0.0 --port 8502 &
VIDEO_PID=$!

# Clean up the video server when Streamlit exits (Ctrl+C)
trap 'kill $VIDEO_PID 2>/dev/null || true' EXIT

# Give the video server a beat to bind
sleep 1

echo "Starting Streamlit on :8501..."
exec "$STREAMLIT" run app.py --server.address 0.0.0.0 --server.port 8501
