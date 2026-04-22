#!/usr/bin/env bash
# run_demo.sh -- launch the unified Muse demo.
#
# Spins up two processes:
#   - sidecar video HTTP server on $VIDEO_PORT (default 8602)
#   - Streamlit app on $APP_PORT (default 8601)
# Both bind to 0.0.0.0 so you can view from a phone on the same Wi-Fi.
# Ctrl+C in the terminal stops both.
#
# Python path is resolved in this order:
#   1. $MUSE_PYTHON env var, if set
#   2. The `eyeball` conda env under ~/anaconda3
#   3. `python3` on PATH
#
# Usage:
#   ./run_demo.sh
#   MUSE_PYTHON=/path/to/venv/bin/python3 ./run_demo.sh
#   APP_PORT=9000 VIDEO_PORT=9001 ./run_demo.sh

set -euo pipefail

cd "$(dirname "$0")"

APP_PORT="${APP_PORT:-8601}"
VIDEO_PORT="${VIDEO_PORT:-8602}"

resolve_python() {
  if [[ -n "${MUSE_PYTHON:-}" ]]; then
    echo "$MUSE_PYTHON"
    return
  fi
  local candidate="$HOME/anaconda3/envs/eyeball/bin/python3"
  if [[ -x "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  command -v python3
}

PY="$(resolve_python)"
if [[ ! -x "$PY" ]]; then
  echo "Error: no Python interpreter found. Set MUSE_PYTHON or install python3." >&2
  exit 1
fi
STREAMLIT="$(dirname "$PY")/streamlit"
if [[ ! -x "$STREAMLIT" ]]; then
  echo "Error: streamlit not found at $STREAMLIT. Install with: $PY -m pip install streamlit" >&2
  exit 1
fi

echo "Python: $PY"

pkill -f "serve_videos.py.*$VIDEO_PORT" 2>/dev/null || true
pkill -f "streamlit run app.py.*$APP_PORT" 2>/dev/null || true
sleep 1

echo "Starting Muse video server on :$VIDEO_PORT..."
"$PY" serve_videos.py --host 0.0.0.0 --port "$VIDEO_PORT" &
VIDEO_PID=$!

trap 'kill $VIDEO_PID 2>/dev/null || true' EXIT
sleep 1

echo "Starting Muse Streamlit on :$APP_PORT..."
exec "$STREAMLIT" run app.py --server.address 0.0.0.0 --server.port "$APP_PORT"
