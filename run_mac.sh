#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_PYTHON="$BACKEND_DIR/.venv/bin/python"
BACKEND_PORT="${BACKEND_PORT:-8011}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

if [ ! -x "$VENV_PYTHON" ] || [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Dependencies missing. Running install_mac.sh..."
  "$ROOT_DIR/install_mac.sh"
fi

if ! command -v node >/dev/null 2>&1; then
  echo "[ERROR] Node.js 18+ is required and was not found in PATH."
  exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
  echo "[ERROR] Backend venv still missing after install."
  exit 1
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "[ERROR] Frontend dependencies still missing after install."
  exit 1
fi

stop_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -ti tcp:"$port" || true)"
    if [ -n "$pids" ]; then
      echo "Stopping process(es) on port $port: $pids"
      kill $pids >/dev/null 2>&1 || true
      sleep 1
      pids="$(lsof -ti tcp:"$port" || true)"
      if [ -n "$pids" ]; then
        kill -9 $pids >/dev/null 2>&1 || true
      fi
    fi
  fi
}

stop_port "$BACKEND_PORT"
stop_port "$FRONTEND_PORT"

BACKEND_LOG="$BACKEND_DIR/uvicorn.out.log"
FRONTEND_LOG="$FRONTEND_DIR/vite.out.log"
BACKEND_READY_URL="http://127.0.0.1:${BACKEND_PORT}/system/worker-status"

echo "Starting backend on http://localhost:$BACKEND_PORT ..."
(
  cd "$BACKEND_DIR"
  exec "$VENV_PYTHON" -m uvicorn src.main:app --app-dir . --host 0.0.0.0 --port "$BACKEND_PORT"
) >>"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

echo "Waiting for backend to become ready..."
backend_ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "$BACKEND_READY_URL" >/dev/null 2>&1; then
    backend_ready=1
    break
  fi
  sleep 1
done

if [ "$backend_ready" -ne 1 ]; then
  echo "[ERROR] Backend did not become ready in time."
  echo "Check backend log: $BACKEND_LOG"
  exit 1
fi

echo "Starting frontend on http://localhost:$FRONTEND_PORT ..."
(
  cd "$FRONTEND_DIR"
  exec npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
) >>"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

cleanup() {
  echo
  echo "Shutting down..."
  kill "$BACKEND_PID" "$FRONTEND_PID" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

sleep 2
if command -v open >/dev/null 2>&1; then
  open "http://localhost:$FRONTEND_PORT" >/dev/null 2>&1 || true
fi

echo
echo "Frontend: http://localhost:$FRONTEND_PORT"
echo "Backend:  http://localhost:$BACKEND_PORT"
echo "API docs: http://localhost:$BACKEND_PORT/docs"
echo "Logs:"
echo "  $BACKEND_LOG"
echo "  $FRONTEND_LOG"
echo "Press Ctrl+C to stop both services."
echo

wait -n "$BACKEND_PID" "$FRONTEND_PID"
