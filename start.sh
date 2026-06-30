#!/bin/bash
set -e

echo "[DevOps] Starting Python FastAPI backend in the background..."
# Run Uvicorn on localhost port 8000, streaming logs directly to console
.venv/bin/python -m uvicorn src.lib.server:app --host 127.0.0.1 --port 8000 &

# Give Uvicorn a moment to initialize
echo "[DevOps] Waiting for FastAPI server to initialize..."
sleep 2

echo "[DevOps] Starting TanStack Start Node server on port $PORT..."
node .output/server/index.mjs
