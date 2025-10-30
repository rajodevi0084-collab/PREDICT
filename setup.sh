#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON=${PYTHON:-python3}
NODE=${NODE:-npm}

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

$NODE install

(uvicorn backend.main:create_app --host 0.0.0.0 --port 8000 --reload &) 
UVICORN_PID=$!

trap 'kill $UVICORN_PID 2>/dev/null || true' EXIT

npm run dev -- --hostname 0.0.0.0 --port 3000
