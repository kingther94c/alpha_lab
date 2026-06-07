#!/usr/bin/env bash
# One-click mock stack (demo bot + Streamlit cockpit) — thin wrapper over scripts/demo.py.
#
#   ./scripts/demo.sh [up|down|status] [--bot p7_crypto_book] [--port 8501]
#
# No action defaults to `up`. Override the interpreter with PY=... ; it defaults to the py313
# env because plain `python` is a dead Microsoft Store stub on this machine. Works in git-bash
# (Windows) and any Unix shell. We cd to the repo root and pass a *relative* script path so a
# native Windows python.exe doesn't choke on msys-style (/d/...) absolute paths.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-D:/conda/envs/py313/python.exe}"
[ "$#" -eq 0 ] && set -- up
cd "$ROOT"
exec "$PY" scripts/demo.py "$@"
