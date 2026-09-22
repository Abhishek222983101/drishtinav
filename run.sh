#!/usr/bin/env bash
# DrishtiNav launcher (Linux / macOS). Pass --https to serve the phone app over HTTPS.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
echo "Dashboard: http://localhost:5000   Phone app: http://<this-ip>:5000/app/"
exec .venv/bin/python -m drishtinav serve --port 5000 "$@"
