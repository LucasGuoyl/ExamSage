#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
  ".venv/bin/python" scripts/backup_data.py
else
  python3 scripts/backup_data.py
fi
read -r -p "Press Return to close..."
