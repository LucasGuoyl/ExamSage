#!/bin/bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.11 or 3.12 is required. Install it from https://www.python.org/downloads/"
  read -r -p "Press Return to close..."
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] <= (3, 12)))'; then
  echo "ExamSage requires Python 3.11 or 3.12. Current: $(python3 --version)"
  read -r -p "Press Return to close..."
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating ExamSage environment..."
  python3 -m venv .venv
fi

echo "Installing or updating ExamSage..."
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt

echo "Opening ExamSage in your browser..."
".venv/bin/python" -m streamlit run app.py
