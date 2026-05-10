#!/usr/bin/env bash
# SiteSafe — one-command setup
# Installs Python deps, builds the OSHA SQLite DB, and pulls demo assets.
# Does NOT pull the Gemma 4 model (requires Ollama; we print instructions instead).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

green()  { printf '\033[0;32m%s\033[0m\n' "$1"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$1"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$1"; }

green "==> SiteSafe setup starting"

# 1. Python version sanity check
#
# We deliberately prefer `python` over `python3` on Windows: the bare
# `python3` symlink there can resolve to a Microsoft Store launcher that
# spawns a *different* interpreter than `python`, with a *different*
# site-packages directory. Installing into one and running from the other
# leaves you with phantom ImportError: No module named 'gradio'.
if command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    red "python / python3 not found on PATH. Install Python 3.10+ first."
    exit 1
fi
export PYTHON

PY_VER="$("${PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
green "==> Python ${PY_VER} detected at ${PYTHON}"

# 2. Install Python dependencies
green "==> Installing Python dependencies (this can take a couple of minutes)"
"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -r requirements.txt

# 3. Build the OSHA SQLite knowledge base
green "==> Building OSHA regulation database"
"${PYTHON}" data/build_osha_db.py

# 4. Pull demo images (best-effort; non-fatal if offline)
green "==> Fetching sample images for demo"
if bash data/download_datasets.sh; then
    green "    sample images ready under data/sample_images/"
else
    yellow "    download_datasets.sh exited non-zero (likely offline). Skipping."
fi

# 5. Ollama check
if command -v ollama >/dev/null 2>&1; then
    green "==> Ollama detected — you can now register the SiteSafe model:"
    echo "        ollama create sitesafe -f training/Modelfile"
    echo "    (the Modelfile expects a fine-tuned GGUF; see training/README.md)"
else
    yellow "==> Ollama not found on PATH."
    yellow "    Install from https://ollama.com, then run:"
    echo  "        ollama create sitesafe -f training/Modelfile"
fi

green "==> Setup complete"
echo
echo "Run the app with:"
echo "    ${PYTHON} app/sitesafe_app.py"
echo
echo "Open your browser to: http://localhost:7860"
