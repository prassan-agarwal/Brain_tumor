#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <destination_directory>"
  exit 1
fi

DEST="$1"
SRC_DIR="$(pwd)"

if [[ -e "$DEST" ]]; then
  echo "Destination already exists: $DEST"
  exit 1
fi

mkdir -p "$DEST"
rsync -a --exclude '.git' --exclude '.venv' --exclude '__pycache__' "$SRC_DIR/" "$DEST/"

cd "$DEST"
rm -rf .git
git init -b main
git add .
git commit -m "Initial standalone import from Brain_tumor"

echo "Standalone repository created at: $DEST"
