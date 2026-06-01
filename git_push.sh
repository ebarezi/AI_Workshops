#!/bin/bash
set -e

REMOTE_URL="https://github.com/ebarezi/AI_Workshops.git"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$REPO_DIR"

if [ ! -d ".git" ]; then
  git init
  git remote add origin "$REMOTE_URL"
else
  if ! git remote get-url origin &>/dev/null; then
    git remote add origin "$REMOTE_URL"
  else
    git remote set-url origin "$REMOTE_URL"
  fi
fi

git add .
git status

if [ -n "$1" ]; then
  MSG="$1"
else
  read -rp "Commit message (default: 'Update workshop files'): " MSG
  MSG="${MSG:-Update workshop files}"
fi

git commit -m "$MSG" || echo "Nothing new to commit."

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
if [ "$BRANCH" = "HEAD" ] || [ -z "$BRANCH" ]; then
  BRANCH="main"
fi

git push -u origin "$BRANCH"
echo "Pushed to $REMOTE_URL on branch '$BRANCH'."
