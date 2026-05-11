#!/bin/bash
# Push local changes to git and deploy to Oracle Cloud in one shot.
# Usage: bash deploy/push-deploy.sh ["optional commit message"]

set -e

SSH_KEY="$HOME/Desktop/ssh-key-2026-05-11 (3).key"
SERVER="ubuntu@129.80.108.131"
REPO="/home/ubuntu/Ryo"

MSG="${1:-update}"

echo "==> Committing and pushing..."
git add -A
git diff --cached --quiet && echo "Nothing to commit, skipping." || git commit -m "$MSG"
git push origin pilot

echo "==> Deploying to server..."
ssh -i "$SSH_KEY" "$SERVER" "cd $REPO && git pull && sudo systemctl restart ryo"

echo "==> Done. Tailing logs (Ctrl+C to exit)..."
ssh -i "$SSH_KEY" "$SERVER" "sudo journalctl -u ryo -f --no-pager"
