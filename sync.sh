#!/usr/bin/env bash
# Sync this repo between the Mac and the GPU box. The Mac copy holds the git history.
#   VAST="-p PORT root@IP" ./sync.sh push   # Mac -> box
#   VAST="-p PORT root@IP" ./sync.sh pull   # box -> Mac (results, IR dumps)
set -euo pipefail
: "${VAST:?set VAST=\"-p PORT root@IP\"}"
read -r _ port host <<<"$VAST"
remote_dir=/workspace/llama-matmul-triton
ssh_cmd="ssh -i $HOME/.ssh/vast_ed25519 -o IdentitiesOnly=yes -p $port"
opts=(-az --exclude .git --exclude __pycache__ --exclude .pytest_cache -e "$ssh_cmd")
case "${1:-}" in
  push) rsync "${opts[@]}" ./ "$host:$remote_dir/" ;;
  pull) rsync "${opts[@]}" "$host:$remote_dir/" ./ ;;
  *) echo "usage: sync.sh push|pull" >&2; exit 2 ;;
esac
