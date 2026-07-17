#!/bin/bash
# Sync FailureModeBench TO rolf. RUN FROM YOUR LAPTOP, from the project root.
#
#   bash scripts/sync_to_rolf.sh              # code + labelsets + configs
#   WITH_RESULTS=1 bash scripts/sync_to_rolf.sh   # also push existing results/
#
# Uses ONE shared authenticated SSH connection (multiplexing) so rolf's 2FA/OTP
# prompts a single time for the whole sync. Requires an `rolf` Host entry in
# ~/.ssh/config (HostName rolf.ifi.uzh.ch, User <you>) and campus/VPN reach.
set -e

HOST=${1:-rolf}
REMOTE=FailureModeBench                          # relative to remote home

CM="${TMPDIR:-/tmp}/fmb-cm-$$"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=600"
cleanup() { ssh -o ControlPath="$CM" -O exit "$HOST" 2>/dev/null || true; }
trap cleanup EXIT

echo "== authenticate ONCE (password + OTP) =="
ssh $SSH_OPTS "$HOST" "mkdir -p $REMOTE && echo connected" \
    || { echo "[err] could not connect to $HOST (on campus / VPN?)"; exit 1; }
rs() { rsync -av --progress -e "ssh -o ControlPath=$CM" "$@"; }

echo "== code + labelsets + configs -> ~/$REMOTE =="
rs --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
   --exclude 'results' --exclude 'predictions' --exclude '.DS_Store' \
   ./ "$HOST:$REMOTE/"

if [ -n "$WITH_RESULTS" ]; then
    echo "== results -> ~/$REMOTE/results =="
    rs results/ "$HOST:$REMOTE/results/"
fi

echo
echo "SYNC DONE (single authentication). Next, on rolf:"
echo "  cd ~/FailureModeBench && bash scripts/bootstrap_rolf.sh"
echo "  screen -S fmb && bash scripts/run_rolf.sh"
