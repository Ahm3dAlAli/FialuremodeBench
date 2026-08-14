#!/usr/bin/env bash
# Wait for a GPU with >= MIN_FREE MiB free, then launch a 4-bit LLaVA-1.6-7B
# recognition run pinned to it. Runs unattended (nohup). Contention on the shared
# box fluctuates; this grabs the first window instead of babysitting.
#   nohup scripts/launch_when_free.sh > guard.log 2>&1 &
# NOTE: no `set -u` -- conda activate references unset vars (e.g. $PS1) in a
# detached non-interactive shell, which -u would make fatal before any output.
set -o pipefail
MIN_FREE="${MIN_FREE:-6600}"
RUN="${RUN:-llava7b}"
MODEL="${MODEL:-llava16_7b}"
MAX_WAIT="${MAX_WAIT:-14400}"     # give up after 4h
source /local/scratch/alali/miniconda/etc/profile.d/conda.sh
conda activate fmb
cd ~/FailureModeBench
export HF_HOME=/local/scratch/alali/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
waited=0
while :; do
  # pick the GPU with the most free memory
  read -r GID FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
     | sort -t, -k2 -rn | head -1 | tr -d ',')
  if [ "${FREE:-0}" -ge "$MIN_FREE" ]; then
    echo "[$(date +%T)] GPU $GID has ${FREE} MiB free (>= $MIN_FREE) -> launching"
    export CUDA_VISIBLE_DEVICES="$GID"
    exec python -m failuremodebench.cli --run "$RUN" --models "$MODEL" \
         --limit 1000 --embeddings infer-recognition
  fi
  echo "[$(date +%T)] best free=${FREE} MiB on GPU $GID (<$MIN_FREE); waiting ${waited}s"
  sleep 120; waited=$((waited+120))
  [ "$waited" -ge "$MAX_WAIT" ] && { echo "gave up after ${waited}s"; exit 1; }
done
