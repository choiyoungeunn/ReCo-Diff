# Batch script that runs test_err_cfg.sh for every .pkl checkpoint in ckpt_dir.

#!/usr/bin/env bash
set -euo pipefail

# Enter the checkpoint folder path of the trained model.
ckpt_dir="/path/to/your/checkpoints"

# Stop early if the checkpoint directory does not exist.
if [[ ! -d "$ckpt_dir" ]]; then
  echo "Checkpoint dir not found: $ckpt_dir" >&2
  exit 1
fi

# Collect all .pkl checkpoint files in the directory.
shopt -s nullglob
pkls=("$ckpt_dir"/*.pkl)
shopt -u nullglob

# Stop if no checkpoint files are found.
if (( ${#pkls[@]} == 0 )); then
  echo "No .pkl files found in $ckpt_dir" >&2
  exit 1
fi

# Run test_err_cfg.sh for each checkpoint and save output to a separate log file.
for pkl in "${pkls[@]}"; do
  base="$(basename "$pkl" .pkl)"
  log="log_reset_sampling_18v36v72v_${base}.txt"
  echo "Running test_err_cfg.sh with $pkl -> $log"
  NET_CHECKPATH="$pkl" ./test_err_cfg.sh > "$log"
done
