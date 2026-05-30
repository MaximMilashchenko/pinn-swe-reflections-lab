#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"

JOBS=(
  # "training_jobs/train_swe_pinn.py"
  # "training_jobs/train_swe_gpinn.py"
  # "training_jobs/train_swe_rad_pinn.py"
  # "training_jobs/train_swe_rar_d_pinn.py"
  # "training_jobs/train_swe_euler_transition_pinn.py"
  "training_jobs/train_swe_integral_conservation_pinn.py"
)

if command -v uv >/dev/null 2>&1 && [ -f "uv.lock" ]; then
  PYTHON_CMD=(uv run python -u)
else
  PYTHON_CMD=(python -u)
fi

for job in "${JOBS[@]}"; do
  name="$(basename "$job" .py)"
  log_file="logs/${name}.txt"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $job"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $job" > "$log_file"

  "${PYTHON_CMD[@]}" "$job" >> "$log_file" 2>&1
  status=$?

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXIT $status $job" >> "$log_file"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXIT $status $job"

  if [ "$status" -ne 0 ]; then
    echo "Job failed: $job. See $log_file"
    exit "$status"
  fi
done
