#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-train}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA_DIR=${DATA_DIR:-$ROOT/data/divvy}
PY=${PYTHON:-python}
SEED=${SEED:-42}
TASKS=${TASKS:-0}

if [[ "$MODE" == train ]]; then
  OUTPUT=${OUTPUT:-$ROOT/outputs/divvy_seed${SEED}}
  STEPS=${STEPS:-20000}
  BATCH_SIZE=${BATCH_SIZE:-256}
  MICROBATCH_SIZE=${MICROBATCH_SIZE:-64}
  EVAL_EVERY=${EVAL_EVERY:-1000}
  exec "$PY" -u "$ROOT/src/divvy.py" --mode train --data-dir "$DATA_DIR" \
    --output "$OUTPUT" --seed "$SEED" --tasks "$TASKS" --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" --microbatch-size "$MICROBATCH_SIZE" \
    --eval-every "$EVAL_EVERY"
elif [[ "$MODE" == test ]]; then
  CHECKPOINT=${CHECKPOINT:-$ROOT/checkpoints/divvy.pt}
  OUTPUT=${OUTPUT:-$ROOT/divvy_results.json}
  MC_SAMPLES=${MC_SAMPLES:-300}
  exec "$PY" -u "$ROOT/src/divvy.py" --mode test --data-dir "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" --output "$OUTPUT" --seed "$SEED" \
    --tasks "$TASKS" --mc-samples "$MC_SAMPLES"
else
  echo "MODE must be train or test" >&2
  exit 2
fi
