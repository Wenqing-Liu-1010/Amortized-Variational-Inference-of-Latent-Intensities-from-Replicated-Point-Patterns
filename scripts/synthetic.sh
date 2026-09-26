#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-train}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}

if [[ "$MODE" == train ]]; then
  DATA=${DATA:-$ROOT/data/synthetic.pt}
  TOTAL_K=${TOTAL_K:-4}
  SEED=${SEED:-42}
  TASKS=${TASKS:-0}
  OUTPUT=${OUTPUT:-$ROOT/outputs/k${TOTAL_K}_seed${SEED}}
  STEPS=${STEPS:-20000}
  BATCH_SIZE=${BATCH_SIZE:-256}
  MICROBATCH_SIZE=${MICROBATCH_SIZE:-64}
  EVAL_EVERY=${EVAL_EVERY:-1000}
  exec "$PY" -u "$ROOT/src/train.py" \
    --data "$DATA" --output "$OUTPUT" --total-k "$TOTAL_K" --seed "$SEED" \
    --steps "$STEPS" --batch-size "$BATCH_SIZE" \
    --microbatch-size "$MICROBATCH_SIZE" --eval-every "$EVAL_EVERY" --tasks "$TASKS"
elif [[ "$MODE" == test ]]; then
  CHECKPOINT=${CHECKPOINT:-$ROOT/checkpoints/synthetic.pt}
  OUTPUT=${OUTPUT:-$ROOT/synthetic_results.csv}
  SPLIT=${SPLIT:-test}
  TASKS=${TASKS:-0}
  SEQUENCES=${SEQUENCES:-4}
  if [[ "$SPLIT" == test && "$SEQUENCES" -gt 16 ]]; then
    DATA=${DATA:-$ROOT/data/synthetic_test_64.pt}
  else
    DATA=${DATA:-$ROOT/data/synthetic.pt}
  fi
  SAMPLES=${SAMPLES:-300}
  BATCH_SIZE=${BATCH_SIZE:-16}
  exec "$PY" -u "$ROOT/src/test_checkpoint.py" \
    --data "$DATA" --checkpoint "$CHECKPOINT" --output "$OUTPUT" \
    --split "$SPLIT" --tasks "$TASKS" --sequences "$SEQUENCES" --samples "$SAMPLES" \
    --batch-size "$BATCH_SIZE"
else
  echo "MODE must be train or test" >&2
  exit 2
fi
