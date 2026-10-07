#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

RAPO_CFG=${RAPO_CFG:-"$SCRIPT_DIR/../rapo_cfg.json"}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2-VL-7B-Instruct}
SAVE_ROOT=${SAVE_ROOT:-$ROOT_DIR/saves/main_results}
NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
ACTOR_GLOBAL_BATCH_SIZE=${ACTOR_GLOBAL_BATCH_SIZE:-8}
ACTOR_MICRO_BATCH_SIZE_UPDATE=${ACTOR_MICRO_BATCH_SIZE_UPDATE:-1}
ACTOR_MICRO_BATCH_SIZE_EXPERIENCE=${ACTOR_MICRO_BATCH_SIZE_EXPERIENCE:-2}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
PROJECT_NAME=${PROJECT_NAME:-rft_continual_learning}
KL_COEF=${KL_COEF:-2e-3}
DATA_ROOT=${DATA_ROOT:-$ROOT_DIR/data/object_det_cil_dataset}
COCO_IMAGE_ROOT=${COCO_IMAGE_ROOT:-}
if [[ -z "$COCO_IMAGE_ROOT" ]]; then
  echo "Set COCO_IMAGE_ROOT to the local COCO root containing train2017/ and val2017/." >&2
  exit 1
fi
TRAIN_FILES=${TRAIN_FILES:-$DATA_ROOT/train_5shots.jsonl}
VAL_FILES=${VAL_FILES:-$DATA_ROOT/val.jsonl}
CATEGORIES=${CATEGORIES:-$DATA_ROOT/categories.json}
BASE_CLASSES=16
INCREMENTAL_CLASSES=16
TASK_SETTING=5task

CLASS_ORDER_SEEDS=(136 377 639)
for CLASS_ORDER_SEED in "${CLASS_ORDER_SEEDS[@]}"; do
  EXP_NAME="object_det_cil_rapo_${TASK_SETTING}_s${CLASS_ORDER_SEED}"
  OUT_DIR="$SAVE_ROOT/object_det_cil/rapo/$TASK_SETTING/seed$CLASS_ORDER_SEED"
  mkdir -p "$OUT_DIR"

  python -m examples.baselines.cil_det.image_det_cil_rapo \
    --base_classes "$BASE_CLASSES" \
    --incremental_classes "$INCREMENTAL_CLASSES" \
    --class_order_seed "$CLASS_ORDER_SEED" \
    --categories "$CATEGORIES" \
    --save_dir "$OUT_DIR" \
    --launch_script "${BASH_SOURCE[0]}" \
    --prompt_seen_labels \
    --det_only \
    --cil_cfg "$RAPO_CFG" \
    config=examples/config_det.yaml \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.image_dir="$COCO_IMAGE_ROOT" \
    data.rollout_batch_size="$ROLLOUT_BATCH_SIZE" \
    algorithm.kl_coef="$KL_COEF" \
    worker.actor.model.model_path="$MODEL_PATH" \
    worker.actor.global_batch_size="$ACTOR_GLOBAL_BATCH_SIZE" \
    worker.actor.micro_batch_size_per_device_for_update="$ACTOR_MICRO_BATCH_SIZE_UPDATE" \
    worker.actor.micro_batch_size_per_device_for_experience="$ACTOR_MICRO_BATCH_SIZE_EXPERIENCE" \
    worker.rollout.n="$ROLLOUT_N" \
    worker.rollout.tensor_parallel_size="$TENSOR_PARALLEL_SIZE" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="$NUM_GPUS"
done
