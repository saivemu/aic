#!/usr/bin/env bash
# Collect a scored exact-config-centered midcourse recovery dataset chunk and,
# only if the score gate passes, fine-tune ACT from the Plan-D checkpoint.

set -u
set -o pipefail

ROOT="/home/saivemu/code/aic"
RUN_NAME="${RUN_NAME:-exact_midcourse30_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/saivemu/aic_runs/$RUN_NAME}"
CONFIG="${CONFIG:-$ROOT/aic_engine/config/exact_jitter_train30.yaml}"
NUM_EPISODES="${NUM_EPISODES:-30}"
REPO_ID="${REPO_ID:-saivemu/aic_${RUN_NAME}}"
RESULTS_DIR="$RUN_ROOT/results"
DATASET_ROOT="$RUN_ROOT/lerobot"
LOG_DIR="$RUN_ROOT/logs"
STATUS_FILE="$RUN_ROOT/status.env"
VALIDATION_JSON="$RUN_ROOT/validation.json"
AUDIT_JSON="$RUN_ROOT/audit.json"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-$ROOT/outputs/train/act_${RUN_NAME}}"

mkdir -p "$RESULTS_DIR" "$(dirname "$DATASET_ROOT")" "$LOG_DIR"
rm -f "$STATUS_FILE"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_DIR/supervisor.log"
}

if [[ -e "$DATASET_ROOT" ]]; then
  log "dataset_root already exists; refusing to overwrite: $DATASET_ROOT"
  exit 1
fi

cleanup_bags_loop() {
  while true; do
    docker run --rm -v "$RESULTS_DIR:/target" alpine:latest \
      sh -c 'find /target -mindepth 1 -maxdepth 1 -type d -name "bag_trial_*" -mmin +1 -print -exec rm -rf {} +' \
      >> "$LOG_DIR/cleanup.log" 2>&1 || true
    sleep 30
  done
}

cleanup_processes() {
  set +e
  trap - EXIT INT TERM HUP
  if [[ -n "${EVAL_PID:-}" ]]; then kill -TERM -- "-$EVAL_PID" >/dev/null 2>&1 || kill "$EVAL_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${MODEL_PID:-}" ]]; then kill "$MODEL_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${CLEANUP_PID:-}" ]]; then kill "$CLEANUP_PID" >/dev/null 2>&1 || true; fi
  pkill -TERM -f "aic_engine_config_file:=$CONFIG" >/dev/null 2>&1 || true
  if [[ -n "${EVAL_PID:-}" ]]; then wait "$EVAL_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${RECORDER_PID:-}" ]]; then wait "$RECORDER_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${MODEL_PID:-}" ]]; then wait "$MODEL_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${CLEANUP_PID:-}" ]]; then wait "$CLEANUP_PID" >/dev/null 2>&1 || true; fi
}
trap cleanup_processes EXIT INT TERM HUP

log "run_name=$RUN_NAME"
log "run_root=$RUN_ROOT"
log "config=$CONFIG"
log "dataset_root=$DATASET_ROOT"
log "results_dir=$RESULTS_DIR"
log "train_output=$TRAIN_OUTPUT"
log "sc_final_perturb_scale=${AIC_CHEATCODE_SC_FINAL_PERTURB_SCALE:-1.0}"

{
  echo "RUN_NAME=$RUN_NAME"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "CONFIG=$CONFIG"
  echo "REPO_ID=$REPO_ID"
  echo "DATASET_ROOT=$DATASET_ROOT"
  echo "RESULTS_DIR=$RESULTS_DIR"
  echo "TRAIN_OUTPUT=$TRAIN_OUTPUT"
  echo "AIC_CHEATCODE_SC_FINAL_PERTURB_SCALE=${AIC_CHEATCODE_SC_FINAL_PERTURB_SCALE:-1.0}"
} >> "$STATUS_FILE"

cleanup_bags_loop &
CLEANUP_PID=$!
echo "CLEANUP_PID=$CLEANUP_PID" >> "$STATUS_FILE"

log "starting eval stack"
setsid env AIC_RESULTS_DIR="$RESULTS_DIR" \
  distrobox enter aic_eval -- /entrypoint.sh \
    gazebo_gui:=false \
    launch_rviz:=false \
    ground_truth:=true \
    start_aic_engine:=true \
    shutdown_on_aic_engine_exit:=true \
    model_discovery_timeout_seconds:=600 \
    aic_engine_config_file:="$CONFIG" \
  > "$LOG_DIR/eval.log" 2>&1 &
EVAL_PID=$!
echo "EVAL_PID=$EVAL_PID" >> "$STATUS_FILE"

sleep 3

log "starting recorder"
pixi run python "$ROOT/aic_utils/lerobot_robot_aic/scripts/record_dataset.py" \
  --repo-id "$REPO_ID" \
  --root "$DATASET_ROOT" \
  --num-episodes "$NUM_EPISODES" \
  --episode-idle-timeout 2.0 \
  --max-action-age 0.25 \
  --trials-config "$CONFIG" \
  --perturbing-topic "/aic/cheatcode/perturbing" \
  > "$LOG_DIR/recorder.log" 2>&1 &
RECORDER_PID=$!
echo "RECORDER_PID=$RECORDER_PID" >> "$STATUS_FILE"

sleep 3

log "starting CheatCode model with perturb_mode=${AIC_CHEATCODE_PERTURB_MODE:-midcourse}"
pixi run env \
  PYTHONPATH="$ROOT/aic_example_policies:$ROOT/.pixi/envs/default/lib/python/site-packages:$ROOT/.pixi/envs/default/lib/python3.12/site-packages" \
  RMW_IMPLEMENTATION=rmw_zenoh_cpp \
  ZENOH_ROUTER_CHECK_ATTEMPTS=-1 \
  ZENOH_CONFIG_OVERRIDE=transport/shared_memory/enabled=false \
  AIC_CHEATCODE_PERTURB_MODE="${AIC_CHEATCODE_PERTURB_MODE:-midcourse}" \
  AIC_CHEATCODE_PERTURB_PROB="${AIC_CHEATCODE_PERTURB_PROB:-1.0}" \
  AIC_CHEATCODE_PERTURB_XY_MIN_M="${AIC_CHEATCODE_PERTURB_XY_MIN_M:-0.005}" \
  AIC_CHEATCODE_PERTURB_XY_MAX_M="${AIC_CHEATCODE_PERTURB_XY_MAX_M:-0.015}" \
  AIC_CHEATCODE_SC_FINAL_PERTURB_SCALE="${AIC_CHEATCODE_SC_FINAL_PERTURB_SCALE:-1.0}" \
  AIC_CHEATCODE_PERTURB_Z_MAX_M="${AIC_CHEATCODE_PERTURB_Z_MAX_M:-0.0}" \
  AIC_CHEATCODE_PERTURB_DURATION_S="${AIC_CHEATCODE_PERTURB_DURATION_S:-0.50}" \
  AIC_CHEATCODE_PERTURB_SEED="${AIC_CHEATCODE_PERTURB_SEED:-37}" \
  ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCode \
  > "$LOG_DIR/model.log" 2>&1 &
MODEL_PID=$!
echo "MODEL_PID=$MODEL_PID" >> "$STATUS_FILE"

wait "$EVAL_PID"
EVAL_RC=$?
echo "EVAL_RC=$EVAL_RC" >> "$STATUS_FILE"
log "eval exited rc=$EVAL_RC"

wait "$RECORDER_PID"
RECORDER_RC=$?
echo "RECORDER_RC=$RECORDER_RC" >> "$STATUS_FILE"
log "recorder exited rc=$RECORDER_RC"

kill "$MODEL_PID" "$CLEANUP_PID" >/dev/null 2>&1 || true
wait "$MODEL_PID" >/dev/null 2>&1 || true
wait "$CLEANUP_PID" >/dev/null 2>&1 || true

if [[ "$EVAL_RC" -ne 0 || "$RECORDER_RC" -ne 0 ]]; then
  log "collection failed before validation"
  exit 1
fi

if [[ ! -f "$RESULTS_DIR/scoring.yaml" ]]; then
  log "missing scoring.yaml; refusing to train"
  exit 1
fi

log "validating scoring and dataset episode count"
pixi run python "$ROOT/aic_utils/lerobot_robot_aic/scripts/validate_scored_dataset.py" \
  --scoring-yaml "$RESULTS_DIR/scoring.yaml" \
  --dataset-root "$DATASET_ROOT" \
  --expected-trials "$NUM_EPISODES" \
  --min-total 90.0 \
  --min-tier3 75.0 \
  --require-no-contacts \
  --require-no-force-penalty \
  --output-json "$VALIDATION_JSON" \
  > "$LOG_DIR/validation.log" 2>&1
VALIDATE_RC=$?
echo "VALIDATE_RC=$VALIDATE_RC" >> "$STATUS_FILE"
if [[ "$VALIDATE_RC" -ne 0 ]]; then
  log "score gate failed; dataset kept for inspection but training skipped"
  exit 2
fi

log "auditing accepted dataset"
pixi run python "$ROOT/aic_utils/lerobot_robot_aic/scripts/audit_dataset.py" \
  --dataset-repo-id "$REPO_ID" \
  --dataset-root "$DATASET_ROOT" \
  --output-json "$AUDIT_JSON" \
  > "$LOG_DIR/audit.log" 2>&1
AUDIT_RC=$?
echo "AUDIT_RC=$AUDIT_RC" >> "$STATUS_FILE"
if [[ "$AUDIT_RC" -ne 0 ]]; then
  log "audit failed; refusing to train"
  exit 3
fi

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  log "SKIP_TRAIN=1; collection, validation, and audit complete"
  exit 0
fi

log "starting ACT fine-tune"
pixi run lerobot-train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.pretrained_path="$ROOT/outputs/plan_d/pretrained_model" \
  --policy.device=cuda \
  --output_dir="$TRAIN_OUTPUT" \
  --job_name="act_${RUN_NAME}" \
  --batch_size=1 \
  --steps="${TRAIN_STEPS:-10000}" \
  --save_freq="${TRAIN_SAVE_FREQ:-2500}" \
  --eval_freq=0 \
  --num_workers=0 \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --policy.repo_id="saivemu/act_${RUN_NAME}" \
  > "$LOG_DIR/train_act.log" 2>&1
TRAIN_RC=$?
echo "TRAIN_RC=$TRAIN_RC" >> "$STATUS_FILE"
log "ACT training exited rc=$TRAIN_RC"

exit "$TRAIN_RC"
