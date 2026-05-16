#!/usr/bin/env bash
# Score a local RunACT-compatible checkpoint against an AIC config.

set -u
set -o pipefail

ROOT="/home/saivemu/code/aic"
RUN_NAME="${RUN_NAME:-score_policy_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/saivemu/aic_runs/$RUN_NAME}"
CONFIG="${CONFIG:-$ROOT/aic_engine/config/sample_config.yaml}"
POLICY_PATH="${POLICY_PATH:-${AIC_POLICY_PATH:-}}"
RESULTS_DIR="$RUN_ROOT/results"
LOG_DIR="$RUN_ROOT/logs"
STATUS_FILE="$RUN_ROOT/status.env"

if [[ -z "$POLICY_PATH" ]]; then
  echo "POLICY_PATH or AIC_POLICY_PATH is required" >&2
  exit 64
fi
if [[ ! -d "$POLICY_PATH" ]]; then
  echo "POLICY_PATH does not exist: $POLICY_PATH" >&2
  exit 66
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "run root already exists; refusing to overwrite: $RUN_ROOT" >&2
  exit 73
fi

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_DIR/supervisor.log"
}

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
  if [[ -n "${MODEL_PID:-}" ]]; then wait "$MODEL_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${CLEANUP_PID:-}" ]]; then wait "$CLEANUP_PID" >/dev/null 2>&1 || true; fi
}
trap cleanup_processes EXIT INT TERM HUP

log "run_name=$RUN_NAME"
log "run_root=$RUN_ROOT"
log "config=$CONFIG"
log "policy_path=$POLICY_PATH"
log "results_dir=$RESULTS_DIR"

{
  echo "RUN_NAME=$RUN_NAME"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "CONFIG=$CONFIG"
  echo "POLICY_PATH=$POLICY_PATH"
  echo "RESULTS_DIR=$RESULTS_DIR"
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

log "starting RunACT model"
pixi run env \
  PYTHONPATH="$ROOT/aic_example_policies:$ROOT/aic_utils/lerobot_robot_aic:$ROOT/.pixi/envs/default/lib/python/site-packages:$ROOT/.pixi/envs/default/lib/python3.12/site-packages" \
  RMW_IMPLEMENTATION=rmw_zenoh_cpp \
  ZENOH_ROUTER_CHECK_ATTEMPTS=-1 \
  ZENOH_CONFIG_OVERRIDE=transport/shared_memory/enabled=false \
  AIC_POLICY_PATH="$POLICY_PATH" \
  ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.RunACT \
  > "$LOG_DIR/model.log" 2>&1 &
MODEL_PID=$!
echo "MODEL_PID=$MODEL_PID" >> "$STATUS_FILE"

wait "$EVAL_PID"
EVAL_RC=$?
echo "EVAL_RC=$EVAL_RC" >> "$STATUS_FILE"
log "eval exited rc=$EVAL_RC"

kill "$MODEL_PID" "$CLEANUP_PID" >/dev/null 2>&1 || true
wait "$MODEL_PID" >/dev/null 2>&1 || true
wait "$CLEANUP_PID" >/dev/null 2>&1 || true

if [[ -f "$RESULTS_DIR/scoring.yaml" ]]; then
  echo "SCORING_YAML=$RESULTS_DIR/scoring.yaml" >> "$STATUS_FILE"
  log "scoring_yaml=$RESULTS_DIR/scoring.yaml"
else
  log "missing scoring.yaml"
fi

exit "$EVAL_RC"
