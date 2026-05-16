#!/usr/bin/env bash
# Run a scored CheatCode recollection over the first 30 Plan-D trials.
#
# This is intentionally a supervisor script: it launches the eval stack,
# recorder, CheatCode model, and bag cleanup watcher, then tears down helpers
# when the scored run finishes. Logs and pids are written under /tmp.

set -u

ROOT="/home/saivemu/code/aic"
CONFIG="$ROOT/aic_engine/config/random_trials_300_v2_first30.yaml"
RESULTS_DIR="/tmp/aic_plan_d_rerun_first30_results"
DATASET_ROOT="/tmp/aic_plan_d_rerun_first30_lerobot"
REPO_ID="saivemu/aic_plan_d_rerun_first30"
LOG_DIR="/tmp/aic_plan_d_rerun_first30_logs"
STATUS_FILE="$LOG_DIR/status.txt"
PID_FILE="$LOG_DIR/pids.env"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
rm -f "$STATUS_FILE" "$PID_FILE"

echo "started_at=$(date -Is)" | tee -a "$STATUS_FILE"
echo "config=$CONFIG" | tee -a "$STATUS_FILE"
echo "results_dir=$RESULTS_DIR" | tee -a "$STATUS_FILE"
echo "dataset_root=$DATASET_ROOT" | tee -a "$STATUS_FILE"

AIC_CTR=aic_eval AIC_RESULTS_DIR="$RESULTS_DIR" \
  "$ROOT/aic_utils/lerobot_robot_aic/scripts/cleanup_engine_bags.sh" "$RESULTS_DIR" \
  > "$LOG_DIR/cleanup.log" 2>&1 &
CLEANUP_PID=$!
echo "CLEANUP_PID=$CLEANUP_PID" >> "$PID_FILE"

env AIC_RESULTS_DIR="$RESULTS_DIR" \
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
echo "EVAL_PID=$EVAL_PID" >> "$PID_FILE"

sleep 3

pixi run python "$ROOT/aic_utils/lerobot_robot_aic/scripts/record_dataset.py" \
  --repo-id "$REPO_ID" \
  --root "$DATASET_ROOT" \
  --num-episodes 30 \
  --episode-idle-timeout 2.0 \
  --max-action-age 0.25 \
  --trials-config "$CONFIG" \
  --perturbing-topic "" \
  > "$LOG_DIR/recorder.log" 2>&1 &
RECORDER_PID=$!
echo "RECORDER_PID=$RECORDER_PID" >> "$PID_FILE"

sleep 3

pixi run env \
  PYTHONPATH="$ROOT/aic_example_policies:$ROOT/.pixi/envs/default/lib/python/site-packages:$ROOT/.pixi/envs/default/lib/python3.12/site-packages" \
  RMW_IMPLEMENTATION=rmw_zenoh_cpp \
  ZENOH_ROUTER_CHECK_ATTEMPTS=-1 \
  ZENOH_CONFIG_OVERRIDE=transport/shared_memory/enabled=false \
  ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCode \
  > "$LOG_DIR/model.log" 2>&1 &
MODEL_PID=$!
echo "MODEL_PID=$MODEL_PID" >> "$PID_FILE"

wait "$EVAL_PID"
EVAL_RC=$?

wait "$RECORDER_PID"
RECORDER_RC=$?

kill "$MODEL_PID" "$CLEANUP_PID" >/dev/null 2>&1 || true
wait "$MODEL_PID" >/dev/null 2>&1 || true
wait "$CLEANUP_PID" >/dev/null 2>&1 || true

echo "finished_at=$(date -Is)" | tee -a "$STATUS_FILE"
echo "eval_rc=$EVAL_RC" | tee -a "$STATUS_FILE"
echo "recorder_rc=$RECORDER_RC" | tee -a "$STATUS_FILE"
if [[ -f "$RESULTS_DIR/scoring.yaml" ]]; then
  echo "scoring_yaml=$RESULTS_DIR/scoring.yaml" | tee -a "$STATUS_FILE"
fi
if [[ -d "$DATASET_ROOT" ]]; then
  echo "dataset_root=$DATASET_ROOT" | tee -a "$STATUS_FILE"
fi

exit $(( EVAL_RC != 0 || RECORDER_RC != 0 ))
