#!/usr/bin/env bash
# Run a 3-round compose gate for an experiment label.
# Usage:
#   LABEL=t11_assist_c075 \
#   COMPOSE_OVERLAY=outputs/experiments/vision_servo_labels/docker-compose.eval-visual-direction.yaml \
#   ENV_VARS_FILE=outputs/experiments/overnight_2026_05_14/env_t11_assist_c075.env \
#   bash outputs/experiments/overnight_2026_05_14/run_compose_gate.sh
#
# Score is grepped from the eval container's `Total Score: %.3f` line.

set -u

LABEL="${LABEL:?LABEL is required}"
COMPOSE_OVERLAY="${COMPOSE_OVERLAY:-outputs/experiments/vision_servo_labels/docker-compose.eval-visual-direction.yaml}"
ENV_VARS_FILE="${ENV_VARS_FILE:-}"
N_RUNS="${N_RUNS:-3}"

REPO_ROOT="/home/saivemu/code/aic"
LOG_DIR="$REPO_ROOT/outputs/experiments/overnight_2026_05_14/logs"
RES_DIR="$REPO_ROOT/outputs/experiments/overnight_2026_05_14/results"
mkdir -p "$LOG_DIR" "$RES_DIR"
cd "$REPO_ROOT"

SUMMARY_FILE="$RES_DIR/${LABEL}_summary.txt"
echo "=== ${LABEL} :: $(date -Iseconds) ===" > "$SUMMARY_FILE"

if [[ -n "$ENV_VARS_FILE" && -f "$ENV_VARS_FILE" ]]; then
  echo "Env vars file: $ENV_VARS_FILE" >> "$SUMMARY_FILE"
  cat "$ENV_VARS_FILE" >> "$SUMMARY_FILE"
  echo "---" >> "$SUMMARY_FILE"
  # shellcheck disable=SC1090
  set -a; source "$ENV_VARS_FILE"; set +a
fi

scores=()
for i in $(seq 1 "$N_RUNS"); do
  log="$LOG_DIR/${LABEL}_run${i}.log"
  echo ">>> ${LABEL} run ${i}/${N_RUNS} starting at $(date -Iseconds)" | tee -a "$SUMMARY_FILE"

  # Ensure no stale containers
  docker compose \
    -f docker/docker-compose.yaml \
    -f docker/docker-compose.override.yaml \
    -f "$COMPOSE_OVERLAY" \
    down --remove-orphans >>"$log" 2>&1 || true

  # Run with a hard wall-clock timeout in case eval hangs (45 min max)
  timeout 2700 docker compose \
    -f docker/docker-compose.yaml \
    -f docker/docker-compose.override.yaml \
    -f "$COMPOSE_OVERLAY" \
    up --abort-on-container-exit --exit-code-from eval >>"$log" 2>&1
  ec=$?

  score=$(grep -oE "Total Score: [0-9]+\.[0-9]+" "$log" | tail -1 | awk '{print $3}')
  trial_breakdown=$(grep -E "(trial[_:].*tier_|Trial [0-9].*Score)" "$log" | tail -20 || true)
  if [[ -z "$score" ]]; then
    echo "  run ${i}: NO_SCORE (exit=$ec) — see $log" | tee -a "$SUMMARY_FILE"
    score="NA"
  else
    echo "  run ${i}: score=${score} (exit=$ec)" | tee -a "$SUMMARY_FILE"
  fi
  scores+=("$score")

  docker compose \
    -f docker/docker-compose.yaml \
    -f docker/docker-compose.override.yaml \
    -f "$COMPOSE_OVERLAY" \
    down --remove-orphans >>"$log" 2>&1 || true
done

echo "---" >> "$SUMMARY_FILE"
echo "all_scores: ${scores[*]}" >> "$SUMMARY_FILE"

# Compute min / mean / max if all numeric
nums_ok=true
for s in "${scores[@]}"; do
  [[ "$s" =~ ^[0-9]+\.[0-9]+$ ]] || nums_ok=false
done
if [[ "$nums_ok" == "true" ]]; then
  awk_input=$(IFS=$'\n'; echo "${scores[*]}")
  echo "$awk_input" | awk '
    NR==1 {mn=mx=$1; s=0; n=0}
    {s+=$1; n+=1; if($1<mn)mn=$1; if($1>mx)mx=$1}
    END {printf "min=%.3f mean=%.3f max=%.3f n=%d\n", mn, s/n, mx, n}
  ' | tee -a "$SUMMARY_FILE"
fi

echo "=== ${LABEL} done :: $(date -Iseconds) ===" | tee -a "$SUMMARY_FILE"
