#!/usr/bin/env bash
# Plan G-lite autopilot. Runs end-to-end once both pixel_delta trainings finish:
#   1. Compare pixel_to_base_xy calibration matrices (silent-foot-gun check)
#   2. Re-establish F-safe baseline (5 runs)
#   3. Build G-lite image, 5-run gate (gate: mean >= F-safe + 5, sigma < 2)
#   4. If G-lite is positive, build G-lite-control (old data, 60 ep) and 3-run
#      to attribute "wider distribution" vs "longer training"
#   5. Print final ship recommendation
#
# Outputs land in:
#   logs:    outputs/experiments/overnight_2026_05_14/logs/autopilot_*.log
#   summaries: outputs/experiments/overnight_2026_05_14/results/*_summary.txt
#   final:   outputs/experiments/overnight_2026_05_14/results/AUTOPILOT_REPORT.md

set -u

REPO=/home/saivemu/code/aic
EXP_DIR=$REPO/outputs/experiments/overnight_2026_05_14
LOG_DIR=$EXP_DIR/logs
RES_DIR=$EXP_DIR/results
REPORT=$RES_DIR/AUTOPILOT_REPORT.md
mkdir -p "$LOG_DIR" "$RES_DIR"
cd "$REPO"

NEW_HEAD=$REPO/outputs/experiments/vision_servo_labels/models/visual_pixel_delta_longrange40_o80_e60
OLD_HEAD_60=$REPO/outputs/experiments/vision_servo_labels/models/visual_pixel_delta_balanced20_o25_s2_e60

log() { echo "[$(date -Iseconds | cut -c12-19)] $*" | tee -a "$LOG_DIR/autopilot.log"; }

# ---------- Stage 0: wait for both trainings ----------
log "stage 0: waiting for both pixel_delta trainings to finish..."
while pgrep -f train_visual_servo.py >/dev/null; do
  sleep 60
done
log "trainings finished"

# Sanity check: both checkpoints exist
for d in "$NEW_HEAD" "$OLD_HEAD_60"; do
  if [[ ! -f "$d/best_visual_servo.pt" ]]; then
    log "FATAL: checkpoint missing at $d/best_visual_servo.pt — aborting"
    echo "FAIL: missing checkpoint" > "$REPORT"
    exit 1
  fi
done
log "checkpoints verified"

# ---------- Stage 1: compare calibration matrices ----------
log "stage 1: comparing pixel_to_base_xy matrices"
pixi run --as-is python outputs/experiments/overnight_2026_05_14/compare_pixel_to_base.py > "$LOG_DIR/autopilot_calib_compare.log" 2>&1 || true
cp "$LOG_DIR/autopilot_calib_compare.log" "$RES_DIR/calib_compare.txt"
if grep -q "WARN" "$LOG_DIR/autopilot_calib_compare.log"; then
  log "WARN: calibration matrices diverge significantly — check $RES_DIR/calib_compare.txt"
fi

# ---------- Stage 2: F-safe 5-run baseline ----------
log "stage 2: F-safe 5-run baseline"
LABEL=fsafe_baseline_v2 N_RUNS=5 \
  COMPOSE_OVERLAY=$EXP_DIR/docker-compose.use-assist-pixel-zstiff.yaml \
  ENV_VARS_FILE=$EXP_DIR/env_assist_pixel_zstiff.env \
  bash $EXP_DIR/run_compose_gate.sh > "$LOG_DIR/autopilot_fsafe.log" 2>&1
FSAFE_STATS=$(grep -oE "min=[0-9.]+ mean=[0-9.]+ max=[0-9.]+ n=[0-9]+" "$RES_DIR/fsafe_baseline_v2_summary.txt" | tail -1)
log "F-safe baseline: $FSAFE_STATS"
if [[ -z "$FSAFE_STATS" ]]; then
  log "FATAL: F-safe baseline did not produce a score — aborting"
  echo "FAIL: F-safe baseline empty" > "$REPORT"
  exit 1
fi
FSAFE_MEAN=$(echo "$FSAFE_STATS" | grep -oE "mean=[0-9.]+" | cut -d= -f2)
log "F-safe mean = $FSAFE_MEAN"

# ---------- Stage 3: build + eval G-lite ----------
log "stage 3: building aic-runact:glite-pixel-longrange-v1"
docker build -t aic-runact:glite-pixel-longrange-v1 \
  -f docker/aic_model/Dockerfile.glite_pixel_longrange . \
  > "$LOG_DIR/autopilot_build_glite.log" 2>&1
if [[ $? -ne 0 ]]; then
  log "FATAL: G-lite build failed"
  echo "FAIL: G-lite build" > "$REPORT"
  exit 1
fi
log "G-lite built"

# Create the use overlay
cat > "$EXP_DIR/docker-compose.use-glite-pixel-longrange.yaml" <<'YAML'
services:
  model:
    image: aic-runact:glite-pixel-longrange-v1
YAML

LABEL=glite_pixel_longrange N_RUNS=5 \
  COMPOSE_OVERLAY=$EXP_DIR/docker-compose.use-glite-pixel-longrange.yaml \
  ENV_VARS_FILE=$EXP_DIR/env_assist_pixel_zstiff.env \
  bash $EXP_DIR/run_compose_gate.sh > "$LOG_DIR/autopilot_glite.log" 2>&1
GLITE_STATS=$(grep -oE "min=[0-9.]+ mean=[0-9.]+ max=[0-9.]+ n=[0-9]+" "$RES_DIR/glite_pixel_longrange_summary.txt" | tail -1)
log "G-lite stats: $GLITE_STATS"
GLITE_MEAN=$(echo "$GLITE_STATS" | grep -oE "mean=[0-9.]+" | cut -d= -f2)

# Compute sigma manually
GLITE_SIGMA=$(grep -E "run [0-9]+: score=" "$RES_DIR/glite_pixel_longrange_summary.txt" \
  | grep -oE "score=[0-9.]+" | cut -d= -f2 \
  | awk -v m="$GLITE_MEAN" '{s+=($1-m)^2; n+=1} END {if (n>1) print sqrt(s/(n-1)); else print 0}')
log "G-lite sigma = $GLITE_SIGMA"

DELTA=$(awk -v a="$GLITE_MEAN" -v b="$FSAFE_MEAN" 'BEGIN{print a-b}')
log "G-lite vs F-safe delta = $DELTA"

# Ship gate: mean >= F-safe + 5, sigma < 2
SHIP_GATE=$(awk -v d="$DELTA" -v s="$GLITE_SIGMA" 'BEGIN{print (d>=5 && s<2)?"PASS":"FAIL"}')
INVESTIGATE_GATE=$(awk -v d="$DELTA" 'BEGIN{print (d>=1)?"PASS":"FAIL"}')
log "ship gate (delta>=5 sigma<2): $SHIP_GATE"
log "investigate gate (delta>=1): $INVESTIGATE_GATE"

# ---------- Stage 4: only run G-lite-control if G-lite is at least "investigate" ----------
CONTROL_STATS=""
CONTROL_MEAN=""
if [[ "$INVESTIGATE_GATE" == "PASS" ]]; then
  log "stage 4: building aic-runact:glite-control-old60-v1 (attribution control)"
  docker build -t aic-runact:glite-control-old60-v1 \
    -f docker/aic_model/Dockerfile.glite_control_old60 . \
    > "$LOG_DIR/autopilot_build_control.log" 2>&1
  if [[ $? -eq 0 ]]; then
    cat > "$EXP_DIR/docker-compose.use-glite-control-old60.yaml" <<'YAML'
services:
  model:
    image: aic-runact:glite-control-old60-v1
YAML
    LABEL=glite_control_old60 N_RUNS=3 \
      COMPOSE_OVERLAY=$EXP_DIR/docker-compose.use-glite-control-old60.yaml \
      ENV_VARS_FILE=$EXP_DIR/env_assist_pixel_zstiff.env \
      bash $EXP_DIR/run_compose_gate.sh > "$LOG_DIR/autopilot_control.log" 2>&1
    CONTROL_STATS=$(grep -oE "min=[0-9.]+ mean=[0-9.]+ max=[0-9.]+ n=[0-9]+" "$RES_DIR/glite_control_old60_summary.txt" | tail -1)
    CONTROL_MEAN=$(echo "$CONTROL_STATS" | grep -oE "mean=[0-9.]+" | cut -d= -f2)
    log "control stats: $CONTROL_STATS"
  else
    log "WARN: control image build failed; skipping attribution"
  fi
else
  log "stage 4: SKIPPED (G-lite did not clear investigate gate)"
fi

# ---------- Stage 5: write final report + auto-commit ----------
cat > "$REPORT" <<EOF
# Plan G-lite autopilot report — $(date -Iseconds)

## Baselines

| config | min | mean | max | n |
|---|---:|---:|---:|---:|
| F-safe (re-run baseline) | $FSAFE_STATS |
| **G-lite (NEW longrange40_o80 head)** | $GLITE_STATS |
$([[ -n "$CONTROL_STATS" ]] && echo "| G-lite-control (OLD-data 60ep head) | $CONTROL_STATS |")

G-lite − F-safe = **${DELTA}** (sigma=${GLITE_SIGMA})

## Gates

- Ship gate (delta>=5, sigma<2): **$SHIP_GATE**
- Investigate gate (delta>=1): **$INVESTIGATE_GATE**

## Calibration matrix check

See \`calib_compare.txt\`. Any "WARN" lines indicate sign flips or >3x divergence in the pixel_to_base_xy linear map between OLD and NEW heads.

## Recommendation

EOF

if [[ "$SHIP_GATE" == "PASS" ]]; then
  cat >> "$REPORT" <<EOF
**SHIP G-lite.** Tag and push to ECR:

\`\`\`bash
docker tag aic-runact:glite-pixel-longrange-v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/bot-squad-l2-learning-loop:glite-pixel-longrange-v1
aws --profile bot-squad-l2-learning-loop ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/bot-squad-l2-learning-loop:glite-pixel-longrange-v1
\`\`\`

Paste the URI into the submission portal.
EOF
elif [[ "$INVESTIGATE_GATE" == "PASS" ]]; then
  cat >> "$REPORT" <<EOF
**INVESTIGATE FURTHER, do not ship yet.** G-lite is above F-safe but does not meet the strict ship gate. Options:
- Run 5 more compose rounds to see if sigma tightens
- Tune ASSIST hyperparameters (xy speed cap, z-stiffness) on this head
- If control mean is close to G-lite, the gain is "longer training" not "wider data" — easier intervention
EOF
else
  cat >> "$REPORT" <<EOF
**FALL BACK TO F-SAFE for shipping.** The new long-range pixel_delta head did NOT outperform F-safe. The 50-90 mm port-localization gap is likely NOT the dominant bottleneck, OR pixel_delta architecture saturates on wider data. Next step: Plan H (re-record \`aic_act_v2++\` with wider offsets, retrain ACT/Diffusion directly).
EOF
fi

log "report written: $REPORT"

# Auto-commit + push
git add "$EXP_DIR/results/AUTOPILOT_REPORT.md" \
        "$EXP_DIR/results/calib_compare.txt" \
        "$EXP_DIR/results/fsafe_baseline_v2_summary.txt" \
        "$EXP_DIR/results/glite_pixel_longrange_summary.txt" 2>/dev/null
[[ -n "$CONTROL_STATS" ]] && git add "$EXP_DIR/results/glite_control_old60_summary.txt" 2>/dev/null
git add "$EXP_DIR/docker-compose.use-glite-pixel-longrange.yaml" 2>/dev/null
[[ -n "$CONTROL_STATS" ]] && git add "$EXP_DIR/docker-compose.use-glite-control-old60.yaml" 2>/dev/null
git commit -m "checkpoint: Plan G-lite autopilot results — $SHIP_GATE/$INVESTIGATE_GATE" 2>&1 | tee -a "$LOG_DIR/autopilot.log"
git push origin main 2>&1 | tee -a "$LOG_DIR/autopilot.log"

log "============================================"
log "  AUTOPILOT COMPLETE — $SHIP_GATE / $INVESTIGATE_GATE"
log "============================================"
log "Final report: $REPORT"
