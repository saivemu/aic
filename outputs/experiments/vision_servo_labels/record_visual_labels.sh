set -euo pipefail

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/eval:7447"];transport/shared_memory/enabled=false'

pixi run --as-is python /tmp/record_visual_servo_dataset.py \
  --root "${AIC_VISUAL_LABEL_ROOT}" \
  --num-episodes "${AIC_VISUAL_LABEL_EPISODES}" \
  --trials-config /tmp/aic_random_trials.yaml \
  --episode-idle-timeout 2.0 \
  --max-action-age 0.25 \
  --min-frame-index "${AIC_VISUAL_LABEL_MIN_FRAME_INDEX:-120}" \
  --sample-every "${AIC_VISUAL_LABEL_SAMPLE_EVERY:-2}" \
  --image-scale "${AIC_VISUAL_LABEL_IMAGE_SCALE:-0.5}" \
  --cameras "${AIC_VISUAL_LABEL_CAMERAS:-center}" \
  --require-visible "${AIC_VISUAL_LABEL_REQUIRE_VISIBLE:-center}" \
  --jpeg-quality "${AIC_VISUAL_LABEL_JPEG_QUALITY:-90}" \
  --max-labels-per-episode "${AIC_VISUAL_LABEL_MAX_LABELS_PER_EPISODE:-0}" \
  --overwrite
